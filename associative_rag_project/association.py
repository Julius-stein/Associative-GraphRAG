"""Graph association logic.

The association stage is explicitly treated as a divergence stage.

Current design uses two structural relations:
- graph adjacency over entity-relation edges
- provenance adjacency over chunk <-> node/edge links, plus neighboring chunks

On top of these two structures, the code performs two kinds of association:
- structural association: reward candidates that create new bridges
- semantic/coverage association: reward candidates that introduce new evidence

This yields a practical 2 x 2 scheme:
- graph-bridge association
- chunk-bridge association
- graph-coverage association
- chunk-coverage association

关联阶段依据图结构与 provenance 关系扩展证据，兼顾桥接能力与覆盖新信息。
"""

import math
from collections import Counter, defaultdict

import networkx as nx

from .common import edge_key, lexical_overlap_score, normalize_text, tokenize
from .retrieval import (
    _query_focus_terms,
    normalize_relation_category,
    score_root_edges,
    score_root_nodes,
    select_diverse_root_chunks,
    support_score,
)


def _chunk_doc_ids(chunk_ids, chunk_store):
    return {
        chunk_store.get(chunk_id, {}).get("full_doc_id")
        for chunk_id in chunk_ids
        if chunk_store.get(chunk_id, {}).get("full_doc_id")
    }


def _chunk_preview_text(chunk_ids, chunk_store, limit=2):
    parts = []
    for chunk_id in list(chunk_ids)[:limit]:
        content = normalize_text(chunk_store.get(chunk_id, {}).get("content", ""))
        if content:
            parts.append(content[:180])
    return " ".join(parts)


def _edge_preview_text(edge_id, graph):
    edge_data = graph.get_edge_data(edge_id[0], edge_id[1]) or {}
    return normalize_text(
        " ".join(
            [
                edge_id[0],
                edge_id[1],
                edge_data.get("keywords", ""),
                edge_data.get("description", ""),
            ]
        )
    )


def _node_preview_text(node_id, graph):
    node_data = graph.nodes[node_id]
    return normalize_text(
        " ".join(
            [
                node_id,
                node_data.get("entity_type", ""),
                node_data.get("description", ""),
            ]
        )
    )


def _candidate_contract_features(query, query_contract, source_chunk_ids, preview_text, root_chunk_ids, root_chunk_score_lookup, chunk_store):
    source_docs = _chunk_doc_ids(source_chunk_ids, chunk_store)
    root_docs = _chunk_doc_ids(root_chunk_ids, chunk_store)
    _, root_alignment = support_score(source_chunk_ids, root_chunk_score_lookup)
    query_overlap = lexical_overlap_score(query, preview_text)
    same_root_doc = 1 if source_docs and root_docs and source_docs & root_docs else 0
    section_consistent = 1 if query_contract != "section-grounded" else int(bool(source_docs) and source_docs.issubset(root_docs))
    if query_contract == "section-grounded":
        allowed = bool(section_consistent)
    elif query_contract == "mechanism-grounded":
        allowed = bool(same_root_doc or root_alignment >= 0.08 or query_overlap >= 0.035)
    elif query_contract == "comparison-grounded":
        allowed = bool(same_root_doc or root_alignment >= 0.05 or query_overlap >= 0.025)
    else:
        allowed = bool(same_root_doc or root_alignment >= 0.03 or query_overlap >= 0.02)
    return {
        "allowed": allowed,
        "source_docs": sorted(source_docs),
        "root_docs": sorted(root_docs),
        "query_overlap": round(query_overlap, 6),
        "root_alignment": round(root_alignment, 6),
        "same_root_doc": same_root_doc,
        "section_consistent": section_consistent,
    }


def _contract_sort_key(query_contract, item, primary_fields):
    if query_contract == "section-grounded":
        return (
            -item.get("section_consistent", 0),
            -item.get("same_root_doc", 0),
            -item.get("root_alignment", 0.0),
            -item.get("query_overlap", 0.0),
            *primary_fields,
        )
    if query_contract == "mechanism-grounded":
        return (
            *primary_fields,
            -item.get("query_overlap", 0.0),
            -item.get("same_root_doc", 0),
            -item.get("root_alignment", 0.0),
        )
    if query_contract == "comparison-grounded":
        return (
            *primary_fields,
            -item.get("same_root_doc", 0),
            -item.get("query_overlap", 0.0),
            -item.get("root_alignment", 0.0),
        )
    return (
        *primary_fields,
        -item.get("query_overlap", 0.0),
        -item.get("same_root_doc", 0),
        -item.get("root_alignment", 0.0),
    )


def build_root_components(root_nodes, root_edges):
    """Materialize connected components for the current root graph.

    参数:
        root_nodes: 当前根节点列表。
        root_edges: 当前根边列表。

    返回:
        三元组 (root_graph, components, node_to_component)，分别为根子图、连通组件列表和节点到组件索引映射。
    """
    root_graph = nx.Graph()
    root_graph.add_nodes_from(root_nodes)
    root_graph.add_edges_from(root_edges)
    components = list(nx.connected_components(root_graph))
    node_to_component = {}
    for component_index, nodes in enumerate(components):
        for node_id in nodes:
            node_to_component[node_id] = component_index
    return root_graph, components, node_to_component


def _chunk_union_nodes_edges(chunk_ids, chunk_to_nodes, chunk_to_edges):
    node_ids = set()
    edge_ids = set()
    for chunk_id in chunk_ids:
        node_ids.update(chunk_to_nodes.get(chunk_id, set()))
        edge_ids.update(chunk_to_edges.get(chunk_id, set()))
    return node_ids, edge_ids


def _covered_chunk_ids(current_nodes, current_edges, node_to_chunks, edge_to_chunks):
    covered = set()
    for node_id in current_nodes:
        covered.update(node_to_chunks.get(node_id, set()))
    for edge_id in current_edges:
        covered.update(edge_to_chunks.get(edge_id, set()))
    return covered


def _path_chunk_ids(path, node_to_chunks, edge_to_chunks):
    chunk_ids = set()
    for node_id in path:
        chunk_ids.update(node_to_chunks.get(node_id, set()))
    for index in range(len(path) - 1):
        chunk_ids.update(edge_to_chunks.get(edge_key(path[index], path[index + 1]), set()))
    return chunk_ids


def _chunk_structural_neighbors(chunk_id, chunk_to_nodes, chunk_to_edges, node_to_chunks, edge_to_chunks, chunk_neighbors):
    neighbors = set(chunk_neighbors.get(chunk_id, set()))
    for node_id in chunk_to_nodes.get(chunk_id, set()):
        neighbors.update(node_to_chunks.get(node_id, set()))
    neighbors.discard(chunk_id)
    return neighbors


def _chunk_provenance_overlap(chunk_id_a, chunk_id_b, chunk_to_nodes, chunk_to_edges):
    left = set(chunk_to_nodes.get(chunk_id_a, set())) | set(chunk_to_edges.get(chunk_id_a, set()))
    right = set(chunk_to_nodes.get(chunk_id_b, set())) | set(chunk_to_edges.get(chunk_id_b, set()))
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left | right), 1)


def _inverse_sqrt_degree(size):
    return 1.0 / math.sqrt(max(float(size), 1.0))


def _weighted_node_mass(node_ids, node_to_chunks):
    return sum(_inverse_sqrt_degree(len(node_to_chunks.get(node_id, set()))) for node_id in node_ids)


def _weighted_edge_mass(edge_ids, edge_to_chunks):
    return sum(_inverse_sqrt_degree(len(edge_to_chunks.get(edge_id, set()))) for edge_id in edge_ids)


def _chunk_structure_penalty(chunk_id, chunk_to_nodes, chunk_to_edges, node_to_chunks, edge_to_chunks):
    node_ids = set(chunk_to_nodes.get(chunk_id, set()))
    edge_ids = set(chunk_to_edges.get(chunk_id, set()))
    raw_mass = len(node_ids) + len(edge_ids)
    if raw_mass <= 0:
        return 1.0
    weighted_mass = _weighted_node_mass(node_ids, node_to_chunks) + _weighted_edge_mass(edge_ids, edge_to_chunks)
    degree_term = weighted_mass / max(raw_mass, 1)
    size_term = 1.0 / math.sqrt(1.0 + (raw_mass / 6.0))
    return max(0.18, min(1.0, degree_term * size_term))


def _chunk_link_features(chunk_id, anchor_chunk_ids, chunk_store, chunk_to_nodes, chunk_to_edges, node_to_chunks, edge_to_chunks, chunk_neighbors):
    if not anchor_chunk_ids:
        return {
            "linked_chunk_count": 0,
            "linked_strength": 0.0,
            "distinct_doc_touch": 0,
            "shared_node_count": 0,
            "shared_edge_count": 0,
            "same_doc_band": 0,
        }
    chunk_nodes = set(chunk_to_nodes.get(chunk_id, set()))
    chunk_edges = set(chunk_to_edges.get(chunk_id, set()))
    linked_chunk_count = 0
    linked_strength = 0.0
    touched_docs = set()
    shared_node_count = 0
    shared_edge_count = 0
    same_doc_band = 0
    for other_chunk_id in anchor_chunk_ids:
        other_nodes = set(chunk_to_nodes.get(other_chunk_id, set()))
        other_edges = set(chunk_to_edges.get(other_chunk_id, set()))
        shared_nodes = len(chunk_nodes & other_nodes)
        shared_edges = len(chunk_edges & other_edges)
        neighbor_touch = int(
            other_chunk_id in chunk_neighbors.get(chunk_id, set())
            or chunk_id in chunk_neighbors.get(other_chunk_id, set())
        )
        if shared_nodes > 0 or shared_edges > 0 or neighbor_touch:
            linked_chunk_count += 1
            linked_strength += _weighted_node_mass(chunk_nodes & other_nodes, node_to_chunks)
            linked_strength += _weighted_edge_mass(chunk_edges & other_edges, edge_to_chunks)
            if neighbor_touch:
                linked_strength += 0.35
            other_doc = chunk_store.get(other_chunk_id, {}).get("full_doc_id")
            if other_doc:
                touched_docs.add(other_doc)
        shared_node_count += shared_nodes
        shared_edge_count += shared_edges
        if _chunk_same_doc_band(chunk_store, chunk_id, other_chunk_id):
            same_doc_band += 1
    return {
        "linked_chunk_count": linked_chunk_count,
        "linked_strength": round(linked_strength, 6),
        "distinct_doc_touch": len(touched_docs),
        "shared_node_count": shared_node_count,
        "shared_edge_count": shared_edge_count,
        "same_doc_band": same_doc_band,
    }


def _rank_theme_bridge_chunks(
    query,
    graph,
    frontier_chunk_ids,
    selected_chunk_ids,
    query_chunk_score_lookup,
    chunk_store,
    chunk_to_nodes,
    chunk_to_edges,
    node_to_chunks,
    edge_to_chunks,
    chunk_neighbors,
    limit,
):
    covered_nodes, covered_edges = _chunk_union_nodes_edges(selected_chunk_ids, chunk_to_nodes, chunk_to_edges)
    scored = []
    for chunk_id in frontier_chunk_ids:
        if chunk_id in selected_chunk_ids:
            continue
        structure_penalty = _chunk_structure_penalty(
            chunk_id,
            chunk_to_nodes,
            chunk_to_edges,
            node_to_chunks,
            edge_to_chunks,
        )
        features = _chunk_link_features(
            chunk_id,
            selected_chunk_ids,
            chunk_store,
            chunk_to_nodes,
            chunk_to_edges,
            node_to_chunks,
            edge_to_chunks,
            chunk_neighbors,
        )
        if features["linked_chunk_count"] <= 0:
            continue
        candidate_nodes = set(chunk_to_nodes.get(chunk_id, set()))
        candidate_edges = set(chunk_to_edges.get(chunk_id, set()))
        introduced_nodes = candidate_nodes - covered_nodes
        introduced_edges = candidate_edges - covered_edges
        introduced_node_query_rel = sum(
            lexical_overlap_score(query, normalize_text(str(node_id)))
            * _inverse_sqrt_degree(len(node_to_chunks.get(node_id, set())))
            for node_id in introduced_nodes
        )
        introduced_edge_query_rel = 0.0
        for edge_id in introduced_edges:
            edge_data = graph.get_edge_data(edge_id[0], edge_id[1]) or {}
            edge_text = " ".join(
                [
                    normalize_text(str(edge_id[0])),
                    normalize_text(str(edge_id[1])),
                    normalize_text(edge_data.get("relation", "")),
                    normalize_text(edge_data.get("keywords", "")),
                ]
            ).strip()
            if edge_text:
                introduced_edge_query_rel += (
                    lexical_overlap_score(query, edge_text)
                    * _inverse_sqrt_degree(len(edge_to_chunks.get(edge_id, set())))
                )
        introduced_query_rel = (introduced_node_query_rel + introduced_edge_query_rel) * structure_penalty
        if introduced_query_rel <= 0:
            continue
        query_alignment = float(query_chunk_score_lookup.get(chunk_id, 0.0))
        scored.append(
            {
                "chunk_id": chunk_id,
                "frontier_touch": round(features["linked_strength"] * structure_penalty, 6),
                "linked_chunk_count": features["linked_chunk_count"],
                "structure_penalty": round(structure_penalty, 6),
                "introduced_query_rel": round(introduced_query_rel, 6),
                "introduced_node_query_rel": round(introduced_node_query_rel, 6),
                "introduced_edge_query_rel": round(introduced_edge_query_rel, 6),
                "introduced_node_count": len(introduced_nodes),
                "introduced_edge_count": len(introduced_edges),
                "introduced_node_ids": sorted(introduced_nodes),
                "introduced_edge_ids": sorted(introduced_edges),
                "query_alignment": round(query_alignment, 6),
                "node_ids": sorted(chunk_to_nodes.get(chunk_id, set())),
                "edge_ids": sorted(chunk_to_edges.get(chunk_id, set())),
            }
        )
    scored.sort(
        key=lambda item: (
            -item["frontier_touch"],
            -item["introduced_query_rel"],
            item["chunk_id"],
        )
    )
    return scored[:limit]


def _rank_theme_support_chunks(
    query,
    graph,
    candidate_chunk_ids,
    selected_chunk_ids,
    query_chunk_score_lookup,
    chunk_store,
    chunk_to_nodes,
    chunk_to_edges,
    node_to_chunks,
    edge_to_chunks,
    chunk_neighbors,
    limit,
):
    covered_nodes, covered_edges = _chunk_union_nodes_edges(selected_chunk_ids, chunk_to_nodes, chunk_to_edges)
    covered_relation_types = set()
    for edge_id in covered_edges:
        edge_data = graph.get_edge_data(edge_id[0], edge_id[1]) if graph is not None and graph.has_edge(*edge_id) else {}
        covered_relation_types.add(normalize_relation_category(edge_data or {}))
    scored = []
    for chunk_id in candidate_chunk_ids:
        if chunk_id in selected_chunk_ids:
            continue
        structure_penalty = _chunk_structure_penalty(
            chunk_id,
            chunk_to_nodes,
            chunk_to_edges,
            node_to_chunks,
            edge_to_chunks,
        )
        chunk_text = normalize_text(chunk_store.get(chunk_id, {}).get("content", ""))
        query_alignment = max(float(query_chunk_score_lookup.get(chunk_id, 0.0)), lexical_overlap_score(query, chunk_text))
        if query_alignment <= 0:
            continue
        features = _chunk_link_features(
            chunk_id,
            selected_chunk_ids,
            chunk_store,
            chunk_to_nodes,
            chunk_to_edges,
            node_to_chunks,
            edge_to_chunks,
            chunk_neighbors,
        )
        if features["linked_chunk_count"] <= 0 and features["same_doc_band"] <= 0:
            continue
        candidate_nodes = set(chunk_to_nodes.get(chunk_id, set()))
        candidate_edges = set(chunk_to_edges.get(chunk_id, set()))
        introduced_nodes = candidate_nodes - covered_nodes
        introduced_edges = candidate_edges - covered_edges
        if not introduced_nodes and not introduced_edges:
            continue
        introduced_relation_types = set()
        for edge_id in introduced_edges:
            edge_data = graph.get_edge_data(edge_id[0], edge_id[1]) if graph is not None and graph.has_edge(*edge_id) else {}
            introduced_relation_types.add(normalize_relation_category(edge_data or {}))
        new_relation_count = len(introduced_relation_types - covered_relation_types)
        introduced_mass = (
            _weighted_node_mass(introduced_nodes, node_to_chunks)
            + _weighted_edge_mass(introduced_edges, edge_to_chunks)
        ) * structure_penalty
        scored.append(
            {
                "chunk_id": chunk_id,
                "query_alignment": round(query_alignment, 6),
                "introduced_relation_count": new_relation_count,
                "introduced_node_count": len(introduced_nodes),
                "introduced_edge_count": len(introduced_edges),
                "introduced_mass": round(introduced_mass, 6),
                "frontier_touch": round(features["linked_strength"] * structure_penalty, 6),
                "linked_chunk_count": features["linked_chunk_count"],
                "structure_penalty": round(structure_penalty, 6),
                "node_ids": sorted(chunk_to_nodes.get(chunk_id, set())),
                "edge_ids": sorted(chunk_to_edges.get(chunk_id, set())),
            }
        )
    scored.sort(
        key=lambda item: (
            -item["introduced_relation_count"],
            -item["introduced_mass"],
            -item["query_alignment"],
            -item["frontier_touch"],
            item["chunk_id"],
        )
    )
    return scored[:limit]


def _select_diverse_support_chunks(
    scored_candidates,
    selected_chunk_ids,
    chunk_store,
    chunk_to_nodes,
    chunk_to_edges,
    limit,
):
    chosen = []
    for item in scored_candidates:
        if len(chosen) >= limit:
            break
        chunk_id = item["chunk_id"]
        max_overlap = 0.0
        for other_id in list(selected_chunk_ids) + [row["chunk_id"] for row in chosen]:
            overlap = _chunk_provenance_overlap(
                chunk_id,
                other_id,
                chunk_to_nodes,
                chunk_to_edges,
            )
            if overlap > max_overlap:
                max_overlap = overlap
        if max_overlap > 0.72:
            continue
        chosen.append({**item, "max_support_overlap": round(max_overlap, 6)})
    return chosen


def _rank_theme_peripheral_chunks(
    seed_chunk_ids,
    selected_chunk_ids,
    query_chunk_score_lookup,
    chunk_store,
    chunk_to_nodes,
    chunk_to_edges,
    node_to_chunks,
    edge_to_chunks,
    chunk_neighbors,
    limit,
):
    candidates = set()
    for chunk_id in seed_chunk_ids:
        candidates.update(chunk_neighbors.get(chunk_id, set()))
    scored = []
    for chunk_id in candidates:
        if chunk_id in selected_chunk_ids:
            continue
        structure_penalty = _chunk_structure_penalty(
            chunk_id,
            chunk_to_nodes,
            chunk_to_edges,
            node_to_chunks,
            edge_to_chunks,
        )
        features = _chunk_link_features(
            chunk_id,
            seed_chunk_ids,
            chunk_store,
            chunk_to_nodes,
            chunk_to_edges,
            node_to_chunks,
            edge_to_chunks,
            chunk_neighbors,
        )
        if features["linked_chunk_count"] <= 0:
            continue
        query_alignment = float(query_chunk_score_lookup.get(chunk_id, 0.0))
        scored.append(
            {
                "chunk_id": chunk_id,
                "query_alignment": round(query_alignment, 6),
                "frontier_touch": round(features["linked_strength"] * structure_penalty, 6),
                "linked_chunk_count": features["linked_chunk_count"],
                "structure_penalty": round(structure_penalty, 6),
                "node_ids": sorted(chunk_to_nodes.get(chunk_id, set())),
                "edge_ids": sorted(chunk_to_edges.get(chunk_id, set())),
            }
        )
    scored.sort(
        key=lambda item: (
            -item["frontier_touch"],
            -item["query_alignment"],
            item["chunk_id"],
        )
    )
    return scored[:limit]


def _trace_round_limits(root_count, association_rounds, path_budget, semantic_edge_budget, semantic_node_budget):
    denominator = max(root_count * max(association_rounds, 1), 1)
    bridge_limit = max(1, math.ceil(path_budget / denominator))
    support_limit = max(1, math.ceil(semantic_edge_budget / denominator))
    context_limit = max(1, math.ceil(semantic_node_budget / max(denominator * 2, 1)))
    return bridge_limit, support_limit, context_limit


def _trace_frontier(
    trace_chunk_ids,
    chunk_to_nodes,
    chunk_to_edges,
    node_to_chunks,
    edge_to_chunks,
    chunk_neighbors,
    allowed_doc_ids,
    chunk_store,
):
    frontier = set()
    for chunk_id in trace_chunk_ids:
        frontier.update(
            _filter_allowed_chunks(
                _chunk_structural_neighbors(
                    chunk_id,
                    chunk_to_nodes,
                    chunk_to_edges,
                    node_to_chunks,
                    edge_to_chunks,
                    chunk_neighbors,
                ),
                allowed_doc_ids,
                chunk_store,
            )
        )
    frontier.difference_update(trace_chunk_ids)
    return frontier


def _top_trace_candidates(candidate_chunk_ids, query_chunk_score_lookup, limit):
    return set(
        chunk_id
        for chunk_id, _ in sorted(
            ((chunk_id, float(query_chunk_score_lookup.get(chunk_id, 0.0))) for chunk_id in candidate_chunk_ids),
            key=lambda item: (-item[1], item[0]),
        )[:limit]
    )


def _chunk_query_term_hits(
    chunk_id,
    query_terms,
    graph,
    chunk_store,
    chunk_to_nodes,
    chunk_to_edges,
):
    return set(
        _theme_chunk_query_terms(
            chunk_id,
            query_terms,
            graph,
            chunk_store,
            chunk_to_nodes,
            chunk_to_edges,
        )
    )


def _refresh_trace_query_scores(
    *,
    trace,
    query_terms,
    graph,
    query_chunk_score_lookup,
    chunk_store,
    chunk_to_nodes,
    chunk_to_edges,
    node_to_chunks,
    edge_to_chunks,
    chunk_neighbors,
    allowed_doc_ids,
):
    if not query_terms:
        return {"updated_count": 0, "preview": []}
    covered_terms = set()
    for chunk_id in trace.get("selected_chunk_ids", []):
        covered_terms.update(
            _chunk_query_term_hits(
                chunk_id,
                query_terms,
                graph,
                chunk_store,
                chunk_to_nodes,
                chunk_to_edges,
            )
        )
    seed_chunk_ids = (
        trace.get("bridge_chunk_ids", [])[-2:]
        + trace.get("support_chunk_ids", [])[-2:]
        + trace.get("context_chunk_ids", [])[-1:]
    )
    refresh_pool = set(seed_chunk_ids)
    refresh_pool.update(
        _trace_frontier(
            set(seed_chunk_ids) | set(trace.get("selected_chunk_ids", [])),
            chunk_to_nodes,
            chunk_to_edges,
            node_to_chunks,
            edge_to_chunks,
            chunk_neighbors,
            allowed_doc_ids,
            chunk_store,
        )
    )
    refresh_pool = _top_trace_candidates(refresh_pool, query_chunk_score_lookup, limit=120)
    changed = []
    for chunk_id in sorted(refresh_pool):
        term_hits = _chunk_query_term_hits(
            chunk_id,
            query_terms,
            graph,
            chunk_store,
            chunk_to_nodes,
            chunk_to_edges,
        )
        if not term_hits:
            continue
        uncovered_hits = term_hits - covered_terms
        old_score = float(query_chunk_score_lookup.get(chunk_id, 0.0))
        refreshed_score = max(
            old_score,
            len(term_hits) / max(len(query_terms), 1),
            len(uncovered_hits) / max(len(query_terms), 1),
        )
        if refreshed_score > old_score + 1e-9:
            query_chunk_score_lookup[chunk_id] = round(min(1.0, refreshed_score), 6)
            changed.append(
                {
                    "chunk_id": chunk_id,
                    "old": round(old_score, 6),
                    "new": round(query_chunk_score_lookup[chunk_id], 6),
                }
            )
    return {"updated_count": len(changed), "preview": changed[:5]}


def _reseed_trace_root(
    *,
    trace,
    query,
    query_terms,
    graph,
    query_chunk_score_lookup,
    chunk_store,
    chunk_to_nodes,
    chunk_to_edges,
    node_to_chunks,
    edge_to_chunks,
    chunk_neighbors,
    allowed_doc_ids,
):
    covered_terms = set()
    for chunk_id in trace.get("selected_chunk_ids", []):
        covered_terms.update(
            _chunk_query_term_hits(
                chunk_id,
                query_terms,
                graph,
                chunk_store,
                chunk_to_nodes,
                chunk_to_edges,
            )
        )
    current_root_ids = set(trace.get("active_root_chunk_ids", []))
    round_seed_chunk_ids = (
        trace.get("bridge_chunk_ids", [])[-3:]
        + trace.get("support_chunk_ids", [])[-3:]
        + trace.get("context_chunk_ids", [])[-1:]
    )
    candidate_chunk_ids = _trace_frontier(
        set(round_seed_chunk_ids) | set(trace.get("selected_chunk_ids", [])),
        chunk_to_nodes,
        chunk_to_edges,
        node_to_chunks,
        edge_to_chunks,
        chunk_neighbors,
        allowed_doc_ids,
        chunk_store,
    )
    candidate_chunk_ids.update(round_seed_chunk_ids)
    candidate_chunk_ids = _top_trace_candidates(candidate_chunk_ids, query_chunk_score_lookup, limit=120)
    best_item = None
    best_key = None
    for chunk_id in sorted(candidate_chunk_ids):
        if chunk_id in current_root_ids:
            continue
        term_hits = _chunk_query_term_hits(
            chunk_id,
            query_terms,
            graph,
            chunk_store,
            chunk_to_nodes,
            chunk_to_edges,
        )
        if not term_hits:
            continue
        uncovered_hits = term_hits - covered_terms
        if not uncovered_hits:
            continue
        link_features = _chunk_link_features(
            chunk_id,
            set(trace.get("selected_chunk_ids", [])),
            chunk_store,
            chunk_to_nodes,
            chunk_to_edges,
            node_to_chunks,
            edge_to_chunks,
            chunk_neighbors,
        )
        if link_features["linked_chunk_count"] <= 0 and link_features["same_doc_band"] <= 0:
            continue
        structure_penalty = _chunk_structure_penalty(
            chunk_id,
            chunk_to_nodes,
            chunk_to_edges,
            node_to_chunks,
            edge_to_chunks,
        )
        query_alignment = float(query_chunk_score_lookup.get(chunk_id, 0.0))
        doc_novelty = int(
            bool(
                chunk_store.get(chunk_id, {}).get("full_doc_id")
                and chunk_store.get(chunk_id, {}).get("full_doc_id")
                not in _chunk_doc_ids(trace.get("active_root_chunk_ids", []), chunk_store)
            )
        )
        sort_key = (
            len(uncovered_hits),
            link_features.get("linked_strength", 0.0) * structure_penalty,
            len(term_hits),
            doc_novelty,
            query_alignment,
            chunk_id,
        )
        if best_key is None or sort_key > best_key:
            best_key = sort_key
            best_item = {
                "chunk_id": chunk_id,
                "query_alignment": round(query_alignment, 6),
                "new_query_terms": sorted(uncovered_hits),
                "all_query_terms": sorted(term_hits),
                "linked_chunk_count": link_features["linked_chunk_count"],
                "linked_strength": round(link_features.get("linked_strength", 0.0) * structure_penalty, 6),
                "structure_penalty": round(structure_penalty, 6),
                "doc_novelty": doc_novelty,
            }
    return best_item


def _blend_query_chunk_score(score_lookup, chunk_id, signal, alpha=0.38):
    """Blend one crawl signal into query relevance while keeping monotonicity."""
    if signal <= 0:
        return float(score_lookup.get(chunk_id, 0.0))
    old_score = float(score_lookup.get(chunk_id, 0.0))
    bounded_signal = max(0.0, min(1.0, float(signal)))
    if old_score <= 0:
        new_score = bounded_signal
    else:
        new_score = max(old_score, (1.0 - alpha) * old_score + alpha * bounded_signal)
    score_lookup[chunk_id] = round(min(1.0, new_score), 6)
    return score_lookup[chunk_id]


def _update_theme_query_scores_from_round(
    *,
    bridge_items,
    support_items,
    peripheral_items,
    query_chunk_score_lookup,
    candidate_chunk_ids,
    chunk_to_nodes,
    chunk_to_edges,
    node_to_chunks,
    edge_to_chunks,
    chunk_neighbors,
):
    """Update chunk-query relevance after each theme round.

    This makes theme retrieval behave like progressive graph crawling:
    newly introduced bridge/support evidence boosts nearby chunks for the next
    round instead of keeping query relevance frozen.
    """
    updates = {}
    seed_items = []
    for item in bridge_items:
        introduced = min(float(item.get("introduced_query_rel", 0.0)) / 6.0, 1.0)
        seed_signal = max(float(item.get("query_alignment", 0.0)), introduced, 0.1)
        seed_items.append((item.get("chunk_id"), seed_signal))
    for item in support_items:
        seed_signal = max(float(item.get("query_alignment", 0.0)), 0.08)
        seed_items.append((item.get("chunk_id"), seed_signal))
    for item in peripheral_items:
        seed_signal = max(float(item.get("query_alignment", 0.0)), 0.06)
        seed_items.append((item.get("chunk_id"), seed_signal))

    for chunk_id, seed_signal in seed_items:
        if not chunk_id:
            continue
        updates[chunk_id] = max(updates.get(chunk_id, 0.0), seed_signal)

        for neighbor_chunk_id in _chunk_structural_neighbors(
            chunk_id,
            chunk_to_nodes,
            chunk_to_edges,
            node_to_chunks,
            edge_to_chunks,
            chunk_neighbors,
        ):
            updates[neighbor_chunk_id] = max(updates.get(neighbor_chunk_id, 0.0), seed_signal * 0.72)

        shared_counter = Counter()
        for node_id in chunk_to_nodes.get(chunk_id, set()):
            for neighbor_chunk_id in node_to_chunks.get(node_id, set()):
                if neighbor_chunk_id != chunk_id:
                    shared_counter[neighbor_chunk_id] += 1
        for edge_id in chunk_to_edges.get(chunk_id, set()):
            for neighbor_chunk_id in edge_to_chunks.get(edge_id, set()):
                if neighbor_chunk_id != chunk_id:
                    shared_counter[neighbor_chunk_id] += 1
        for neighbor_chunk_id, overlap_count in shared_counter.most_common(80):
            overlap_signal = seed_signal * min(0.65, 0.45 + 0.05 * min(overlap_count, 4))
            updates[neighbor_chunk_id] = max(updates.get(neighbor_chunk_id, 0.0), overlap_signal)

    changed = []
    for chunk_id, signal in sorted(updates.items(), key=lambda item: (-item[1], item[0]))[:420]:
        old_score = float(query_chunk_score_lookup.get(chunk_id, 0.0))
        new_score = _blend_query_chunk_score(query_chunk_score_lookup, chunk_id, signal)
        if new_score > old_score + 1e-9:
            changed.append((chunk_id, old_score, new_score))
        candidate_chunk_ids.add(chunk_id)
        candidate_chunk_ids.update(chunk_neighbors.get(chunk_id, set()))

    return {
        "updated_count": len(changed),
        "preview": [
            {"chunk_id": chunk_id, "old": round(old, 6), "new": round(new, 6)}
            for chunk_id, old, new in changed[:5]
        ],
    }


def _theme_chunk_query_terms(
    chunk_id,
    query_terms,
    graph,
    chunk_store,
    chunk_to_nodes,
    chunk_to_edges,
):
    if not query_terms:
        return []
    observed_terms = set(tokenize(normalize_text(chunk_store.get(chunk_id, {}).get("content", ""))))
    for node_id in chunk_to_nodes.get(chunk_id, set()):
        observed_terms.update(tokenize(normalize_text(str(node_id))))
        if graph is not None and graph.has_node(node_id):
            node_data = graph.nodes[node_id] or {}
            observed_terms.update(tokenize(normalize_text(node_data.get("entity_type", ""))))
            observed_terms.update(tokenize(normalize_text(node_data.get("description", ""))))
    for edge_id in chunk_to_edges.get(chunk_id, set()):
        observed_terms.update(tokenize(normalize_text(str(edge_id[0]))))
        observed_terms.update(tokenize(normalize_text(str(edge_id[1]))))
        edge_data = graph.get_edge_data(edge_id[0], edge_id[1]) if graph is not None else {}
        observed_terms.update(tokenize(normalize_text(normalize_relation_category(edge_data or {}))))
        observed_terms.update(tokenize(normalize_text((edge_data or {}).get("keywords", ""))))
    return sorted(query_terms & observed_terms)[:8]


def _build_theme_reseed_candidate_hits(
    *,
    query,
    graph,
    pool_chunk_ids,
    favored_chunk_ids,
    active_root_chunk_ids,
    seen_root_chunk_ids,
    local_query_chunk_score_lookup,
    chunk_store,
    chunk_to_nodes,
    chunk_to_edges,
    chunk_neighbors,
):
    query_terms = _query_focus_terms(query)
    active_root_doc_ids = _chunk_doc_ids(active_root_chunk_ids, chunk_store)
    candidate_hits = []
    for chunk_id in sorted(pool_chunk_ids):
        base_score = float(local_query_chunk_score_lookup.get(chunk_id, 0.0))
        if base_score <= 0:
            continue
        chunk = chunk_store.get(chunk_id, {})
        same_doc_neighbors = 0
        for active_root_id in active_root_chunk_ids:
            if _chunk_same_doc_band(chunk_store, chunk_id, active_root_id):
                same_doc_neighbors += 1
        query_terms_hit = _theme_chunk_query_terms(
            chunk_id,
            query_terms,
            graph,
            chunk_store,
            chunk_to_nodes,
            chunk_to_edges,
        )
        favored_bonus = 0.0
        if chunk_id in favored_chunk_ids:
            favored_bonus += 0.12
        if chunk_id not in seen_root_chunk_ids:
            favored_bonus += 0.08
        if chunk.get("full_doc_id") and chunk.get("full_doc_id") not in active_root_doc_ids:
            favored_bonus += 0.05
        if same_doc_neighbors <= 0:
            favored_bonus += 0.03
        local_density = len(chunk_neighbors.get(chunk_id, set()))
        candidate_hits.append(
            {
                "chunk_id": chunk_id,
                "score_norm": round(min(1.0, base_score + favored_bonus), 6),
                "retrieval_score": round(min(1.0, base_score + favored_bonus), 6),
                "graph_focus_score_norm": round(base_score, 6),
                "graph_focus_hit_terms": query_terms_hit,
                "dense_score_norm": round(base_score, 6) if base_score >= 0.18 else 0.0,
                "full_doc_id": chunk.get("full_doc_id"),
                "chunk_order_index": chunk.get("chunk_order_index", -1),
                "local_density": local_density,
            }
        )
    return candidate_hits


def _reseed_theme_root_chunks(
    *,
    query,
    graph,
    selected_chunk_ids,
    round_chunk_ids,
    candidate_chunk_ids,
    active_root_chunk_ids,
    seen_root_chunk_ids,
    local_query_chunk_score_lookup,
    chunk_store,
    chunk_to_nodes,
    chunk_to_edges,
    chunk_neighbors,
    top_k,
):
    seed_pool = set(candidate_chunk_ids) | set(selected_chunk_ids)
    for chunk_id in list(round_chunk_ids):
        seed_pool.add(chunk_id)
        seed_pool.update(chunk_neighbors.get(chunk_id, set()))
    candidate_hits = _build_theme_reseed_candidate_hits(
        query=query,
        graph=graph,
        pool_chunk_ids=seed_pool,
        favored_chunk_ids=set(round_chunk_ids),
        active_root_chunk_ids=active_root_chunk_ids,
        seen_root_chunk_ids=seen_root_chunk_ids,
        local_query_chunk_score_lookup=local_query_chunk_score_lookup,
        chunk_store=chunk_store,
        chunk_to_nodes=chunk_to_nodes,
        chunk_to_edges=chunk_to_edges,
        chunk_neighbors=chunk_neighbors,
    )
    if not candidate_hits:
        return []

    reseeded = select_diverse_root_chunks(
        query=query,
        candidate_hits=candidate_hits,
        chunk_store=chunk_store,
        chunk_to_nodes=chunk_to_nodes,
        chunk_to_edges=chunk_to_edges,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        top_k=top_k,
        graph=graph,
    )
    if not reseeded:
        return []

    novel = [item for item in reseeded if item["chunk_id"] not in seen_root_chunk_ids]
    if len(novel) >= top_k:
        return novel[:top_k]
    stable = [item for item in reseeded if item["chunk_id"] in seen_root_chunk_ids]
    return (novel + stable)[:top_k]


def _expand_theme_chunk_graph(
    *,
    query,
    graph,
    root_chunk_ids,
    root_chunk_score_lookup,
    query_chunk_score_lookup,
    chunk_to_nodes,
    chunk_to_edges,
    node_to_chunks,
    edge_to_chunks,
    chunk_neighbors,
    chunk_store,
    association_rounds,
    path_budget,
    semantic_edge_budget,
    semantic_node_budget,
    allowed_doc_ids=None,
):
    root_chunk_ids = list(
        _filter_allowed_chunks(root_chunk_ids, allowed_doc_ids, chunk_store)
    )
    query_terms = _query_focus_terms(query)
    local_query_chunk_score_lookup = {
        chunk_id: score
        for chunk_id, score in query_chunk_score_lookup.items()
        if (not allowed_doc_ids) or (chunk_store.get(chunk_id, {}).get("full_doc_id") in allowed_doc_ids)
    }
    bridge_limit, support_limit, context_limit = _trace_round_limits(
        len(root_chunk_ids),
        association_rounds,
        path_budget,
        semantic_edge_budget,
        semantic_node_budget,
    )

    trace_map = {}
    for root_chunk_id in root_chunk_ids:
        trace_map[root_chunk_id] = {
            "root_chunk_id": root_chunk_id,
            "selected_chunk_ids": [root_chunk_id],
            "active_root_chunk_ids": [root_chunk_id],
            "promoted_root_chunk_ids": [],
            "bridge_chunk_ids": [],
            "support_chunk_ids": [],
            "context_chunk_ids": [],
            "rounds": [],
        }

    rounds = []
    bridge_records = []
    support_records = []
    context_records = []

    for round_index in range(1, max(association_rounds, 1) + 1):
        round_bridge_items = []
        round_support_items = []
        round_context_items = []

        for root_chunk_id in root_chunk_ids:
            trace = trace_map[root_chunk_id]
            trace_selected_set = set(trace["selected_chunk_ids"])
            active_root_chunk_ids = set(trace.get("active_root_chunk_ids", [])) or {root_chunk_id}
            frontier_chunk_ids = _trace_frontier(
                active_root_chunk_ids,
                chunk_to_nodes,
                chunk_to_edges,
                node_to_chunks,
                edge_to_chunks,
                chunk_neighbors,
                allowed_doc_ids,
                chunk_store,
            )
            frontier_chunk_ids = _top_trace_candidates(frontier_chunk_ids, local_query_chunk_score_lookup, limit=120)
            bridge_this_round = _rank_theme_bridge_chunks(
                query=query,
                graph=graph,
                frontier_chunk_ids=frontier_chunk_ids,
                selected_chunk_ids=trace_selected_set,
                query_chunk_score_lookup=local_query_chunk_score_lookup,
                chunk_store=chunk_store,
                chunk_to_nodes=chunk_to_nodes,
                chunk_to_edges=chunk_to_edges,
                node_to_chunks=node_to_chunks,
                edge_to_chunks=edge_to_chunks,
                chunk_neighbors=chunk_neighbors,
                limit=bridge_limit,
            )
            selected_plus_bridge = trace_selected_set | {item["chunk_id"] for item in bridge_this_round}
            support_frontier = _trace_frontier(
                selected_plus_bridge,
                chunk_to_nodes,
                chunk_to_edges,
                node_to_chunks,
                edge_to_chunks,
                chunk_neighbors,
                allowed_doc_ids,
                chunk_store,
            )
            support_frontier = _top_trace_candidates(support_frontier, local_query_chunk_score_lookup, limit=160)
            support_candidates = _rank_theme_support_chunks(
                query=query,
                graph=graph,
                candidate_chunk_ids=support_frontier,
                selected_chunk_ids=selected_plus_bridge,
                query_chunk_score_lookup=local_query_chunk_score_lookup,
                chunk_store=chunk_store,
                chunk_to_nodes=chunk_to_nodes,
                chunk_to_edges=chunk_to_edges,
                node_to_chunks=node_to_chunks,
                edge_to_chunks=edge_to_chunks,
                chunk_neighbors=chunk_neighbors,
                limit=max(support_limit * 3, support_limit),
            )
            support_this_round = _select_diverse_support_chunks(
                scored_candidates=support_candidates,
                selected_chunk_ids=selected_plus_bridge,
                chunk_store=chunk_store,
                chunk_to_nodes=chunk_to_nodes,
                chunk_to_edges=chunk_to_edges,
                limit=support_limit,
            )
            selected_plus_support = selected_plus_bridge | {item["chunk_id"] for item in support_this_round}
            context_this_round = _rank_theme_peripheral_chunks(
                seed_chunk_ids=[item["chunk_id"] for item in bridge_this_round + support_this_round],
                selected_chunk_ids=selected_plus_support,
                query_chunk_score_lookup=local_query_chunk_score_lookup,
                chunk_store=chunk_store,
                chunk_to_nodes=chunk_to_nodes,
                chunk_to_edges=chunk_to_edges,
                node_to_chunks=node_to_chunks,
                edge_to_chunks=edge_to_chunks,
                chunk_neighbors=chunk_neighbors,
                limit=context_limit,
            )

            for item in bridge_this_round:
                if item["chunk_id"] not in trace["selected_chunk_ids"]:
                    trace["selected_chunk_ids"].append(item["chunk_id"])
                if item["chunk_id"] not in trace["bridge_chunk_ids"]:
                    trace["bridge_chunk_ids"].append(item["chunk_id"])
                round_bridge_items.append(
                    {
                        **item,
                        "chunk_role": "bridge",
                        "root_chunk_id": root_chunk_id,
                    }
                )
            for item in support_this_round:
                if item["chunk_id"] not in trace["selected_chunk_ids"]:
                    trace["selected_chunk_ids"].append(item["chunk_id"])
                if item["chunk_id"] not in trace["support_chunk_ids"]:
                    trace["support_chunk_ids"].append(item["chunk_id"])
                round_support_items.append(
                    {
                        **item,
                        "chunk_role": "support",
                        "root_chunk_id": root_chunk_id,
                    }
                )
            for item in context_this_round:
                if item["chunk_id"] not in trace["selected_chunk_ids"]:
                    trace["selected_chunk_ids"].append(item["chunk_id"])
                if item["chunk_id"] not in trace["context_chunk_ids"]:
                    trace["context_chunk_ids"].append(item["chunk_id"])
                round_context_items.append(
                    {
                        **item,
                        "chunk_role": "context",
                        "root_chunk_id": root_chunk_id,
                    }
                )

            score_update = _refresh_trace_query_scores(
                trace=trace,
                query_terms=query_terms,
                graph=graph,
                query_chunk_score_lookup=local_query_chunk_score_lookup,
                chunk_store=chunk_store,
                chunk_to_nodes=chunk_to_nodes,
                chunk_to_edges=chunk_to_edges,
                node_to_chunks=node_to_chunks,
                edge_to_chunks=edge_to_chunks,
                chunk_neighbors=chunk_neighbors,
                allowed_doc_ids=allowed_doc_ids,
            )
            next_root = _reseed_trace_root(
                trace=trace,
                query=query,
                query_terms=query_terms,
                graph=graph,
                query_chunk_score_lookup=local_query_chunk_score_lookup,
                chunk_store=chunk_store,
                chunk_to_nodes=chunk_to_nodes,
                chunk_to_edges=chunk_to_edges,
                node_to_chunks=node_to_chunks,
                edge_to_chunks=edge_to_chunks,
                chunk_neighbors=chunk_neighbors,
                allowed_doc_ids=allowed_doc_ids,
            )
            promoted_this_round = []
            if next_root is not None:
                trace["active_root_chunk_ids"] = [next_root["chunk_id"]]
                if next_root["chunk_id"] not in trace["selected_chunk_ids"]:
                    trace["selected_chunk_ids"].append(next_root["chunk_id"])
                if next_root["chunk_id"] not in trace["promoted_root_chunk_ids"]:
                    trace["promoted_root_chunk_ids"].append(next_root["chunk_id"])
                promoted_this_round.append(next_root)
            else:
                trace["active_root_chunk_ids"] = list(active_root_chunk_ids)

            trace["rounds"].append(
                {
                    "round": round_index,
                    "bridge_chunk_ids": [item["chunk_id"] for item in bridge_this_round],
                    "support_chunk_ids": [item["chunk_id"] for item in support_this_round],
                    "context_chunk_ids": [item["chunk_id"] for item in context_this_round],
                    "promoted_root_chunk_ids": [item["chunk_id"] for item in promoted_this_round],
                    "query_score_update_count": score_update["updated_count"],
                    "query_score_update_preview": score_update["preview"],
                }
            )

        bridge_records.extend(round_bridge_items)
        support_records.extend(round_support_items)
        context_records.extend(round_context_items)

        structural_chunk_ids = [item["chunk_id"] for item in round_bridge_items + round_context_items]
        semantic_chunk_ids = [item["chunk_id"] for item in round_support_items]
        structural_nodes, structural_edges = _chunk_union_nodes_edges(structural_chunk_ids, chunk_to_nodes, chunk_to_edges)
        semantic_nodes, semantic_edges = _chunk_union_nodes_edges(semantic_chunk_ids, chunk_to_nodes, chunk_to_edges)
        current_selected_chunk_ids = {
            chunk_id
            for trace in trace_map.values()
            for chunk_id in trace["selected_chunk_ids"]
        }
        current_nodes, current_edges = _chunk_union_nodes_edges(
            current_selected_chunk_ids,
            chunk_to_nodes,
            chunk_to_edges,
        )
        rounds.append(
            {
                "round": round_index,
                "structural_path_count": 0,
                "structural_chunk_bridge_count": len(round_bridge_items) + len(round_context_items),
                "structural_added_node_count": len(structural_nodes),
                "structural_added_edge_count": len(structural_edges),
                "semantic_added_node_count": len(semantic_nodes),
                "semantic_added_edge_count": len(semantic_edges),
                "semantic_chunk_coverage_count": len(round_support_items),
                "current_node_count": len(current_nodes),
                "current_edge_count": len(current_edges),
                "promoted_root_count": 0,
                "promoted_root_node_count": 0,
                "promoted_root_edge_count": 0,
                "structural_path_preview": [],
                "structural_chunk_preview": round_bridge_items[:3] + round_context_items[:2],
                "semantic_edge_preview": [],
                "semantic_node_preview": [],
                "semantic_chunk_preview": round_support_items[:3],
                "promoted_root_preview": [
                    {
                        "root_chunk_id": root_chunk_id,
                        "chunk_id": trace_map[root_chunk_id].get("active_root_chunk_ids", [root_chunk_id])[0],
                    }
                    for root_chunk_id in root_chunk_ids[:4]
                ],
                "query_score_update_count": sum(
                    trace_round.get("query_score_update_count", 0)
                    for trace_round in (
                        trace_map[root_chunk_id]["rounds"][-1]
                        for root_chunk_id in root_chunk_ids
                        if trace_map[root_chunk_id]["rounds"]
                    )
                ),
                "query_score_update_preview": [
                    item
                    for root_chunk_id in root_chunk_ids
                    for item in trace_map[root_chunk_id]["rounds"][-1].get("query_score_update_preview", [])[:1]
                ][:5],
                "active_root_preview": [
                    trace_map[root_chunk_id].get("active_root_chunk_ids", [root_chunk_id])[0]
                    for root_chunk_id in root_chunk_ids[:4]
                ],
            }
        )
        if not round_bridge_items and not round_support_items and not round_context_items:
            break

    final_selected_chunk_ids = {
        chunk_id
        for trace in trace_map.values()
        for chunk_id in trace["selected_chunk_ids"]
    }
    final_nodes, final_edges = _chunk_union_nodes_edges(final_selected_chunk_ids, chunk_to_nodes, chunk_to_edges)
    structural_chunk_ids = [item["chunk_id"] for item in bridge_records + context_records]
    semantic_chunk_ids = [item["chunk_id"] for item in support_records]
    structural_nodes, structural_edges = _chunk_union_nodes_edges(structural_chunk_ids, chunk_to_nodes, chunk_to_edges)
    semantic_nodes, semantic_edges = _chunk_union_nodes_edges(semantic_chunk_ids, chunk_to_nodes, chunk_to_edges)

    root_traces = []
    promoted_root_chunks = []
    for root_chunk_id in root_chunk_ids:
        trace = trace_map[root_chunk_id]
        trace_nodes, trace_edges = _chunk_union_nodes_edges(
            trace["selected_chunk_ids"],
            chunk_to_nodes,
            chunk_to_edges,
        )
        root_traces.append(
            {
                **trace,
                "node_ids": sorted(trace_nodes),
                "edge_ids": sorted(trace_edges),
            }
        )
        for promoted_chunk_id in trace.get("promoted_root_chunk_ids", []):
            promoted_root_chunks.append(
                {
                    "chunk_id": promoted_chunk_id,
                    "score_norm": round(float(local_query_chunk_score_lookup.get(promoted_chunk_id, 0.0)), 6),
                    "selection_score": round(float(local_query_chunk_score_lookup.get(promoted_chunk_id, 0.0)), 6),
                    "query_alignment": round(float(local_query_chunk_score_lookup.get(promoted_chunk_id, 0.0)), 6),
                    "root_role": "structural",
                }
            )

    last_structural_output = {
        "selected_chunks": bridge_records + context_records,
        "selected_paths": [],
        "selected_path_nodes": [],
        "selected_path_edges": [],
        "root_traces": root_traces,
    }
    return {
        "final_nodes": sorted(final_nodes),
        "final_edges": sorted(final_edges),
        "all_structural_nodes": sorted(structural_nodes),
        "all_structural_edges": sorted(structural_edges),
        "all_semantic_nodes": sorted(semantic_nodes),
        "all_semantic_edges": [{"edge": edge_id, "coverage_gain": 1.0} for edge_id in sorted(semantic_edges)],
        "promoted_root_chunks": promoted_root_chunks,
        "effective_root_chunk_ids": list(dict.fromkeys(root_chunk_ids + [item["chunk_id"] for item in promoted_root_chunks])),
        "effective_root_chunk_score_lookup": {
            chunk_id: max(
                root_chunk_score_lookup.get(chunk_id, 0.0),
                float(local_query_chunk_score_lookup.get(chunk_id, 0.0)),
            )
            for chunk_id in list(dict.fromkeys(root_chunk_ids + [item["chunk_id"] for item in promoted_root_chunks]))
        },
        "effective_query_chunk_score_lookup": local_query_chunk_score_lookup,
        "rounds": rounds,
        "last_structural_output": last_structural_output,
        "root_traces": root_traces,
        "theme_selected_chunks": {
            "core": list(dict.fromkeys(root_chunk_ids + [item["chunk_id"] for item in promoted_root_chunks])),
            "bridge": [item["chunk_id"] for item in bridge_records],
            "support": [item["chunk_id"] for item in support_records],
            "context": [item["chunk_id"] for item in context_records],
        },
    }


def _chunk_doc_ids(chunk_ids, chunk_store):
    return {
        chunk_store.get(chunk_id, {}).get("full_doc_id")
        for chunk_id in chunk_ids
        if chunk_store.get(chunk_id, {}).get("full_doc_id")
    }


def _path_hub_penalty(path, graph):
    internal = path[1:-1]
    if not internal:
        return 0.0
    penalties = [math.log1p(max(graph.degree(node_id), 1)) for node_id in internal]
    return sum(penalties) / max(len(penalties), 1)


def _path_specificity_gain(path, graph):
    internal = path[1:-1]
    if not internal:
        return 0.0
    values = [1.0 / (1.0 + math.log1p(max(graph.degree(node_id), 1))) for node_id in internal]
    return sum(values) / max(len(values), 1)


def _bounded_gain(value, cap=5):
    return min(value, cap)


def _path_bridge_signature(path, node_to_component):
    components = {
        node_to_component.get(path[0]),
        node_to_component.get(path[-1]),
    }
    return len({item for item in components if item is not None})


def _graph_bridge_association(
    query,
    query_contract,
    graph,
    current_nodes,
    current_edges,
    root_chunk_ids,
    root_chunk_score_lookup,
    top_nodes,
    top_edges,
    node_to_chunks,
    edge_to_chunks,
    chunk_store,
    max_hop,
    path_budget,
):
    """Expand by graph paths that connect disconnected evidence regions.

    Primary objective:
    - bridge separate current components

    Tie-break objectives:
    - introduce more new supporting chunks
    - prefer shorter paths
    """
    if not current_nodes:
        return {
            "current_components": [],
            "selected_paths": [],
            "selected_path_nodes": [],
            "selected_path_edges": [],
        }

    _, components, node_to_component = build_root_components(current_nodes, current_edges)
    current_edge_set = set(current_edges)
    covered_chunks = _covered_chunk_ids(current_nodes, current_edges, node_to_chunks, edge_to_chunks)
    current_doc_ids = _chunk_doc_ids(covered_chunks, chunk_store)

    seed_nodes = []
    seen = set()
    for item in top_nodes:
        if item["id"] not in seen:
            seed_nodes.append(item["id"])
            seen.add(item["id"])
    for item in top_edges:
        for node_id in item["edge"]:
            if node_id not in seen:
                seed_nodes.append(node_id)
                seen.add(node_id)

    candidate_paths = []
    dedupe_paths = set()
    for source_id in seed_nodes:
        source_component = node_to_component.get(source_id)
        paths = nx.single_source_shortest_path(graph, source_id, cutoff=max_hop)
        for target_id in current_nodes:
            if target_id == source_id:
                continue
            target_component = node_to_component.get(target_id)
            if source_component is None or target_component is None or source_component == target_component:
                continue
            path = paths.get(target_id)
            if path is None or len(path) <= 1:
                continue
            canonical = tuple(path) if tuple(path) < tuple(reversed(path)) else tuple(reversed(path))
            if canonical in dedupe_paths:
                continue
            path_edges = {edge_key(path[index], path[index + 1]) for index in range(len(path) - 1)}
            if path_edges.issubset(current_edge_set):
                continue
            dedupe_paths.add(canonical)
            path_chunks = _path_chunk_ids(path, node_to_chunks, edge_to_chunks)
            preview_text = " ".join(
                [
                    normalize_text(" ".join(path[:6])),
                    _chunk_preview_text(path_chunks, chunk_store),
                ]
            )
            contract_features = _candidate_contract_features(
                query=query,
                query_contract=query_contract,
                source_chunk_ids=path_chunks,
                preview_text=preview_text,
                root_chunk_ids=root_chunk_ids,
                root_chunk_score_lookup=root_chunk_score_lookup,
                chunk_store=chunk_store,
            )
            if not contract_features["allowed"]:
                continue
            new_source_count = _bounded_gain(len(path_chunks - covered_chunks))
            path_doc_gain = len(_chunk_doc_ids(path_chunks, chunk_store) - current_doc_ids)
            hub_penalty = _path_hub_penalty(path, graph)
            specificity_gain = _path_specificity_gain(path, graph)
            candidate_paths.append(
                {
                    "path": path,
                    "bridge_gain": _path_bridge_signature(path, node_to_component),
                    "new_source_count": new_source_count,
                    "path_doc_gain": path_doc_gain,
                    "hub_penalty": round(hub_penalty, 6),
                    "specificity_gain": round(specificity_gain, 6),
                    "path_length": len(path) - 1,
                    **contract_features,
                }
            )

    candidate_paths.sort(
        key=lambda item: (
            -item["bridge_gain"],
            -item["path_doc_gain"],
            -item["specificity_gain"],
            item["hub_penalty"],
            -item["new_source_count"],
            item["path_length"],
            item["path"],
        )
    )
    selected_paths = candidate_paths[:path_budget]
    selected_nodes = set()
    selected_edges = set()
    for item in selected_paths:
        selected_nodes.update(item["path"])
        selected_edges.update(edge_key(item["path"][index], item["path"][index + 1]) for index in range(len(item["path"]) - 1))
    return {
        "current_components": [sorted(component) for component in components],
        "selected_paths": selected_paths,
        "selected_path_nodes": sorted(selected_nodes),
        "selected_path_edges": sorted(selected_edges),
    }


def _chunk_component_ids(chunk_id, current_nodes, chunk_to_nodes, chunk_to_edges, node_to_component):
    component_ids = set()
    for node_id in chunk_to_nodes.get(chunk_id, set()):
        if node_id in current_nodes and node_id in node_to_component:
            component_ids.add(node_to_component[node_id])
    for edge_id in chunk_to_edges.get(chunk_id, set()):
        for node_id in edge_id:
            if node_id in current_nodes and node_id in node_to_component:
                component_ids.add(node_to_component[node_id])
    return component_ids


def _chunk_bridge_touch(chunk_id, current_nodes, current_edges, chunk_to_nodes, chunk_to_edges, chunk_neighbors, covered_band):
    touched = 0
    touched += len(set(chunk_to_nodes.get(chunk_id, set())) & set(current_nodes))
    touched += len(set(chunk_to_edges.get(chunk_id, set())) & set(current_edges))
    touched += len(set(chunk_neighbors.get(chunk_id, set())) & set(covered_band))
    return touched


def _chunk_bridge_association(
    graph,
    current_nodes,
    current_edges,
    root_chunk_ids,
    root_chunk_score_lookup,
    chunk_to_nodes,
    chunk_to_edges,
    node_to_chunks,
    edge_to_chunks,
    chunk_neighbors,
    chunk_store,
    chunk_budget,
):
    """Expand by chunk-side structural adjacency.

    A candidate chunk is considered a structural bridge when it is reachable via
    chunk neighborhoods/provenance and helps connect multiple currently active
    components or strengthens the current frontier with new chunk-supported
    graph content.
    """
    if chunk_budget <= 0:
        return {
            "selected_chunks": [],
            "selected_chunk_nodes": [],
            "selected_chunk_edges": [],
        }

    _, _, node_to_component = build_root_components(current_nodes, current_edges)
    covered_chunks = _covered_chunk_ids(current_nodes, current_edges, node_to_chunks, edge_to_chunks)
    covered_band = _expand_chunk_band(covered_chunks, chunk_neighbors)
    covered_doc_ids = _chunk_doc_ids(covered_band, chunk_store)
    candidate_chunks = set(covered_band)
    for chunk_id in list(covered_band):
        candidate_chunks.update(chunk_neighbors.get(chunk_id, set()))
    candidate_chunks -= covered_chunks

    scored_chunks = []
    current_node_set = set(current_nodes)
    current_edge_set = set(current_edges)
    root_chunk_id_set = set(root_chunk_ids)
    for chunk_id in candidate_chunks:
        chunk_nodes = set(chunk_to_nodes.get(chunk_id, set()))
        chunk_edges = set(chunk_to_edges.get(chunk_id, set()))
        new_nodes = chunk_nodes - current_node_set
        new_edges = chunk_edges - current_edge_set
        if not new_nodes and not new_edges:
            continue
        component_ids = _chunk_component_ids(
            chunk_id=chunk_id,
            current_nodes=current_nodes,
            chunk_to_nodes=chunk_to_nodes,
            chunk_to_edges=chunk_to_edges,
            node_to_component=node_to_component,
        )
        bridge_gain = 1 if len(component_ids) >= 2 else 0
        frontier_touch = _bounded_gain(
            _chunk_bridge_touch(
                chunk_id=chunk_id,
                current_nodes=current_nodes,
                current_edges=current_edges,
                chunk_to_nodes=chunk_to_nodes,
                chunk_to_edges=chunk_to_edges,
                chunk_neighbors=chunk_neighbors,
                covered_band=covered_band,
            )
        )
        new_source_count = _bounded_gain(len(_expand_chunk_band({chunk_id}, chunk_neighbors) - covered_band))
        doc_gain = len(_chunk_doc_ids({chunk_id}, chunk_store) - covered_doc_ids)
        root_overlap = 1 if chunk_id in root_chunk_id_set else 0
        root_band_alignment = max(
            [root_chunk_score_lookup.get(neighbor_id, 0.0) for neighbor_id in _expand_chunk_band({chunk_id}, chunk_neighbors)],
            default=0.0,
        )
        specificity_gain = 0.0
        if graph is not None:
            attached_nodes = sorted(chunk_nodes)
            if attached_nodes:
                values = [1.0 / (1.0 + math.log1p(max(graph.degree(node_id), 1))) for node_id in attached_nodes]
                specificity_gain = sum(values) / max(len(values), 1)
        scored_chunks.append(
            {
                "chunk_id": chunk_id,
                "bridge_gain": bridge_gain,
                "frontier_touch": frontier_touch,
                "new_source_count": new_source_count,
                "doc_gain": doc_gain,
                "specificity_gain": round(specificity_gain, 6),
                "new_node_count": len(new_nodes),
                "new_edge_count": len(new_edges),
                "root_overlap": root_overlap,
                "root_band_alignment": round(root_band_alignment, 6),
                "node_ids": sorted(chunk_nodes),
                "edge_ids": sorted(chunk_edges),
                **contract_features,
            }
        )

    scored_chunks.sort(
        key=lambda item: (
            -item["bridge_gain"],
            -item["doc_gain"],
            -item["specificity_gain"],
            -item["frontier_touch"],
            -item["new_source_count"],
            -(item["new_node_count"] + item["new_edge_count"]),
            -item["root_overlap"],
            -item["root_band_alignment"],
            item["chunk_id"],
        )
    )
    selected_chunks = scored_chunks[:chunk_budget]
    selected_nodes = set()
    selected_edges = set()
    for item in selected_chunks:
        selected_nodes.update(item["node_ids"])
        selected_edges.update(item["edge_ids"])
    return {
        "selected_chunks": selected_chunks,
        "selected_chunk_nodes": sorted(selected_nodes),
        "selected_chunk_edges": sorted(selected_edges),
    }


def bridge_association(
    query,
    query_contract,
    graph,
    current_nodes,
    current_edges,
    root_chunk_ids,
    root_chunk_score_lookup,
    top_nodes,
    top_edges,
    chunk_to_nodes,
    chunk_to_edges,
    node_to_chunks,
    edge_to_chunks,
    chunk_neighbors,
    chunk_store,
    max_hop,
    path_budget,
):
    """Expand structurally using both graph paths and chunk-side bridges."""
    graph_bridge_budget = max(1, path_budget // 2)
    chunk_bridge_budget = max(1, path_budget - graph_bridge_budget)
    graph_output = _graph_bridge_association(
        query=query,
        query_contract=query_contract,
        graph=graph,
        current_nodes=current_nodes,
        current_edges=current_edges,
        root_chunk_ids=root_chunk_ids,
        root_chunk_score_lookup=root_chunk_score_lookup,
        top_nodes=top_nodes,
        top_edges=top_edges,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        chunk_store=chunk_store,
        max_hop=max_hop,
        path_budget=graph_bridge_budget,
    )
    chunk_output = _chunk_bridge_association(
        graph=graph,
        current_nodes=current_nodes,
        current_edges=current_edges,
        root_chunk_ids=root_chunk_ids,
        root_chunk_score_lookup=root_chunk_score_lookup,
        chunk_to_nodes=chunk_to_nodes,
        chunk_to_edges=chunk_to_edges,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        chunk_neighbors=chunk_neighbors,
        chunk_store=chunk_store,
        chunk_budget=chunk_bridge_budget,
    )
    selected_nodes = set(graph_output["selected_path_nodes"]) | set(chunk_output["selected_chunk_nodes"])
    selected_edges = set(graph_output["selected_path_edges"]) | set(chunk_output["selected_chunk_edges"])
    return {
        "current_components": graph_output["current_components"],
        "selected_paths": graph_output["selected_paths"],
        "selected_path_nodes": sorted(selected_nodes),
        "selected_path_edges": sorted(selected_edges),
        "selected_chunks": chunk_output["selected_chunks"],
        "selected_chunk_nodes": chunk_output["selected_chunk_nodes"],
        "selected_chunk_edges": chunk_output["selected_chunk_edges"],
    }


def build_current_relation_categories(edge_ids, graph):
    """Collect coarse relation labels from the current edge set."""
    categories = []
    for edge_id in edge_ids:
        edge_data = graph.get_edge_data(edge_id[0], edge_id[1]) or {}
        categories.append(normalize_relation_category(edge_data))
    return categories


def _expand_chunk_band(chunk_ids, chunk_neighbors):
    expanded = set(chunk_ids)
    for chunk_id in list(chunk_ids):
        expanded.update(chunk_neighbors.get(chunk_id, set()))
    return expanded


def _filter_allowed_chunks(chunk_ids, allowed_doc_ids, chunk_store):
    if not allowed_doc_ids:
        return set(chunk_ids)
    return {
        chunk_id
        for chunk_id in chunk_ids
        if chunk_store.get(chunk_id, {}).get("full_doc_id") in allowed_doc_ids
    }


def _query_band_alignment(chunk_ids, query_chunk_score_lookup, chunk_neighbors):
    if not query_chunk_score_lookup or not chunk_ids:
        return 0.0
    expanded = _expand_chunk_band(set(chunk_ids), chunk_neighbors)
    return max((query_chunk_score_lookup.get(chunk_id, 0.0) for chunk_id in expanded), default=0.0)


def _edge_query_alignment(edge_id, query, graph):
    edge_data = graph.get_edge_data(edge_id[0], edge_id[1]) or {}
    edge_text = " ".join(
        [
            normalize_text(edge_id[0]),
            normalize_text(edge_id[1]),
            normalize_text(edge_data.get("keywords", "")),
            normalize_text(edge_data.get("description", "")),
        ]
    )
    return lexical_overlap_score(query, edge_text)


def _node_query_alignment(node_id, query, graph):
    node_data = graph.nodes[node_id]
    node_text = " ".join(
        [
            normalize_text(node_id),
            normalize_text(node_data.get("entity_type", "")),
            normalize_text(node_data.get("description", "")),
        ]
    )
    return lexical_overlap_score(query, node_text)


def _chunk_same_doc_band(chunk_store, left_chunk_id, right_chunk_id, window=1):
    left = chunk_store.get(left_chunk_id, {})
    right = chunk_store.get(right_chunk_id, {})
    if left.get("full_doc_id") != right.get("full_doc_id"):
        return False
    left_order = left.get("chunk_order_index")
    right_order = right.get("chunk_order_index")
    if left_order is None or right_order is None:
        return False
    return abs(left_order - right_order) <= window


def _chunk_provenance_overlap(left_chunk_id, right_chunk_id, chunk_to_nodes, chunk_to_edges):
    left = set(chunk_to_nodes.get(left_chunk_id, set())) | set(chunk_to_edges.get(left_chunk_id, set()))
    right = set(chunk_to_nodes.get(right_chunk_id, set())) | set(chunk_to_edges.get(right_chunk_id, set()))
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left | right), 1)


def _chunk_signature_tokens(chunk_id, chunk_store, chunk_to_nodes, chunk_to_edges):
    chunk = chunk_store.get(chunk_id, {})
    parts = [normalize_text(chunk.get("content", ""))]
    parts.extend(normalize_text(node_id) for node_id in list(chunk_to_nodes.get(chunk_id, set()))[:10])
    parts.extend(
        normalize_text(normalize_relation_category({"keywords": " ".join(edge_id)}))
        for edge_id in list(chunk_to_edges.get(chunk_id, set()))[:10]
    )
    tokens = {
        token
        for token in tokenize(" ".join(parts))
        if len(token) >= 4 and not token.isdigit()
    }
    return tokens


def _token_overlap_ratio(left_tokens, right_tokens):
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)


def _theme_frontier_chunk_scores(structural_output, node_to_chunks, edge_to_chunks, chunk_neighbors):
    frontier_scores = defaultdict(float)
    for item in structural_output.get("selected_chunks", []):
        chunk_id = item["chunk_id"]
        base = 1.0 + 0.35 * item.get("bridge_gain", 0) + 0.08 * item.get("frontier_touch", 0) + 0.08 * item.get("new_source_count", 0)
        frontier_scores[chunk_id] = max(frontier_scores[chunk_id], base)
        for neighbor_chunk_id in chunk_neighbors.get(chunk_id, set()):
            frontier_scores[neighbor_chunk_id] = max(frontier_scores[neighbor_chunk_id], base * 0.55)

    for path_item in structural_output.get("selected_paths", []):
        path_chunks = _path_chunk_ids(path_item["path"], node_to_chunks, edge_to_chunks)
        base = 0.8 + 0.25 * path_item.get("bridge_gain", 0) + 0.05 * path_item.get("new_source_count", 0)
        for chunk_id in path_chunks:
            frontier_scores[chunk_id] = max(frontier_scores[chunk_id], base)
            for neighbor_chunk_id in chunk_neighbors.get(chunk_id, set()):
                frontier_scores[neighbor_chunk_id] = max(frontier_scores[neighbor_chunk_id], base * 0.45)
    return frontier_scores


def _promote_theme_frontier_roots(
    frontier_scores,
    active_root_chunk_score_lookup,
    query_chunk_score_lookup,
    chunk_store,
    chunk_to_nodes,
    chunk_to_edges,
    chunk_neighbors,
    core_chunk_ids,
    max_new_roots=2,
):
    existing_root_ids = list(active_root_chunk_score_lookup)
    promoted = []
    if not frontier_scores:
        return promoted
    core_token_set = set()
    for core_chunk_id in core_chunk_ids[:3]:
        core_token_set.update(_chunk_signature_tokens(core_chunk_id, chunk_store, chunk_to_nodes, chunk_to_edges))
    active_doc_counts = Counter(
        chunk_store.get(chunk_id, {}).get("full_doc_id")
        for chunk_id in existing_root_ids
        if chunk_store.get(chunk_id, {}).get("full_doc_id")
    )
    active_doc_ids = set(active_doc_counts)
    promoted_doc_counts = Counter()

    for chunk_id, frontier_score in sorted(
        frontier_scores.items(),
        key=lambda item: (
            -int(chunk_store.get(item[0], {}).get("full_doc_id") not in active_doc_ids),
            -item[1],
            -query_chunk_score_lookup.get(item[0], 0.0),
            item[0],
        ),
    ):
        if chunk_id in active_root_chunk_score_lookup:
            continue
        query_alignment = float(query_chunk_score_lookup.get(chunk_id, 0.0))
        expanded_band = _expand_chunk_band({chunk_id}, chunk_neighbors)
        root_band_alignment = max((active_root_chunk_score_lookup.get(item, 0.0) for item in expanded_band), default=0.0)
        if query_alignment < 0.05 and root_band_alignment < 0.18:
            continue
        candidate_doc_id = chunk_store.get(chunk_id, {}).get("full_doc_id")
        doc_novelty = 1 if candidate_doc_id and active_doc_counts.get(candidate_doc_id, 0) == 0 and promoted_doc_counts.get(candidate_doc_id, 0) == 0 else 0
        core_overlap = _token_overlap_ratio(
            _chunk_signature_tokens(chunk_id, chunk_store, chunk_to_nodes, chunk_to_edges),
            core_token_set,
        )
        if doc_novelty and core_overlap > 0.32:
            continue
        if not doc_novelty and core_overlap > 0.18:
            continue

        compare_ids = existing_root_ids + [item["chunk_id"] for item in promoted]
        same_band = any(_chunk_same_doc_band(chunk_store, chunk_id, existing_chunk_id) for existing_chunk_id in compare_ids)
        overlaps = [
            _chunk_provenance_overlap(chunk_id, existing_chunk_id, chunk_to_nodes, chunk_to_edges)
            for existing_chunk_id in compare_ids
        ]
        max_overlap = max(overlaps) if overlaps else 0.0
        if same_band or (doc_novelty and max_overlap > 0.7) or (not doc_novelty and max_overlap > 0.45):
            continue

        frontier_component = min(frontier_score / 2.0, 1.0)
        score_norm = max(
            0.12,
            min(
                1.0,
                0.35 * frontier_component
                + 0.2 * doc_novelty
                + 0.35 * max(query_alignment, root_band_alignment)
                + 0.1 * root_band_alignment,
            ),
        )
        promoted.append(
            {
                "chunk_id": chunk_id,
                "score_norm": round(score_norm, 6),
                "selection_score": round(score_norm, 6),
                "query_alignment": round(query_alignment, 6),
                "root_band_alignment": round(root_band_alignment, 6),
                "frontier_score": round(frontier_score, 6),
                "doc_novelty": doc_novelty,
                "core_overlap": round(core_overlap, 6),
                "full_doc_id": chunk_store.get(chunk_id, {}).get("full_doc_id"),
                "chunk_order_index": chunk_store.get(chunk_id, {}).get("chunk_order_index", -1),
                "root_role": "structural",
                "novelty_gain": len(chunk_to_nodes.get(chunk_id, set())) + len(chunk_to_edges.get(chunk_id, set())),
                "max_selected_overlap": round(max_overlap, 6),
            }
        )
        if candidate_doc_id:
            promoted_doc_counts[candidate_doc_id] += 1
        if len(promoted) >= max_new_roots:
            break
    return promoted


def _chunk_coverage_association(
    query,
    query_contract,
    graph,
    query,
    current_nodes,
    current_edges,
    current_categories,
    root_chunk_ids,
    root_chunk_score_lookup,
    query_chunk_score_lookup,
    chunk_to_nodes,
    chunk_to_edges,
    node_to_chunks,
    edge_to_chunks,
    chunk_neighbors,
    chunk_store,
    chunk_budget,
    allowed_doc_ids=None,
    query_priority=True,
):
    """Expand by informative adjacent/provenance chunk bands."""
    if chunk_budget <= 0:
        return {"selected_chunks": [], "selected_chunk_nodes": [], "selected_chunk_edges": []}

    covered_chunks = _covered_chunk_ids(current_nodes, current_edges, node_to_chunks, edge_to_chunks)
    covered_band = _expand_chunk_band(covered_chunks, chunk_neighbors)
    current_node_set = set(current_nodes)
    current_edge_set = set(current_edges)
    root_chunk_id_set = set(root_chunk_ids)

    candidate_chunks = set()
    for chunk_id in covered_band:
        candidate_chunks.update(chunk_neighbors.get(chunk_id, set()))
    candidate_chunks -= covered_band
    candidate_chunks = _filter_allowed_chunks(candidate_chunks, allowed_doc_ids, chunk_store)

    scored_chunks = []
    for chunk_id in candidate_chunks:
        chunk_nodes = set(chunk_to_nodes.get(chunk_id, set()))
        chunk_edges = set(chunk_to_edges.get(chunk_id, set()))
        new_nodes = chunk_nodes - current_node_set
        new_edges = chunk_edges - current_edge_set
        if not new_nodes and not new_edges:
            continue
        new_relation_count = 0
        for edge_id in new_edges:
            edge_data = graph.get_edge_data(edge_id[0], edge_id[1]) or {}
            category = normalize_relation_category(edge_data)
            if category not in current_categories:
                new_relation_count += 1
        new_source_count = _bounded_gain(len(_expand_chunk_band({chunk_id}, chunk_neighbors) - covered_band))
        root_overlap = 1 if chunk_id in root_chunk_id_set else 0
        root_band_alignment = max(
            [root_chunk_score_lookup.get(neighbor_id, 0.0) for neighbor_id in _expand_chunk_band({chunk_id}, chunk_neighbors)],
            default=0.0,
        )
        query_alignment = _query_band_alignment({chunk_id}, query_chunk_score_lookup, chunk_neighbors)
        if query_alignment < 0.05 and root_band_alignment < 0.15:
            continue
        scored_chunks.append(
            {
                "chunk_id": chunk_id,
                "new_source_count": new_source_count,
                "new_node_count": len(new_nodes),
                "new_edge_count": len(new_edges),
                "new_relation_count": new_relation_count,
                "root_overlap": root_overlap,
                "root_band_alignment": round(root_band_alignment, 6),
                "query_alignment": round(query_alignment, 6),
                "node_ids": sorted(chunk_nodes),
                "edge_ids": sorted(chunk_edges),
                **contract_features,
            }
        )

    if query_priority:
        scored_chunks.sort(
            key=lambda item: (
                -item["query_alignment"],
                -(item["new_node_count"] + item["new_edge_count"]),
                -item["new_relation_count"],
                -item["new_source_count"],
                -item["root_overlap"],
                -item["root_band_alignment"],
                item["chunk_id"],
            )
        )
    else:
        scored_chunks.sort(
            key=lambda item: (
                -(item["new_node_count"] + item["new_edge_count"]),
                -item["new_relation_count"],
                -item["new_source_count"],
                -item["root_overlap"],
                -item["root_band_alignment"],
                -item["query_alignment"],
                item["chunk_id"],
            )
        )
    selected_chunks = scored_chunks[:chunk_budget]
    selected_nodes = set()
    selected_edges = set()
    for item in selected_chunks:
        selected_nodes.update(item["node_ids"])
        selected_edges.update(item["edge_ids"])
    return {
        "selected_chunks": selected_chunks,
        "selected_chunk_nodes": sorted(selected_nodes),
        "selected_chunk_edges": sorted(selected_edges),
    }


def coverage_association(
    query,
    query_contract,
    graph,
    query,
    current_nodes,
    current_edges,
    root_chunk_ids,
    root_chunk_score_lookup,
    query_chunk_score_lookup,
    chunk_to_nodes,
    chunk_to_edges,
    node_to_chunks,
    edge_to_chunks,
    chunk_neighbors,
    chunk_store,
    semantic_edge_budget,
    semantic_node_budget,
    semantic_edge_min_score,
    semantic_node_min_score,
    allowed_doc_ids=None,
    query_priority=True,
):
    """Expand by evidence gain rather than local query similarity.

    Primary signal:
    - how many new supporting chunks / chunk-band items are introduced

    Tie-break signals:
    - whether a new relation category is introduced
    - whether the candidate still overlaps with rooted evidence
    """
    current_nodes = set(current_nodes)
    current_edges = set(current_edges)
    current_categories = set(build_current_relation_categories(current_edges, graph))
    covered_chunks = _covered_chunk_ids(current_nodes, current_edges, node_to_chunks, edge_to_chunks)
    covered_chunk_band = _expand_chunk_band(covered_chunks, chunk_neighbors)
    root_chunk_id_set = set(root_chunk_ids)

    candidate_edges = set()
    candidate_nodes = set()
    candidate_chunks = set(covered_chunk_band)

    for node_id in current_nodes:
        for neighbor_id in graph.neighbors(node_id):
            ek = edge_key(node_id, neighbor_id)
            if ek not in current_edges:
                candidate_edges.add(ek)
            if neighbor_id not in current_nodes:
                candidate_nodes.add(neighbor_id)

    for chunk_id in covered_chunk_band:
        candidate_nodes.update(node_id for node_id in chunk_to_nodes.get(chunk_id, set()) if node_id not in current_nodes)
        candidate_edges.update(edge_id for edge_id in chunk_to_edges.get(chunk_id, set()) if edge_id not in current_edges)
        candidate_chunks.update(chunk_neighbors.get(chunk_id, set()))

    scored_edges = []
    for edge_id in candidate_edges:
        edge_data = graph.get_edge_data(edge_id[0], edge_id[1]) or {}
        source_chunks = _filter_allowed_chunks(set(edge_to_chunks.get(edge_id, set())), allowed_doc_ids, chunk_store)
        if allowed_doc_ids and not source_chunks:
            continue
        local_source_chunks = source_chunks & candidate_chunks if candidate_chunks else source_chunks
        expanded_band = _expand_chunk_band(local_source_chunks, chunk_neighbors)
        new_source_count = _bounded_gain(len(expanded_band - covered_chunk_band))
        category = normalize_relation_category(edge_data)
        new_relation = 1 if category not in current_categories else 0
        _, chunk_alignment = support_score(source_chunks, root_chunk_score_lookup)
        root_overlap = len(source_chunks & root_chunk_id_set)
        query_alignment = max(
            _query_band_alignment(source_chunks, query_chunk_score_lookup, chunk_neighbors),
            _edge_query_alignment(edge_id, query, graph),
        )
        if query_alignment < 0.05 and root_overlap <= 0 and chunk_alignment < 0.12:
            continue
        coverage_gain = new_source_count if new_source_count > 0 else new_relation
        contract_features = _candidate_contract_features(
            query=query,
            query_contract=query_contract,
            source_chunk_ids=source_chunks,
            preview_text=_edge_preview_text(edge_id, graph),
            root_chunk_ids=root_chunk_ids,
            root_chunk_score_lookup=root_chunk_score_lookup,
            chunk_store=chunk_store,
        )
        if not contract_features["allowed"]:
            continue
        if coverage_gain < semantic_edge_min_score:
            continue
        if coverage_gain <= 0:
            continue
        scored_edges.append(
            {
                "edge": edge_id,
                "score": float(coverage_gain),
                "coverage_gain": float(coverage_gain),
                "new_source_count": new_source_count,
                "new_relation": new_relation,
                "root_overlap": root_overlap,
                "chunk_alignment": round(chunk_alignment, 6),
                "query_alignment": round(query_alignment, 6),
                "category": category,
                "keywords": edge_data.get("keywords", ""),
                "description": edge_data.get("description", ""),
                "source_chunk_ids": sorted(source_chunks),
                **contract_features,
            }
        )
    if query_priority:
        scored_edges.sort(
            key=lambda item: (
                -item["query_alignment"],
                -item["coverage_gain"],
                -item["new_relation"],
                -item["root_overlap"],
                -item["chunk_alignment"],
                item["edge"],
            )
        )
    else:
        scored_edges.sort(
            key=lambda item: (
                -item["coverage_gain"],
                -item["new_relation"],
                -item["root_overlap"],
                -item["chunk_alignment"],
                -item["query_alignment"],
                item["edge"],
            )
        )
    selected_edges = scored_edges[:semantic_edge_budget]

    expanded_nodes = set()
    for item in selected_edges:
        expanded_nodes.update(item["edge"])

    scored_nodes = []
    for node_id in candidate_nodes:
        node_data = graph.nodes[node_id]
        source_chunks = _filter_allowed_chunks(set(node_to_chunks.get(node_id, set())), allowed_doc_ids, chunk_store)
        if allowed_doc_ids and not source_chunks:
            continue
        local_source_chunks = source_chunks & candidate_chunks if candidate_chunks else source_chunks
        expanded_band = _expand_chunk_band(local_source_chunks, chunk_neighbors)
        new_source_count = _bounded_gain(len(expanded_band - covered_chunk_band))
        relation_categories = set()
        bridge_strength = 0
        for neighbor_id in graph.neighbors(node_id):
            if neighbor_id in current_nodes or neighbor_id in expanded_nodes:
                bridge_strength += 1
            edge_data = graph.get_edge_data(node_id, neighbor_id) or {}
            relation_categories.add(normalize_relation_category(edge_data))
        new_relation_count = len(relation_categories - current_categories)
        _, chunk_alignment = support_score(source_chunks, root_chunk_score_lookup)
        root_overlap = len(source_chunks & root_chunk_id_set)
        query_alignment = max(
            _query_band_alignment(source_chunks, query_chunk_score_lookup, chunk_neighbors),
            _node_query_alignment(node_id, query, graph),
        )
        if query_alignment < 0.05 and root_overlap <= 0 and chunk_alignment < 0.12:
            continue
        coverage_gain = new_source_count if new_source_count > 0 else new_relation_count
        contract_features = _candidate_contract_features(
            query=query,
            query_contract=query_contract,
            source_chunk_ids=source_chunks,
            preview_text=_node_preview_text(node_id, graph),
            root_chunk_ids=root_chunk_ids,
            root_chunk_score_lookup=root_chunk_score_lookup,
            chunk_store=chunk_store,
        )
        if not contract_features["allowed"]:
            continue
        if coverage_gain < semantic_node_min_score:
            continue
        if coverage_gain <= 0 and bridge_strength <= 0:
            continue
        scored_nodes.append(
            {
                "id": node_id,
                "score": float(coverage_gain),
                "coverage_gain": float(coverage_gain),
                "new_source_count": new_source_count,
                "new_relation_count": new_relation_count,
                "bridge_strength": bridge_strength,
                "root_overlap": root_overlap,
                "chunk_alignment": round(chunk_alignment, 6),
                "query_alignment": round(query_alignment, 6),
                "entity_type": node_data.get("entity_type", "UNKNOWN"),
                "description": node_data.get("description", ""),
                "source_chunk_ids": sorted(source_chunks),
                **contract_features,
            }
        )
    if query_priority:
        scored_nodes.sort(
            key=lambda item: (
                -item["query_alignment"],
                -item["coverage_gain"],
                -item["bridge_strength"],
                -item["root_overlap"],
                -item["chunk_alignment"],
                item["id"],
            )
        )
    else:
        scored_nodes.sort(
            key=lambda item: (
                -item["coverage_gain"],
                -item["bridge_strength"],
                -item["root_overlap"],
                -item["chunk_alignment"],
                -item["query_alignment"],
                item["id"],
            )
        )
    selected_nodes = scored_nodes[:semantic_node_budget]

    semantic_chunk_budget = max(1, min(semantic_edge_budget, semantic_node_budget) // 2)
    chunk_output = _chunk_coverage_association(
        query=query,
        query_contract=query_contract,
        graph=graph,
        query=query,
        current_nodes=current_nodes,
        current_edges=current_edges,
        current_categories=current_categories,
        root_chunk_ids=root_chunk_ids,
        root_chunk_score_lookup=root_chunk_score_lookup,
        query_chunk_score_lookup=query_chunk_score_lookup,
        chunk_to_nodes=chunk_to_nodes,
        chunk_to_edges=chunk_to_edges,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        chunk_neighbors=chunk_neighbors,
        chunk_store=chunk_store,
        chunk_budget=semantic_chunk_budget,
        allowed_doc_ids=allowed_doc_ids,
        query_priority=query_priority,
    )

    final_node_ids = set(current_nodes)
    final_edge_ids = set(current_edges)
    for item in selected_edges:
        final_edge_ids.add(item["edge"])
        final_node_ids.update(item["edge"])
    for item in selected_nodes:
        final_node_ids.add(item["id"])
    final_node_ids.update(chunk_output["selected_chunk_nodes"])
    final_edge_ids.update(chunk_output["selected_chunk_edges"])

    return {
        "selected_edges": selected_edges,
        "selected_nodes": selected_nodes,
        "selected_chunks": chunk_output["selected_chunks"],
        "final_nodes": sorted(final_node_ids),
        "final_edges": sorted(final_edge_ids),
    }


def build_node_role_sets(root_nodes, structural_nodes, semantic_nodes):
    """Tag final nodes by how they entered the graph for later presentation."""
    roles = {}
    for node_id in root_nodes:
        roles[node_id] = "root"
    for node_id in structural_nodes:
        if node_id not in roles:
            roles[node_id] = "structural"
    for node_id in semantic_nodes:
        if node_id not in roles:
            roles[node_id] = "semantic"
    return roles


def build_edge_role_sets(root_edges, structural_edges, semantic_edges):
    """Tag final edges by origin: root, structural bridge, or coverage expansion."""
    roles = {}
    for edge_id in root_edges:
        roles[edge_id] = "root"
    for edge_id in structural_edges:
        if edge_id not in roles:
            roles[edge_id] = "structural"
    for item in semantic_edges:
        edge_id = item["edge"]
        if edge_id not in roles:
            roles[edge_id] = "semantic"
    return roles


def _support_candidate_sort_key(item):
    return (
        -item["query_alignment"],
        -item["chunk_alignment"],
        -len(item["supporting_chunk_ids"]),
        item["label"],
    )


def _collect_support_candidates(
    *,
    query,
    graph,
    final_nodes,
    final_edges,
    node_to_chunks,
    edge_to_chunks,
    chunk_neighbors,
    root_chunk_score_lookup,
    query_chunk_score_lookup,
    chunk_store,
    allowed_doc_ids=None,
):
    point_candidates = []

    for edge_id in set(final_edges):
        source_chunks = _filter_allowed_chunks(set(edge_to_chunks.get(edge_id, set())), allowed_doc_ids, chunk_store)
        if not source_chunks:
            continue
        edge_data = graph.get_edge_data(edge_id[0], edge_id[1]) or {}
        relation_label = normalize_relation_category(edge_data)
        label = relation_label if relation_label not in {"unknown_relation", "unknown relation"} else f"{edge_id[0]} <-> {edge_id[1]}"
        _, chunk_alignment = support_score(source_chunks, root_chunk_score_lookup)
        query_alignment = max(
            _query_band_alignment(source_chunks, query_chunk_score_lookup, chunk_neighbors),
            _edge_query_alignment(edge_id, query, graph),
        )
        point_candidates.append(
            {
                "point_type": "edge",
                "label": label,
                "query_alignment": round(query_alignment, 6),
                "chunk_alignment": round(chunk_alignment, 6),
                "supporting_chunk_ids": sorted(source_chunks),
                "node_ids": sorted(set(edge_id)),
                "edge_ids": [edge_id],
                "doc_ids": sorted(
                    {
                        chunk_store.get(chunk_id, {}).get("full_doc_id")
                        for chunk_id in source_chunks
                        if chunk_store.get(chunk_id, {}).get("full_doc_id")
                    }
                ),
            }
        )

    for node_id in set(final_nodes):
        source_chunks = _filter_allowed_chunks(set(node_to_chunks.get(node_id, set())), allowed_doc_ids, chunk_store)
        if not source_chunks:
            continue
        _, chunk_alignment = support_score(source_chunks, root_chunk_score_lookup)
        query_alignment = max(
            _query_band_alignment(source_chunks, query_chunk_score_lookup, chunk_neighbors),
            _node_query_alignment(node_id, query, graph),
        )
        point_candidates.append(
            {
                "point_type": "node",
                "label": normalize_text(node_id),
                "query_alignment": round(query_alignment, 6),
                "chunk_alignment": round(chunk_alignment, 6),
                "supporting_chunk_ids": sorted(source_chunks),
                "node_ids": [node_id],
                "edge_ids": [],
                "doc_ids": sorted(
                    {
                        chunk_store.get(chunk_id, {}).get("full_doc_id")
                        for chunk_id in source_chunks
                        if chunk_store.get(chunk_id, {}).get("full_doc_id")
                    }
                ),
            }
        )

    point_candidates.sort(key=_support_candidate_sort_key)
    return point_candidates


def _aspect_candidate_related(left, right):
    left_chunks = set(left.get("supporting_chunk_ids", []))
    right_chunks = set(right.get("supporting_chunk_ids", []))
    if left_chunks & right_chunks:
        return True

    left_nodes = set(left.get("node_ids", []))
    right_nodes = set(right.get("node_ids", []))
    if left_nodes & right_nodes:
        return True

    left_edges = {tuple(edge_id) for edge_id in left.get("edge_ids", [])}
    right_edges = {tuple(edge_id) for edge_id in right.get("edge_ids", [])}
    if left_edges & right_edges:
        return True

    left_docs = set(left.get("doc_ids", []))
    right_docs = set(right.get("doc_ids", []))
    if left_docs & right_docs and lexical_overlap_score(left.get("label", ""), right.get("label", "")) >= 0.18:
        return True

    if left.get("point_type") == "edge" and right.get("point_type") == "edge" and left.get("label") == right.get("label"):
        return True

    return False


def _aspect_label_from_cluster(cluster_items, query, chunk_store):
    edge_labels = [
        normalize_text(item.get("label", ""))
        for item in cluster_items
        if item.get("point_type") == "edge"
        and normalize_text(item.get("label", "")) not in {"unknown relation", "unknown_relation"}
    ]
    if edge_labels:
        edge_counter = Counter(edge_labels)
        primary, _ = edge_counter.most_common(1)[0]
        secondary = None
        for label, _ in edge_counter.most_common(4):
            if label != primary:
                secondary = label
                break
        if secondary and lexical_overlap_score(primary, secondary) < 0.75:
            return f"{primary} | {secondary}"
        return primary

    query_tokens = set(tokenize(query))
    token_counter = Counter()
    for item in cluster_items:
        token_counter.update(token for token in tokenize(item.get("label", "")) if token not in query_tokens)
        for chunk_id in item.get("supporting_chunk_ids", [])[:3]:
            chunk = chunk_store.get(chunk_id, {})
            token_counter.update(token for token in tokenize(chunk.get("content", "")) if token not in query_tokens)
    aspect_tokens = []
    for token, _ in token_counter.most_common(6):
        if token in query_tokens:
            continue
        if token not in aspect_tokens:
            aspect_tokens.append(token)
        if len(aspect_tokens) >= 4:
            break
    if aspect_tokens:
        return " ".join(aspect_tokens)

    labels = [normalize_text(item.get("label", "")) for item in cluster_items if normalize_text(item.get("label", ""))]
    return labels[0] if labels else "aspect"


def _build_aspect_points(query, support_candidates, chunk_store, top_k):
    shortlist = support_candidates[: max(top_k * 4, 16)]
    if not shortlist:
        return []

    adjacency = {index: set() for index in range(len(shortlist))}
    for left_index, left in enumerate(shortlist):
        for right_index in range(left_index + 1, len(shortlist)):
            right = shortlist[right_index]
            if _aspect_candidate_related(left, right):
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)

    visited = set()
    clusters = []
    for start_index in range(len(shortlist)):
        if start_index in visited:
            continue
        stack = [start_index]
        cluster_indices = []
        visited.add(start_index)
        while stack:
            current = stack.pop()
            cluster_indices.append(current)
            for neighbor in adjacency[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)
        clusters.append([shortlist[index] for index in sorted(cluster_indices)])

    aspect_points = []
    for cluster_index, cluster_items in enumerate(clusters, start=1):
        supporting_chunk_ids = sorted({chunk_id for item in cluster_items for chunk_id in item.get("supporting_chunk_ids", [])})
        node_ids = sorted({node_id for item in cluster_items for node_id in item.get("node_ids", [])})
        edge_ids = []
        seen_edges = set()
        for item in cluster_items:
            for edge_id in item.get("edge_ids", []):
                canonical = tuple(edge_id)
                if canonical in seen_edges:
                    continue
                seen_edges.add(canonical)
                edge_ids.append(edge_id)
        doc_ids = sorted({doc_id for item in cluster_items for doc_id in item.get("doc_ids", []) if doc_id})
        support_labels = [normalize_text(item.get("label", "")) for item in cluster_items if normalize_text(item.get("label", ""))]
        aspect_points.append(
            {
                "point_type": "aspect",
                "label": _aspect_label_from_cluster(cluster_items, query, chunk_store),
                "query_alignment": round(sum(item.get("query_alignment", 0.0) for item in cluster_items) / len(cluster_items), 6),
                "chunk_alignment": round(max(item.get("chunk_alignment", 0.0) for item in cluster_items), 6),
                "supporting_chunk_ids": supporting_chunk_ids,
                "node_ids": node_ids,
                "edge_ids": edge_ids,
                "doc_ids": doc_ids,
                "support_labels": support_labels[:6],
                "support_count": len(cluster_items),
                "support_types": dict(Counter(item.get("point_type", "") for item in cluster_items)),
                "cluster_id": f"aspect-{cluster_index:02d}",
            }
        )

    aspect_points.sort(
        key=lambda item: (
            -item["query_alignment"],
            -item["chunk_alignment"],
            -len(item.get("doc_ids", [])),
            -len(item.get("supporting_chunk_ids", [])),
            -item.get("support_count", 0),
            item["label"],
        )
    )
    return aspect_points[:top_k]


def extract_candidate_points(
    *,
    query,
    graph,
    final_nodes,
    final_edges,
    node_to_chunks,
    edge_to_chunks,
    chunk_neighbors,
    root_chunk_score_lookup,
    query_chunk_score_lookup,
    chunk_store,
    allowed_doc_ids=None,
    top_k=8,
):
    """Expose reusable evidence points without forcing final answer structure.

    These points are intended as a lightweight intermediate product:
    - retrieval/association should recall them
    - later organization / prompting can choose which ones to expose
    """
    point_candidates = _collect_support_candidates(
        query=query,
        graph=graph,
        final_nodes=final_nodes,
        final_edges=final_edges,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        chunk_neighbors=chunk_neighbors,
        root_chunk_score_lookup=root_chunk_score_lookup,
        query_chunk_score_lookup=query_chunk_score_lookup,
        chunk_store=chunk_store,
        allowed_doc_ids=allowed_doc_ids,
    )

    return _build_aspect_points(
        query=query,
        support_candidates=point_candidates,
        chunk_store=chunk_store,
        top_k=top_k,
    )


def expand_associative_graph(
    query,
    query_contract,
    graph,
    root_nodes,
    root_edges,
    root_chunk_ids,
    root_chunk_score_lookup,
    query_chunk_score_lookup,
    chunk_to_nodes,
    chunk_to_edges,
    node_to_chunks,
    edge_to_chunks,
    chunk_neighbors,
    chunk_store,
    top_root_nodes,
    top_root_edges,
    max_hop,
    path_budget,
    semantic_edge_budget,
    semantic_node_budget,
    association_rounds,
    semantic_edge_min_score,
    semantic_node_min_score,
    allowed_doc_ids=None,
):
    """Run alternating bridge-association and coverage-association rounds."""
    return _expand_theme_chunk_graph(
        query=query,
        graph=graph,
        root_chunk_ids=root_chunk_ids,
        root_chunk_score_lookup=root_chunk_score_lookup,
        query_chunk_score_lookup=query_chunk_score_lookup,
        chunk_to_nodes=chunk_to_nodes,
        chunk_to_edges=chunk_to_edges,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        chunk_neighbors=chunk_neighbors,
        chunk_store=chunk_store,
        association_rounds=association_rounds,
        path_budget=path_budget,
        semantic_edge_budget=semantic_edge_budget,
        semantic_node_budget=semantic_node_budget,
        allowed_doc_ids=allowed_doc_ids,
    )
