"""Graph association logic.

Structural association tries to bridge disconnected root components with short,
query-relevant paths. Semantic association then adds nearby nodes and edges
that improve coverage or relation diversity without expanding blindly.
"""

from collections import Counter

import networkx as nx

from .common import edge_key, normalize_text, safe_mean, lexical_overlap_score
from .retrieval import normalize_relation_category, relation_entropy, score_root_edges, score_root_nodes, support_score


def build_root_components(root_nodes, root_edges):
    """Materialize connected components for the current root graph."""
    root_graph = nx.Graph()
    root_graph.add_nodes_from(root_nodes)
    root_graph.add_edges_from(root_edges)
    components = list(nx.connected_components(root_graph))
    node_to_component = {}
    for component_index, nodes in enumerate(components):
        for node_id in nodes:
            node_to_component[node_id] = component_index
    return root_graph, components, node_to_component


def path_edge_relevance(path, graph, query):
    """Average lexical relevance of the edges along a candidate path."""
    if len(path) < 2:
        return 0.0
    values = []
    for index in range(len(path) - 1):
        edge_data = graph.get_edge_data(path[index], path[index + 1]) or {}
        edge_text = " ".join(
            [
                normalize_text(path[index]),
                normalize_text(path[index + 1]),
                normalize_text(edge_data.get("keywords", "")),
                normalize_text(edge_data.get("description", "")),
            ]
        )
        values.append(lexical_overlap_score(query, edge_text))
    return safe_mean(values)


def path_support_span(path, node_to_chunks, edge_to_chunks):
    """Estimate how many supporting chunks a path touches."""
    chunk_ids = set()
    for node_id in path:
        chunk_ids.update(node_to_chunks.get(node_id, set()))
    for index in range(len(path) - 1):
        chunk_ids.update(edge_to_chunks.get(edge_key(path[index], path[index + 1]), set()))
    return len(chunk_ids)


def build_path_score(path, graph, query, root_chunk_ids, node_to_chunks, edge_to_chunks, source_component, target_component):
    """Score a bridging path by usefulness, support span, and length cost."""
    root_reach = 1.0 if source_component != target_component else 0.4
    support_span = min(path_support_span(path, node_to_chunks, edge_to_chunks) / max(len(root_chunk_ids), 1), 2.0)
    rel_path = path_edge_relevance(path, graph, query)
    length_penalty = len(path) - 1
    score = 0.45 * root_reach + 0.30 * support_span + 0.35 * rel_path - 0.10 * length_penalty
    return {
        "score": round(score, 6),
        "root_reach": round(root_reach, 6),
        "support_span": round(support_span, 6),
        "rel_path": round(rel_path, 6),
        "length_penalty": length_penalty,
    }


def structural_association(
    query,
    graph,
    current_nodes,
    current_edges,
    top_nodes,
    top_edges,
    node_to_chunks,
    edge_to_chunks,
    root_chunk_ids,
    max_hop,
    path_budget,
):
    """Select a small set of graph paths that connect different root components."""
    _, components, node_to_component = build_root_components(current_nodes, current_edges)
    if not current_nodes:
        return {
            "current_components": [],
            "selected_paths": [],
            "selected_path_nodes": [],
            "selected_path_edges": [],
        }

    current_edge_set = set(current_edges)
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

    # Search from strong node/edge anchors toward already selected nodes. This
    # keeps the path search rooted in evidence instead of drifting globally.
    candidate_paths = []
    dedupe_paths = set()
    for source_id in seed_nodes:
        source_component = node_to_component.get(source_id)
        paths = nx.single_source_shortest_path(graph, source_id, cutoff=max_hop)
        for target_id in current_nodes:
            if target_id == source_id:
                continue
            target_component = node_to_component.get(target_id)
            if target_component is None or target_component == source_component:
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
            path_score = build_path_score(
                path=path,
                graph=graph,
                query=query,
                root_chunk_ids=root_chunk_ids,
                node_to_chunks=node_to_chunks,
                edge_to_chunks=edge_to_chunks,
                source_component=source_component,
                target_component=target_component,
            )
            candidate_paths.append(
                {
                    "path": path,
                    "source_component": source_component,
                    "target_component": target_component,
                    **path_score,
                }
            )
    candidate_paths.sort(key=lambda item: (-item["score"], len(item["path"]), item["path"]))
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


def build_current_relation_categories(edge_ids, graph):
    """Collect coarse relation labels from the currently selected edge set."""
    categories = []
    for edge_id in edge_ids:
        edge_data = graph.get_edge_data(edge_id[0], edge_id[1]) or {}
        categories.append(normalize_relation_category(edge_data))
    return categories


def semantic_association(
    query,
    graph,
    current_nodes,
    current_edges,
    root_chunk_score_lookup,
    chunk_to_nodes,
    chunk_to_edges,
    node_to_chunks,
    edge_to_chunks,
    semantic_edge_budget,
    semantic_node_budget,
    semantic_edge_min_score,
    semantic_node_min_score,
):
    """Add semantically useful local graph content around the current frontier."""
    current_nodes = set(current_nodes)
    current_edges = set(current_edges)
    current_categories = build_current_relation_categories(current_edges, graph)
    current_entropy = relation_entropy(current_categories)

    # Candidates come from the immediate neighborhood of the current graph and
    # from chunks already supporting the selected nodes/edges.
    candidate_edges = set()
    candidate_nodes = set()
    candidate_chunks = set()
    for node_id in current_nodes:
        candidate_chunks.update(node_to_chunks.get(node_id, set()))
        for neighbor_id in graph.neighbors(node_id):
            ek = edge_key(node_id, neighbor_id)
            if ek not in current_edges:
                candidate_edges.add(ek)
            if neighbor_id not in current_nodes:
                candidate_nodes.add(neighbor_id)

    for edge_id in current_edges:
        candidate_chunks.update(edge_to_chunks.get(edge_id, set()))
    for chunk_id in candidate_chunks:
        candidate_nodes.update(node_id for node_id in chunk_to_nodes.get(chunk_id, set()) if node_id not in current_nodes)
        candidate_edges.update(edge_id for edge_id in chunk_to_edges.get(chunk_id, set()) if edge_id not in current_edges)

    scored_edges = []
    for edge_id in candidate_edges:
        edge_data = graph.get_edge_data(edge_id[0], edge_id[1]) or {}
        edge_text = " ".join(
            [
                normalize_text(edge_id[0]),
                normalize_text(edge_id[1]),
                normalize_text(edge_data.get("keywords", "")),
                normalize_text(edge_data.get("description", "")),
            ]
        )
        query_rel = lexical_overlap_score(query, edge_text)
        support, chunk_alignment = support_score(edge_to_chunks.get(edge_id, set()), root_chunk_score_lookup)
        category = normalize_relation_category(edge_data)
        new_entropy = relation_entropy(current_categories + [category])
        info_gain = max(new_entropy - current_entropy, 0.0)
        score = 0.55 * query_rel + 0.20 * support + 0.15 * chunk_alignment + 0.10 * info_gain
        scored_edges.append(
            {
                "edge": edge_id,
                "score": round(score, 6),
                "query_rel": round(query_rel, 6),
                "support": round(support, 6),
                "chunk_alignment": round(chunk_alignment, 6),
                "info_gain": round(info_gain, 6),
                "category": category,
                "keywords": edge_data.get("keywords", ""),
                "description": edge_data.get("description", ""),
                "source_chunk_ids": sorted(edge_to_chunks.get(edge_id, set())),
            }
        )
    scored_edges.sort(key=lambda item: (-item["score"], item["edge"]))
    filtered_edges = [
        item
        for item in scored_edges
        if item["score"] >= semantic_edge_min_score
        and (
            item["query_rel"] >= semantic_edge_min_score / 2
            or item["support"] > 0
            or item["chunk_alignment"] >= 0.6
            or item["info_gain"] >= 0.15
        )
    ]
    selected_edges = filtered_edges[:semantic_edge_budget]

    expanded_nodes = set()
    for item in selected_edges:
        expanded_nodes.update(item["edge"])

    scored_nodes = []
    for node_id in candidate_nodes:
        node_data = graph.nodes[node_id]
        source_chunks = node_to_chunks.get(node_id, set())
        query_rel = lexical_overlap_score(
            query,
            " ".join(
                [
                    normalize_text(node_id),
                    normalize_text(node_data.get("entity_type", "")),
                    normalize_text(node_data.get("description", "")),
                ]
            ),
        )
        support, chunk_alignment = support_score(source_chunks, root_chunk_score_lookup)
        bridge_strength = sum(1 for neighbor_id in graph.neighbors(node_id) if neighbor_id in current_nodes or neighbor_id in expanded_nodes)
        bridge_strength = min(bridge_strength / 4.0, 1.0)
        if bridge_strength <= 0 and not source_chunks.intersection(root_chunk_score_lookup):
            continue
        score = 0.50 * query_rel + 0.20 * support + 0.15 * chunk_alignment + 0.15 * bridge_strength
        scored_nodes.append(
            {
                "id": node_id,
                "score": round(score, 6),
                "query_rel": round(query_rel, 6),
                "support": round(support, 6),
                "chunk_alignment": round(chunk_alignment, 6),
                "bridge_strength": round(bridge_strength, 6),
                "entity_type": node_data.get("entity_type", "UNKNOWN"),
                "description": node_data.get("description", ""),
                "source_chunk_ids": sorted(node_to_chunks.get(node_id, set())),
            }
        )
    scored_nodes.sort(key=lambda item: (-item["score"], item["id"]))
    selected_nodes = [item for item in scored_nodes if item["score"] >= semantic_node_min_score][:semantic_node_budget]

    final_node_ids = set(current_nodes)
    final_edge_ids = set(current_edges)
    for item in selected_edges:
        final_edge_ids.add(item["edge"])
        final_node_ids.update(item["edge"])
    for item in selected_nodes:
        final_node_ids.add(item["id"])

    return {
        "selected_edges": selected_edges,
        "selected_nodes": selected_nodes,
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
    """Tag final edges by origin: root, structural bridge, or semantic gain."""
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


def expand_associative_graph(
    query,
    graph,
    root_nodes,
    root_edges,
    root_chunk_ids,
    root_chunk_score_lookup,
    chunk_to_nodes,
    chunk_to_edges,
    node_to_chunks,
    edge_to_chunks,
    top_root_nodes,
    top_root_edges,
    max_hop,
    path_budget,
    semantic_edge_budget,
    semantic_node_budget,
    association_rounds,
    semantic_edge_min_score,
    semantic_node_min_score,
):
    """Run alternating structural + semantic expansion rounds.

    The output preserves round-level traces so later analysis can inspect how
    the graph grew and which phase introduced which evidence.
    """
    current_nodes = set(root_nodes)
    current_edges = set(root_edges)
    all_structural_nodes = set()
    all_structural_edges = set()
    all_semantic_nodes = set()
    all_semantic_edges = []
    round_outputs = []
    last_structural_output = {
        "current_components": [],
        "selected_paths": [],
        "selected_path_nodes": [],
        "selected_path_edges": [],
    }

    for round_index in range(1, max(association_rounds, 1) + 1):
        scored_current_nodes = score_root_nodes(
            query=query,
            root_nodes=current_nodes,
            graph=graph,
            node_to_chunks=node_to_chunks,
            root_chunk_score_lookup=root_chunk_score_lookup,
        )
        scored_current_edges = score_root_edges(
            query=query,
            root_edges=current_edges,
            graph=graph,
            edge_to_chunks=edge_to_chunks,
            root_chunk_score_lookup=root_chunk_score_lookup,
        )
        structural_output = structural_association(
            query=query,
            graph=graph,
            current_nodes=current_nodes,
            current_edges=current_edges,
            top_nodes=scored_current_nodes[:top_root_nodes],
            top_edges=scored_current_edges[:top_root_edges],
            node_to_chunks=node_to_chunks,
            edge_to_chunks=edge_to_chunks,
            root_chunk_ids=root_chunk_ids,
            max_hop=max_hop,
            path_budget=path_budget,
        )
        structural_new_nodes = set(structural_output["selected_path_nodes"]) - current_nodes
        structural_new_edges = set(structural_output["selected_path_edges"]) - current_edges
        current_nodes = current_nodes.union(structural_output["selected_path_nodes"])
        current_edges = current_edges.union(structural_output["selected_path_edges"])

        semantic_output = semantic_association(
            query=query,
            graph=graph,
            current_nodes=current_nodes,
            current_edges=current_edges,
            root_chunk_score_lookup=root_chunk_score_lookup,
            chunk_to_nodes=chunk_to_nodes,
            chunk_to_edges=chunk_to_edges,
            node_to_chunks=node_to_chunks,
            edge_to_chunks=edge_to_chunks,
            semantic_edge_budget=semantic_edge_budget,
            semantic_node_budget=semantic_node_budget,
            semantic_edge_min_score=semantic_edge_min_score,
            semantic_node_min_score=semantic_node_min_score,
        )
        semantic_new_nodes = set(semantic_output["final_nodes"]) - current_nodes
        semantic_new_edges = set(semantic_output["final_edges"]) - current_edges
        current_nodes = set(semantic_output["final_nodes"])
        current_edges = set(semantic_output["final_edges"])

        all_structural_nodes.update(structural_new_nodes)
        all_structural_edges.update(structural_new_edges)
        all_semantic_nodes.update(semantic_new_nodes)
        all_semantic_edges.extend(item for item in semantic_output["selected_edges"] if item["edge"] in semantic_new_edges)
        round_outputs.append(
            {
                "round": round_index,
                "structural_path_count": len(structural_output["selected_paths"]),
                "structural_added_node_count": len(structural_new_nodes),
                "structural_added_edge_count": len(structural_new_edges),
                "semantic_added_node_count": len(semantic_new_nodes),
                "semantic_added_edge_count": len(semantic_new_edges),
                "current_node_count": len(current_nodes),
                "current_edge_count": len(current_edges),
                "structural_path_preview": [
                    {
                        "path": item["path"],
                        "score": item["score"],
                        "support_span": item["support_span"],
                        "rel_path": item["rel_path"],
                    }
                    for item in structural_output["selected_paths"][:3]
                ],
                "semantic_edge_preview": [
                    {
                        "edge": item["edge"],
                        "score": item["score"],
                        "category": item["category"],
                        "query_rel": item["query_rel"],
                        "support": item["support"],
                    }
                    for item in semantic_output["selected_edges"][:3]
                ],
                "semantic_node_preview": [
                    {
                        "id": item["id"],
                        "score": item["score"],
                        "bridge_strength": item["bridge_strength"],
                        "query_rel": item["query_rel"],
                    }
                    for item in semantic_output["selected_nodes"][:3]
                ],
            }
        )
        last_structural_output = structural_output
        if not structural_new_nodes and not structural_new_edges and not semantic_new_nodes and not semantic_new_edges:
            break

    return {
        "final_nodes": sorted(current_nodes),
        "final_edges": sorted(current_edges),
        "all_structural_nodes": sorted(all_structural_nodes),
        "all_structural_edges": sorted(all_structural_edges),
        "all_semantic_nodes": sorted(all_semantic_nodes),
        "all_semantic_edges": all_semantic_edges,
        "rounds": round_outputs,
        "last_structural_output": last_structural_output,
    }
