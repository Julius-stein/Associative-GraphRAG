"""Deep Evidence Tracing retrieval prototype.

This module keeps the online method deliberately small:

1. extract a light research goal from the query;
2. select goal-covering anchor chunks;
3. trace neighboring chunks through shared KG nodes;
4. update the next search goal from the discovered evidence graph;
5. present each trace as a chunk-labeled evidence path.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from time import perf_counter

import numpy as np

from .common import approx_word_count, lexical_overlap_score, normalize_text, tokenize


_QUERY_TASK_WORDS = {
    "compare",
    "contrast",
    "explain",
    "describe",
    "discuss",
    "summarize",
    "identify",
    "list",
    "cause",
    "causes",
    "effect",
    "effects",
    "impact",
    "impacts",
    "influence",
    "influenced",
    "relationship",
    "role",
    "roles",
    "ways",
}

_QUERY_SHELL_WORDS = {
    "main",
    "major",
    "primary",
    "reason",
    "reasons",
    "factor",
    "factors",
    "issue",
    "issues",
    "thing",
    "things",
    "various",
    "different",
    "section",
    "sections",
    "part",
    "parts",
    "example",
    "examples",
    "novice",  # retained in phrases such as "novice beekeepers", not as a standalone atom.
    "may",
    "might",
}


def _answer_type(query):
    tokens = set(tokenize(query))
    if {"compare", "contrast"} & tokens:
        return "comparison"
    if {"cause", "causes", "why", "influence", "influenced", "impact", "effects"} & tokens:
        return "causal synthesis"
    if {"list", "identify", "which", "what"} & tokens:
        return "evidence listing"
    if {"how", "explain", "describe"} & tokens:
        return "explanatory synthesis"
    return "source-grounded synthesis"


def _clean_goal_terms(text, limit=8):
    raw_tokens = [
        token
        for token in tokenize(text)
        if token not in _QUERY_TASK_WORDS and token not in (_QUERY_SHELL_WORDS - {"novice"})
    ]
    phrase_tokens = [token for token in raw_tokens if token not in _QUERY_SHELL_WORDS or token == "novice"]
    phrases = []
    for ngram_size in (3, 2):
        for index in range(0, max(len(phrase_tokens) - ngram_size + 1, 0)):
            ngram = phrase_tokens[index : index + ngram_size]
            if all(token in _QUERY_SHELL_WORDS for token in ngram):
                continue
            phrase = " ".join(ngram)
            if phrase not in phrases:
                phrases.append(phrase)
    terms = []
    for phrase in phrases:
        if phrase not in terms:
            terms.append(phrase)
        if len(terms) >= limit:
            return terms
    for token in raw_tokens:
        if token in _QUERY_SHELL_WORDS:
            continue
        if token not in terms:
            terms.append(token)
        if len(terms) >= limit:
            break
    return terms


def extract_research_goal(query, graph=None, atom_limit=8, entity_limit=6):
    """Create the initial research goal without making answer contracts."""
    if graph is None:
        raise ValueError("Deep Evidence Tracing requires a knowledge graph for research-goal extraction.")
    query_atoms = _clean_goal_terms(query, limit=atom_limit)
    target_entities = []
    scored = []
    for node_id, node_data in graph.nodes(data=True):
        text = " ".join(
            [
                normalize_text(node_id),
                normalize_text(node_data.get("entity_type", "")),
                normalize_text(node_data.get("description", ""))[:240],
            ]
        )
        score = lexical_overlap_score(query, text)
        if score > 0:
            scored.append((score, str(node_id)))
    scored.sort(key=lambda item: (-item[0], item[1]))
    target_entities = [node_id for _, node_id in scored[:entity_limit]]
    return {
        "answer_type": _answer_type(query),
        "query_atoms": query_atoms,
        "target_entities": target_entities,
    }


def build_evidence_chunk_graph(chunk_store, chunk_to_nodes, node_to_chunks, chunk_neighbors=None):
    """Build the dataset-level chunk graph used by evidence tracing.

    Edges have two possible provenance channels:
    - shared KG nodes, when two chunks mention the same graph node;
    - document context, when chunks are adjacent in the original document.
    """
    chunk_graph = {
        chunk_id: {}
        for chunk_id in chunk_store
    }

    def ensure(left, right):
        if left == right or left not in chunk_graph or right not in chunk_graph:
            return None
        slot = chunk_graph[left].setdefault(right, {"shared_nodes": set(), "doc_context": False})
        chunk_graph[right].setdefault(left, {"shared_nodes": set(), "doc_context": False})
        return slot

    for node_id, source_chunks in node_to_chunks.items():
        source_chunks = [chunk_id for chunk_id in source_chunks if chunk_id in chunk_graph]
        for left_index, left in enumerate(source_chunks):
            for right in source_chunks[left_index + 1 :]:
                left_slot = ensure(left, right)
                if left_slot is None:
                    continue
                left_slot["shared_nodes"].add(node_id)
                chunk_graph[right][left]["shared_nodes"].add(node_id)

    for left, neighbors in (chunk_neighbors or {}).items():
        for right in neighbors:
            left_slot = ensure(left, right)
            if left_slot is None:
                continue
            left_slot["doc_context"] = True
            chunk_graph[right][left]["doc_context"] = True

    return chunk_graph


def _chunk_degree(chunk_id, chunk_graph):
    return len(chunk_graph.get(chunk_id, {}))


def _degree_penalty(chunk_id, chunk_graph):
    return 1.0 / max(math.log1p(_chunk_degree(chunk_id, chunk_graph) + 1), 1.0)


def _chunk_vector_lookup(dense_index):
    if dense_index is None or dense_index.normalized_matrix.size == 0:
        raise ValueError("Deep Evidence Tracing requires a non-empty dense chunk index.")
    return {chunk_id: idx for idx, chunk_id in enumerate(dense_index.chunk_ids)}, dense_index.normalized_matrix


def _normalize_vector(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector
    return vector / norm


def _embed_goal_items(goal_items, embedding_client):
    if embedding_client is None:
        raise ValueError("Deep Evidence Tracing requires an embedding client.")
    if not goal_items:
        raise ValueError("Deep Evidence Tracing received no goal items to embed.")
    vectors = embedding_client.embed_texts(goal_items)
    return {item: _normalize_vector(vector) for item, vector in zip(goal_items, vectors)}


def _dense_sim_text_to_chunk(goal, chunk_id, goal_vectors, chunk_index_lookup, chunk_matrix):
    if chunk_matrix is None:
        raise ValueError("Deep Evidence Tracing requires a dense chunk matrix.")
    if goal not in goal_vectors:
        raise ValueError(f"Missing embedding vector for goal item: {goal}")
    index = chunk_index_lookup.get(chunk_id)
    if index is None:
        raise ValueError(f"Chunk {chunk_id} is missing from the dense chunk index.")
    return float(chunk_matrix[index] @ goal_vectors[goal])


def _dense_sim_texts(left, right, embedding_client):
    if embedding_client is None:
        raise ValueError("Deep Evidence Tracing requires an embedding client.")
    left_vec, right_vec = embedding_client.embed_texts([left, right])
    return float(_normalize_vector(left_vec) @ _normalize_vector(right_vec))


def _ensure_text_vectors(texts, embedding_client, text_vector_cache):
    if embedding_client is None:
        raise ValueError("Deep Evidence Tracing requires an embedding client.")
    missing = [text for text in dict.fromkeys(texts) if text and text not in text_vector_cache]
    if not missing:
        return
    for text, vector in zip(missing, embedding_client.embed_texts(missing)):
        text_vector_cache[text] = _normalize_vector(vector)


def _chunk_goal_score(chunk_id, goal_items, chunk_store, goal_vectors, chunk_index_lookup, chunk_matrix):
    best_score = 0.0
    best_goal = ""
    for goal in goal_items:
        dense_score = _dense_sim_text_to_chunk(goal, chunk_id, goal_vectors, chunk_index_lookup, chunk_matrix)
        score = dense_score
        if score > best_score:
            best_score = score
            best_goal = goal
    return best_score, best_goal


def _allowed_chunk_ids(allowed_doc_ids, chunk_store):
    if allowed_doc_ids is None:
        return None
    allowed_doc_ids = set(allowed_doc_ids)
    return {
        chunk_id
        for chunk_id, chunk in chunk_store.items()
        if chunk.get("full_doc_id") in allowed_doc_ids
    }


def select_goal_covering_anchors(
    *,
    query,
    goal,
    chunk_retriever,
    chunk_store,
    chunk_to_nodes,
    chunk_graph,
    top_k,
    candidate_pool_size,
    allowed_doc_ids=None,
):
    """Select anchors by assigning at least one entry to each retained goal item."""
    raw_goal_items = [query] + list(goal.get("query_atoms", [])) + list(goal.get("target_entities", []))
    goal_items = list(dict.fromkeys(item for item in raw_goal_items if normalize_text(item)))
    if not goal_items:
        goal_items = [query]
    goal_items = goal_items[: max(top_k, 1)]

    selected = []
    selected_ids = set()
    candidates_by_goal = {}
    for goal_item in goal_items:
        hits = chunk_retriever.search(
            goal_item,
            top_k=max(candidate_pool_size, top_k * 4),
            allowed_doc_ids=allowed_doc_ids,
            chunk_store=chunk_store,
        )
        ranked = []
        for hit in hits:
            chunk_id = hit["chunk_id"]
            score = float(hit.get("score_norm", hit.get("dense_score_norm", hit.get("score", 0.0))))
            score *= _degree_penalty(chunk_id, chunk_graph)
            ranked.append((score, hit))
        ranked.sort(key=lambda item: (-item[0], item[1]["chunk_id"]))
        candidates_by_goal[goal_item] = ranked

    for goal_item in goal_items:
        if len(selected) >= top_k:
            break
        for score, hit in candidates_by_goal.get(goal_item, []):
            chunk_id = hit["chunk_id"]
            if chunk_id in selected_ids:
                continue
            selected_ids.add(chunk_id)
            selected.append(
                {
                    **hit,
                    "chunk_id": chunk_id,
                    "score": score,
                    "score_norm": score,
                    "anchor_label": goal_item,
                    "root_role": "primary",
                    "graph_mass": len(chunk_to_nodes.get(chunk_id, set())),
                    "chunk_graph_degree": _chunk_degree(chunk_id, chunk_graph),
                    "full_doc_id": chunk_store.get(chunk_id, {}).get("full_doc_id"),
                    "chunk_order_index": chunk_store.get(chunk_id, {}).get("chunk_order_index", -1),
                }
            )
            break

    if len(selected) < top_k:
        merged = []
        for ranked in candidates_by_goal.values():
            merged.extend(ranked)
        merged.sort(key=lambda item: (-item[0], item[1]["chunk_id"]))
        for score, hit in merged:
            if len(selected) >= top_k:
                break
            chunk_id = hit["chunk_id"]
            if chunk_id in selected_ids:
                continue
            selected_ids.add(chunk_id)
            selected.append(
                {
                    **hit,
                    "chunk_id": chunk_id,
                    "score": score,
                    "score_norm": score,
                    "anchor_label": "query",
                    "root_role": "primary",
                    "graph_mass": len(chunk_to_nodes.get(chunk_id, set())),
                    "chunk_graph_degree": _chunk_degree(chunk_id, chunk_graph),
                    "full_doc_id": chunk_store.get(chunk_id, {}).get("full_doc_id"),
                    "chunk_order_index": chunk_store.get(chunk_id, {}).get("chunk_order_index", -1),
                }
            )
    required_anchor_count = min(top_k, len(goal_items))
    if len(selected) < required_anchor_count:
        raise ValueError(
            "Deep Evidence Tracing could not select one dense anchor per retained goal item: "
            f"selected={len(selected)} required={required_anchor_count}"
        )
    top_score = max((item["score"] for item in selected), default=1.0)
    for item in selected:
        item["score_norm"] = item["score"] / max(top_score, 1e-9)
    return selected, goal_items, candidates_by_goal


def _trace_frontier(trace_chunk_ids, chunk_graph, allowed_chunks=None):
    """Return candidate chunks from the fixed dataset-level chunk graph."""
    frontier = {}
    trace_set = set(trace_chunk_ids)

    for chunk_id in trace_chunk_ids:
        for neighbor_chunk_id, edge_data in chunk_graph.get(chunk_id, {}).items():
            if neighbor_chunk_id in trace_set:
                continue
            if allowed_chunks is not None and neighbor_chunk_id not in allowed_chunks:
                continue
            slot = frontier.setdefault(neighbor_chunk_id, {"shared_nodes": set(), "doc_context": False})
            slot["shared_nodes"].update(edge_data.get("shared_nodes", set()))
            slot["doc_context"] = slot["doc_context"] or bool(edge_data.get("doc_context"))
    return frontier


def _node_text(node_id, graph):
    if graph is None or not graph.has_node(node_id):
        raise ValueError(f"Missing graph node required for evidence tracing: {node_id}")
    data = graph.nodes[node_id]
    return " ".join(
        [
            normalize_text(node_id),
            normalize_text(data.get("entity_type", "")),
            normalize_text(data.get("description", ""))[:180],
        ]
    )


def _link_score(shared_nodes, goal_items, graph, embedding_client, goal_vectors=None, text_vector_cache=None):
    best_score = 0.0
    best_node = ""
    if embedding_client is None:
        raise ValueError("Deep Evidence Tracing requires an embedding client.")
    if not goal_vectors:
        raise ValueError("Deep Evidence Tracing requires embedded goal vectors for link scoring.")
    if text_vector_cache is None:
        raise ValueError("Deep Evidence Tracing requires a node text vector cache for link scoring.")
    for node_id in shared_nodes:
        text = _node_text(node_id, graph)
        for goal in goal_items:
            if text not in text_vector_cache:
                raise ValueError(f"Missing embedding vector for frontier node text: {text[:80]}")
            if goal not in goal_vectors:
                raise ValueError(f"Missing embedding vector for goal item: {goal}")
            score = float(text_vector_cache[text] @ goal_vectors[goal])
            if score > best_score:
                best_score = score
                best_node = node_id
    return best_score, best_node


def _trace_one_anchor(
    *,
    anchor_hit,
    goal_items,
    graph,
    chunk_store,
    chunk_to_edges,
    chunk_graph,
    dense_index,
    embedding_client,
    max_steps,
    frontier_edge_top_k,
    allowed_chunks=None,
):
    chunk_index_lookup, chunk_matrix = _chunk_vector_lookup(dense_index)
    goal_vectors = _embed_goal_items(goal_items, embedding_client)
    text_vector_cache = {}
    selected_chunk_ids = [anchor_hit["chunk_id"]]
    steps = []
    labels = {anchor_hit["chunk_id"]: anchor_hit.get("anchor_label", "query")}

    for step_index in range(1, max_steps + 1):
        frontier = _trace_frontier(
            selected_chunk_ids,
            chunk_graph,
            allowed_chunks=allowed_chunks,
        )
        if not frontier:
            break
        _ensure_text_vectors(
            [
                _node_text(node_id, graph)
                for lead in frontier.values()
                for node_id in lead.get("shared_nodes", set())
            ],
            embedding_client,
            text_vector_cache,
        )
        scored_links = []
        for candidate_id, lead in frontier.items():
            shared_nodes = lead.get("shared_nodes", set())
            link_score, best_node = _link_score(
                shared_nodes,
                goal_items,
                graph,
                embedding_client,
                goal_vectors=goal_vectors,
                text_vector_cache=text_vector_cache,
            )
            link_type = "shared_node" if shared_nodes else "document_context"
            scored_links.append((link_score, candidate_id, best_node, shared_nodes, link_type, lead.get("doc_context", False)))
        scored_links.sort(key=lambda item: (item[4] != "shared_node", -item[0], item[1]))
        pruned = scored_links[: max(frontier_edge_top_k, 1)]

        best = None
        best_key = None
        for link_score, candidate_id, best_node, shared_nodes, link_type, doc_context in pruned:
            chunk_score, best_goal = _chunk_goal_score(
                candidate_id,
                goal_items,
                chunk_store,
                goal_vectors,
                chunk_index_lookup,
                chunk_matrix,
            )
            score = chunk_score * (1.0 + max(link_score, 0.0))
            score *= _degree_penalty(candidate_id, chunk_graph)
            key = (
                score,
                chunk_score,
                link_type == "document_context",
                link_score,
                -_chunk_degree(candidate_id, chunk_graph),
                candidate_id,
            )
            if best_key is None or key > best_key:
                best_key = key
                best = {
                    "chunk_id": candidate_id,
                    "step": step_index,
                    "score": round(score, 6),
                    "chunk_goal_score": round(chunk_score, 6),
                    "link_score": round(link_score, 6),
                    "label": best_goal or best_node or anchor_hit.get("anchor_label", "query"),
                    "link_type": link_type,
                    "doc_context": bool(doc_context),
                    "shared_nodes": sorted(shared_nodes),
                    "best_shared_node": best_node,
                }
        if best is None or best["score"] <= 0:
            break
        selected_chunk_ids.append(best["chunk_id"])
        labels[best["chunk_id"]] = best["label"]
        steps.append(best)

    return {
        "root_chunk_id": anchor_hit["chunk_id"],
        "anchor_label": anchor_hit.get("anchor_label", "query"),
        "selected_chunk_ids": selected_chunk_ids,
        "evidence_chunk_ids": selected_chunk_ids[1:],
        "chunk_labels": labels,
        "steps": steps,
        # Kept only for legacy consumers; the method treats all non-root chunks as evidence chunks.
        "bridge_chunk_ids": [],
        "support_chunk_ids": selected_chunk_ids[1:],
        "context_chunk_ids": [],
    }


def _covered_goal_items(trace_chunk_ids, goal_items, dense_index, embedding_client):
    chunk_index_lookup, chunk_matrix = _chunk_vector_lookup(dense_index)
    goal_vectors = _embed_goal_items(goal_items, embedding_client)
    covered = set()
    for goal in goal_items:
        for chunk_id in trace_chunk_ids:
            if _dense_sim_text_to_chunk(goal, chunk_id, goal_vectors, chunk_index_lookup, chunk_matrix) > 0:
                covered.add(goal)
                break
    return covered


def _emergent_goal_items(
    *,
    active_goal_items,
    trace_chunk_ids,
    graph,
    chunk_to_nodes,
    embedding_client,
    top_k,
):
    if graph is None:
        raise ValueError("Deep Evidence Tracing requires a knowledge graph for goal update.")
    goal_vectors = _embed_goal_items(active_goal_items, embedding_client)
    active_norm = {normalize_text(item).lower() for item in active_goal_items}
    seed_nodes = set()
    for chunk_id in trace_chunk_ids:
        seed_nodes.update(chunk_to_nodes.get(chunk_id, set()))
    candidates = []
    for node_id in seed_nodes:
        if not graph.has_node(node_id):
            continue
        node_degree = max(graph.degree(node_id), 1)
        for neighbor_id in graph.neighbors(node_id):
            normalized_neighbor = normalize_text(neighbor_id).lower()
            if not normalized_neighbor or normalized_neighbor in active_norm or neighbor_id in seed_nodes:
                continue
            neighbor_degree = max(graph.degree(neighbor_id), 1)
            candidates.append((node_id, neighbor_id, node_degree, neighbor_degree, _node_text(neighbor_id, graph)))
    text_vector_cache = {}
    _ensure_text_vectors([item[4] for item in candidates], embedding_client, text_vector_cache)
    scores = Counter()
    for _node_id, neighbor_id, node_degree, neighbor_degree, neighbor_text in candidates:
        if neighbor_text not in text_vector_cache:
            raise ValueError(f"Missing embedding vector for emergent node text: {neighbor_text[:80]}")
        rel = max(float(text_vector_cache[neighbor_text] @ goal_vector) for goal_vector in goal_vectors.values())
        if rel <= 0:
            continue
        scores[neighbor_id] += rel / math.sqrt(node_degree * neighbor_degree)
    return [node_id for node_id, _ in scores.most_common(top_k)]


def _trace_edges(trace, chunk_to_edges):
    edges = []
    for chunk_id in trace.get("selected_chunk_ids", []):
        for edge_id in chunk_to_edges.get(chunk_id, set()):
            if edge_id not in edges:
                edges.append(edge_id)
    return edges


def _trace_nodes(trace, chunk_to_nodes):
    nodes = []
    for chunk_id in trace.get("selected_chunk_ids", []):
        for node_id in chunk_to_nodes.get(chunk_id, set()):
            if node_id not in nodes:
                nodes.append(node_id)
    return nodes


def _relation_theme(edge_id, graph):
    if graph is None:
        raise ValueError("Deep Evidence Tracing requires a knowledge graph for path organization.")
    data = graph.get_edge_data(edge_id[0], edge_id[1])
    if data is None:
        raise ValueError(f"Missing graph edge required for path organization: {edge_id}")
    return normalize_text(data.get("keywords") or data.get("description") or "")


def build_trace_groups(query, traces, graph, chunk_store, chunk_to_nodes, chunk_to_edges, group_limit, chunks_per_group=8):
    groups = []
    seen_chunks = set()
    for index, trace in enumerate(traces, start=1):
        chunk_ids = list(dict.fromkeys(trace.get("selected_chunk_ids", [])))[:chunks_per_group]
        if not chunk_ids:
            continue
        overlap = len(set(chunk_ids) & seen_chunks) / max(len(set(chunk_ids)), 1)
        if overlap >= 0.75:
            continue
        seen_chunks.update(chunk_ids)
        nodes = _trace_nodes({"selected_chunk_ids": chunk_ids}, chunk_to_nodes)
        edges = _trace_edges({"selected_chunk_ids": chunk_ids}, chunk_to_edges)
        relation_themes = [theme for theme in (_relation_theme(edge_id, graph) for edge_id in edges[:6]) if theme]
        labels = trace.get("chunk_labels", {})
        focus = list(dict.fromkeys([trace.get("anchor_label", "")] + [labels.get(chunk_id, "") for chunk_id in chunk_ids]))
        focus = [item for item in focus if normalize_text(item)]
        label = normalize_text(focus[0]) if focus else f"evidence trace {index}"
        evidence_dossier = [
            {
                "chunk_id": chunk_id,
                "role": "anchor" if chunk_id == trace.get("root_chunk_id") else "evidence",
                "covered_atoms": [labels.get(chunk_id, "")] if labels.get(chunk_id) else [],
                "coverage_reasons": [f"trace label: {labels.get(chunk_id, label)}"],
                "answerability": {
                    "query_rel": lexical_overlap_score(query, chunk_store.get(chunk_id, {}).get("content", "")),
                    "signal_count": 1,
                    "specific_count": 1,
                },
                "word_count": approx_word_count(chunk_store.get(chunk_id, {}).get("content", "")),
            }
            for chunk_id in chunk_ids
        ]
        groups.append(
            {
                "group_id": f"kg-{len(groups) + 1:02d}",
                "facet_label": label,
                "primary_theme": label,
                "group_summary": (
                    f"Chunk-labeled evidence path from anchor '{trace.get('anchor_label', label)}' "
                    f"through {len(chunk_ids)} source chunks."
                ),
                "group_score": round(max((item["answerability"]["query_rel"] for item in evidence_dossier), default=0.0), 6),
                "query_rel": round(max((item["answerability"]["query_rel"] for item in evidence_dossier), default=0.0), 6),
                "edge_query_rel": 0.0,
                "dossier_query_rel": round(max((item["answerability"]["query_rel"] for item in evidence_dossier), default=0.0), 6),
                "dossier_answerability": len(evidence_dossier),
                "dossier_specificity": len(evidence_dossier),
                "anchor_support": 1,
                "root_anchor_count": 1,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "relation_themes": relation_themes[:5],
                "focus_entities": focus[:6],
                "supporting_chunk_ids": chunk_ids,
                "anchor_chunk_ids": [trace["root_chunk_id"]],
                "source_previews": [
                    {
                        "chunk_id": chunk_id,
                        "preview": normalize_text(chunk_store.get(chunk_id, {}).get("content", ""))[:220],
                    }
                    for chunk_id in chunk_ids[:3]
                ],
                "evidence_dossier": evidence_dossier,
                "growth_traces": [
                    f"anchor={trace.get('root_chunk_id')}",
                    f"label={trace.get('anchor_label', label)}",
                    f"chunks={len(chunk_ids)}",
                ],
                "nodes": nodes,
                "edges": edges,
                "edge_skeleton": [
                    {
                        "edge": edge_id,
                        "relation_theme": _relation_theme(edge_id, graph),
                        "keywords": _relation_theme(edge_id, graph),
                        "source_chunk_ids": [
                            chunk_id for chunk_id in chunk_ids if edge_id in chunk_to_edges.get(chunk_id, set())
                        ][:3],
                        "role": "evidence",
                        "kind": "answer",
                        "query_rel": round(lexical_overlap_score(query, _relation_theme(edge_id, graph)), 6),
                    }
                    for edge_id in edges[:12]
                ],
                "trace_steps": trace.get("steps", []),
                "chunk_labels": labels,
            }
        )
        if len(groups) >= group_limit:
            break
    return groups


def build_trace_prompt_context(query_row, groups, root_chunk_hits, chunk_store, max_source_chunks, max_source_word_budget):
    """Render a simple trace/path evidence package for the answer LLM."""
    selected_ids = []
    seen = set()
    for group in groups:
        for chunk_id in group.get("supporting_chunk_ids", [])[:2]:
            if chunk_id not in seen:
                seen.add(chunk_id)
                selected_ids.append(chunk_id)
    for group in groups:
        for chunk_id in group.get("supporting_chunk_ids", [])[2:]:
            if chunk_id not in seen:
                seen.add(chunk_id)
                selected_ids.append(chunk_id)
            if len(selected_ids) >= max_source_chunks:
                break
        if len(selected_ids) >= max_source_chunks:
            break
    for item in root_chunk_hits:
        chunk_id = item["chunk_id"]
        if chunk_id not in seen:
            seen.add(chunk_id)
            selected_ids.append(chunk_id)
        if len(selected_ids) >= max_source_chunks:
            break

    source_sections = []
    used_words = 0
    source_id_map = {}
    for chunk_id in selected_ids:
        if len(source_sections) >= max_source_chunks:
            break
        chunk = chunk_store.get(chunk_id)
        if not chunk:
            continue
        content = chunk.get("content", "").strip()
        word_count = approx_word_count(content)
        if source_sections and used_words + word_count > max_source_word_budget:
            break
        source_id = f"src-{len(source_sections) + 1:02d}"
        source_id_map[chunk_id] = source_id
        used_words += word_count
        source_sections.append(
            "\n".join(
                [
                    f"[{source_id}] chunk_id={chunk_id}; words={word_count}",
                    content,
                ]
            )
        )

    path_sections = []
    for group in groups:
        chunk_path = " -> ".join(
            f"{source_id_map.get(chunk_id, chunk_id)}({group.get('chunk_labels', {}).get(chunk_id, '')})"
            for chunk_id in group.get("supporting_chunk_ids", [])
            if chunk_id in source_id_map
        )
        skeleton = []
        for unit in group.get("edge_skeleton", [])[:8]:
            edge = unit.get("edge", ["", ""])
            sources = [source_id_map.get(chunk_id, chunk_id) for chunk_id in unit.get("source_chunk_ids", []) if chunk_id in source_id_map]
            skeleton.append(f"- {edge[0]} -> {edge[1]} :: {unit.get('relation_theme', '')[:90]} [{', '.join(sources)}]")
        path_sections.append(
            "\n".join(
                [
                    f"[{group['group_id']}] {group.get('facet_label', '')}",
                    f"Covered focus: {' | '.join(group.get('focus_entities', [])[:6]) or 'n/a'}",
                    f"Chunk path: {chunk_path or 'n/a'}",
                    "Graph clues:",
                    *(skeleton or ["- n/a"]),
                ]
            )
        )

    context = f"""
-----Research Goal-----
Question: {query_row["query"]}
-----Chunk-Labeled Evidence Paths-----
{chr(10).join(path_sections) if path_sections else '- n/a'}
-----Source Chunks-----
{chr(10).join(source_sections) if source_sections else '- n/a'}
""".strip()
    return {
        "context": context,
        "selected_source_word_count": used_words,
        "selected_source_chunk_count": len(source_sections),
    }


def run_deep_evidence_tracing(
    *,
    query,
    graph,
    chunk_store,
    chunk_retriever,
    dense_index,
    embedding_client,
    chunk_to_nodes,
    chunk_to_edges,
    node_to_chunks,
    chunk_graph,
    top_k,
    candidate_pool_size,
    association_rounds,
    max_steps,
    frontier_edge_top_k,
    group_limit,
    max_source_chunks,
    max_source_word_budget,
    allowed_doc_ids=None,
    query_row=None,
):
    if graph is None:
        raise ValueError("Deep Evidence Tracing requires the LightRAG knowledge graph.")
    if getattr(chunk_retriever, "mode", None) != "dense":
        raise ValueError("Deep Evidence Tracing requires dense chunk retrieval.")
    if dense_index is None or dense_index.normalized_matrix.size == 0:
        raise ValueError("Deep Evidence Tracing requires vdb_chunks.json and dense chunk embeddings.")
    if embedding_client is None:
        raise ValueError("Deep Evidence Tracing requires an embedding client.")
    if not chunk_graph:
        raise ValueError("Deep Evidence Tracing requires a prebuilt corpus-level chunk graph.")
    if getattr(dense_index, "normalized_matrix", None) is None:
        raise ValueError("Deep Evidence Tracing requires normalized dense chunk embeddings.")
    timings = Counter()
    goal_start = perf_counter()
    goal = extract_research_goal(query, graph=graph, atom_limit=max(top_k - 1, 1))
    timings["anchor_seconds"] += perf_counter() - goal_start
    allowed_chunks = _allowed_chunk_ids(allowed_doc_ids, chunk_store)
    all_traces = []
    all_root_hits = []
    rounds = []
    active_goal_items = [query] + goal["query_atoms"] + goal["target_entities"]
    active_goal_items = list(dict.fromkeys(item for item in active_goal_items if normalize_text(item)))[: max(top_k, 1)]
    used_anchor_ids = set()

    for round_index in range(1, max(association_rounds, 1) + 1):
        round_goal = {
            "answer_type": goal["answer_type"],
            "query_atoms": [item for item in active_goal_items if item != query][: max(top_k, 1)],
            "target_entities": [],
        }
        anchor_start = perf_counter()
        anchors, retained_goal_items, _ = select_goal_covering_anchors(
            query=query,
            goal=round_goal,
            chunk_retriever=chunk_retriever,
            chunk_store=chunk_store,
            chunk_to_nodes=chunk_to_nodes,
            chunk_graph=chunk_graph,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
            allowed_doc_ids=allowed_doc_ids,
        )
        timings["anchor_seconds"] += perf_counter() - anchor_start
        new_anchors = [item for item in anchors if item["chunk_id"] not in used_anchor_ids]
        if not new_anchors:
            break
        used_anchor_ids.update(item["chunk_id"] for item in new_anchors)
        all_root_hits.extend(new_anchors)
        round_traces = []
        expand_start = perf_counter()
        for anchor in new_anchors:
            trace = _trace_one_anchor(
                anchor_hit=anchor,
                goal_items=retained_goal_items,
                graph=graph,
                chunk_store=chunk_store,
                chunk_to_edges=chunk_to_edges,
                chunk_graph=chunk_graph,
                dense_index=dense_index,
                embedding_client=embedding_client,
                max_steps=max_steps,
                frontier_edge_top_k=frontier_edge_top_k,
                allowed_chunks=allowed_chunks,
            )
            round_traces.append(trace)
            all_traces.append(trace)
        round_chunk_ids = list(dict.fromkeys(chunk_id for trace in round_traces for chunk_id in trace["selected_chunk_ids"]))
        covered = _covered_goal_items(
            round_chunk_ids,
            retained_goal_items,
            dense_index=dense_index,
            embedding_client=embedding_client,
        )
        emergent = _emergent_goal_items(
            active_goal_items=retained_goal_items,
            trace_chunk_ids=round_chunk_ids,
            graph=graph,
            chunk_to_nodes=chunk_to_nodes,
            embedding_client=embedding_client,
            top_k=top_k,
        )
        timings["expand_seconds"] += perf_counter() - expand_start
        uncovered = [item for item in retained_goal_items if item not in covered and item != query]
        next_goal_items = list(dict.fromkeys(uncovered + emergent))[: max(top_k, 1)]
        rounds.append(
            {
                "round": round_index,
                "goal_items": retained_goal_items,
                "anchor_count": len(new_anchors),
                "trace_count": len(round_traces),
                "selected_chunk_count": len(round_chunk_ids),
                "covered_goal_items": sorted(covered),
                "uncovered_goal_items": uncovered,
                "emergent_goal_items": emergent,
            }
        )
        if not next_goal_items:
            break
        active_goal_items = next_goal_items

    organize_start = perf_counter()
    final_chunk_ids = list(dict.fromkeys(chunk_id for trace in all_traces for chunk_id in trace["selected_chunk_ids"]))
    root_nodes = sorted({node_id for chunk_id in [item["chunk_id"] for item in all_root_hits] for node_id in chunk_to_nodes.get(chunk_id, set())})
    root_edges = sorted({edge_id for chunk_id in [item["chunk_id"] for item in all_root_hits] for edge_id in chunk_to_edges.get(chunk_id, set())})
    final_nodes = sorted({node_id for chunk_id in final_chunk_ids for node_id in chunk_to_nodes.get(chunk_id, set())})
    final_edges = sorted({edge_id for chunk_id in final_chunk_ids for edge_id in chunk_to_edges.get(chunk_id, set())})
    groups = build_trace_groups(
        query=query,
        traces=all_traces,
        graph=graph,
        chunk_store=chunk_store,
        chunk_to_nodes=chunk_to_nodes,
        chunk_to_edges=chunk_to_edges,
        group_limit=group_limit,
    )
    prompt_payload = build_trace_prompt_context(
        query_row=query_row or {"query": query},
        groups=groups,
        root_chunk_hits=all_root_hits,
        chunk_store=chunk_store,
        max_source_chunks=max_source_chunks,
        max_source_word_budget=max_source_word_budget,
    )
    timings["organize_seconds"] += perf_counter() - organize_start
    return {
        "research_goal": goal,
        "root_chunk_hits": all_root_hits,
        "root_chunk_ids": [item["chunk_id"] for item in all_root_hits],
        "root_nodes": root_nodes,
        "root_edges": root_edges,
        "root_traces": all_traces,
        "rounds": rounds,
        "final_chunk_ids": final_chunk_ids,
        "final_nodes": final_nodes,
        "final_edges": final_edges,
        "knowledge_groups": groups,
        "prompt_payload": prompt_payload,
        "timings": {
            "anchor_seconds": float(timings["anchor_seconds"]),
            "expand_seconds": float(timings["expand_seconds"]),
            "organize_seconds": float(timings["organize_seconds"]),
        },
    }
