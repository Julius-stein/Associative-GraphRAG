"""Turn the final graph into LLM-facing evidence groups and source bundles."""

from collections import Counter

import networkx as nx

from .common import (
    approx_word_count,
    build_csv,
    lexical_overlap_score,
    normalize_text,
    query_prefers_technical_content,
    technical_density,
)
from .retrieval import normalize_relation_category


def rank_supporting_chunks(final_nodes, final_edges, root_chunk_ids, node_to_chunks, edge_to_chunks):
    """Rank chunks by how much final graph evidence they support.

    Root chunks are strongly boosted so the answer always remains grounded in
    the initial dense retrieval rather than drifting to graph-only evidence.
    """
    chunk_scores = Counter()
    for node_id in final_nodes:
        for chunk_id in node_to_chunks.get(node_id, set()):
            chunk_scores[chunk_id] += 1
    for edge_id in final_edges:
        for chunk_id in edge_to_chunks.get(edge_id, set()):
            chunk_scores[chunk_id] += 2
    for chunk_id in root_chunk_ids:
        chunk_scores[chunk_id] += 1000
    return [chunk_id for chunk_id, _ in sorted(chunk_scores.items(), key=lambda item: (-item[1], item[0]))]


def truncate_source_chunks(chunk_ids, chunk_store, max_source_chunks, max_source_word_budget):
    """Simple budget-based chunk truncation helper."""
    selected = []
    used_words = 0
    for chunk_id in chunk_ids:
        if len(selected) >= max_source_chunks:
            break
        chunk = chunk_store.get(chunk_id)
        if chunk is None:
            continue
        content = chunk.get("content", "")
        word_count = approx_word_count(content)
        if selected and used_words + word_count > max_source_word_budget:
            break
        selected.append((chunk_id, chunk, word_count))
        used_words += word_count
    return selected, used_words


def _ranked_group_chunks(supporting_chunk_ids, rank_index, root_chunk_id_set):
    """Order group chunks to preserve grounding while surfacing non-root support."""
    ranked = sorted(
        supporting_chunk_ids,
        key=lambda chunk_id: (rank_index.get(chunk_id, 10**9), chunk_id),
    )
    roots = [chunk_id for chunk_id in ranked if chunk_id in root_chunk_id_set]
    supports = [chunk_id for chunk_id in ranked if chunk_id not in root_chunk_id_set]
    return roots, supports


def choose_diverse_source_chunks(
    knowledge_groups,
    ranked_chunk_ids,
    chunk_store,
    root_chunk_ids,
    max_source_chunks,
    max_source_word_budget,
):
    """Pick source chunks with group diversity before falling back to global rank.

    Each group first gets a chance to contribute one grounded root chunk and one
    non-root support chunk. This keeps the final prompt centered on dense roots
    while ensuring graph-expanded evidence survives into the final package.
    """
    selected_ids = []
    used_words = 0
    seen = set()
    root_chunk_id_set = set(root_chunk_ids)
    rank_index = {chunk_id: idx for idx, chunk_id in enumerate(ranked_chunk_ids)}

    def try_add(chunk_id):
        nonlocal used_words
        if chunk_id in seen or len(selected_ids) >= max_source_chunks:
            return
        chunk = chunk_store.get(chunk_id)
        if chunk is None:
            return
        word_count = approx_word_count(chunk.get("content", ""))
        if selected_ids and used_words + word_count > max_source_word_budget:
            return
        selected_ids.append(chunk_id)
        seen.add(chunk_id)
        used_words += word_count

    for group in knowledge_groups:
        roots, supports = _ranked_group_chunks(group["supporting_chunk_ids"], rank_index, root_chunk_id_set)
        if roots:
            try_add(roots[0])
        elif supports:
            try_add(supports[0])
        if supports:
            try_add(supports[0])
        if len(selected_ids) >= max_source_chunks:
            break

    for chunk_id in ranked_chunk_ids:
        try_add(chunk_id)
        if len(selected_ids) >= max_source_chunks:
            break

    selected = [(chunk_id, chunk_store[chunk_id], approx_word_count(chunk_store[chunk_id].get("content", ""))) for chunk_id in selected_ids]
    return selected, used_words


def _build_group_summary(component_nodes, relation_categories, source_previews, chunk_count):
    """Produce a short orienting summary for one knowledge group."""
    lead_entities = ", ".join(component_nodes[:3]) if component_nodes else "the retrieved evidence"
    themes = [theme for theme, _ in relation_categories.most_common(3) if theme]
    summary_parts = [f"This group centers on {lead_entities}."]
    if themes:
        summary_parts.append(f"It mainly connects evidence through {', '.join(themes)}.")
    if chunk_count > 1:
        summary_parts.append(f"It is supported by {chunk_count} chunks.")
    if source_previews:
        summary_parts.append(f"Representative evidence: {source_previews[0]['preview'][:180]}")
    return " ".join(summary_parts)


def build_knowledge_groups(
    query,
    graph,
    final_nodes,
    final_edges,
    node_roles,
    edge_roles,
    node_to_chunks,
    edge_to_chunks,
    chunk_store,
    group_limit,
):
    """Package connected subgraphs into query-facing knowledge groups."""
    group_graph = nx.Graph()
    group_graph.add_nodes_from(final_nodes)
    group_graph.add_edges_from(final_edges)
    groups = []
    query_is_technical = query_prefers_technical_content(query)
    components = sorted(nx.connected_components(group_graph), key=lambda item: (-len(item), sorted(item)[0]))
    for component in components:
        component_nodes = sorted(component)
        component_edges = sorted(edge_id for edge_id in final_edges if edge_id[0] in component and edge_id[1] in component)
        if len(component_nodes) < 2 and len(component_edges) == 0:
            continue
        chunk_ids = set()
        relation_categories = Counter()
        for node_id in component_nodes:
            chunk_ids.update(node_to_chunks.get(node_id, set()))
        for edge_id in component_edges:
            chunk_ids.update(edge_to_chunks.get(edge_id, set()))
            edge_data = graph.get_edge_data(edge_id[0], edge_id[1]) or {}
            relation_categories[normalize_relation_category(edge_data)] += 1
        sorted_chunk_ids = sorted(chunk_ids)
        source_previews = []
        for chunk_id in sorted_chunk_ids[:4]:
            chunk = chunk_store.get(chunk_id, {})
            source_previews.append(
                {
                    "chunk_id": chunk_id,
                    "preview": " ".join(chunk.get("content", "").split())[:200],
                }
            )
        root_role_count = sum(1 for node_id in component_nodes if node_roles.get(node_id) == "root") + sum(
            1 for edge_id in component_edges if edge_roles.get(edge_id) == "root"
        )
        structural_role_count = sum(1 for node_id in component_nodes if node_roles.get(node_id) == "structural") + sum(
            1 for edge_id in component_edges if edge_roles.get(edge_id) == "structural"
        )
        group_text = " ".join(
            component_nodes[:12]
            + [theme for theme, _ in relation_categories.most_common(5)]
            + [item["preview"] for item in source_previews]
        )
        query_rel = lexical_overlap_score(query, group_text)
        root_density = root_role_count / max(len(component_nodes) + len(component_edges), 1)
        structure_density = structural_role_count / max(len(component_nodes) + len(component_edges), 1)
        support_span = min(len(sorted_chunk_ids) / 8.0, 1.0)
        size_term = min((len(component_nodes) + len(component_edges)) / 14.0, 1.0)
        relation_term = min(len(relation_categories) / 4.0, 1.0)
        technical_penalty = 0.0 if query_is_technical else 0.18 * technical_density(group_text)
        # A group should be relevant, supported by multiple chunks, reasonably
        # sized, and not dominated by technical metadata unless the query asks
        # for technical content.
        group_score = (
            0.42 * query_rel
            + 0.18 * support_span
            + 0.12 * size_term
            + 0.10 * relation_term
            + 0.10 * root_density
            + 0.08 * structure_density
            - technical_penalty
        )
        if len(sorted_chunk_ids) < 2 and len(component_nodes) < 4:
            group_score -= 0.06
        if not relation_categories and query_rel < 0.05:
            group_score -= 0.04
        groups.append(
            {
                "node_count": len(component_nodes),
                "edge_count": len(component_edges),
                "group_score": round(group_score, 6),
                "query_rel": round(query_rel, 6),
                "technical_penalty": round(technical_penalty, 6),
                "node_roles": dict(Counter(node_roles.get(node_id, "other") for node_id in component_nodes)),
                "edge_roles": dict(Counter(edge_roles.get(edge_id, "other") for edge_id in component_edges)),
                "relation_themes": [theme for theme, _ in relation_categories.most_common(5)],
                "supporting_chunk_ids": sorted_chunk_ids,
                "source_previews": source_previews,
                "group_summary": _build_group_summary(component_nodes, relation_categories, source_previews, len(sorted_chunk_ids)),
                "nodes": component_nodes,
                "edges": component_edges,
            }
        )
    groups.sort(key=lambda item: (-item["group_score"], -item["node_count"], item["nodes"][0] if item["nodes"] else ""))
    for group_index, group in enumerate(groups[:group_limit], start=1):
        group["group_id"] = f"kg-{group_index:02d}"
    groups = groups[:group_limit]
    return groups


def build_prompt_context(
    query_row,
    root_chunk_hits,
    top_root_nodes,
    top_root_edges,
    node_roles,
    edge_roles,
    semantic_nodes,
    semantic_edges,
    final_nodes,
    final_edges,
    knowledge_groups,
    chunk_store,
    node_to_chunks,
    edge_to_chunks,
    max_source_chunks,
    max_source_word_budget,
):
    """Assemble the final evidence package shown to the answer-generation LLM."""
    root_chunk_ids = [item["chunk_id"] for item in root_chunk_hits]
    chunk_rank = rank_supporting_chunks(final_nodes, final_edges, root_chunk_ids, node_to_chunks, edge_to_chunks)
    selected_source_chunks, used_words = choose_diverse_source_chunks(
        knowledge_groups=knowledge_groups,
        ranked_chunk_ids=chunk_rank,
        chunk_store=chunk_store,
        root_chunk_ids=root_chunk_ids,
        max_source_chunks=max_source_chunks,
        max_source_word_budget=max_source_word_budget,
    )
    root_chunk_id_set = set(root_chunk_ids)
    chunk_to_group_ids = {}
    for group in knowledge_groups:
        for chunk_id in group["supporting_chunk_ids"]:
            chunk_to_group_ids.setdefault(chunk_id, []).append(group["group_id"])
    source_id_map = {chunk_id: f"src-{index:02d}" for index, (chunk_id, _, _) in enumerate(selected_source_chunks, start=1)}

    root_chunk_rows = [["id", "score", "chunk_order", "content_preview"]]
    for item in root_chunk_hits:
        chunk = chunk_store[item["chunk_id"]]
        root_chunk_rows.append(
            [
                item["chunk_id"],
                f"{item['score_norm']:.4f}",
                chunk.get("chunk_order_index", -1),
                " ".join(chunk.get("content", "").split())[:220],
            ]
        )

    node_lookup = {item["id"]: item for item in top_root_nodes}
    node_lookup.update({item["id"]: item for item in semantic_nodes})
    root_edge_lookup = {item["edge"]: item for item in top_root_edges}
    semantic_edge_lookup = {item["edge"]: item for item in semantic_edges}
    role_priority = {"root": 3, "structural": 2, "semantic": 1, "other": 0}
    entity_rows = [["id", "entity", "role", "source_chunk_count", "description"]]
    ranked_nodes = sorted(
        final_nodes,
        key=lambda node_id: (
            -lexical_overlap_score(query_row["query"], f"{node_id} {normalize_text(node_lookup.get(node_id, {}).get('description', ''))}"),
            -role_priority.get(node_roles.get(node_id, "other"), 0),
            -len(node_to_chunks.get(node_id, set())),
            node_id,
        ),
    )[:30]
    for node_id in ranked_nodes:
        item = node_lookup.get(node_id, {"description": ""})
        entity_rows.append(
            [
                len(entity_rows) - 1,
                node_id,
                node_roles.get(node_id, "other"),
                len(node_to_chunks.get(node_id, set())),
                normalize_text(item.get("description", ""))[:200],
            ]
        )

    relation_rows = [["id", "source", "target", "role", "source_chunk_count", "keywords"]]
    ranked_edges = sorted(
        final_edges,
        key=lambda edge_id: (
            -lexical_overlap_score(
                query_row["query"],
                " ".join(
                    [
                        edge_id[0],
                        edge_id[1],
                        normalize_text((semantic_edge_lookup.get(edge_id) or root_edge_lookup.get(edge_id) or {}).get("keywords", "")),
                    ]
                ),
            ),
            -role_priority.get(edge_roles.get(edge_id, "other"), 0),
            -len(edge_to_chunks.get(edge_id, set())),
            edge_id,
        ),
    )[:30]
    for edge_id in ranked_edges:
        item = semantic_edge_lookup.get(edge_id) or root_edge_lookup.get(edge_id) or {}
        relation_rows.append(
            [
                len(relation_rows) - 1,
                edge_id[0],
                edge_id[1],
                edge_roles.get(edge_id, "other"),
                len(edge_to_chunks.get(edge_id, set())),
                normalize_text(item.get("keywords", ""))[:120],
            ]
        )

    group_rows = [["id", "score", "node_count", "edge_count", "themes", "key_entities", "key_relations", "supporting_chunks"]]
    group_sections = []
    for group in knowledge_groups:
        key_entities = sorted(
            group["nodes"],
            key=lambda node_id: (
                -lexical_overlap_score(query_row["query"], f"{node_id} {normalize_text(node_lookup.get(node_id, {}).get('description', ''))}"),
                -role_priority.get(node_roles.get(node_id, "other"), 0),
                -len(node_to_chunks.get(node_id, set())),
                node_id,
            ),
        )[:6]
        key_relations = sorted(
            group["edges"],
            key=lambda edge_id: (
                -lexical_overlap_score(
                    query_row["query"],
                    " ".join(
                        [
                            edge_id[0],
                            edge_id[1],
                            normalize_text((semantic_edge_lookup.get(edge_id) or root_edge_lookup.get(edge_id) or {}).get("keywords", "")),
                        ]
                    ),
                ),
                -role_priority.get(edge_roles.get(edge_id, "other"), 0),
                -len(edge_to_chunks.get(edge_id, set())),
                edge_id,
            ),
        )[:5]
        group_rows.append(
            [
                group["group_id"],
                f"{group['group_score']:.4f}",
                group["node_count"],
                group["edge_count"],
                " | ".join(group["relation_themes"]),
                " | ".join(key_entities),
                " | ".join(
                    f"{edge_id[0]} -> {edge_id[1]} ({normalize_text((semantic_edge_lookup.get(edge_id) or root_edge_lookup.get(edge_id) or {}).get('keywords',''))[:40]})"
                    for edge_id in key_relations
                ),
                len(group["supporting_chunk_ids"]),
            ]
        )
        linked_source_rows = [["source_id", "role", "word_count", "content_preview"]]
        linked_chunk_ids = sorted(
            (chunk_id for chunk_id in group["supporting_chunk_ids"] if chunk_id in source_id_map),
            key=lambda chunk_id: source_id_map[chunk_id],
        )[:3]
        for chunk_id in linked_chunk_ids:
            chunk_data = chunk_store.get(chunk_id, {})
            linked_source_rows.append(
                [
                    source_id_map[chunk_id],
                    "root" if chunk_id in root_chunk_id_set else "support",
                    approx_word_count(chunk_data.get("content", "")),
                    " ".join(chunk_data.get("content", "").split())[:220],
                ]
            )
        group_section = [
            f"[{group['group_id']}] score={group['group_score']:.4f} nodes={group['node_count']} edges={group['edge_count']}",
            f"Summary: {group['group_summary']}",
            f"Themes: {' | '.join(group['relation_themes']) or 'n/a'}",
            f"Key Entities: {' | '.join(key_entities) or 'n/a'}",
            "Key Relations:",
        ]
        if key_relations:
            group_section.extend(
                f"- {edge_id[0]} -> {edge_id[1]} ({normalize_text((semantic_edge_lookup.get(edge_id) or root_edge_lookup.get(edge_id) or {}).get('keywords',''))[:60]})"
                for edge_id in key_relations
            )
        else:
            group_section.append("- n/a")
        group_section.append("Linked Sources:")
        group_section.append("```csv")
        group_section.append(build_csv(linked_source_rows))
        group_section.append("```")
        group_sections.append("\n".join(group_section))

    source_rows = [["id", "role", "linked_groups", "word_count", "content"]]
    for index, (chunk_id, chunk_data, word_count) in enumerate(selected_source_chunks):
        source_rows.append(
            [
                source_id_map[chunk_id],
                "root" if chunk_id in root_chunk_id_set else "support",
                " | ".join(chunk_to_group_ids.get(chunk_id, [])[:4]),
                word_count,
                chunk_data.get("content", "").replace("\n", " "),
            ]
        )

    context = f"""
-----Root Chunks-----
```csv
{build_csv(root_chunk_rows)}
```
-----Focused Entities-----
```csv
{build_csv(entity_rows)}
```
-----Focused Relations-----
```csv
{build_csv(relation_rows)}
```
-----Knowledge Groups-----
```csv
{build_csv(group_rows)}
```
-----Knowledge Group Dossiers-----
{chr(10).join(group_sections)}
-----Sources-----
```csv
{build_csv(source_rows)}
```
""".strip()

    return {
        "context": context,
        "selected_source_word_count": used_words,
        "selected_source_chunk_count": len(selected_source_chunks),
    }
