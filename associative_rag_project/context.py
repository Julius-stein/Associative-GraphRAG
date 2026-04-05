"""Turn the final graph into LLM-facing evidence groups and source bundles."""

from collections import Counter, defaultdict

import networkx as nx

from .common import approx_word_count, build_csv, lexical_overlap_score, normalize_text
from .retrieval import normalize_relation_category


def rank_supporting_chunks(final_nodes, final_edges, root_chunk_ids, node_to_chunks, edge_to_chunks):
    """Rank chunks by direct support of the final graph.

    Root chunks are no longer forced to dominate via a huge constant boost.
    Grounding is preserved later through anchor selection inside each group.
    """
    chunk_scores = Counter()
    for node_id in final_nodes:
        for chunk_id in node_to_chunks.get(node_id, set()):
            chunk_scores[chunk_id] += 1
    for edge_id in final_edges:
        for chunk_id in edge_to_chunks.get(edge_id, set()):
            chunk_scores[chunk_id] += 1
    for chunk_id in root_chunk_ids:
        chunk_scores[chunk_id] += 1
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


def _chunk_doc_and_order(chunk_store, chunk_id):
    chunk = chunk_store.get(chunk_id, {})
    return chunk.get("full_doc_id"), chunk.get("chunk_order_index")


def _violates_local_band_cap(chunk_store, chunk_id, selected_orders_by_doc, band_radius=1, band_cap=2):
    doc_id, order = _chunk_doc_and_order(chunk_store, chunk_id)
    if doc_id is None or order is None:
        return False
    nearby = sum(1 for existing_order in selected_orders_by_doc.get(doc_id, []) if abs(existing_order - order) <= band_radius)
    return nearby >= band_cap


def choose_diverse_source_chunks(
    facet_groups,
    ranked_chunk_ids,
    chunk_store,
    root_chunk_ids,
    max_source_chunks,
    max_source_word_budget,
):
    """Pick source chunks with group diversity before falling back to global rank.

    Selection policy:
    - each facet group gets a first chance to contribute one chunk
    - then each facet can contribute one more chunk if budget remains
    - finally global rank fills the rest

    Adjacent chunks from the same document are capped so one dense support
    chain cannot crowd out other useful facets.
    """
    selected_ids = []
    used_words = 0
    seen = set()
    root_chunk_id_set = set(root_chunk_ids)
    rank_index = {chunk_id: idx for idx, chunk_id in enumerate(ranked_chunk_ids)}
    selected_orders_by_doc = defaultdict(list)
    group_candidates = []

    for group in facet_groups:
        roots, supports = _ranked_group_chunks(group["supporting_chunk_ids"], rank_index, root_chunk_id_set)
        ordered = []
        for chunk_id in roots[:2] + supports[:6]:
            if chunk_id not in ordered:
                ordered.append(chunk_id)
        group_candidates.append(ordered)

    def try_add(chunk_id):
        nonlocal used_words
        if chunk_id in seen or len(selected_ids) >= max_source_chunks:
            return False
        chunk = chunk_store.get(chunk_id)
        if chunk is None:
            return False
        if _violates_local_band_cap(chunk_store, chunk_id, selected_orders_by_doc):
            return False
        word_count = approx_word_count(chunk.get("content", ""))
        if selected_ids and used_words + word_count > max_source_word_budget:
            return False
        selected_ids.append(chunk_id)
        seen.add(chunk_id)
        used_words += word_count
        doc_id, order = _chunk_doc_and_order(chunk_store, chunk_id)
        if doc_id is not None and order is not None:
            selected_orders_by_doc[doc_id].append(order)
        return True

    # Round 1: let each facet secure one chunk if possible.
    for ordered_chunk_ids in group_candidates:
        for chunk_id in ordered_chunk_ids:
            if try_add(chunk_id):
                break
        if len(selected_ids) >= max_source_chunks:
            break

    # Round 2: allow each facet one more chunk before global fallback.
    if len(selected_ids) < max_source_chunks:
        for ordered_chunk_ids in group_candidates:
            added = 0
            for chunk_id in ordered_chunk_ids:
                if chunk_id in seen:
                    continue
                if try_add(chunk_id):
                    added += 1
                if added >= 1:
                    break
            if len(selected_ids) >= max_source_chunks:
                break

    for chunk_id in ranked_chunk_ids:
        try_add(chunk_id)
        if len(selected_ids) >= max_source_chunks:
            break

    selected = [(chunk_id, chunk_store[chunk_id], approx_word_count(chunk_store[chunk_id].get("content", ""))) for chunk_id in selected_ids]
    return selected, used_words


def _build_group_summary(component_nodes, relation_categories, source_previews, anchor_count):
    """Produce a short orienting summary for one knowledge group.

    The summary is intentionally anchor-based rather than component-based:
    it should describe what the selected evidence anchors say, not summarize
    every chunk connected to a large component.
    """
    lead_entities = ", ".join(component_nodes[:3]) if component_nodes else "the retrieved evidence"
    themes = [theme for theme, _ in relation_categories.most_common(3) if theme]
    summary_parts = [f"This group centers on {lead_entities}."]
    if themes:
        summary_parts.append(f"It mainly connects evidence through {', '.join(themes)}.")
    if anchor_count > 1:
        summary_parts.append(f"It is anchored by {anchor_count} evidence chunks.")
    if source_previews:
        summary_parts.append(f"Representative anchor evidence: {source_previews[0]['preview'][:180]}")
    return " ".join(summary_parts)


def _group_overlap_ratio(left, right):
    if not left or not right:
        return 0.0
    left = set(left)
    right = set(right)
    return len(left & right) / max(len(left | right), 1)


def _build_anchor_previews(anchor_chunk_ids, chunk_store, limit=2):
    previews = []
    for chunk_id in anchor_chunk_ids[:limit]:
        chunk = chunk_store.get(chunk_id, {})
        previews.append(
            {
                "chunk_id": chunk_id,
                "preview": " ".join(chunk.get("content", "").split())[:220],
            }
        )
    return previews


def _coverage_checklist_lines(facet_groups, limit=6):
    lines = []
    seen = set()
    for group in facet_groups:
        label = normalize_text(group.get("facet_label", ""))
        if not label or label in seen:
            continue
        seen.add(label)
        themes = [normalize_text(theme) for theme in group.get("relation_themes", [])[:2] if normalize_text(theme)]
        theme_suffix = f" [{', '.join(themes)}]" if themes else ""
        lines.append(f"- {label}{theme_suffix}")
        if len(lines) >= limit:
            break
    return lines


def _build_chunk_local_index(final_nodes, final_edges, node_to_chunks, edge_to_chunks):
    chunk_to_final_nodes = defaultdict(set)
    chunk_to_final_edges = defaultdict(set)
    for node_id in final_nodes:
        for chunk_id in node_to_chunks.get(node_id, set()):
            chunk_to_final_nodes[chunk_id].add(node_id)
    for edge_id in final_edges:
        for chunk_id in edge_to_chunks.get(edge_id, set()):
            chunk_to_final_edges[chunk_id].add(edge_id)
    return dict(chunk_to_final_nodes), dict(chunk_to_final_edges)


def _rank_local_support_chunks(query, root_chunk_id, seed_nodes, seed_edges, node_to_chunks, edge_to_chunks, chunk_store):
    support_chunk_ids = set()
    for node_id in seed_nodes:
        support_chunk_ids.update(node_to_chunks.get(node_id, set()))
    for edge_id in seed_edges:
        support_chunk_ids.update(edge_to_chunks.get(edge_id, set()))
    support_chunk_ids.discard(root_chunk_id)
    return sorted(
        support_chunk_ids,
        key=lambda chunk_id: (
            -lexical_overlap_score(query, chunk_store.get(chunk_id, {}).get("content", "")),
            -approx_word_count(chunk_store.get(chunk_id, {}).get("content", "")),
            chunk_id,
        ),
    )


def _build_anchor_local_group(
    query,
    root_chunk_id,
    graph,
    final_nodes,
    final_edges,
    node_roles,
    edge_roles,
    node_to_chunks,
    edge_to_chunks,
    chunk_store,
    chunk_to_final_nodes,
    chunk_to_final_edges,
):
    seed_nodes = set(chunk_to_final_nodes.get(root_chunk_id, set()))
    seed_edges = set(chunk_to_final_edges.get(root_chunk_id, set()))
    if not seed_nodes and not seed_edges:
        return None

    support_chunks = _rank_local_support_chunks(
        query=query,
        root_chunk_id=root_chunk_id,
        seed_nodes=seed_nodes,
        seed_edges=seed_edges,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        chunk_store=chunk_store,
    )
    anchor_chunk_ids = [root_chunk_id] + support_chunks[:2]
    local_nodes = set(seed_nodes)
    local_edges = set(seed_edges)
    for chunk_id in anchor_chunk_ids[1:]:
        local_nodes.update(chunk_to_final_nodes.get(chunk_id, set()))
        local_edges.update(chunk_to_final_edges.get(chunk_id, set()))
    for edge_id in list(local_edges):
        local_nodes.update(edge_id[:2])
    local_edges.update(
        edge_id for edge_id in final_edges if edge_id[0] in local_nodes and edge_id[1] in local_nodes
    )
    supporting_chunk_ids = set(anchor_chunk_ids)
    relation_categories = Counter()
    for node_id in local_nodes:
        supporting_chunk_ids.update(node_to_chunks.get(node_id, set()))
    for edge_id in local_edges:
        supporting_chunk_ids.update(edge_to_chunks.get(edge_id, set()))
        edge_data = graph.get_edge_data(edge_id[0], edge_id[1]) or {}
        relation_categories[normalize_relation_category(edge_data)] += 1

    component_nodes = sorted(local_nodes)
    component_edges = sorted(local_edges)
    sorted_chunk_ids = sorted(supporting_chunk_ids)
    source_previews = _build_anchor_previews(anchor_chunk_ids, chunk_store, limit=3)
    group_text = " ".join(
        component_nodes[:12]
        + [theme for theme, _ in relation_categories.most_common(5)]
        + [item["preview"] for item in source_previews]
    )
    query_rel = lexical_overlap_score(query, group_text)
    cohesion = len(component_edges) / max(len(component_nodes), 1)
    return {
        "node_count": len(component_nodes),
        "edge_count": len(component_edges),
        "group_score": round(query_rel, 6),
        "query_rel": round(query_rel, 6),
        "cohesion": round(cohesion, 6),
        "anchor_support": len(anchor_chunk_ids),
        "has_root_anchor": True,
        "near_root": True,
        "node_roles": dict(Counter(node_roles.get(node_id, "other") for node_id in component_nodes)),
        "edge_roles": dict(Counter(edge_roles.get(edge_id, "other") for edge_id in component_edges)),
        "relation_themes": [theme for theme, _ in relation_categories.most_common(5)],
        "supporting_chunk_ids": sorted_chunk_ids,
        "anchor_chunk_ids": anchor_chunk_ids,
        "source_previews": source_previews,
        "group_summary": _build_group_summary(component_nodes, relation_categories, source_previews, len(anchor_chunk_ids)),
        "nodes": component_nodes,
        "edges": component_edges,
    }


def build_knowledge_groups(
    query,
    graph,
    final_nodes,
    final_edges,
    root_chunk_ids,
    node_roles,
    edge_roles,
    node_to_chunks,
    edge_to_chunks,
    chunk_store,
    group_limit,
):
    """Package connected subgraphs into query-facing knowledge groups.

    The organization stage is intentionally different from association:
    - association expands by evidence gain
    - organization builds anchor-local groups around grounded root chunks
      and only then orders/deduplicates them
    """
    chunk_to_final_nodes, chunk_to_final_edges = _build_chunk_local_index(
        final_nodes=final_nodes,
        final_edges=final_edges,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
    )
    candidate_groups = []
    for root_chunk_id in root_chunk_ids:
        group = _build_anchor_local_group(
            query=query,
            root_chunk_id=root_chunk_id,
            graph=graph,
            final_nodes=final_nodes,
            final_edges=final_edges,
            node_roles=node_roles,
            edge_roles=edge_roles,
            node_to_chunks=node_to_chunks,
            edge_to_chunks=edge_to_chunks,
            chunk_store=chunk_store,
            chunk_to_final_nodes=chunk_to_final_nodes,
            chunk_to_final_edges=chunk_to_final_edges,
        )
        if group is not None:
            candidate_groups.append(group)

    def _group_rank_key(item):
        return (
            -item["query_rel"],
            -item["anchor_support"],
            -item["node_count"],
            item["nodes"][0] if item["nodes"] else "",
        )

    candidate_groups.sort(key=_group_rank_key)

    groups = []
    for candidate in candidate_groups:
        if len(groups) >= group_limit:
            break
        redundant = False
        for chosen in groups:
            chunk_overlap = _group_overlap_ratio(candidate["supporting_chunk_ids"], chosen["supporting_chunk_ids"])
            if chunk_overlap >= 0.60:
                redundant = True
                break
        if redundant:
            continue
        groups.append(candidate)

    for group_index, group in enumerate(groups, start=1):
        group["group_id"] = f"kg-{group_index:02d}"
    return groups


def build_prompt_context(
    query_row,
    query_contract,
    root_chunk_hits,
    top_root_nodes,
    top_root_edges,
    node_roles,
    edge_roles,
    semantic_nodes,
    semantic_edges,
    final_nodes,
    final_edges,
    facet_groups,
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
        facet_groups=facet_groups,
        ranked_chunk_ids=chunk_rank,
        chunk_store=chunk_store,
        root_chunk_ids=root_chunk_ids,
        max_source_chunks=max_source_chunks,
        max_source_word_budget=max_source_word_budget,
    )
    root_chunk_id_set = set(root_chunk_ids)
    chunk_to_group_ids = {}
    for group in facet_groups:
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

    group_rows = [["id", "score", "node_count", "edge_count", "facet", "key_entities", "key_relations", "supporting_chunks"]]
    group_sections = []
    for group in facet_groups:
        key_entities = sorted(
            group.get("focus_entities") or group["nodes"],
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
                f"{group.get('facet_label', group.get('primary_theme', 'facet'))} :: {' / '.join(group.get('region_kinds', []))}",
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
            f"Facet: {group.get('facet_label', group.get('primary_theme', 'facet'))}",
            f"Facet Prompt: {group.get('facet_prompt', '')}",
            f"Summary: {group['group_summary']}",
            f"Themes: {' | '.join(group['relation_themes']) or 'n/a'}",
            f"Evidence Roles: {' | '.join(group.get('region_kinds', [])) or 'n/a'}",
            f"Key Entities: {' | '.join(key_entities) or 'n/a'}",
            "Structural Growth Trace:",
        ]
        if group.get("growth_traces"):
            group_section.extend(f"- {trace}" for trace in group["growth_traces"][:4])
        else:
            group_section.append("- n/a")
        group_section.extend(
            [
            "Anchor Evidence:",
            ]
        )
        if group.get("anchor_chunk_ids"):
            for chunk_id in group["anchor_chunk_ids"][:2]:
                chunk_data = chunk_store.get(chunk_id, {})
                role = "root" if chunk_id in root_chunk_id_set else "support"
                group_section.append(f"- {role} {chunk_id}: {' '.join(chunk_data.get('content', '').split())[:220]}")
        else:
            group_section.append("- n/a")
        group_section.extend(
            [
            "Key Relations:",
            ]
        )
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

    coverage_checklist = _coverage_checklist_lines(facet_groups)

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
-----Facet Groups-----
```csv
{build_csv(group_rows)}
```
-----Coverage Checklist-----
{chr(10).join(coverage_checklist) if coverage_checklist else '- n/a'}
-----Facet Group Dossiers-----
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
