"""Turn the final graph into LLM-facing evidence groups and source bundles.

将最终证据图组织成面向 LLM 的知识组与来源包，方便生成阶段使用。
"""

from collections import Counter, defaultdict

import networkx as nx

from .common import approx_word_count, lexical_overlap_score, normalize_text
from .retrieval import normalize_relation_category


def rank_supporting_chunks(final_nodes, final_edges, root_chunk_ids, node_to_chunks, edge_to_chunks):
    """Rank chunks by direct support of the final graph.

    Root chunks are no longer forced to dominate via a huge constant boost.
    Grounding is preserved later through anchor selection inside each group.

    按最终节点/边对 chunk 的直接支持度排序，便于后续来源选择。
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


def _truncate_to_words(text, max_words):
    if max_words is None or max_words <= 0:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _group_source_priority(group):
    """Lexicographic quality key for deciding which groups receive text budget."""
    return (
        group.get("dossier_query_rel", group.get("query_rel", 0.0)),
        group.get("dossier_answerability", 0),
        group.get("dossier_specificity", 0),
        group.get("edge_query_rel", 0.0),
        group.get("edge_count", 0),
        -len(group.get("supporting_chunk_ids", [])),
        group.get("group_id", ""),
    )


def _ordered_groups_for_source_budget(facet_groups):
    return sorted(facet_groups, key=_group_source_priority, reverse=True)


def _dossier_source_quota(group_rank):
    if group_rank <= 4:
        return 4
    if group_rank <= 6:
        return 1
    return 0


def _dossier_item_priority(item):
    profile = item.get("answerability", {})
    return (
        profile.get("query_rel", 0.0),
        profile.get("signal_count", 0),
        profile.get("specific_count", 0),
        item.get("word_count", 0),
        item.get("chunk_id", ""),
    )


def _ordered_dossier_items(group):
    return sorted(group.get("evidence_dossier", []), key=_dossier_item_priority, reverse=True)


_THEME_SUMMARY_BUCKETS = {
    "context_event": {
        "political",
        "war",
        "revolution",
        "regime",
        "state",
        "society",
        "historical",
        "period",
        "context",
        "crisis",
    },
    "movement_example": {
        "movement",
        "movements",
        "style",
        "school",
        "artist",
        "artists",
        "example",
        "examples",
        "avant",
        "modernism",
    },
    "driver_mechanism": {
        "influence",
        "influenced",
        "response",
        "reaction",
        "mechanism",
        "driver",
        "cause",
        "causal",
        "shift",
        "change",
    },
    "outcome_impact": {
        "impact",
        "effects",
        "effect",
        "outcome",
        "transformation",
        "emergence",
        "result",
        "consequence",
        "value",
        "perception",
    },
    "institution_channel": {
        "museum",
        "academy",
        "institution",
        "market",
        "publication",
        "media",
        "patronage",
        "education",
        "audience",
        "circulation",
    },
}


def _theme_query_bucket_weights(query_text):
    query_tokens = set(normalize_text(query_text).split())
    weights = {}
    for bucket, keywords in _THEME_SUMMARY_BUCKETS.items():
        overlap = len(query_tokens & keywords)
        weights[bucket] = 1.0 + (0.3 if overlap > 0 else 0.0)
    return weights


def _theme_bucket_scores(text):
    tokens = set(normalize_text(text).split())
    scores = {}
    for bucket, keywords in _THEME_SUMMARY_BUCKETS.items():
        scores[bucket] = len(tokens & keywords)
    return scores


def choose_diverse_source_chunks(
    facet_groups,
    ranked_chunk_ids,
    chunk_store,
    root_chunk_ids,
    max_source_chunks,
    max_source_word_budget,
    preferred_chunk_ids=None,
    per_chunk_word_cap=None,
    query_text="",
    chunk_role_lookup=None,
    chunk_to_group_ids=None,
    group_label_lookup=None,
):
    """Pick source chunks with group diversity before falling back to global rank.

    Selection policy:
    - each facet group gets a first chance to contribute one chunk
    - then each facet can contribute one more chunk if budget remains
    - finally global rank fills the rest

    Adjacent chunks from the same document are capped so one dense support
    chain cannot crowd out other useful facets.

    优先从不同知识组中选取来源 chunks，再补齐全局高排序 chunk。
    """
    selected_ids = []
    used_words = 0
    seen = set()
    root_chunk_id_set = set(root_chunk_ids)
    preferred_chunk_ids = list(dict.fromkeys(preferred_chunk_ids or []))
    rank_index = {chunk_id: idx for idx, chunk_id in enumerate(ranked_chunk_ids)}
    selected_orders_by_doc = defaultdict(list)
    group_candidates = []
    chunk_role_lookup = chunk_role_lookup or {}
    chunk_to_group_ids = chunk_to_group_ids or {}
    group_label_lookup = group_label_lookup or {}

    for group in _ordered_groups_for_source_budget(facet_groups):
        dossier_chunk_ids = [item.get("chunk_id") for item in _ordered_dossier_items(group) if item.get("chunk_id")]
        roots, supports = _ranked_group_chunks(group["supporting_chunk_ids"], rank_index, root_chunk_id_set)
        ordered = []
        for chunk_id in dossier_chunk_ids + roots[:2] + supports[:6]:
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
        content = _truncate_to_words(chunk.get("content", ""), per_chunk_word_cap)
        word_count = approx_word_count(content)
        if selected_ids and used_words + word_count > max_source_word_budget:
            return False
        selected_ids.append(chunk_id)
        seen.add(chunk_id)
        used_words += word_count
        doc_id, order = _chunk_doc_and_order(chunk_store, chunk_id)
        if doc_id is not None and order is not None:
            selected_orders_by_doc[doc_id].append(order)
        return True

    for chunk_id in preferred_chunk_ids:
        try_add(chunk_id)
        if len(selected_ids) >= max_source_chunks:
            break

    if len(selected_ids) < max_source_chunks:
        query_bucket_weights = _theme_query_bucket_weights(query_text)
        bucket_best = {}
        for chunk_id in ranked_chunk_ids[:240]:
            if chunk_id in seen:
                continue
            chunk = chunk_store.get(chunk_id)
            if chunk is None:
                continue
            linked_groups = chunk_to_group_ids.get(chunk_id, [])
            linked_labels = " ".join(group_label_lookup.get(group_id, "") for group_id in linked_groups)
            text = f"{chunk.get('content', '')} {linked_labels}".strip()
            bucket_scores = _theme_bucket_scores(text)
            if not any(score > 0 for score in bucket_scores.values()):
                continue
            query_overlap = lexical_overlap_score(query_text, text)
            roles = set(chunk_role_lookup.get(chunk_id, {}).get("roles", []))
            role_rank = 0
            if "query-root" in roles:
                role_rank = max(role_rank, 1)
            if "bridge-chunk" in roles:
                role_rank = max(role_rank, 2)
            if "support-chunk" in roles:
                role_rank = max(role_rank, 3)
            for bucket, bucket_score in bucket_scores.items():
                if bucket_score <= 0:
                    continue
                final_score = (
                    query_bucket_weights.get(bucket, 1.0) > 1.0,
                    bucket_score,
                    query_overlap,
                    role_rank,
                    -rank_index.get(chunk_id, 10**9),
                )
                current = bucket_best.get(bucket)
                if current is None or final_score > current["score"]:
                    bucket_best[bucket] = {"chunk_id": chunk_id, "score": final_score}
        for bucket in sorted(
            bucket_best.keys(),
            key=lambda name: bucket_best[name]["score"],
            reverse=True,
        ):
            try_add(bucket_best[bucket]["chunk_id"])
            if len(selected_ids) >= max_source_chunks:
                break

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

    selected = []
    for chunk_id in selected_ids:
        chunk_data = dict(chunk_store[chunk_id])
        chunk_data["content"] = _truncate_to_words(chunk_data.get("content", ""), per_chunk_word_cap)
        selected.append((chunk_id, chunk_data, approx_word_count(chunk_data.get("content", ""))))
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


def _candidate_point_rows(candidate_points):
    rows = [["id", "type", "label", "query_alignment", "chunk_alignment", "doc_count", "supporting_chunks"]]
    for index, point in enumerate(candidate_points, start=1):
        support_size = point.get("support_count", len(point.get("supporting_chunk_ids", [])))
        rows.append(
            [
                f"pt-{index:02d}",
                point.get("point_type", ""),
                normalize_text(point.get("label", ""))[:120],
                f"{point.get('query_alignment', 0.0):.4f}",
                f"{point.get('chunk_alignment', 0.0):.4f}",
                len(point.get("doc_ids", [])),
                support_size,
            ]
        )
    return rows


def _candidate_point_lines(candidate_points):
    lines = []
    for index, point in enumerate(candidate_points, start=1):
        doc_suffix = f" docs={len(point.get('doc_ids', []))}" if point.get("doc_ids") else ""
        if point.get("point_type") == "aspect":
            support_labels = ", ".join(normalize_text(label) for label in point.get("support_labels", [])[:4])
            support_suffix = f" supports={support_labels}" if support_labels else ""
            lines.append(
                f"- pt-{index:02d}: {normalize_text(point.get('label', ''))[:160]} "
                f"[aspect; q={point.get('query_alignment', 0.0):.3f}; "
                f"c={point.get('chunk_alignment', 0.0):.3f}; support_items={point.get('support_count', 0)}{doc_suffix}]{support_suffix}"
            )
            continue
        lines.append(
            f"- pt-{index:02d}: {normalize_text(point.get('label', ''))[:160]} "
            f"[{point.get('point_type', '')}; q={point.get('query_alignment', 0.0):.3f}; "
            f"c={point.get('chunk_alignment', 0.0):.3f}; chunks={len(point.get('supporting_chunk_ids', []))}{doc_suffix}]"
        )
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
    candidate_points,
    chunk_roles,
    theme_selected_chunks,
    chunk_store,
    node_to_chunks,
    edge_to_chunks,
    max_source_chunks,
    max_source_word_budget,
):
    """Assemble the final evidence package shown to the answer-generation LLM.

    组合知识组、来源列表和源 chunk，生成最终给 LLM 的上下文。
    """
    root_chunk_ids = [item["chunk_id"] for item in root_chunk_hits]
    chunk_role_lookup = {item["chunk_id"]: item for item in (chunk_roles or [])}
    preferred_chunk_ids = []
    per_chunk_word_cap = 520
    source_chunk_limit = max(max_source_chunks, 12)
    chunk_to_group_ids = {}
    ranked_source_groups = _ordered_groups_for_source_budget(facet_groups)
    for group_rank, group in enumerate(ranked_source_groups, start=1):
        for dossier_item in _ordered_dossier_items(group)[: _dossier_source_quota(group_rank)]:
            chunk_id = dossier_item.get("chunk_id")
            if chunk_id:
                preferred_chunk_ids.append(chunk_id)
        for chunk_id in group["supporting_chunk_ids"]:
            chunk_to_group_ids.setdefault(chunk_id, []).append(group["group_id"])
        for dossier_item in group.get("evidence_dossier", []):
            chunk_id = dossier_item.get("chunk_id")
            if chunk_id:
                chunk_to_group_ids.setdefault(chunk_id, []).append(group["group_id"])
    if theme_selected_chunks:
        for bucket in ("core", "bridge", "support", "context"):
            preferred_chunk_ids.extend(theme_selected_chunks.get(bucket, []))
    group_label_lookup = {
        group["group_id"]: normalize_text(group.get("facet_label", ""))
        for group in facet_groups
    }
    chunk_rank = rank_supporting_chunks(final_nodes, final_edges, root_chunk_ids, node_to_chunks, edge_to_chunks)
    selected_source_chunks, used_words = choose_diverse_source_chunks(
        facet_groups=facet_groups,
        ranked_chunk_ids=chunk_rank,
        chunk_store=chunk_store,
        root_chunk_ids=root_chunk_ids,
        max_source_chunks=source_chunk_limit,
        max_source_word_budget=max_source_word_budget,
        preferred_chunk_ids=preferred_chunk_ids,
        per_chunk_word_cap=per_chunk_word_cap,
        query_text=query_row["query"],
        chunk_role_lookup=chunk_role_lookup,
        chunk_to_group_ids=chunk_to_group_ids,
        group_label_lookup=group_label_lookup,
    )
    root_chunk_id_set = set(root_chunk_ids)
    source_id_map = {chunk_id: f"src-{index:02d}" for index, (chunk_id, _, _) in enumerate(selected_source_chunks, start=1)}

    node_lookup = {item["id"]: item for item in top_root_nodes}
    node_lookup.update({item["id"]: item for item in semantic_nodes})
    root_edge_lookup = {item["edge"]: item for item in top_root_edges}
    semantic_edge_lookup = {item["edge"]: item for item in semantic_edges}
    role_priority = {"root": 3, "structural": 2, "semantic": 1, "other": 0}
    group_index_lines = []
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
        linked_chunk_ids = sorted(
            (chunk_id for chunk_id in group["supporting_chunk_ids"] if chunk_id in source_id_map),
            key=lambda chunk_id: source_id_map[chunk_id],
        )[:8]
        dossier_rows = []
        for item in group.get("evidence_dossier", []):
            chunk_id = item.get("chunk_id")
            if not chunk_id:
                continue
            source_id = source_id_map.get(chunk_id)
            if not source_id:
                continue
            reasons = "; ".join(item.get("coverage_reasons", [])[:4])
            dossier_rows.append(
                f"- {source_id} [{item.get('role', 'support')}; {item.get('word_count', 0)} words]: {reasons or 'adds branch evidence'}"
            )
        group_label = group.get("facet_label", group.get("primary_theme", "facet"))
        group_index_lines.append(
            f"- {group['group_id']}: {group_label} :: {group['group_summary']}"
        )
        group_section = [
            f"[{group['group_id']}] {group_label}",
            f"Aspect gist: {group['group_summary']}",
            f"Key entities: {' | '.join(key_entities) or 'n/a'}",
            f"Key relations: {' | '.join(group['relation_themes']) or 'n/a'}",
            f"Linked source ids: {' | '.join(source_id_map[chunk_id] for chunk_id in linked_chunk_ids) or 'n/a'}",
        ]
        group_section.append("Relational spine:")
        if group.get("edge_skeleton"):
            for unit in group["edge_skeleton"][:12]:
                source_ids = [
                    source_id_map.get(chunk_id, chunk_id)
                    for chunk_id in unit.get("source_chunk_ids", [])[:3]
                ]
                edge_id = unit.get("edge", ["", ""])
                group_section.append(
                    f"- {edge_id[0]} -> {edge_id[1]} "
                    f"({normalize_text(unit.get('relation_theme') or unit.get('keywords') or '')[:70]}; "
                    f"{unit.get('kind', 'answer')}; sources={', '.join(source_ids) or 'n/a'})"
                )
        else:
            group_section.append("- n/a")
        group_section.append("Evidence dossier:")
        if dossier_rows:
            group_section.extend(dossier_rows[:6])
        else:
            group_section.append("- n/a")
        group_sections.append("\n".join(group_section))

    source_sections = []
    for index, (chunk_id, chunk_data, word_count) in enumerate(selected_source_chunks):
        role_text = " | ".join(chunk_role_lookup.get(chunk_id, {}).get("roles", [])) or (
            "root" if chunk_id in root_chunk_id_set else "support"
        )
        linked_groups = " | ".join(chunk_to_group_ids.get(chunk_id, [])[:4]) or "n/a"
        content = chunk_data.get("content", "").strip()
        source_sections.append(
            "\n".join(
                [
                    f"[{source_id_map[chunk_id]}] role={role_text}; linked_groups={linked_groups}; words={word_count}",
                    content,
                ]
            )
        )
    source_bank_text = "\n\n".join(source_sections)

    context = f"""
-----Group Index-----
{chr(10).join(group_index_lines) if group_index_lines else '- n/a'}
-----Knowledge Group Dossiers-----
{chr(10).join(group_sections)}
-----Source Excerpts-----
{source_bank_text}
""".strip()

    return {
        "context": context,
        "selected_source_word_count": used_words,
        "selected_source_chunk_count": len(selected_source_chunks),
    }
