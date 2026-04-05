"""Query-contract-driven organization over the final evidence subgraph.

Organization is the second stage after association:

- association expands toward high-recall evidence
- organization chooses one organization contract for the query and packages
  the final evidence graph accordingly

Supported contracts:
- section-grounded
- mechanism-grounded
- comparison-grounded
- theme-grounded
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from .common import edge_key, lexical_overlap_score, normalize_text
from .retrieval import normalize_relation_category


QUERY_CONTRACTS = [
    "section-grounded",
    "mechanism-grounded",
    "comparison-grounded",
    "theme-grounded",
]


SECTION_EXPLICIT_PHRASES = (
    "what sections",
    "which sections",
    "what section",
    "which section",
    "different sections",
    "across sections",
    "across different sections",
    "what parts",
    "which parts",
    "what passages",
    "which passages",
    "where in the corpus",
    "where in the text",
    "where does the book",
    "where does the text",
    "periods discussed",
)

SECTION_LIST_HEADS = (
    "what are the",
    "which are the",
    "what is the",
    "which is the",
)

SECTION_LIST_NOUNS = (
    "steps",
    "step",
    "tasks",
    "task",
    "pieces of equipment",
    "equipment",
    "materials",
    "parts",
    "sections",
    "resources",
)

COMPARISON_PHRASES = (
    "compare",
    "difference",
    "differences",
    "distinguish",
    "distinguishes",
    "contrast",
    "contrasts",
    "versus",
    "vs.",
    "vs ",
    "similarities and differences",
    "balance between",
)

MECHANISM_PROCESS_PATTERNS = (
    "in what ways did",
    "in what ways does",
    "in what ways do",
    "how did",
    "how does",
    "how do",
    "how are",
    "how is",
    "how were",
    "how have",
    "how should",
    "how can",
)

MECHANISM_LINK_VERBS = (
    "affect",
    "affected",
    "affects",
    "change",
    "changed",
    "changes",
    "influence",
    "influenced",
    "influences",
    "shape",
    "shaped",
    "shapes",
    "foster",
    "fostered",
    "fosters",
    "lead to",
    "leads to",
    "produce",
    "produced",
    "produces",
    "result in",
    "results in",
    "used to",
    "support",
    "supports",
    "maintain",
    "ensuring",
    "ensure",
    "preparation",
    "prepare",
    "engage",
    "engages",
    "engaging",
    "enhance",
    "evaluate",
    "convey",
    "transform",
    "increase",
    "encourage",
    "transition",
    "innovate",
    "innovated",
    "innovates",
    "innovation",
)

THEME_REASON_CUES = (
    "what are the primary reasons",
    "what are the reasons",
    "why is",
    "why are",
    "what strategies",
    "what role",
    "what patterns",
    "what themes",
    "what examples",
    "how does the book suggest",
    "what policy recommendations",
    "what role does",
)

MECHANISM_EXPLICIT_CUES = (
    "what impact did",
    "what impact does",
    "what impact do",
    "what role did",
    "what role does",
    "what role do",
)

BROAD_SCOPE_MARKERS = (
    "different ",
    "various ",
    "across ",
    "throughout ",
    "patterns",
    "themes",
    "types of",
    "regions",
    "periods",
    "social issues",
    "historical and political impact",
    "according to the dataset",
)


@dataclass
class EvidenceRegion:
    """One overlapping evidence region extracted from the final subgraph."""

    region_id: str
    region_kind: str
    root_chunk_ids: list[str]
    anchor_chunk_ids: list[str]
    supporting_chunk_ids: list[str]
    node_ids: list[str]
    edge_ids: list[tuple[str, str]]
    relation_themes: list[str]
    focus_entities: list[str]
    descriptor_text: str
    root_connected: bool
    doc_ids: list[str]
    growth_traces: list[str]


def _evidence_descriptor_text(label, relation_themes, focus_entities, anchor_chunk_ids, chunk_store):
    parts = []
    if label:
        parts.append(label)
    parts.extend(theme for theme in relation_themes[:4] if theme)
    parts.extend(entity for entity in focus_entities[:6] if entity)
    parts.extend(
        normalize_text(chunk_store.get(chunk_id, {}).get("content", ""))[:180]
        for chunk_id in anchor_chunk_ids[:2]
        if chunk_store.get(chunk_id, {}).get("content")
    )
    return " ".join(parts)


def detect_query_contract(query: str) -> str:
    """Pick exactly one organization contract for the query."""
    query_lower = query.lower()
    query_lower = " ".join(query_lower.split())
    starts_how = query_lower.startswith(("how ", "how did", "how does", "how do", "how are", "how is", "how should", "how can", "how have"))
    starts_in_what_ways = query_lower.startswith(("in what ways ", "in what way "))
    starts_what = query_lower.startswith(("what ", "which "))
    if any(phrase in query_lower for phrase in SECTION_EXPLICIT_PHRASES):
        return "section-grounded"
    if any(phrase in query_lower for phrase in COMPARISON_PHRASES):
        return "comparison-grounded"
    if query_lower.startswith(SECTION_LIST_HEADS) and any(noun in query_lower for noun in SECTION_LIST_NOUNS):
        return "section-grounded"
    if query_lower.startswith(("what seasonal", "what initial", "which seasonal", "which initial")):
        return "section-grounded"
    if query_lower.startswith("what role did") and any(
        token in query_lower for token in ("play in", "shape", "influence", "affect", "evolution")
    ):
        return "mechanism-grounded"
    if any(cue in query_lower for cue in MECHANISM_EXPLICIT_CUES) and any(
        verb in query_lower for verb in MECHANISM_LINK_VERBS
    ):
        return "mechanism-grounded"
    if (starts_how or starts_in_what_ways) and any(pattern in query_lower for pattern in MECHANISM_PROCESS_PATTERNS) and any(
        verb in query_lower for verb in MECHANISM_LINK_VERBS
    ):
        return "mechanism-grounded"
    if any(cue in query_lower for cue in THEME_REASON_CUES):
        return "theme-grounded"
    if starts_what and " and how " in query_lower:
        return "theme-grounded"
    if starts_how and any(marker in query_lower for marker in BROAD_SCOPE_MARKERS) and not any(
        token in query_lower for token in ("process", "procedure", "preparation", "approach", "evaluate", "enhance", "convey", "transform")
    ):
        return "theme-grounded"
    return "theme-grounded"


def _chunk_local_index(final_nodes, final_edges, node_to_chunks, edge_to_chunks):
    chunk_to_nodes = defaultdict(set)
    chunk_to_edges = defaultdict(set)
    for node_id in final_nodes:
        for chunk_id in node_to_chunks.get(node_id, set()):
            chunk_to_nodes[chunk_id].add(node_id)
    for edge_id in final_edges:
        for chunk_id in edge_to_chunks.get(edge_id, set()):
            chunk_to_edges[chunk_id].add(edge_id)
    return dict(chunk_to_nodes), dict(chunk_to_edges)


def _root_chunk_band(root_chunk_ids, chunk_neighbors):
    band = set(root_chunk_ids)
    for chunk_id in list(root_chunk_ids):
        band.update(chunk_neighbors.get(chunk_id, set()))
    return band


def _supporting_chunks_for_region(node_ids, edge_ids, node_to_chunks, edge_to_chunks):
    chunk_ids = set()
    for node_id in node_ids:
        chunk_ids.update(node_to_chunks.get(node_id, set()))
    for edge_id in edge_ids:
        chunk_ids.update(edge_to_chunks.get(edge_id, set()))
    return chunk_ids


def _relation_themes(edge_ids, graph, limit=5):
    counter = Counter()
    for edge_id in edge_ids:
        edge_data = graph.get_edge_data(edge_id[0], edge_id[1]) or {}
        counter[normalize_relation_category(edge_data)] += 1
    return [theme for theme, _ in counter.most_common(limit)]


def _focus_entities(query, node_ids, graph, limit=6):
    return sorted(
        node_ids,
        key=lambda node_id: (
            -lexical_overlap_score(
                query,
                f"{node_id} {normalize_text((graph.nodes.get(node_id) or {}).get('description', ''))}",
            ),
            node_id,
        ),
    )[:limit]


def _rank_chunks(query, chunk_ids, chunk_store, root_chunk_id_set):
    return sorted(
        chunk_ids,
        key=lambda chunk_id: (
            -(1 if chunk_id in root_chunk_id_set else 0),
            -lexical_overlap_score(query, chunk_store.get(chunk_id, {}).get("content", "")),
            chunk_id,
        ),
    )


def _doc_ids(chunk_ids, chunk_store):
    values = []
    for chunk_id in chunk_ids:
        doc_id = chunk_store.get(chunk_id, {}).get("full_doc_id")
        if doc_id and doc_id not in values:
            values.append(doc_id)
    return values


def _make_region(
    *,
    region_id,
    region_kind,
    query,
    root_chunk_ids,
    node_ids,
    edge_ids,
    graph,
    node_to_chunks,
    edge_to_chunks,
    chunk_store,
    root_chunk_band,
    growth_traces=None,
):
    node_ids = sorted(set(node_ids))
    edge_ids = sorted(set(edge_ids))
    if not node_ids and not edge_ids:
        return None
    supporting_chunk_ids = sorted(_supporting_chunks_for_region(node_ids, edge_ids, node_to_chunks, edge_to_chunks))
    if not supporting_chunk_ids:
        return None
    root_chunk_id_set = set(root_chunk_ids)
    anchor_chunk_ids = _rank_chunks(query, supporting_chunk_ids, chunk_store, root_chunk_id_set)[:5]
    relation_themes = _relation_themes(edge_ids, graph) if graph is not None else []
    focus_entities = _focus_entities(query, node_ids, graph) if graph is not None else sorted(node_ids)[:6]
    descriptor_text = _evidence_descriptor_text("", relation_themes, focus_entities, anchor_chunk_ids, chunk_store)
    root_connected = bool(set(anchor_chunk_ids) & root_chunk_id_set or set(supporting_chunk_ids) & root_chunk_band)
    return EvidenceRegion(
        region_id=region_id,
        region_kind=region_kind,
        root_chunk_ids=sorted(set(root_chunk_ids)),
        anchor_chunk_ids=anchor_chunk_ids,
        supporting_chunk_ids=supporting_chunk_ids,
        node_ids=node_ids,
        edge_ids=edge_ids,
        relation_themes=relation_themes,
        focus_entities=focus_entities,
        descriptor_text=descriptor_text,
        root_connected=root_connected,
        doc_ids=_doc_ids(supporting_chunk_ids, chunk_store),
        growth_traces=list(growth_traces or []),
    )


def _collect_root_regions(
    *,
    query,
    root_chunk_ids,
    graph,
    final_edges,
    chunk_to_final_nodes,
    chunk_to_final_edges,
    node_to_chunks,
    edge_to_chunks,
    chunk_store,
    root_chunk_band,
):
    regions = []
    final_edge_set = set(final_edges)
    for index, root_chunk_id in enumerate(root_chunk_ids, start=1):
        seed_nodes = set(chunk_to_final_nodes.get(root_chunk_id, set()))
        seed_edges = set(chunk_to_final_edges.get(root_chunk_id, set()))
        if not seed_nodes and not seed_edges:
            continue
        support_chunks = sorted(
            _supporting_chunks_for_region(seed_nodes, seed_edges, node_to_chunks, edge_to_chunks) - {root_chunk_id},
            key=lambda chunk_id: (
                -lexical_overlap_score(query, chunk_store.get(chunk_id, {}).get("content", "")),
                chunk_id,
            ),
        )
        local_nodes = set(seed_nodes)
        local_edges = set(seed_edges)
        for chunk_id in support_chunks[:4]:
            local_nodes.update(chunk_to_final_nodes.get(chunk_id, set()))
            local_edges.update(chunk_to_final_edges.get(chunk_id, set()))
        local_edges.update(edge_id for edge_id in final_edge_set if edge_id[0] in local_nodes and edge_id[1] in local_nodes)
        region = _make_region(
            region_id=f"root-{index:02d}",
            region_kind="root",
            query=query,
            root_chunk_ids=[root_chunk_id],
            node_ids=local_nodes,
            edge_ids=local_edges,
            graph=graph,
            node_to_chunks=node_to_chunks,
            edge_to_chunks=edge_to_chunks,
            chunk_store=chunk_store,
            root_chunk_band=root_chunk_band,
            growth_traces=[f"Root-anchored around {root_chunk_id}"],
        )
        if region is not None:
            regions.append(region)
    return regions


def _path_chunk_ids(path, node_to_chunks, edge_to_chunks):
    chunk_ids = set()
    for node_id in path:
        chunk_ids.update(node_to_chunks.get(node_id, set()))
    for idx in range(len(path) - 1):
        chunk_ids.update(edge_to_chunks.get(edge_key(path[idx], path[idx + 1]), set()))
    return chunk_ids


def _collect_bridge_regions(
    *,
    query,
    root_chunk_ids,
    last_structural_output,
    graph,
    final_edges,
    node_to_chunks,
    edge_to_chunks,
    chunk_store,
    root_chunk_band,
):
    final_edge_set = set(final_edges)
    regions = []
    for index, item in enumerate(last_structural_output.get("selected_paths", []), start=1):
        path = item.get("path") or []
        if len(path) < 2:
            continue
        node_ids = set(path)
        edge_ids = {edge_key(path[idx], path[idx + 1]) for idx in range(len(path) - 1)}
        path_chunks = _path_chunk_ids(path, node_to_chunks, edge_to_chunks)
        ranked = _rank_chunks(query, path_chunks, chunk_store, set(root_chunk_ids))
        for chunk_id in ranked[:3]:
            node_ids.update(node_to_chunks.get(chunk_id, set()))
            edge_ids.update(edge_id for edge_id in edge_to_chunks.get(chunk_id, set()) if edge_id in final_edge_set)
        edge_ids.update(edge_id for edge_id in final_edge_set if edge_id[0] in node_ids and edge_id[1] in node_ids)
        region = _make_region(
            region_id=f"bridge-{index:02d}",
            region_kind="bridge",
            query=query,
            root_chunk_ids=sorted(set(root_chunk_ids) & set(ranked[:2])),
            node_ids=node_ids,
            edge_ids=edge_ids,
            graph=graph,
            node_to_chunks=node_to_chunks,
            edge_to_chunks=edge_to_chunks,
            chunk_store=chunk_store,
            root_chunk_band=root_chunk_band,
            growth_traces=[
                f"Graph bridge path: {' -> '.join(path)}",
                f"Bridge introduced {item.get('new_source_count', 0)} new source chunks",
            ],
        )
        if region is not None and region.root_connected:
            regions.append(region)
    for index, item in enumerate(last_structural_output.get("selected_chunks", []), start=1):
        node_ids = set(item.get("node_ids", []))
        edge_ids = set(edge_id for edge_id in item.get("edge_ids", []) if edge_id in final_edge_set)
        if not node_ids and not edge_ids:
            continue
        region = _make_region(
            region_id=f"bridge-chunk-{index:02d}",
            region_kind="bridge",
            query=query,
            root_chunk_ids=sorted(set(root_chunk_ids) & {item.get("chunk_id")}),
            node_ids=node_ids,
            edge_ids=edge_ids,
            graph=graph,
            node_to_chunks=node_to_chunks,
            edge_to_chunks=edge_to_chunks,
            chunk_store=chunk_store,
            root_chunk_band=root_chunk_band,
            growth_traces=[
                f"Chunk-side bridge via {item.get('chunk_id')}",
                f"Frontier touch={item.get('frontier_touch', 0)}, new sources={item.get('new_source_count', 0)}",
            ],
        )
        if region is not None and region.root_connected:
            regions.append(region)
    return regions


def _collect_theme_regions(
    *,
    query,
    root_chunk_ids,
    graph,
    final_nodes,
    final_edges,
    node_to_chunks,
    edge_to_chunks,
    chunk_store,
    root_chunk_band,
    max_themes=5,
):
    theme_to_edges = defaultdict(list)
    final_edge_set = set(final_edges)
    final_node_set = set(final_nodes)
    for edge_id in final_edges:
        theme = normalize_relation_category(graph.get_edge_data(edge_id[0], edge_id[1]) or {})
        if theme in {"unknown_relation", "unknown relation"}:
            continue
        theme_to_edges[theme].append(edge_id)
    theme_items = sorted(
        theme_to_edges.items(),
        key=lambda item: (-lexical_overlap_score(query, item[0]), -len(item[1]), item[0]),
    )[:max_themes]
    regions = []
    for index, (theme, edges) in enumerate(theme_items, start=1):
        node_ids = set()
        edge_ids = set(edges)
        for edge_id in edges:
            node_ids.update(edge_id)
        support_chunks = _supporting_chunks_for_region(node_ids, edge_ids, node_to_chunks, edge_to_chunks)
        if not (support_chunks & root_chunk_band):
            continue
        ranked = _rank_chunks(query, support_chunks, chunk_store, set(root_chunk_ids))
        for chunk_id in ranked[:4]:
            node_ids.update(node_to_chunks.get(chunk_id, set()) & final_node_set)
            edge_ids.update(edge_id for edge_id in edge_to_chunks.get(chunk_id, set()) if edge_id in final_edge_set)
        region = _make_region(
            region_id=f"theme-{index:02d}",
            region_kind="theme",
            query=query,
            root_chunk_ids=sorted(set(root_chunk_ids) & set(ranked[:2])),
            node_ids=node_ids,
            edge_ids=edge_ids,
            graph=graph,
            node_to_chunks=node_to_chunks,
            edge_to_chunks=edge_to_chunks,
            chunk_store=chunk_store,
            root_chunk_band=root_chunk_band,
            growth_traces=[f"Theme expansion around relation theme '{theme}'"],
        )
        if region is not None and region.root_connected:
            regions.append(region)
    return regions


def collect_overlapping_regions(
    *,
    query,
    root_chunk_ids,
    graph,
    final_nodes,
    final_edges,
    last_structural_output,
    chunk_neighbors,
    node_to_chunks,
    edge_to_chunks,
    chunk_store,
):
    chunk_to_final_nodes, chunk_to_final_edges = _chunk_local_index(
        final_nodes=final_nodes,
        final_edges=final_edges,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
    )
    root_chunk_band = _root_chunk_band(root_chunk_ids, chunk_neighbors)
    root_regions = _collect_root_regions(
        query=query,
        root_chunk_ids=root_chunk_ids,
        graph=graph,
        final_edges=final_edges,
        chunk_to_final_nodes=chunk_to_final_nodes,
        chunk_to_final_edges=chunk_to_final_edges,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        chunk_store=chunk_store,
        root_chunk_band=root_chunk_band,
    )
    bridge_regions = _collect_bridge_regions(
        query=query,
        root_chunk_ids=root_chunk_ids,
        last_structural_output=last_structural_output,
        graph=graph,
        final_edges=final_edges,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        chunk_store=chunk_store,
        root_chunk_band=root_chunk_band,
    )
    theme_regions = _collect_theme_regions(
        query=query,
        root_chunk_ids=root_chunk_ids,
        graph=graph,
        final_nodes=final_nodes,
        final_edges=final_edges,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        chunk_store=chunk_store,
        root_chunk_band=root_chunk_band,
    )
    return root_regions, bridge_regions, theme_regions


def _primary_theme(region: EvidenceRegion):
    for theme in region.relation_themes:
        if theme and theme not in {"unknown_relation", "unknown relation"}:
            return theme
    if region.focus_entities:
        return normalize_text(region.focus_entities[0])
    return region.region_kind


def _facet_prompt(query, contract, label):
    if contract == "section-grounded":
        return f"What section-like evidence block in the corpus best grounds the aspect '{label}' for: {query}"
    if contract == "mechanism-grounded":
        return f"What mechanism, cause, or process evidence in the corpus supports the aspect '{label}' for: {query}"
    if contract == "comparison-grounded":
        return f"What side, contrast, or comparison angle in the corpus supports the aspect '{label}' for: {query}"
    return f"What major theme in the corpus supports the aspect '{label}' for: {query}"


def _build_group(query, contract, group_id, label, regions, chunk_store):
    root_chunk_ids = set()
    anchor_chunk_ids = []
    supporting_chunk_ids = set()
    node_ids = set()
    edge_ids = set()
    relation_counter = Counter()
    focus_counter = Counter()
    region_kinds = []
    doc_ids = []
    growth_traces = []
    for region in regions:
        root_chunk_ids.update(region.root_chunk_ids)
        node_ids.update(region.node_ids)
        edge_ids.update(region.edge_ids)
        supporting_chunk_ids.update(region.supporting_chunk_ids)
        region_kinds.append(region.region_kind)
        relation_counter.update(region.relation_themes)
        focus_counter.update(region.focus_entities)
        for doc_id in region.doc_ids:
            if doc_id not in doc_ids:
                doc_ids.append(doc_id)
        for trace in region.growth_traces:
            if trace not in growth_traces:
                growth_traces.append(trace)
        for chunk_id in region.anchor_chunk_ids:
            if chunk_id not in anchor_chunk_ids:
                anchor_chunk_ids.append(chunk_id)
    anchor_chunk_ids = _rank_chunks(query, anchor_chunk_ids, chunk_store, root_chunk_ids)[:5]
    relation_themes = [theme for theme, _ in relation_counter.most_common(5)]
    focus_entities = [entity for entity, _ in focus_counter.most_common(8)]
    descriptor_text = _evidence_descriptor_text(
        label,
        relation_themes,
        focus_entities,
        anchor_chunk_ids,
        chunk_store,
    )
    summary = (
        f"This {contract} facet centers on {label}. "
        f"It is grounded in {len(anchor_chunk_ids)} anchor chunks and combines "
        f"{', '.join(sorted(set(region_kinds)))} evidence."
    )
    return {
        "group_id": group_id,
        "facet_label": label,
        "organization_contract": contract,
        "facet_prompt": _facet_prompt(query, contract, label),
        "group_score": round(lexical_overlap_score(query, descriptor_text), 6),
        "query_rel": round(lexical_overlap_score(query, descriptor_text), 6),
        "anchor_support": len(anchor_chunk_ids),
        "root_anchor_count": len(set(anchor_chunk_ids) & root_chunk_ids),
        "node_count": len(node_ids),
        "edge_count": len(edge_ids),
        "root_chunk_ids": sorted(root_chunk_ids),
        "doc_ids": doc_ids,
        "region_kinds": sorted(set(region_kinds)),
        "relation_themes": relation_themes,
        "focus_entities": focus_entities,
        "supporting_chunk_ids": sorted(supporting_chunk_ids),
        "anchor_chunk_ids": anchor_chunk_ids,
        "source_previews": [
            {
                "chunk_id": chunk_id,
                "preview": normalize_text(chunk_store.get(chunk_id, {}).get("content", ""))[:220],
            }
            for chunk_id in anchor_chunk_ids[:3]
        ],
        "growth_traces": growth_traces[:4],
        "group_summary": summary,
        "nodes": sorted(node_ids),
        "edges": sorted(edge_ids),
    }


def _group_rank_key(group):
    return (
        -group["root_anchor_count"],
        -group["query_rel"],
        -len(group["region_kinds"]),
        -group["anchor_support"],
        -group["node_count"],
        group["facet_label"],
    )


def _group_coverage_keys(group):
    keys = []
    label = normalize_text(group.get("facet_label", ""))
    if label:
        keys.append(("label", label))
    for theme in group.get("relation_themes", [])[:2]:
        theme = normalize_text(theme)
        if theme:
            keys.append(("theme", theme))
    for doc_id in group.get("doc_ids", [])[:2]:
        if doc_id:
            keys.append(("doc", doc_id))
    for root_chunk_id in group.get("root_chunk_ids", [])[:2]:
        if root_chunk_id:
            keys.append(("root", root_chunk_id))
    region_kinds = tuple(sorted(group.get("region_kinds", [])))
    if region_kinds:
        keys.append(("kinds", region_kinds))
    return tuple(keys)


def _select_groups(groups, limit, distinct_key_fn, coverage_key_fn=None):
    groups = sorted(groups, key=_group_rank_key)
    coverage_key_fn = coverage_key_fn or _group_coverage_keys
    selected = []
    used = set()
    covered = set()

    # Select groups greedily by new facet coverage first, then by rank.
    while len(selected) < limit:
        best_group = None
        best_gain = 0
        best_rank = None
        for group in groups:
            if group in selected:
                continue
            key = distinct_key_fn(group)
            if key in used:
                continue
            gain = len(set(coverage_key_fn(group)) - covered)
            if gain <= 0:
                continue
            rank = _group_rank_key(group)
            if gain > best_gain or (gain == best_gain and (best_rank is None or rank < best_rank)):
                best_group = group
                best_gain = gain
                best_rank = rank
        if best_group is None:
            break
        selected.append(best_group)
        used.add(distinct_key_fn(best_group))
        covered.update(coverage_key_fn(best_group))

    for group in groups:
        if len(selected) >= limit:
            break
        if group in selected:
            continue
        key = distinct_key_fn(group)
        if key in used:
            continue
        if any(set(group["anchor_chunk_ids"]) == set(other["anchor_chunk_ids"]) for other in selected):
            continue
        selected.append(group)
        used.add(key)
    return selected


def _build_doc_chunk_index(chunk_store):
    doc_to_chunks = defaultdict(list)
    for chunk_id, chunk in chunk_store.items():
        doc_id = chunk.get("full_doc_id")
        order = chunk.get("chunk_order_index")
        if doc_id is None or order is None:
            continue
        doc_to_chunks[doc_id].append((order, chunk_id))
    for doc_id in doc_to_chunks:
        doc_to_chunks[doc_id].sort()
    return dict(doc_to_chunks)


def _chunk_order_bounds(chunk_ids, chunk_store):
    orders = [chunk_store.get(chunk_id, {}).get("chunk_order_index") for chunk_id in chunk_ids]
    orders = [order for order in orders if order is not None]
    if not orders:
        return None, None
    return min(orders), max(orders)


def _section_band_chunk_ids(seed_chunk_ids, chunk_store, doc_to_chunks, radius=2):
    selected = set()
    for chunk_id in seed_chunk_ids:
        chunk = chunk_store.get(chunk_id, {})
        doc_id = chunk.get("full_doc_id")
        order = chunk.get("chunk_order_index")
        if doc_id is None or order is None:
            continue
        for candidate_order, candidate_chunk_id in doc_to_chunks.get(doc_id, []):
            if abs(candidate_order - order) <= radius:
                selected.add(candidate_chunk_id)
    return selected


def _build_section_groups(
    query,
    root_regions,
    chunk_store,
    chunk_to_nodes,
    chunk_to_edges,
    node_to_chunks,
    edge_to_chunks,
    final_nodes,
    final_edges,
    group_limit,
):
    doc_to_chunks = _build_doc_chunk_index(chunk_store)
    final_node_set = set(final_nodes)
    final_edge_set = set(final_edges)
    candidates = []
    for region in root_regions:
        seed_chunk_ids = region.anchor_chunk_ids[:3]
        band_chunk_ids = _section_band_chunk_ids(seed_chunk_ids, chunk_store, doc_to_chunks, radius=2)
        if not band_chunk_ids:
            continue
        doc_ids = _doc_ids(band_chunk_ids, chunk_store)
        if not doc_ids:
            continue
        node_ids = set()
        edge_ids = set()
        for chunk_id in band_chunk_ids:
            node_ids.update(chunk_to_nodes.get(chunk_id, set()) & final_node_set)
            edge_ids.update(edge_id for edge_id in chunk_to_edges.get(chunk_id, set()) if edge_id in final_edge_set)
        refined = _make_region(
            region_id=f"section-{region.region_id}",
            region_kind="section",
            query=query,
            root_chunk_ids=region.root_chunk_ids,
            node_ids=node_ids,
            edge_ids=edge_ids,
            graph=None,
            node_to_chunks=node_to_chunks,
            edge_to_chunks=edge_to_chunks,
            chunk_store=chunk_store,
            root_chunk_band=set(band_chunk_ids),
        )
        if refined is None:
            continue
        start_order, end_order = _chunk_order_bounds(refined.anchor_chunk_ids or refined.supporting_chunk_ids, chunk_store)
        if start_order is not None and end_order is not None:
            label = f"section band {doc_ids[0][-6:]}:{start_order}-{end_order}"
        else:
            label = f"section band {doc_ids[0][-6:]}"
        candidates.append(_build_group(query, "section-grounded", f"facet-{len(candidates)+1:02d}", label, [refined], chunk_store))
    return _select_groups(
        candidates,
        group_limit,
        lambda group: (
            tuple(group["doc_ids"][:1]),
            tuple(group["anchor_chunk_ids"][:2]),
        ),
        coverage_key_fn=lambda group: (
            ("doc", tuple(group["doc_ids"][:1])),
            ("band", tuple(group["anchor_chunk_ids"][:2])),
            ("theme", tuple(group["relation_themes"][:1])),
        ),
    )


def _build_mechanism_groups(query, root_regions, bridge_regions, theme_regions, chunk_store, group_limit):
    candidates = []
    root_by_chunk = {}
    for region in root_regions:
        for chunk_id in region.root_chunk_ids:
            root_by_chunk.setdefault(chunk_id, []).append(region)

    for region in bridge_regions:
        if not region.root_connected:
            continue
        related = [region]
        for chunk_id in region.root_chunk_ids:
            related.extend(root_by_chunk.get(chunk_id, []))
        for theme_region in theme_regions:
            if not theme_region.root_connected:
                continue
            if lexical_overlap_score(query, theme_region.descriptor_text) < 0.06:
                continue
            if set(theme_region.root_chunk_ids) & set(region.root_chunk_ids):
                related.append(theme_region)
                continue
            if set(theme_region.supporting_chunk_ids) & set(region.supporting_chunk_ids):
                related.append(theme_region)
        deduped = []
        seen = set()
        for item in related:
            if item.region_id in seen:
                continue
            seen.add(item.region_id)
            deduped.append(item)
        primary = _primary_theme(region)
        secondary = None
        for theme in region.relation_themes[1:]:
            if theme and theme != primary:
                secondary = theme
                break
        label = primary if secondary is None else f"{primary} -> {secondary}"
        candidates.append(
            _build_group(
                query,
                "mechanism-grounded",
                f"facet-{len(candidates)+1:02d}",
                label,
                deduped,
                chunk_store,
            )
        )

    if not candidates:
        grouped = defaultdict(list)
        for region in root_regions + theme_regions:
            if not region.root_connected:
                continue
            if lexical_overlap_score(query, region.descriptor_text) < 0.05:
                continue
            grouped[_primary_theme(region)].append(region)
        for label, regions in grouped.items():
            candidates.append(
                _build_group(
                    query,
                    "mechanism-grounded",
                    f"facet-{len(candidates)+1:02d}",
                    label,
                    regions,
                    chunk_store,
                )
            )

    filtered = []
    for group in candidates:
        if "bridge" not in group["region_kinds"] and group["edge_count"] < 3:
            continue
        filtered.append(group)
    return _select_groups(
        filtered or candidates,
        group_limit,
        lambda group: group["facet_label"],
        coverage_key_fn=lambda group: (
            ("label", group["facet_label"]),
            ("root", tuple(group["root_chunk_ids"][:2])),
            ("doc", tuple(group["doc_ids"][:1])),
            ("kinds", tuple(group["region_kinds"])),
        ),
    )


def _build_comparison_groups(query, root_regions, bridge_regions, chunk_store, group_limit):
    candidates = []
    for index, region in enumerate(root_regions, start=1):
        label = _primary_theme(region)
        candidates.append(_build_group(query, "comparison-grounded", f"facet-{index:02d}", label, [region], chunk_store))
    for region in bridge_regions:
        if len(region.root_chunk_ids) >= 2:
            label = f"contrast around {_primary_theme(region)}"
            candidates.append(
                _build_group(
                    query,
                    "comparison-grounded",
                    f"facet-x-{len(candidates)+1:02d}",
                    label,
                    [region],
                    chunk_store,
                )
            )
    return _select_groups(
        candidates,
        group_limit,
        lambda group: tuple(group["root_chunk_ids"][:2]) or group["facet_label"],
        coverage_key_fn=lambda group: (
            ("root", tuple(group["root_chunk_ids"][:2])),
            ("theme", tuple(group["relation_themes"][:2])),
            ("doc", tuple(group["doc_ids"][:1])),
        ),
    )


def _build_theme_groups(query, root_regions, bridge_regions, theme_regions, chunk_store, group_limit):
    grouped = defaultdict(list)
    for region in root_regions + theme_regions + bridge_regions:
        if not region.root_connected:
            continue
        region_rel = lexical_overlap_score(query, region.descriptor_text)
        if region.region_kind == "bridge" and region_rel < 0.12:
            continue
        if region.region_kind in {"root", "theme"} and region_rel < 0.05:
            continue
        root_key = tuple(region.root_chunk_ids[:1]) if region.root_chunk_ids else tuple(region.doc_ids[:1])
        grouped[(_primary_theme(region), root_key)].append(region)
    groups = []
    for index, ((label, _root_key), regions) in enumerate(grouped.items(), start=1):
        groups.append(_build_group(query, "theme-grounded", f"facet-{index:02d}", label, regions, chunk_store))
    return _select_groups(
        groups,
        group_limit,
        lambda group: (
            group["facet_label"],
            tuple(group["root_chunk_ids"][:1]),
        ),
        coverage_key_fn=lambda group: (
            ("label", group["facet_label"]),
            ("root", tuple(group["root_chunk_ids"][:1])),
            ("doc", tuple(group["doc_ids"][:1])),
            ("kinds", tuple(group["region_kinds"])),
        ),
    )


def build_answer_facet_groups(
    *,
    query,
    query_contract=None,
    root_chunk_ids,
    graph,
    final_nodes,
    final_edges,
    last_structural_output,
    chunk_neighbors,
    chunk_to_nodes,
    chunk_to_edges,
    node_to_chunks,
    edge_to_chunks,
    chunk_store,
    group_limit,
):
    """Build one-contract-only facet groups from the final subgraph."""
    contract = query_contract or detect_query_contract(query)
    root_regions, bridge_regions, theme_regions = collect_overlapping_regions(
        query=query,
        root_chunk_ids=root_chunk_ids,
        graph=graph,
        final_nodes=final_nodes,
        final_edges=final_edges,
        last_structural_output=last_structural_output,
        chunk_neighbors=chunk_neighbors,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        chunk_store=chunk_store,
    )
    if contract == "section-grounded":
        groups = _build_section_groups(
            query=query,
            root_regions=root_regions,
            chunk_store=chunk_store,
            chunk_to_nodes=chunk_to_nodes,
            chunk_to_edges=chunk_to_edges,
            node_to_chunks=node_to_chunks,
            edge_to_chunks=edge_to_chunks,
            final_nodes=final_nodes,
            final_edges=final_edges,
            group_limit=group_limit,
        )
    elif contract == "mechanism-grounded":
        groups = _build_mechanism_groups(
            query=query,
            root_regions=root_regions,
            bridge_regions=bridge_regions,
            theme_regions=theme_regions,
            chunk_store=chunk_store,
            group_limit=group_limit,
        )
    elif contract == "comparison-grounded":
        groups = _build_comparison_groups(
            query=query,
            root_regions=root_regions,
            bridge_regions=bridge_regions,
            chunk_store=chunk_store,
            group_limit=group_limit,
        )
    else:
        groups = _build_theme_groups(
            query=query,
            root_regions=root_regions,
            bridge_regions=bridge_regions,
            theme_regions=theme_regions,
            chunk_store=chunk_store,
            group_limit=group_limit,
        )
    for group in groups:
        group["query_contract"] = contract
    return groups, contract
