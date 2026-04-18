"""Evidence organization over the final evidence subgraph.

The current system uses a unified anchor/expand backbone and a single
theme/QFS-oriented organization flow:

- anchor/expand builds a high-recall final evidence graph
- organization compresses root traces into query-facing theme groups

将最终证据子图组织成自然语言可用的知识组，支持问答模型的主题型输出。
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import networkx as nx

from .common import STOPWORDS, edge_key, lexical_overlap_score, normalize_text, tokenize
from .retrieval import normalize_relation_category


ORGANIZATION_MODE = "qfs"


def _normalize_layout_label(raw_value: str | None) -> str | None:
    return ORGANIZATION_MODE


def resolve_organization_layout(query_row, controller="predicted", default_layout=ORGANIZATION_MODE):
    """Resolve the organization layout.

    The current system uses one unified QFS organization path.
    """
    return ORGANIZATION_MODE, {"mode": ORGANIZATION_MODE, "source": "fixed"}


ASPECT_GROUPING_SYSTEM_PROMPT = """You plan query-facing evidence groups.
You must only use the provided evidence units and the provided focus items.
Do not invent new facts, examples, entities, or scenarios.
Choose focus items first, then group units around those focus items.
Return valid JSON only."""


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
    "social change",
    "social commentary",
    "public opinion",
    "societal values",
    "societal disparities",
    "international stage",
    "international relations",
    "global economic",
    "consumerism",
    "art market",
    "market landscape",
    "identity",
    "collective identity",
    "regional art",
    "geopolitical",
    "historical and political impact",
    "according to the dataset",
)

THEME_ROLE_SCOPE_MARKERS = (
    "social",
    "societal",
    "public",
    "regional",
    "international",
    "global",
    "cultural",
    "historical",
    "political",
    "economic",
    "identity",
    "market",
    "institution",
    "values",
    "landscape",
    "development",
)

MECHANISM_HARD_CUES = (
    "mechanism",
    "mechanisms",
    "process",
    "processes",
    "procedure",
    "procedures",
    "workflow",
    "workflows",
    "pathway",
    "pathways",
    "step by step",
    "how to",
    "method",
    "methods",
    "implementation",
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


def _filter_chunk_ids_by_doc(chunk_ids, chunk_store, allowed_doc_ids):
    if not allowed_doc_ids:
        return list(chunk_ids)
    return [
        chunk_id
        for chunk_id in chunk_ids
        if chunk_store.get(chunk_id, {}).get("full_doc_id") in allowed_doc_ids
    ]


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
    allowed_doc_ids=None,
):
    node_ids = sorted(set(node_ids))
    edge_ids = sorted(set(edge_ids))
    if not node_ids and not edge_ids:
        return None
    supporting_chunk_ids = sorted(
        _filter_chunk_ids_by_doc(
            _supporting_chunks_for_region(node_ids, edge_ids, node_to_chunks, edge_to_chunks),
            chunk_store,
            allowed_doc_ids,
        )
    )
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
    allowed_doc_ids=None,
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
            allowed_doc_ids=allowed_doc_ids,
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
    allowed_doc_ids=None,
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
            allowed_doc_ids=allowed_doc_ids,
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
            allowed_doc_ids=allowed_doc_ids,
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
    allowed_doc_ids=None,
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
            allowed_doc_ids=allowed_doc_ids,
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
    allowed_doc_ids=None,
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
        allowed_doc_ids=allowed_doc_ids,
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
        allowed_doc_ids=allowed_doc_ids,
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
        allowed_doc_ids=allowed_doc_ids,
    )
    return root_regions, bridge_regions, theme_regions


def _primary_theme(region: EvidenceRegion):
    for theme in region.relation_themes:
        if theme and theme not in {"unknown_relation", "unknown relation"}:
            return theme
    if region.focus_entities:
        return normalize_text(region.focus_entities[0])
    return region.region_kind


def _facet_prompt(query, label):
    return f"What major theme in the corpus supports the aspect '{label}' for: {query}"


def _build_group(query, group_id, label, regions, chunk_store):
    root_chunk_ids = set()
    anchor_chunk_ids = []
    supporting_chunk_ids = set()
    node_ids = set()
    edge_ids = set()
    relation_counter = Counter()
    focus_counter = Counter()
    region_kind_counter = Counter()
    region_kinds = []
    doc_ids = []
    growth_traces = []
    for region in regions:
        root_chunk_ids.update(region.root_chunk_ids)
        node_ids.update(region.node_ids)
        edge_ids.update(region.edge_ids)
        supporting_chunk_ids.update(region.supporting_chunk_ids)
        region_kinds.append(region.region_kind)
        region_kind_counter[region.region_kind] += 1
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
        f"This QFS evidence group centers on {label}. "
        f"It is grounded in {len(anchor_chunk_ids)} anchor chunks and combines "
        f"{', '.join(sorted(set(region_kinds)))} evidence."
    )
    return {
        "group_id": group_id,
        "facet_label": label,
        "facet_prompt": _facet_prompt(query, label),
        "group_score": round(lexical_overlap_score(query, descriptor_text), 6),
        "query_rel": round(lexical_overlap_score(query, descriptor_text), 6),
        "anchor_support": len(anchor_chunk_ids),
        "root_anchor_count": len(set(anchor_chunk_ids) & root_chunk_ids),
        "node_count": len(node_ids),
        "edge_count": len(edge_ids),
        "region_count": len(regions),
        "unique_doc_count": len(doc_ids),
        "unique_root_count": len(root_chunk_ids),
        "root_chunk_ids": sorted(root_chunk_ids),
        "doc_ids": doc_ids,
        "region_kinds": sorted(set(region_kinds)),
        "region_kind_counts": dict(region_kind_counter),
        "root_region_count": region_kind_counter.get("root", 0),
        "bridge_region_count": region_kind_counter.get("bridge", 0),
        "theme_region_count": region_kind_counter.get("theme", 0),
        "section_region_count": region_kind_counter.get("section", 0),
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
        -group.get("selection_priority", 0),
        -group["root_anchor_count"],
        -group["query_rel"],
        -len(group["region_kinds"]),
        -group.get("bridge_region_count", 0),
        -group.get("theme_region_count", 0),
        -group.get("unique_doc_count", 0),
        -group["anchor_support"],
        -group["node_count"],
        group["facet_label"],
    )


def _theme_representative_regions(query, regions, max_roots=3, max_expansions=2):
    root_like = [region for region in regions if region.region_kind in {"root", "theme"}]
    expansion_like = [region for region in regions if region.region_kind == "bridge"]
    root_like = sorted(
        root_like,
        key=lambda region: (
            -lexical_overlap_score(query, region.descriptor_text),
            -len(region.anchor_chunk_ids),
            -len(region.supporting_chunk_ids),
            region.region_id,
        ),
    )
    expansion_like = sorted(
        expansion_like,
        key=lambda region: (
            -lexical_overlap_score(query, region.descriptor_text),
            -len(region.root_chunk_ids),
            -len(region.supporting_chunk_ids),
            region.region_id,
        ),
    )
    selected = []
    seen = set()
    for region in root_like[:max_roots]:
        if region.region_id not in seen:
            selected.append(region)
            seen.add(region.region_id)
    for region in expansion_like[:max_expansions]:
        if region.region_id not in seen:
            selected.append(region)
            seen.add(region.region_id)
    if not selected:
        selected = root_like[:max_roots] or expansion_like[:max_expansions] or regions[:1]
    return selected


def _retune_theme_group(group, query, representative_regions, chunk_store):
    root_chunk_id_set = set(group.get("root_chunk_ids", []))
    representative_anchor_chunks = []
    for region in representative_regions:
        for chunk_id in region.anchor_chunk_ids:
            if chunk_id not in representative_anchor_chunks:
                representative_anchor_chunks.append(chunk_id)
    stable_anchor_candidates = [
        chunk_id
        for chunk_id in representative_anchor_chunks
        if chunk_id in root_chunk_id_set
    ] or representative_anchor_chunks
    stable_anchor_candidates = _rank_chunks(query, stable_anchor_candidates, chunk_store, root_chunk_id_set)[:5]
    if stable_anchor_candidates:
        group["anchor_chunk_ids"] = stable_anchor_candidates
        group["anchor_support"] = len(stable_anchor_candidates)
        group["root_anchor_count"] = len(set(stable_anchor_candidates) & root_chunk_id_set)
        group["source_previews"] = [
            {
                "chunk_id": chunk_id,
                "preview": normalize_text(chunk_store.get(chunk_id, {}).get("content", ""))[:220],
            }
            for chunk_id in stable_anchor_candidates[:3]
        ]
    representative_kinds = sorted({region.region_kind for region in representative_regions})
    group["group_summary"] = (
        f"This QFS evidence group centers on {group['facet_label']}. "
        f"It uses representative root-connected evidence from {len(group.get('doc_ids', []))} documents "
        f"and synthesizes {', '.join(representative_kinds)} support."
    )
    return group


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
        candidates.append(_build_group(query, f"facet-{len(candidates)+1:02d}", label, [refined], chunk_store))
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
        label = f"pathway: {primary}" if secondary is None else f"pathway: {primary} -> {secondary}"
        candidates.append(
            _build_group(
                query,
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
                    f"facet-{len(candidates)+1:02d}",
                    f"pathway: {label}",
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
    for region in root_regions:
        label = _primary_theme(region)
        candidates.append(
            _build_group(
                query,
                f"facet-{len(candidates)+1:02d}",
                f"comparison side {len(candidates)+1}: {label}",
                [region],
                chunk_store,
            )
        )
    for region in bridge_regions:
        if len(region.root_chunk_ids) >= 2:
            label = f"contrast axis: {_primary_theme(region)}"
            candidates.append(
                _build_group(
                    query,
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


def _build_theme_groups_legacy(query, root_regions, bridge_regions, theme_regions, chunk_store, group_limit):
    grouped = defaultdict(list)
    for region in root_regions + theme_regions + bridge_regions:
        if not region.root_connected:
            continue
        region_rel = lexical_overlap_score(query, region.descriptor_text)
        if region.region_kind == "bridge" and region_rel < 0.06:
            continue
        if region.region_kind in {"root", "theme"} and region_rel < 0.04:
            continue
        grouped[_primary_theme(region)].append(region)

    groups = []
    for index, (label, regions) in enumerate(grouped.items(), start=1):
        representative_regions = _theme_representative_regions(query, regions)
        group = _build_group(query, f"facet-{index:02d}", label, representative_regions, chunk_store)
        group["supporting_chunk_ids"] = sorted(
            {
                chunk_id
                for region in regions
                for chunk_id in region.supporting_chunk_ids
            }
        )
        group["root_chunk_ids"] = sorted(
            {
                chunk_id
                for region in regions
                for chunk_id in region.root_chunk_ids
            }
        )
        group["doc_ids"] = list(
            dict.fromkeys(
                doc_id
                for region in regions
                for doc_id in region.doc_ids
            )
        )
        group["region_kinds"] = sorted({region.region_kind for region in regions})
        group["region_count"] = len(regions)
        group["bridge_region_count"] = sum(1 for region in regions if region.region_kind == "bridge")
        group["theme_region_count"] = sum(1 for region in regions if region.region_kind == "theme")
        group["root_region_count"] = sum(1 for region in regions if region.region_kind == "root")
        group["unique_doc_count"] = len(group["doc_ids"])
        group["unique_root_count"] = len(group["root_chunk_ids"])
        group = _retune_theme_group(group, query, representative_regions, chunk_store)
        group["selection_priority"] = (
            group["bridge_region_count"] * 2
            + group["theme_region_count"] * 2
            + min(group["unique_doc_count"], 3)
            + min(group["unique_root_count"], 3)
            + min(group["region_count"], 4)
        )
        groups.append(group)
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
            ("doc", tuple(group["doc_ids"][:2])),
            ("theme", tuple(group["relation_themes"][:2])),
            ("kinds", tuple(group["region_kinds"])),
        ),
    )


def _contains_any(text, items):
    return any(item in text for item in items)


def _aspect_layout_specs(query, group_limit=5):
    query_lower = " ".join(query.lower().split())
    slots = []
    seen = set()

    def add_slot(key, label, cues):
        if key in seen:
            return
        seen.add(key)
        slots.append(
            {
                "key": key,
                "label": label,
                "cues": tuple(cues),
                "signature": " ".join([query_lower, label.lower(), *cues]),
            }
        )

    if _contains_any(query_lower, ("examples", "example", "types of", "styles", "themes", "movements", "artworks", "artifacts", "models")):
        add_slot(
            "examples",
            "Examples and representative cases",
            ("examples", "cases", "styles", "themes", "movements", "artworks", "resources", "models"),
        )
    if _contains_any(query_lower, ("resources", "support", "support networks", "associations", "organizations", "universities", "agencies", "partnerships")):
        add_slot(
            "support",
            "Support networks and institutions",
            ("support", "resources", "organizations", "associations", "institutions", "agencies", "universities", "partnerships"),
        )
    if _contains_any(query_lower, ("strategies", "tools", "manage", "management", "practices", "recommendations", "how can", "how should", "what role")):
        add_slot(
            "actions",
            "Strategies, practices, and practical tools",
            ("strategies", "tools", "management", "practices", "recommendations", "methods", "actions"),
        )
    if _contains_any(query_lower, ("why", "reasons", "reason", "fail", "failure", "challenges", "risks", "barriers", "importance")):
        add_slot(
            "drivers",
            "Causes, constraints, and driving pressures",
            ("reasons", "causes", "pressures", "constraints", "challenges", "risks", "failures", "barriers"),
        )
    if _contains_any(query_lower, ("benefits", "advantages", "impacts", "influence", "alter", "effects", "outcomes", "value", "importance")):
        add_slot(
            "outcomes",
            "Effects, outcomes, and benefits",
            ("effects", "outcomes", "benefits", "advantages", "value", "impact", "influence"),
        )
    if _contains_any(query_lower, ("policy", "policies", "government", "regulation", "regulatory", "market", "education", "community", "social", "historical", "political", "environmental", "local", "global")):
        add_slot(
            "contexts",
            "Contexts, institutions, and operating conditions",
            ("context", "institutions", "policy", "market", "community", "historical", "political", "environmental"),
        )

    if not slots:
        add_slot(
            "examples",
            "Representative examples and cases",
            ("examples", "cases", "instances", "themes"),
        )
        add_slot(
            "drivers",
            "Causes and driving pressures",
            ("drivers", "causes", "pressures", "forces"),
        )
        add_slot(
            "outcomes",
            "Effects and outcomes",
            ("effects", "outcomes", "results", "benefits"),
        )
        add_slot(
            "contexts",
            "Contexts and conditions",
            ("contexts", "conditions", "settings", "institutions"),
        )

    if len(slots) < 3:
        add_slot(
            "contexts",
            "Contexts and conditions",
            ("contexts", "conditions", "settings", "institutions"),
        )
        add_slot(
            "outcomes",
            "Effects and outcomes",
            ("effects", "outcomes", "results", "benefits"),
        )
        add_slot(
            "examples",
            "Representative examples and cases",
            ("examples", "cases", "instances", "themes"),
        )
    return slots[: max(3, min(group_limit, 5))]


def _region_slot_affinity(query, slot, region):
    descriptor = normalize_text(region.descriptor_text)
    primary = normalize_text(_primary_theme(region))
    support_text = " ".join(
        [primary]
        + [normalize_text(theme) for theme in region.relation_themes[:4]]
        + [normalize_text(entity) for entity in region.focus_entities[:6]]
    )
    slot_signature = slot["signature"]
    cue_hits = sum(1 for cue in slot["cues"] if cue in descriptor or cue in support_text)
    score = 0.0
    score += lexical_overlap_score(slot_signature, descriptor) * 1.4
    score += lexical_overlap_score(slot_signature, support_text) * 1.0
    score += lexical_overlap_score(query, descriptor) * 0.4
    score += cue_hits * 0.03
    if slot["key"] in {"drivers", "outcomes"} and region.region_kind == "bridge":
        score += 0.05
    if slot["key"] in {"examples", "support", "contexts"} and region.region_kind in {"root", "theme"}:
        score += 0.04
    if slot["key"] == "actions" and region.region_kind in {"root", "bridge"}:
        score += 0.04
    return round(score, 6)


def _slot_group_label(query, slot, regions):
    primary_counter = Counter(normalize_text(_primary_theme(region)) for region in regions if normalize_text(_primary_theme(region)))
    dominant = primary_counter.most_common(1)[0][0] if primary_counter else ""
    if dominant and lexical_overlap_score(query, dominant) >= 0.03:
        return f"{slot['label']}: {dominant}"
    return slot["label"]


def _select_regions_for_slot(query, slot, regions, used_region_ids, max_regions=3):
    ranked = sorted(
        regions,
        key=lambda region: (
            -_region_slot_affinity(query, slot, region),
            -lexical_overlap_score(query, region.descriptor_text),
            -len(region.supporting_chunk_ids),
            region.region_id,
        ),
    )
    selected = []
    seen_roots = set()
    seen_docs = set()
    seen_themes = set()
    for region in ranked:
        if region.region_id in used_region_ids:
            continue
        affinity = _region_slot_affinity(query, slot, region)
        if affinity < 0.045:
            continue
        primary = normalize_text(_primary_theme(region))
        if selected and primary in seen_themes and set(region.root_chunk_ids) & seen_roots and set(region.doc_ids) & seen_docs:
            continue
        selected.append(region)
        seen_roots.update(region.root_chunk_ids)
        seen_docs.update(region.doc_ids)
        if primary:
            seen_themes.add(primary)
        if len(selected) >= max_regions:
            break
    return selected


def _build_theme_groups(query, root_regions, bridge_regions, theme_regions, chunk_store, group_limit):
    candidate_regions = []
    for region in root_regions + theme_regions + bridge_regions:
        if not region.root_connected:
            continue
        region_rel = lexical_overlap_score(query, region.descriptor_text)
        if region.region_kind == "bridge" and region_rel < 0.05:
            continue
        if region.region_kind in {"root", "theme"} and region_rel < 0.035:
            continue
        candidate_regions.append(region)

    if not candidate_regions:
        return _build_theme_groups_legacy(query, root_regions, bridge_regions, theme_regions, chunk_store, group_limit)

    slots = _aspect_layout_specs(query, group_limit=group_limit)
    used_region_ids = set()
    candidates = []
    for index, slot in enumerate(slots, start=1):
        max_regions = 4 if slot["key"] in {"contexts", "outcomes", "examples"} else 3
        selected_regions = _select_regions_for_slot(
            query,
            slot,
            candidate_regions,
            used_region_ids,
            max_regions=max_regions,
        )
        if not selected_regions:
            continue
        label = _slot_group_label(query, slot, selected_regions)
        group = _build_group(
            query,
            f"facet-{index:02d}",
            label,
            selected_regions,
            chunk_store,
        )
        representative_regions = _theme_representative_regions(query, selected_regions, max_roots=3, max_expansions=2)
        group = _retune_theme_group(group, query, representative_regions, chunk_store)
        group["focus_items"] = _priority_sort_texts(
            [label] + group.get("relation_themes", []) + group.get("focus_entities", []),
            [slot["label"]] + list(slot["cues"]),
            limit=6,
        )
        group["slot_key"] = slot["key"]
        group["selection_priority"] = (
            min(group.get("unique_doc_count", 0), 4)
            + min(group.get("unique_root_count", 0), 4)
            + min(group.get("region_count", 0), 4)
            + min(group.get("theme_region_count", 0), 3)
            + min(group.get("bridge_region_count", 0), 2)
            + min(group.get("root_region_count", 0), 2)
        )
        candidates.append(group)
        used_region_ids.update(region.region_id for region in selected_regions)
        if len(candidates) >= group_limit:
            break

    if len(candidates) < min(3, group_limit):
        legacy_groups = _build_theme_groups_legacy(
            query,
            root_regions,
            bridge_regions,
            theme_regions,
            chunk_store,
            group_limit,
        )
        existing_labels = {normalize_text(group["facet_label"]) for group in candidates}
        for group in legacy_groups:
            if len(candidates) >= group_limit:
                break
            label = normalize_text(group.get("facet_label", ""))
            if label in existing_labels:
                continue
            candidates.append(group)
            if label:
                existing_labels.add(label)

    return _select_groups(
        candidates,
        group_limit,
        lambda group: (group.get("slot_key", "legacy"), normalize_text(group.get("facet_label", ""))),
        coverage_key_fn=lambda group: (
            ("slot", group.get("slot_key", "legacy")),
            ("label", normalize_text(group.get("facet_label", ""))),
            ("doc", tuple(group.get("doc_ids", [])[:2])),
            ("root", tuple(group.get("root_chunk_ids", [])[:2])),
            ("theme", tuple(group.get("relation_themes", [])[:2])),
        ),
    )


def build_candidate_points_from_groups(facet_groups, top_k=8):
    """Project grouped evidence into LLM-facing aspect points.

    This keeps aspect construction tied to region/group composition instead of
    re-deriving labels from raw chunk text or isolated entities.
    """
    candidate_points = []
    for group in facet_groups[:top_k]:
        support_labels = []
        for label in group.get("focus_items", [])[:4]:
            text = normalize_text(label)
            if text and text not in support_labels:
                support_labels.append(text)
        for label in group.get("relation_themes", [])[:3]:
            text = normalize_text(label)
            if text and text not in support_labels:
                support_labels.append(text)
        for label in group.get("focus_entities", [])[:4]:
            text = normalize_text(label)
            if text and text not in support_labels:
                support_labels.append(text)
        candidate_points.append(
            {
                "point_type": "aspect",
                "label": normalize_text(group.get("facet_label", "")),
                "query_alignment": round(float(group.get("query_rel", group.get("group_score", 0.0))), 6),
                "chunk_alignment": round(
                    min(
                        1.0,
                        max(
                            float(group.get("root_anchor_count", 0)),
                            float(group.get("anchor_support", 0)),
                        )
                        / 3.0,
                    ),
                    6,
                ),
                "supporting_chunk_ids": list(group.get("supporting_chunk_ids", [])),
                "node_ids": list(group.get("nodes", [])),
                "edge_ids": list(group.get("edges", [])),
                "doc_ids": list(group.get("doc_ids", [])),
                "support_labels": support_labels[:6],
                "support_count": max(
                    int(group.get("region_count", 0)),
                    int(group.get("anchor_support", 0)),
                    len(support_labels),
                ),
                "group_id": group.get("group_id"),
                "focus_items": list(group.get("focus_items", []))[:6],
            }
        )
    return candidate_points


def _extract_json(text):
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _group_focus_pool(group, limit=8):
    items = []
    for label in group.get("relation_themes", [])[:4]:
        text = normalize_text(label)
        if text and text not in items:
            items.append(text)
    for label in group.get("focus_entities", [])[:5]:
        text = normalize_text(label)
        if text and text not in items:
            items.append(text)
    facet_label = normalize_text(group.get("facet_label", ""))
    if facet_label and facet_label not in items:
        items.append(facet_label)
    return items[:limit]


def _aspect_grouping_prompt(query, facet_groups, group_limit):
    grouping_hint = "Build broad QFS aspects that cover distinct major sides of the query."
    unit_blocks = []
    for group in facet_groups:
        focus_pool = _group_focus_pool(group)
        unit_blocks.append(
            "\n".join(
                [
                    f"id: {group['group_id']}",
                    f"label: {group.get('facet_label', '')}",
                    f"summary: {group.get('group_summary', '')}",
                    f"themes: {', '.join(group.get('relation_themes', [])[:4])}",
                    f"entities: {', '.join(group.get('focus_entities', [])[:5])}",
                    f"focus_pool: {', '.join(focus_pool)}",
                    f"docs: {', '.join(group.get('doc_ids', [])[:3])}",
                    f"region_kinds: {', '.join(group.get('region_kinds', []))}",
                ]
            )
        )
    max_groups = min(group_limit, max(2, min(6, len(facet_groups))))
    return f"""Plan query-facing evidence groups from the evidence units below.

Requirements:
- {grouping_hint}
- Use only the given unit ids.
- Use only focus items copied exactly from the provided focus_pool lines.
- Select focus items first, then assign unit ids to each group.
- Merge units when they support the same query-facing aspect.
- Omit clearly irrelevant or noisy units instead of forcing them into a group.
- Prefer specific, query-facing aspects over generic broad abstractions.
- Prefer aspect labels that could serve as answer subsection headings and that stay close to the chosen focus items.
- Cover distinct aspects rather than repeating the same one with different wording.
- Return at most {max_groups} aspects.

Query:
{query}

Evidence Units:

{chr(10).join(unit_blocks)}

Return JSON:
{{
  "groups": [
    {{
      "aspect_label": "...",
      "focus_items": ["...", "..."],
      "reason": "...",
      "unit_ids": ["facet-01", "facet-03"]
    }}
  ]
}}
"""


def _priority_sort_texts(values, focus_items, limit):
    focus_text = " ".join(normalize_text(item) for item in focus_items if normalize_text(item))
    return [
        text
        for text, _ in sorted(
            Counter(normalize_text(value) for value in values if normalize_text(value)).items(),
            key=lambda item: (
                -(
                    1.0
                    if item[0] in {normalize_text(focus) for focus in focus_items if normalize_text(focus)}
                    else lexical_overlap_score(focus_text, item[0])
                ),
                -item[1],
                item[0],
            ),
        )[:limit]
    ]


def _merge_groups_to_aspect(query, aspect_index, aspect_label, focus_items, reason, member_groups, chunk_store):
    root_chunk_ids = set()
    supporting_chunk_ids = set()
    anchor_candidates = []
    node_ids = set()
    edge_ids = set()
    relation_counter = Counter()
    focus_counter = Counter()
    region_kind_counter = Counter()
    doc_ids = []
    growth_traces = []
    member_ids = []
    member_labels = []
    region_count = 0
    for group in member_groups:
        root_chunk_ids.update(group.get("root_chunk_ids", []))
        supporting_chunk_ids.update(group.get("supporting_chunk_ids", []))
        node_ids.update(group.get("nodes", []))
        edge_ids.update(group.get("edges", []))
        relation_counter.update(group.get("relation_themes", []))
        focus_counter.update(group.get("focus_entities", []))
        region_count += int(group.get("region_count", 0))
        member_ids.append(group.get("group_id"))
        member_labels.append(group.get("facet_label"))
        for kind, count in group.get("region_kind_counts", {}).items():
            region_kind_counter[kind] += count
        for doc_id in group.get("doc_ids", []):
            if doc_id not in doc_ids:
                doc_ids.append(doc_id)
        for chunk_id in group.get("anchor_chunk_ids", []):
            if chunk_id not in anchor_candidates:
                anchor_candidates.append(chunk_id)
        for trace in group.get("growth_traces", []):
            if trace not in growth_traces:
                growth_traces.append(trace)
    normalized_focus_items = [normalize_text(item) for item in focus_items if normalize_text(item)]
    focus_query = " ".join([query] + normalized_focus_items).strip()
    ranked_supporting_chunks = _rank_chunks(focus_query, supporting_chunk_ids, chunk_store, root_chunk_ids)
    filtered_supporting_chunk_ids = ranked_supporting_chunks[: min(18, max(10, len(anchor_candidates) * 2, len(member_groups) * 4))]
    supporting_chunk_ids = set(filtered_supporting_chunk_ids)
    anchor_chunk_ids = _rank_chunks(focus_query, anchor_candidates, chunk_store, root_chunk_ids)[:5]
    relation_themes = _priority_sort_texts(list(relation_counter.elements()), normalized_focus_items, limit=5)
    focus_entities = _priority_sort_texts(list(focus_counter.elements()), normalized_focus_items, limit=8)
    descriptor_text = _evidence_descriptor_text(
        aspect_label,
        relation_themes,
        focus_entities,
        anchor_chunk_ids,
        chunk_store,
    )
    summary = (
        reason.strip()
        or f"This QFS aspect centers on {aspect_label} and merges related evidence units."
    )
    return {
        "group_id": f"llm-aspect-{aspect_index:02d}",
        "facet_label": normalize_text(aspect_label),
        "facet_prompt": _facet_prompt(query, aspect_label),
        "group_score": round(lexical_overlap_score(query, descriptor_text), 6),
        "query_rel": round(lexical_overlap_score(query, descriptor_text), 6),
        "anchor_support": len(anchor_chunk_ids),
        "root_anchor_count": len(set(anchor_chunk_ids) & root_chunk_ids),
        "node_count": len(node_ids),
        "edge_count": len(edge_ids),
        "region_count": region_count,
        "unique_doc_count": len(doc_ids),
        "unique_root_count": len(root_chunk_ids),
        "root_chunk_ids": sorted(root_chunk_ids),
        "doc_ids": doc_ids,
        "region_kinds": sorted(region_kind_counter),
        "region_kind_counts": dict(region_kind_counter),
        "root_region_count": region_kind_counter.get("root", 0),
        "bridge_region_count": region_kind_counter.get("bridge", 0),
        "theme_region_count": region_kind_counter.get("theme", 0),
        "section_region_count": region_kind_counter.get("section", 0),
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
        "member_group_ids": member_ids,
        "member_facet_labels": member_labels,
        "focus_items": normalized_focus_items[:6],
        "selection_reason": reason.strip(),
        "selection_priority": len(member_groups) + min(len(doc_ids), 3) + min(len(root_chunk_ids), 3),
    }


def regroup_facet_groups_with_llm(query, facet_groups, llm_client, chunk_store, group_limit):
    """Use the LLM to regroup facet units into query-facing aspects.

    This only changes how existing support is grouped. It does not add facts or
    evidence outside the current facet groups.
    """
    if llm_client is None or len(facet_groups) < 2:
        return facet_groups
    try:
        prompt = _aspect_grouping_prompt(query, facet_groups, group_limit)
        raw = llm_client.generate(
            prompt,
            system_prompt=ASPECT_GROUPING_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=900,
        )
        payload = _extract_json(raw)
        requested_groups = payload.get("groups", [])
        if not requested_groups:
            return facet_groups
        group_by_id = {group["group_id"]: group for group in facet_groups}
        regrouped = []
        used = set()
        for index, item in enumerate(requested_groups, start=1):
            unit_ids = [unit_id for unit_id in item.get("unit_ids", []) if unit_id in group_by_id]
            unit_ids = [unit_id for unit_id in unit_ids if unit_id not in used]
            if not unit_ids:
                continue
            focus_items = []
            allowed_focus_pool = {
                focus_item
                for unit_id in unit_ids
                for focus_item in _group_focus_pool(group_by_id[unit_id], limit=12)
            }
            for focus_item in item.get("focus_items", []):
                text = normalize_text(focus_item)
                if text and text in allowed_focus_pool and text not in focus_items:
                    focus_items.append(text)
            if not focus_items:
                focus_items = list(allowed_focus_pool)[:3]
            member_groups = [group_by_id[unit_id] for unit_id in unit_ids]
            regrouped.append(
                _merge_groups_to_aspect(
                    query=query,
                    aspect_index=index,
                    aspect_label=item.get("aspect_label", f"aspect {index}"),
                    focus_items=focus_items,
                    reason=item.get("reason", ""),
                    member_groups=member_groups,
                    chunk_store=chunk_store,
                )
            )
            used.update(unit_ids)
            if len(regrouped) >= group_limit:
                break
        if len(regrouped) < min(3, len(facet_groups)):
            leftovers = [group for group in facet_groups if group["group_id"] not in used]
            for group in leftovers:
                if len(regrouped) >= min(group_limit, 3):
                    break
                regrouped.append(group)
        return regrouped or facet_groups
    except Exception:
        return facet_groups


def _group_overlap_ratio(left, right):
    left = set(left)
    right = set(right)
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left | right), 1)


def _trace_local_graph(
    trace,
    graph,
    final_nodes,
    final_edges,
    chunk_to_nodes,
    chunk_to_edges,
):
    final_node_set = set(final_nodes)
    final_edge_set = set(final_edges)
    node_ids = set()
    edge_ids = set()
    for chunk_id in trace.get("selected_chunk_ids", []):
        node_ids.update(chunk_to_nodes.get(chunk_id, set()) & final_node_set)
        edge_ids.update(edge_id for edge_id in chunk_to_edges.get(chunk_id, set()) if edge_id in final_edge_set)
    local_graph = nx.Graph()
    local_graph.add_nodes_from(sorted(node_ids))
    local_graph.add_edges_from(sorted(edge_ids))
    return local_graph, node_ids, edge_ids


def _node_query_rel(query, node_id, graph):
    node_data = graph.nodes.get(node_id) if graph is not None else {}
    node_text = " ".join(
        [
            normalize_text(str(node_id)),
            normalize_text((node_data or {}).get("entity_type", "")),
            normalize_text((node_data or {}).get("description", "")),
        ]
    ).strip()
    return lexical_overlap_score(query, node_text)


def _query_content_terms(query):
    return [
        token
        for token in tokenize(normalize_text(query))
        if len(token) >= 3 and token not in STOPWORDS
    ]


def _clean_query_keywords(query, limit=8):
    """Keep only high-signal content words for group labels and weighted PPR seeds."""
    keywords = []
    for token in _query_content_terms(query):
        if token in _GROUP_LABEL_STOPWORDS:
            continue
        if token not in keywords:
            keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords


def _clean_label_text(text):
    return normalize_text(str(text)).strip().strip("\"' ")


def _is_label_term(token):
    token = token.strip().lower()
    if len(token) < 4:
        return False
    if not token.isalpha():
        return False
    if not any(vowel in token for vowel in "aeiou"):
        return False
    return True


def _group_query_terms(query, focus_entities, relation_themes, supporting_chunk_ids, chunk_store, limit=3):
    query_terms = _clean_query_keywords(query, limit=limit + 4)
    if not query_terms:
        return []
    support_text = " ".join(
        [_clean_label_text(item) for item in focus_entities]
        + [_clean_label_text(item) for item in relation_themes]
        + [
            normalize_text(chunk_store.get(chunk_id, {}).get("content", ""))[:240]
            for chunk_id in supporting_chunk_ids[:3]
        ]
    )
    return [term for term in query_terms if term in support_text and _is_label_term(term)][:limit]


_GROUP_LABEL_STOPWORDS = set(STOPWORDS) | {
    "art",
    "arts",
    "artist",
    "artists",
    "artwork",
    "artworks",
    "dataset",
    "section",
    "sections",
    "described",
    "different",
    "various",
    "period",
    "periods",
    "work",
    "works",
    "mentioned",
    "covered",
    "using",
    "according",
    "related",
    "based",
    "question",
    "summary",
    "evidence",
    "did",
    "does",
    "role",
    "ways",
    "what",
    "which",
    "how",
    "influence",
    "influenced",
    "emergence",
    "unknown",
    "relation",
    "change",
    "changes",
    "shaped",
    "shape",
    "potential",
    "might",
    "would",
    "could",
    "should",
    "have",
    "has",
    "had",
    "like",
    "than",
    "just",
    "still",
    "through",
    "elephant",
    "elephants",
}


def _group_support_terms(query, focus_entities, relation_themes, supporting_chunk_ids, chunk_store, limit=4):
    """Derive lightweight label terms from retrieved evidence."""
    query_terms = set(_clean_query_keywords(query, limit=12))
    focus_entity_tokens = {
        token
        for item in focus_entities
        for token in tokenize(_clean_label_text(item))
    }
    counter = Counter()
    for item in relation_themes:
        for token in tokenize(_clean_label_text(item)):
            if token in query_terms or token in _GROUP_LABEL_STOPWORDS or not _is_label_term(token):
                continue
            counter[token] += 2
    for chunk_id in supporting_chunk_ids[:4]:
        content = normalize_text(chunk_store.get(chunk_id, {}).get("content", ""))[:320]
        for token in tokenize(content):
            if (
                token in query_terms
                or token in _GROUP_LABEL_STOPWORDS
                or token in focus_entity_tokens
                or not _is_label_term(token)
            ):
                continue
            counter[token] += 1
    terms = []
    for token, _ in counter.most_common(12):
        if token not in terms:
            terms.append(token)
        if len(terms) >= limit:
            break
    return terms


def _trace_seed_nodes(query, trace, final_node_set, chunk_to_nodes, graph):
    root_nodes = set(chunk_to_nodes.get(trace["root_chunk_id"], set())) & set(final_node_set)
    bridge_nodes = set()
    for chunk_id in trace.get("bridge_chunk_ids", []):
        bridge_nodes.update(chunk_to_nodes.get(chunk_id, set()))
    bridge_nodes &= set(final_node_set)
    cleaned_query = " ".join(_clean_query_keywords(query, limit=10))
    seed_weights = {}
    for node_id in (root_nodes | bridge_nodes):
        rel = _node_query_rel(cleaned_query or query, node_id, graph)
        if rel > 0:
            seed_weights[node_id] = rel
    return seed_weights


def _ppr_subgraph_nodes(local_graph, seed_weights):
    if not seed_weights:
        return set()
    if local_graph.number_of_edges() <= 0:
        return set(seed_weights)
    nodes = list(local_graph.nodes())
    total_seed_weight = sum(seed_weights.values()) or 1.0
    personalization = {
        node_id: (seed_weights.get(node_id, 0.0) / total_seed_weight)
        for node_id in nodes
    }
    try:
        scores = nx.pagerank(
            local_graph,
            alpha=0.85,
            personalization=personalization,
            dangling=personalization,
            max_iter=100,
            tol=1.0e-6,
        )
    except Exception:
        # Fallback keeps the pipeline runnable if the fast backend is unavailable.
        scores = dict(personalization)
        alpha = 0.85
        for _ in range(40):
            next_scores = {node_id: (1.0 - alpha) * personalization[node_id] for node_id in nodes}
            dangling_mass = 0.0
            for node_id in nodes:
                degree = local_graph.degree(node_id)
                if degree <= 0:
                    dangling_mass += alpha * scores[node_id]
                    continue
                share = alpha * scores[node_id] / degree
                for neighbor_id in local_graph.neighbors(node_id):
                    next_scores[neighbor_id] += share
            if dangling_mass > 0:
                for node_id in nodes:
                    next_scores[node_id] += dangling_mass * personalization[node_id]
            delta = sum(abs(next_scores[node_id] - scores[node_id]) for node_id in nodes)
            scores = next_scores
            if delta < 1e-8:
                break
    ranked_nodes = [
        node_id
        for node_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ]
    prefix_size = max(len(seed_weights) * 4, 12)
    prefix_size = min(prefix_size, max(24, min(len(ranked_nodes), 48)))
    chosen_subgraph = local_graph.subgraph(ranked_nodes[:prefix_size]).copy()
    if chosen_subgraph.number_of_nodes() <= 0:
        return set(seed_weights)
    if chosen_subgraph.number_of_edges() <= 0:
        return set(chosen_subgraph.nodes())
    largest_component = max(nx.connected_components(chosen_subgraph), key=len)
    return set(largest_component)


def _group_chunks_for_trace_subgraph(trace, chosen_nodes, chosen_edges, chunk_to_nodes, chunk_to_edges):
    chosen_nodes = set(chosen_nodes)
    chosen_edges = set(chosen_edges)
    supporting_chunk_ids = []
    for chunk_id in trace.get("selected_chunk_ids", []):
        chunk_nodes = set(chunk_to_nodes.get(chunk_id, set()))
        chunk_edges = set(chunk_to_edges.get(chunk_id, set()))
        if chunk_nodes & chosen_nodes or chunk_edges & chosen_edges:
            supporting_chunk_ids.append(chunk_id)
    return supporting_chunk_ids


def _trace_group_label(
    group_index,
    label_terms,
    focus_entities,
    relation_themes,
    doc_ids,
    anchor_chunk_ids,
    chunk_store,
):
    focus_entities = [_clean_label_text(item) for item in focus_entities if _clean_label_text(item)]
    relation_themes = [
        _clean_label_text(item)
        for item in relation_themes
        if _clean_label_text(item) and _clean_label_text(item) not in {"unknown relation", "unknown_relation"}
    ]
    label_terms = [term for term in label_terms if term]
    if label_terms and relation_themes:
        return f"{' '.join(label_terms[:3])}: {relation_themes[0]}"
    if label_terms and focus_entities:
        return f"{' '.join(label_terms[:3])}: {focus_entities[0]}"
    if label_terms:
        return " ".join(label_terms[:4])
    if focus_entities and relation_themes:
        return f"{focus_entities[0]} via {relation_themes[0]}"
    if focus_entities:
        return focus_entities[0]
    if relation_themes:
        return relation_themes[0]
    return f"aspect {group_index}"


def _trace_group_gist(label, focus_entities, relation_themes, supporting_chunk_ids, chunk_store):
    """Produce one short evidence gist for generation-time group indexing."""
    focus_entities = [_clean_label_text(item) for item in focus_entities if _clean_label_text(item)]
    relation_themes = [
        _clean_label_text(item)
        for item in relation_themes
        if _clean_label_text(item) and _clean_label_text(item) not in {"unknown relation", "unknown_relation"}
    ]
    entity_part = ", ".join(focus_entities[:3])
    relation_part = relation_themes[0] if relation_themes else ""
    if relation_part and entity_part:
        return f"This group supports the answer with {relation_part} evidence around {entity_part}."
    if relation_part:
        return f"This group supports the answer with {relation_part} evidence."
    if entity_part:
        return f"This group supports the answer with evidence centered on {entity_part}."
    return f"This group supports the answer through {label}."


def _build_trace_group(
    *,
    query,
    group_index,
    trace,
    graph,
    final_nodes,
    final_edges,
    chunk_to_nodes,
    chunk_to_edges,
    chunk_store,
):
    local_graph, trace_nodes, trace_edges = _trace_local_graph(
        trace,
        graph,
        final_nodes,
        final_edges,
        chunk_to_nodes,
        chunk_to_edges,
    )
    if local_graph.number_of_nodes() <= 0:
        return None
    seed_nodes = _trace_seed_nodes(
        query,
        trace,
        trace_nodes,
        chunk_to_nodes,
        graph,
    )
    if not seed_nodes:
        return None
    chosen_nodes = _ppr_subgraph_nodes(local_graph, seed_nodes)
    if not chosen_nodes:
        return None
    chosen_edges = {
        edge_key(left, right)
        for left, right in local_graph.edges()
        if left in chosen_nodes and right in chosen_nodes
    }
    supporting_chunk_ids = _group_chunks_for_trace_subgraph(
        trace,
        chosen_nodes,
        chosen_edges,
        chunk_to_nodes,
        chunk_to_edges,
    )
    if not supporting_chunk_ids:
        return None

    root_chunk_id = trace["root_chunk_id"]
    bridge_chunk_ids = [chunk_id for chunk_id in trace.get("bridge_chunk_ids", []) if chunk_id in supporting_chunk_ids]
    support_chunk_ids = [chunk_id for chunk_id in trace.get("support_chunk_ids", []) if chunk_id in supporting_chunk_ids]
    context_chunk_ids = [chunk_id for chunk_id in trace.get("context_chunk_ids", []) if chunk_id in supporting_chunk_ids]
    anchor_chunk_ids = [root_chunk_id]
    relation_themes = _relation_themes(sorted(chosen_edges), graph) if graph is not None else []
    focus_entities = _focus_entities(query, sorted(chosen_nodes), graph) if graph is not None else sorted(chosen_nodes)[:6]
    query_label_terms = _group_query_terms(
        query,
        focus_entities,
        relation_themes,
        supporting_chunk_ids,
        chunk_store,
    )
    support_label_terms = _group_support_terms(
        query,
        focus_entities,
        relation_themes,
        supporting_chunk_ids,
        chunk_store,
    )
    label_terms = list(dict.fromkeys(query_label_terms + support_label_terms))
    doc_ids = _doc_ids(supporting_chunk_ids, chunk_store)
    label = _trace_group_label(
        group_index,
        label_terms,
        focus_entities,
        relation_themes,
        doc_ids,
        anchor_chunk_ids,
        chunk_store,
    )
    descriptor_text = _evidence_descriptor_text(
        label,
        relation_themes,
        focus_entities,
        anchor_chunk_ids,
        chunk_store,
    )
    group_summary = _trace_group_gist(
        label,
        focus_entities,
        relation_themes,
        supporting_chunk_ids,
        chunk_store,
    )
    growth_traces = [
        f"root={root_chunk_id}",
        f"bridge={len(bridge_chunk_ids)}",
        f"support={len(support_chunk_ids)}",
        f"context={len(context_chunk_ids)}",
    ]
    return {
        "group_id": f"kg-{group_index:02d}",
        "facet_label": label,
        "primary_theme": relation_themes[0] if relation_themes else (focus_entities[0] if focus_entities else label),
        "facet_prompt": _facet_prompt(query, label),
        "group_score": round(lexical_overlap_score(query, descriptor_text), 6),
        "query_rel": round(lexical_overlap_score(query, descriptor_text), 6),
        "anchor_support": len(anchor_chunk_ids),
        "root_anchor_count": 1,
        "node_count": len(chosen_nodes),
        "edge_count": len(chosen_edges),
        "region_count": 1,
        "unique_doc_count": len(doc_ids),
        "unique_root_count": 1,
        "root_chunk_ids": [root_chunk_id],
        "doc_ids": doc_ids,
        "region_kinds": [
            kind
            for kind, chunk_ids in (
                ("root", [root_chunk_id]),
                ("bridge", bridge_chunk_ids),
                ("support", support_chunk_ids),
                ("context", context_chunk_ids),
            )
            if chunk_ids
        ],
        "region_kind_counts": {
            "root": 1,
            "bridge": len(bridge_chunk_ids),
            "support": len(support_chunk_ids),
            "context": len(context_chunk_ids),
        },
        "root_region_count": 1,
        "bridge_region_count": len(bridge_chunk_ids),
        "theme_region_count": 0,
        "section_region_count": 0,
        "relation_themes": relation_themes,
        "focus_entities": focus_entities,
        "supporting_chunk_ids": supporting_chunk_ids,
        "anchor_chunk_ids": anchor_chunk_ids,
        "source_previews": [
            {
                "chunk_id": chunk_id,
                "preview": normalize_text(chunk_store.get(chunk_id, {}).get("content", ""))[:220],
            }
            for chunk_id in supporting_chunk_ids[:3]
        ],
        "growth_traces": growth_traces,
        "group_summary": group_summary,
        "nodes": sorted(chosen_nodes),
        "edges": sorted(chosen_edges),
        "focus_items": list(dict.fromkeys(focus_entities[:4] + relation_themes[:3]))[:6],
    }


def _build_trace_groups(
    *,
    query,
    root_traces,
    graph,
    final_nodes,
    final_edges,
    chunk_to_nodes,
    chunk_to_edges,
    chunk_store,
    group_limit,
):
    candidates = []
    for group_index, trace in enumerate(root_traces, start=1):
        group = _build_trace_group(
            query=query,
            group_index=group_index,
            trace=trace,
            graph=graph,
            final_nodes=final_nodes,
            final_edges=final_edges,
            chunk_to_nodes=chunk_to_nodes,
            chunk_to_edges=chunk_to_edges,
            chunk_store=chunk_store,
        )
        if group is not None:
            candidates.append(group)
    selected = []
    remaining = list(candidates)
    while remaining and len(selected) < group_limit:
        best_index = None
        best_key = None
        for index, candidate in enumerate(remaining):
            max_overlap = 0.0
            for chosen in selected:
                overlap = _group_overlap_ratio(candidate["supporting_chunk_ids"], chosen["supporting_chunk_ids"])
                if overlap > max_overlap:
                    max_overlap = overlap
            if max_overlap >= 0.88:
                continue
            if max_overlap >= 0.72:
                continue
            rank_key = (
                -max_overlap,
                candidate["query_rel"],
                candidate["node_count"],
                candidate["edge_count"],
                -len(candidate["supporting_chunk_ids"]),
            )
            if best_key is None or rank_key > best_key:
                best_key = rank_key
                best_index = index
        if best_index is None:
            break
        chosen = remaining.pop(best_index)
        selected.append(chosen)
    return selected


def build_answer_facet_groups(
    *,
    query,
    root_chunk_ids,
    graph,
    final_nodes,
    final_edges,
    root_traces,
    last_structural_output,
    chunk_neighbors,
    chunk_to_nodes,
    chunk_to_edges,
    node_to_chunks,
    edge_to_chunks,
    chunk_store,
    group_limit,
    allowed_doc_ids=None,
):
    """Build QFS knowledge groups from root traces and the final graph.

    参数:
        query: 查询文本。
        root_chunk_ids: 当前根证据 chunk id 列表。
        graph: 实体关系图对象。
        final_nodes: 选定的最终节点集合。
        final_edges: 选定的最终边集合。
        root_traces: 根 trace 结构列表。
        last_structural_output: 最后一次结构关联输出。
        chunk_neighbors: chunk 邻居映射。
        chunk_to_nodes: chunk 到节点映射。
        chunk_to_edges: chunk 到边映射。
        node_to_chunks: 节点到 chunk 映射。
        edge_to_chunks: 边到 chunk 映射。
        chunk_store: chunk 存储字典。
        group_limit: 要构造的 facet group 上限。
        allowed_doc_ids: 可选的文档白名单，用于过滤支持 chunk。

    返回:
        (facet_groups, organization_mode) 二元组，facet_groups 为分组结果。
    """
    groups = _build_trace_groups(
        query=query,
        root_traces=root_traces,
        graph=graph,
        final_nodes=final_nodes,
        final_edges=final_edges,
        chunk_to_nodes=chunk_to_nodes,
        chunk_to_edges=chunk_to_edges,
        chunk_store=chunk_store,
        group_limit=group_limit,
    )
    return groups, ORGANIZATION_MODE


def build_layout_groups(**kwargs):
    """Thin wrapper that exposes the organization stage in layout terms.

    参数:
        **kwargs: 传递给 build_answer_facet_groups 的组织阶段参数。

    返回:
        与 build_answer_facet_groups 相同的 (facet_groups, organization_mode) 结果。
    """
    return build_answer_facet_groups(**kwargs)
