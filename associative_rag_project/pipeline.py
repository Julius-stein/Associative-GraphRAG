"""End-to-end retrieval pipeline for associative QFS.

端到端检索流水线模块，包含检索、扩展、组织和在线生成流程。
"""

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import tiktoken

from .association import (
    build_edge_role_sets,
    build_node_role_sets,
    expand_associative_graph,
)
from .context import build_prompt_context
from .data import (
    build_chunk_mappings,
    build_chunk_neighborhoods,
    infer_corpus_name,
    load_graph_corpus,
    load_query_rows,
    resolve_corpus_index_dir,
)
from .config import load_llm_config
from .embedding_client import build_embedding_client
from .logging_utils import log, shorten
from .llm_client import generate_one_answer_record
from .organization import (
    build_layout_groups,
    build_candidate_points_from_groups,
    resolve_organization_layout,
)
from .retrieval import (
    BM25Index,
    DenseChunkIndex,
    HybridChunkRetriever,
    build_graph_keyword_index,
    merge_candidate_hits_with_graph,
    search_graph_evidence_chunks,
    score_root_edges,
    score_root_nodes,
    select_anchor_root_chunks,
)


def summarize_results(results):
    """Summarize per-query numeric stats into mean / median / max values."""
    if not results:
        return {}
    stats = defaultdict(list)
    for result in results:
        for key, value in result["stats"].items():
            stats[key].append(value)
    summary = {}
    for key, values in stats.items():
        values = sorted(values)
        summary[key] = {
            "mean": round(sum(values) / len(values), 3),
            "median": values[len(values) // 2],
            "max": values[-1],
        }
    return summary


@lru_cache(maxsize=4)
def _token_encoder_for_model(model_name):
    """Return a tokenizer for the configured generation model."""
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def _true_token_count(text):
    """Tokenizer-based token count for the LLM-facing prompt context."""
    text = text or ""
    if not text:
        return 0
    model_name = load_llm_config().get("model", "gpt-4o-mini")
    encoder = _token_encoder_for_model(model_name)
    return len(encoder.encode(text))


def _format_root_chunk_preview(root_chunk_hits, chunk_store, limit=3):
    previews = []
    for item in root_chunk_hits[:limit]:
        chunk = chunk_store.get(item["chunk_id"], {})
        preview = shorten(chunk.get("content", ""), 90)
        previews.append(
            f"{item['chunk_id']} score={item['score_norm']:.3f} novelty={item.get('novelty_gain', 0)} "
            f"overlap={item.get('max_selected_overlap', 0.0):.3f} basin={item.get('basin_signature', '-')}" 
            f" kw={item.get('keyword_match_score', 0.0):.3f}/{item.get('keyword_hit_count', 0)}"
            f" :: {preview}"
        )
    return previews


def _serialize_root_chunk_hit(item):
    serializable = {}
    for key, value in item.items():
        if key in {"graph_nodes", "graph_edges"}:
            continue
        if isinstance(value, set):
            serializable[key] = sorted(value)
        elif isinstance(value, tuple):
            serializable[key] = list(value)
        else:
            serializable[key] = value
    return serializable


def _build_chunk_roles(root_chunk_hits, promoted_root_chunks, facet_groups, theme_selected_chunks=None):
    role_map = {}

    def ensure(chunk_id):
        role_map.setdefault(chunk_id, {"roles": [], "linked_groups": []})
        return role_map[chunk_id]

    for item in root_chunk_hits:
        slot = ensure(item["chunk_id"])
        if item.get("root_role") == "diversity":
            role = "aspect-root"
        elif item.get("root_role") == "structural":
            role = "structural-root"
        else:
            role = "query-root"
        if role not in slot["roles"]:
            slot["roles"].append(role)

    for item in promoted_root_chunks:
        slot = ensure(item["chunk_id"])
        if "structural-root" not in slot["roles"]:
            slot["roles"].append("structural-root")

    if theme_selected_chunks:
        role_aliases = {
            "core": "query-root",
            "bridge": "bridge-chunk",
            "support": "support-chunk",
            "context": "context-chunk",
        }
        for bucket, chunk_ids in theme_selected_chunks.items():
            role = role_aliases.get(bucket, bucket)
            for chunk_id in chunk_ids:
                slot = ensure(chunk_id)
                if role not in slot["roles"]:
                    slot["roles"].append(role)

    for group in facet_groups:
        for chunk_id in group.get("anchor_chunk_ids", []):
            slot = ensure(chunk_id)
            if "facet-anchor" not in slot["roles"]:
                slot["roles"].append("facet-anchor")
            if group["group_id"] not in slot["linked_groups"]:
                slot["linked_groups"].append(group["group_id"])
        for chunk_id in group.get("supporting_chunk_ids", []):
            slot = ensure(chunk_id)
            if "facet-support" not in slot["roles"]:
                slot["roles"].append("facet-support")
            if group["group_id"] not in slot["linked_groups"]:
                slot["linked_groups"].append(group["group_id"])

    return [
        {
            "chunk_id": chunk_id,
            "roles": payload["roles"],
            "linked_groups": payload["linked_groups"][:6],
        }
        for chunk_id, payload in sorted(role_map.items())
    ]


def _format_round_preview(round_info):
    lines = [
        f"round={round_info['round']} structural_paths={round_info['structural_path_count']} "
        f"structural_chunks={round_info.get('structural_chunk_bridge_count', 0)} "
        f"+nodes={round_info['structural_added_node_count']} +edges={round_info['structural_added_edge_count']} "
        f"semantic_chunks={round_info.get('semantic_chunk_coverage_count', 0)} "
        f"semantic +nodes={round_info['semantic_added_node_count']} +edges={round_info['semantic_added_edge_count']} "
        f"current=({round_info['current_node_count']}n/{round_info['current_edge_count']}e)"
    ]
    for path_item in round_info.get("structural_path_preview", []):
        lines.append(
            f"  structural: bridge={path_item['score']:.3f} new_sources={path_item['new_source_count']} "
            f"path={' -> '.join(path_item['path'])}"
        )
    for chunk_item in round_info.get("structural_chunk_preview", []):
        lines.append(
            f"  structural-chunk: intro_qrel={chunk_item.get('introduced_query_rel', 0.0):.3f} "
            f"touch={chunk_item.get('frontier_touch', 0)} "
            f"nodes={chunk_item.get('introduced_node_count', len(chunk_item.get('introduced_node_ids', [])))} "
            f"edges={chunk_item.get('introduced_edge_count', len(chunk_item.get('introduced_edge_ids', [])))} "
            f"{chunk_item['chunk_id']}"
        )
    for edge_item in round_info.get("semantic_edge_preview", []):
        lines.append(
            f"  coverage-edge: gain={edge_item['score']:.3f} new_sources={edge_item['new_source_count']} "
            f"{edge_item['edge'][0]} -> {edge_item['edge'][1]} [{edge_item['category']}]"
        )
    for node_item in round_info.get("semantic_node_preview", []):
        lines.append(
            f"  coverage-node: gain={node_item['score']:.3f} bridge={node_item['bridge_strength']:.2f} {node_item['id']}"
        )
    for chunk_item in round_info.get("semantic_chunk_preview", []):
        lines.append(
            f"  coverage-chunk: rels={chunk_item.get('introduced_relation_count', 0)} "
            f"nodes={chunk_item.get('introduced_node_count', len(chunk_item.get('node_ids', [])))} "
            f"edges={chunk_item.get('introduced_edge_count', len(chunk_item.get('edge_ids', [])))} "
            f"{chunk_item['chunk_id']}"
        )
    return lines


def _anchor_candidate_top_k(cfg):
    return max(
        cfg["top_chunks"],
        cfg["top_chunks"] * max(cfg["chunk_candidate_multiplier"], 1),
        cfg.get("candidate_pool_size", 30),
    )


def _anchor_root_top_k(cfg):
    return cfg["top_chunks"]


def _expand_max_hop(cfg):
    return cfg["max_hop"]


def _online_stage_clock():
    """Use wall-clock time so query timings match end-to-end runtime perception."""
    return perf_counter()


def _load_runtime_resources(corpus_dir, cfg):
    """Load shared corpus/runtime resources once for either offline or online execution."""
    corpus_dir = Path(corpus_dir)
    index_dir = resolve_corpus_index_dir(corpus_dir)
    log(f"[pipeline] loading corpus={corpus_dir} index={index_dir}")
    graph, chunk_store, index_dir = load_graph_corpus(corpus_dir)
    log(f"[pipeline] graph nodes={graph.number_of_nodes()} edges={graph.number_of_edges()} chunks={len(chunk_store)}")
    bm25_index = BM25Index.build(chunk_store)
    dense_index = DenseChunkIndex.load(index_dir / "vdb_chunks.json") if (index_dir / "vdb_chunks.json").exists() else None
    embedding_client = None
    if cfg["retrieval_mode"] in {"dense", "hybrid"}:
        llm_cfg = load_llm_config()
        for key in (
            "embedding_provider",
            "embedding_model",
            "embedding_base_url",
            "embedding_api_key",
            "local_embedding_model",
            "local_embedding_device",
            "local_embedding_batch_size",
            "local_embedding_max_length",
            "local_embedding_query_instruction",
        ):
            if cfg.get(key) is not None:
                llm_cfg[key] = cfg[key]
        embedding_client = build_embedding_client(llm_cfg)
        log(
            "[pipeline] dense index loaded "
            f"rows={len(dense_index.chunk_ids) if dense_index else 0} "
            f"provider={llm_cfg.get('embedding_provider', 'openai_compatible')}"
        )
    chunk_retriever = HybridChunkRetriever(
        bm25_index=bm25_index,
        dense_index=dense_index,
        embedding_client=embedding_client,
        mode=cfg["retrieval_mode"],
        dense_weight=cfg["dense_weight"],
        bm25_weight=cfg["bm25_weight"],
    )
    chunk_ids = list(chunk_store.keys())
    chunk_to_nodes, chunk_to_edges, node_to_chunks, edge_to_chunks = build_chunk_mappings(graph, chunk_ids)
    chunk_neighbors = build_chunk_neighborhoods(chunk_store, radius=1)
    keyword_index = build_graph_keyword_index(
        graph=graph,
        chunk_to_nodes=chunk_to_nodes,
        chunk_to_edges=chunk_to_edges,
        chunk_store=chunk_store,
    )
    log(
        "[pipeline] keyword-index "
        f"vocab={len(keyword_index.vocabulary)} ignored={len(keyword_index.ignored_terms)} "
        f"df_threshold={keyword_index.ignored_threshold}"
    )
    return {
        "corpus_dir": corpus_dir,
        "index_dir": index_dir,
        "graph": graph,
        "chunk_store": chunk_store,
        "chunk_retriever": chunk_retriever,
        "chunk_to_nodes": chunk_to_nodes,
        "chunk_to_edges": chunk_to_edges,
        "node_to_chunks": node_to_chunks,
        "edge_to_chunks": edge_to_chunks,
        "chunk_neighbors": chunk_neighbors,
        "keyword_index": keyword_index,
    }


def _result_paths(corpus_dir, output_dir, cfg, limit_groups):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_name = infer_corpus_name(corpus_dir)
    limit_suffix = f"_limit{limit_groups}" if limit_groups is not None else ""
    stem = f"{corpus_name}_top{cfg['top_chunks']}_hop{cfg['max_hop']}_assoc_project{limit_suffix}"
    return {
        "output_dir": output_dir,
        "stem": stem,
        "retrieval_path": output_dir / f"{stem}_retrieval.json",
        "performance_path": output_dir / f"{stem}_performance.json",
        "sample_context_path": output_dir / f"{stem}_sample_context.txt",
        "answer_path": output_dir / f"{stem}_answers.json",
    }


def _performance_records(results):
    return {
        "per_query_records": [
            {
                "query_id": result.get("group_id"),
                "anchor_seconds": result.get("stats", {}).get("anchor_seconds"),
                "expand_seconds": result.get("stats", {}).get("expand_seconds"),
                "organize_seconds": result.get("stats", {}).get("organize_seconds"),
                "query_total_seconds": result.get("stats", {}).get("query_total_seconds"),
                "final_node_count": result.get("stats", {}).get("final_node_count"),
                "final_edge_count": result.get("stats", {}).get("final_edge_count"),
                "llm_input_tokens": result.get("stats", {}).get("prompt_context_token_estimate"),
            }
            for result in results
        ],
    }


def _write_retrieval_outputs(*, corpus_dir, output_dir, cfg, limit_groups, results):
    paths = _result_paths(corpus_dir, output_dir, cfg, limit_groups)
    performance_summary = _performance_records(results)
    payload = {
        "corpus_dir": str(corpus_dir),
        "config": cfg,
        "summary": summarize_results(results),
        "performance": performance_summary,
        "results": results,
    }
    paths["retrieval_path"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["performance_path"].write_text(json.dumps(performance_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if results:
        paths["sample_context_path"].write_text(results[0]["prompt_context"], encoding="utf-8")
    log(f"[pipeline] wrote {paths['retrieval_path']}")
    log(f"[pipeline] wrote {paths['performance_path']}")
    return payload, paths


def _pick_section_doc_ids(root_chunk_hits, chunk_store):
    doc_ids = [
        chunk_store.get(item["chunk_id"], {}).get("full_doc_id")
        for item in root_chunk_hits
        if chunk_store.get(item["chunk_id"], {}).get("full_doc_id")
    ]
    if not doc_ids:
        return None
    return {doc_ids[0]}


def _run_anchor_stage(
    *,
    query_row,
    organization_layout,
    graph,
    chunk_store,
    chunk_retriever,
    chunk_to_nodes,
    chunk_to_edges,
    node_to_chunks,
    edge_to_chunks,
    keyword_index,
    cfg,
):
    candidate_top_k = _anchor_candidate_top_k(cfg)
    root_top_k = _anchor_root_top_k(cfg)
    max_hop = _expand_max_hop(cfg)

    candidate_chunk_hits = chunk_retriever.search(query_row["query"], top_k=candidate_top_k)
    graph_evidence_hits = search_graph_evidence_chunks(
        query=query_row["query"],
        graph=graph,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        top_chunk_k=max(candidate_top_k * 2, cfg["top_chunks"] * 10),
    )
    candidate_chunk_hits = merge_candidate_hits_with_graph(
        primary_hits=candidate_chunk_hits,
        graph_hits=graph_evidence_hits,
        graph_weight=0.28,
    )[:candidate_top_k]

    query_chunk_score_lookup = {
        item["chunk_id"]: item.get(
            "graph_evidence_score_norm",
            item.get(
                "graph_focus_score_norm",
                item.get(
                    "graph_keyword_score_norm",
                    item.get("dense_score_norm", item.get("retrieval_score", item.get("score_norm", 0.0))),
                ),
            ),
        )
        for item in candidate_chunk_hits
    }

    root_chunk_hits = select_anchor_root_chunks(
        query=query_row["query"],
        candidate_hits=candidate_chunk_hits,
        chunk_store=chunk_store,
        chunk_to_nodes=chunk_to_nodes,
        chunk_to_edges=chunk_to_edges,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        top_k=root_top_k,
        graph=graph,
        keyword_index=keyword_index,
    )

    allowed_doc_ids = None

    root_chunk_hits = [_serialize_root_chunk_hit(item) for item in root_chunk_hits]

    root_chunk_score_lookup = {item["chunk_id"]: item["score_norm"] for item in root_chunk_hits}
    root_chunk_ids = list(root_chunk_score_lookup)
    root_nodes = set()
    root_edges = set()
    for chunk_id in root_chunk_ids:
        root_nodes.update(chunk_to_nodes.get(chunk_id, set()))
        root_edges.update(chunk_to_edges.get(chunk_id, set()))

    top_root_nodes = score_root_nodes(
        query=query_row["query"],
        root_nodes=root_nodes,
        graph=graph,
        node_to_chunks=node_to_chunks,
        root_chunk_score_lookup=root_chunk_score_lookup,
    )[: cfg["top_root_nodes"]]
    top_root_edges = score_root_edges(
        query=query_row["query"],
        root_edges=root_edges,
        graph=graph,
        edge_to_chunks=edge_to_chunks,
        root_chunk_score_lookup=root_chunk_score_lookup,
    )[: cfg["top_root_edges"]]

    return {
        "retrieval_mode_label": "anchor:dense+graph_evidence",
        "candidate_top_k": candidate_top_k,
        "root_top_k": root_top_k,
        "max_hop": max_hop,
        "candidate_chunk_hits": candidate_chunk_hits,
        "query_chunk_score_lookup": query_chunk_score_lookup,
        "root_chunk_hits": root_chunk_hits,
        "root_chunk_score_lookup": root_chunk_score_lookup,
        "root_chunk_ids": root_chunk_ids,
        "root_nodes": root_nodes,
        "root_edges": root_edges,
        "top_root_nodes": top_root_nodes,
        "top_root_edges": top_root_edges,
        "allowed_doc_ids": allowed_doc_ids,
    }


def _run_expand_stage(
    *,
    query_row,
    organization_layout,
    anchor_state,
    graph,
    chunk_to_nodes,
    chunk_to_edges,
    node_to_chunks,
    edge_to_chunks,
    chunk_store,
    cfg,
):
    expansion = expand_associative_graph(
        query=query_row["query"],
        graph=graph,
        root_nodes=anchor_state["root_nodes"],
        root_edges=anchor_state["root_edges"],
        root_chunk_ids=anchor_state["root_chunk_ids"],
        root_chunk_score_lookup=anchor_state["root_chunk_score_lookup"],
        query_chunk_score_lookup=anchor_state["query_chunk_score_lookup"],
        chunk_to_nodes=chunk_to_nodes,
        chunk_to_edges=chunk_to_edges,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        chunk_neighbors=cfg["chunk_neighbors"],
        chunk_store=chunk_store,
        top_root_nodes=cfg["top_root_nodes"],
        top_root_edges=cfg["top_root_edges"],
        max_hop=anchor_state["max_hop"],
        path_budget=cfg["path_budget"],
        semantic_edge_budget=cfg["semantic_edge_budget"],
        semantic_node_budget=cfg["semantic_node_budget"],
        association_rounds=cfg["association_rounds"],
        semantic_edge_min_score=cfg["semantic_edge_min_score"],
        semantic_node_min_score=cfg["semantic_node_min_score"],
        allowed_doc_ids=anchor_state["allowed_doc_ids"],
    )
    promoted_root_chunks = [_serialize_root_chunk_hit(item) for item in expansion.get("promoted_root_chunks", [])]
    effective_root_chunk_ids = expansion.get("effective_root_chunk_ids", anchor_state["root_chunk_ids"])
    effective_root_chunk_hits = anchor_state["root_chunk_hits"] + promoted_root_chunks
    return {
        "expansion": expansion,
        "promoted_root_chunks": promoted_root_chunks,
        "effective_root_chunk_ids": effective_root_chunk_ids,
        "effective_root_chunk_hits": effective_root_chunk_hits,
    }


def _run_organize_stage(
    *,
    query_row,
    organization_layout,
    anchor_state,
    expand_state,
    graph,
    chunk_store,
    chunk_to_nodes,
    chunk_to_edges,
    node_to_chunks,
    edge_to_chunks,
    cfg,
):
    expansion = expand_state["expansion"]
    node_roles = build_node_role_sets(
        root_nodes=anchor_state["root_nodes"],
        structural_nodes=set(expansion["all_structural_nodes"]),
        semantic_nodes=set(expansion["all_semantic_nodes"]),
    )
    edge_roles = build_edge_role_sets(
        root_edges=anchor_state["root_edges"],
        structural_edges=set(expansion["all_structural_edges"]),
        semantic_edges=expansion["all_semantic_edges"],
    )
    facet_groups, organization_layout = build_layout_groups(
        query=query_row["query"],
        graph=graph,
        final_nodes=expansion["final_nodes"],
        final_edges=expansion["final_edges"],
        root_chunk_ids=expand_state["effective_root_chunk_ids"],
        root_traces=expansion.get("root_traces", []),
        last_structural_output=expansion["last_structural_output"],
        chunk_neighbors=cfg["chunk_neighbors"],
        chunk_to_nodes=chunk_to_nodes,
        chunk_to_edges=chunk_to_edges,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        chunk_store=chunk_store,
        group_limit=cfg["group_limit"],
        allowed_doc_ids=anchor_state["allowed_doc_ids"],
    )
    candidate_points = build_candidate_points_from_groups(facet_groups, top_k=8)
    theme_selected_chunks = expansion.get("theme_selected_chunks", {})
    chunk_roles = _build_chunk_roles(
        expand_state["effective_root_chunk_hits"],
        expand_state["promoted_root_chunks"],
        facet_groups,
        theme_selected_chunks=theme_selected_chunks,
    )
    prompt_payload = build_prompt_context(
        query_row=query_row,
        root_chunk_hits=expand_state["effective_root_chunk_hits"],
        top_root_nodes=anchor_state["top_root_nodes"],
        top_root_edges=anchor_state["top_root_edges"],
        node_roles=node_roles,
        edge_roles=edge_roles,
        semantic_nodes=[{"id": node_id} for node_id in expansion["all_semantic_nodes"]],
        semantic_edges=expansion["all_semantic_edges"],
        final_nodes=expansion["final_nodes"],
        final_edges=expansion["final_edges"],
        facet_groups=facet_groups,
        candidate_points=candidate_points,
        chunk_roles=chunk_roles,
        theme_selected_chunks=theme_selected_chunks,
        chunk_store=chunk_store,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        max_source_chunks=cfg["max_source_chunks"],
        max_source_word_budget=cfg["max_source_word_budget"],
    )
    return {
        "organization_layout": organization_layout,
        "node_roles": node_roles,
        "edge_roles": edge_roles,
        "facet_groups": facet_groups,
        "candidate_points": candidate_points,
        "theme_selected_chunks": theme_selected_chunks,
        "chunk_roles": chunk_roles,
        "prompt_payload": prompt_payload,
    }


def _run_query_states(
    *,
    query_row,
    graph,
    chunk_store,
    chunk_retriever,
    chunk_to_nodes,
    chunk_to_edges,
    node_to_chunks,
    edge_to_chunks,
    keyword_index,
    cfg,
    query_index=None,
    total_queries=None,
):
    """Run anchor -> expand -> organize for one query and return intermediate states."""
    prefix = f"[query {query_index}/{total_queries}] " if query_index is not None and total_queries is not None else ""
    log(f"{prefix}start {query_row['group_id']} :: {query_row['query']}")
    query_start = _online_stage_clock()
    organization_layout, controller_info = resolve_organization_layout(query_row)

    anchor_start = _online_stage_clock()
    anchor_state = _run_anchor_stage(
        query_row=query_row,
        organization_layout=organization_layout,
        graph=graph,
        chunk_store=chunk_store,
        chunk_retriever=chunk_retriever,
        chunk_to_nodes=chunk_to_nodes,
        chunk_to_edges=chunk_to_edges,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        keyword_index=keyword_index,
        cfg=cfg,
    )
    anchor_elapsed = _online_stage_clock() - anchor_start
    log(
        f"{prefix}anchor candidates={len(anchor_state['candidate_chunk_hits'])} "
        f"mode={anchor_state['retrieval_mode_label']} organization={organization_layout} "
        f"controller={controller_info['mode']}/{controller_info['source']}"
    )
    for line in _format_root_chunk_preview(anchor_state["root_chunk_hits"], chunk_store):
        log(f"{prefix}root {line}")

    expand_start = _online_stage_clock()
    expand_state = _run_expand_stage(
        query_row=query_row,
        organization_layout=organization_layout,
        anchor_state=anchor_state,
        graph=graph,
        chunk_to_nodes=chunk_to_nodes,
        chunk_to_edges=chunk_to_edges,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        chunk_store=chunk_store,
        cfg=cfg,
    )
    expand_elapsed = _online_stage_clock() - expand_start
    for round_info in expand_state["expansion"]["rounds"]:
        for line in _format_round_preview(round_info):
            log(f"{prefix}{line}")

    organize_start = _online_stage_clock()
    organize_state = _run_organize_stage(
        query_row=query_row,
        organization_layout=organization_layout,
        anchor_state=anchor_state,
        expand_state=expand_state,
        graph=graph,
        chunk_store=chunk_store,
        chunk_to_nodes=chunk_to_nodes,
        chunk_to_edges=chunk_to_edges,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        cfg=cfg,
    )
    organize_elapsed = _online_stage_clock() - organize_start
    total_elapsed = _online_stage_clock() - query_start
    expansion = expand_state["expansion"]
    prompt_payload = organize_state["prompt_payload"]
    log(
        f"{prefix}done roots=({len(anchor_state['root_nodes'])}n/{len(anchor_state['root_edges'])}e) "
        f"final=({len(expansion['final_nodes'])}n/{len(expansion['final_edges'])}e) "
        f"groups={len(organize_state['facet_groups'])} sources={prompt_payload['selected_source_chunk_count']} "
        f"words={prompt_payload['selected_source_word_count']} "
        f"time={total_elapsed:.2f}s"
    )
    return {
        "controller_info": controller_info,
        "anchor_state": anchor_state,
        "expand_state": expand_state,
        "organize_state": organize_state,
        "timings": {
            "anchor_seconds": anchor_elapsed,
            "expand_seconds": expand_elapsed,
            "organize_seconds": organize_elapsed,
            "query_total_seconds": total_elapsed,
        },
        "prefix": prefix,
    }


def _build_query_record_from_states(query_row, controller_info, anchor_state, expand_state, organize_state, timings):
    expansion = expand_state["expansion"]
    prompt_payload = organize_state["prompt_payload"]
    prompt_context = prompt_payload["context"]
    prompt_word_count = len(prompt_context.split())
    prompt_token_count = _true_token_count(prompt_context)
    effective_root_chunk_hits = expand_state["effective_root_chunk_hits"]
    primary_root_chunks = [item for item in effective_root_chunk_hits if item.get("root_role") == "primary"]
    diversity_root_chunks = [item for item in effective_root_chunk_hits if item.get("root_role") == "diversity"]
    root_basin_count = len({tuple(item.get("basin_key", (item["chunk_id"],))) for item in effective_root_chunk_hits})
    initial_root_basin_count = len({tuple(item.get("basin_key", (item["chunk_id"],))) for item in anchor_state["root_chunk_hits"]})
    return {
        **query_row,
        "candidate_root_chunks": anchor_state["candidate_chunk_hits"],
        "root_chunks": effective_root_chunk_hits,
        "primary_root_chunks": primary_root_chunks,
        "diversity_root_chunks": diversity_root_chunks,
        "promoted_root_chunks": expand_state["promoted_root_chunks"],
        "theme_selected_chunks": organize_state["theme_selected_chunks"],
        "chunk_roles": organize_state["chunk_roles"],
        "organization_mode": organize_state["organization_layout"],
        "stats": {
            "initial_root_chunk_count": len(anchor_state["root_chunk_hits"]),
            "promoted_root_chunk_count": len(expand_state["promoted_root_chunks"]),
            "root_chunk_count": len(effective_root_chunk_hits),
            "primary_root_chunk_count": len(primary_root_chunks),
            "diversity_root_chunk_count": len(diversity_root_chunks),
            "initial_root_basin_count": initial_root_basin_count,
            "root_basin_count": root_basin_count,
            "anchor_root_top_k": anchor_state["root_top_k"],
            "expand_max_hop": anchor_state["max_hop"],
            "root_node_count": len(anchor_state["root_nodes"]),
            "root_edge_count": len(anchor_state["root_edges"]),
            "top_root_node_count": len(anchor_state["top_root_nodes"]),
            "top_root_edge_count": len(anchor_state["top_root_edges"]),
            "association_round_count": len(expansion["rounds"]),
            "structural_path_count": sum(item["structural_path_count"] for item in expansion["rounds"]),
            "structural_chunk_bridge_count": sum(item.get("structural_chunk_bridge_count", 0) for item in expansion["rounds"]),
            "structural_added_node_count": len(expansion["all_structural_nodes"]),
            "structural_added_edge_count": len(expansion["all_structural_edges"]),
            "semantic_added_node_count": len(expansion["all_semantic_nodes"]),
            "semantic_added_edge_count": len({tuple(item["edge"]) for item in expansion["all_semantic_edges"]}),
            "semantic_chunk_coverage_count": sum(item.get("semantic_chunk_coverage_count", 0) for item in expansion["rounds"]),
            "final_node_count": len(expansion["final_nodes"]),
            "final_edge_count": len(expansion["final_edges"]),
            "facet_group_count": len(organize_state["facet_groups"]),
            "knowledge_group_count": len(organize_state["facet_groups"]),
            "candidate_point_count": len(organize_state["candidate_points"]),
            "selected_source_chunk_count": prompt_payload["selected_source_chunk_count"],
            "selected_source_word_count": prompt_payload["selected_source_word_count"],
            "prompt_context_word_count": prompt_word_count,
            "prompt_context_token_estimate": prompt_token_count,
            "prompt_context_token_count": prompt_token_count,
            "prompt_context_char_count": len(prompt_context),
            "anchor_seconds": round(timings["anchor_seconds"], 4),
            "expand_seconds": round(timings["expand_seconds"], 4),
            "organize_seconds": round(timings["organize_seconds"], 4),
            "query_total_seconds": round(timings["query_total_seconds"], 4),
        },
        "top_root_nodes": anchor_state["top_root_nodes"],
        "top_root_edges": anchor_state["top_root_edges"],
        "rounds": expansion["rounds"],
        "structural_association": expansion["last_structural_output"],
        "semantic_association": {
            "selected_edges": expansion["all_semantic_edges"],
            "selected_nodes": [{"id": node_id} for node_id in expansion["all_semantic_nodes"]],
        },
        "candidate_points": organize_state["candidate_points"],
        "facet_groups": organize_state["facet_groups"],
        "knowledge_groups": organize_state["facet_groups"],
        "prompt_context": prompt_context,
    }


def run_query_online(
    *,
    query_row,
    graph,
    chunk_store,
    chunk_retriever,
    chunk_to_nodes,
    chunk_to_edges,
    node_to_chunks,
    edge_to_chunks,
    keyword_index,
    cfg,
    llm_client,
    query_index=None,
    total_queries=None,
):
    """Run one query end-to-end: retrieve -> answer.

    参数:
        query_row: 包含 query 和 group_id 的单条查询记录。
        graph: 实体关系图对象。
        chunk_store: chunk 存储字典。
        chunk_retriever: 用于根候选检索的检索器对象。
        chunk_to_nodes: chunk 到节点映射。
        chunk_to_edges: chunk 到边映射。
        node_to_chunks: 节点到 chunk 映射。
        edge_to_chunks: 边到 chunk 映射。
        keyword_index: 图关键词索引对象。
        cfg: 配置字典。
        llm_client: LLM 客户端实例。
        query_index: 可选的查询序号，用于日志记录。
        total_queries: 可选的总问题数，用于日志比例。

    返回:
        一个元组 (retrieval_record, answer_record)，分别为检索结果和生成答案结果。

    对单条问题执行线上检索和答案生成全流程。
    """
    query_wall_start = perf_counter()
    query_state = _run_query_states(
        query_row=query_row,
        graph=graph,
        chunk_store=chunk_store,
        chunk_retriever=chunk_retriever,
        chunk_to_nodes=chunk_to_nodes,
        chunk_to_edges=chunk_to_edges,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        keyword_index=keyword_index,
        cfg=cfg,
        query_index=query_index,
        total_queries=total_queries,
    )
    controller_info = query_state["controller_info"]
    anchor_state = query_state["anchor_state"]
    expand_state = query_state["expand_state"]
    organize_state = query_state["organize_state"]
    timings = dict(query_state["timings"])
    prefix = query_state["prefix"]

    retrieval_record = _build_query_record_from_states(
        query_row,
        controller_info,
        anchor_state,
        expand_state,
        organize_state,
        timings,
    )
    answer_start = perf_counter()
    answer_record = generate_one_answer_record(retrieval_record, llm_client)
    answer_elapsed = perf_counter() - answer_start
    query_wall_elapsed = perf_counter() - query_wall_start
    retrieval_record["stats"]["answer_seconds"] = round(answer_elapsed, 4)
    retrieval_record["stats"]["query_total_seconds"] = round(query_wall_elapsed, 4)
    answer_record["stats"] = retrieval_record["stats"]
    log(
        f"{prefix}answer chars={len(answer_record.get('model_answer', ''))} "
        f"answer_time={answer_elapsed:.2f}s total={query_wall_elapsed:.2f}s"
    )
    return retrieval_record, answer_record


def run_query(
    query_row,
    graph,
    chunk_store,
    chunk_retriever,
    chunk_to_nodes,
    chunk_to_edges,
    node_to_chunks,
    edge_to_chunks,
    keyword_index,
    cfg,
    query_index=None,
    total_queries=None,
):
    """Run the full retrieval pipeline for a single query.

    参数:
        query_row: 单条查询记录。
        graph: 实体关系图对象。
        chunk_store: chunk 存储字典。
        chunk_retriever: 根候选检索器对象。
        chunk_to_nodes: chunk 到节点映射。
        chunk_to_edges: chunk 到边映射。
        node_to_chunks: 节点到 chunk 映射。
        edge_to_chunks: 边到 chunk 映射。
        keyword_index: 图关键词索引对象。
        cfg: 配置字典。
        query_index: 可选 query 序号。
        total_queries: 可选问题总数。

    返回:
        检索记录 dict，包含组织好的 prompt_context 和统计信息。

    Flow:
    chunk retrieval -> diverse root selection -> grounded graph association ->
    query-aware organization -> final prompt context assembly
    """
    query_state = _run_query_states(
        query_row=query_row,
        graph=graph,
        chunk_store=chunk_store,
        chunk_retriever=chunk_retriever,
        chunk_to_nodes=chunk_to_nodes,
        chunk_to_edges=chunk_to_edges,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        keyword_index=keyword_index,
        cfg=cfg,
        query_index=query_index,
        total_queries=total_queries,
    )
    return _build_query_record_from_states(
        query_row,
        query_state["controller_info"],
        query_state["anchor_state"],
        query_state["expand_state"],
        query_state["organize_state"],
        query_state["timings"],
    )


def retrieve_corpus_queries(
    corpus_dir,
    rewrites_file=None,
    questions_file=None,
    limit_groups=None,
    output_dir="associative_rag_project/runs",
    **cfg,
):
    """Run retrieval for every question in a corpus and save one retrieval JSON.

    参数:
        corpus_dir: 语料库根目录。
        rewrites_file: 可选的重写组文件路径。
        questions_file: 可选的问题文件路径。
        limit_groups: 限制检索的 query 数量。
        output_dir: 保存结果的输出目录。
        **cfg: 其他运行配置，例如检索模式、group_limit 等。

    返回:
        (payload, retrieval_path)，其中 payload 是元数据汇总，retrieval_path 是检索输出文件路径。

    批量运行检索任务，并将结构化检索结果保存为 JSON 文件。
    """
    corpus_dir = Path(corpus_dir)
    runtime = _load_runtime_resources(corpus_dir, cfg)
    query_rows = load_query_rows(
        rewrites_file=Path(rewrites_file) if rewrites_file else None,
        questions_file=Path(questions_file) if questions_file else None,
        limit_groups=limit_groups,
    )
    log(f"[retrieve] queries={len(query_rows)} retrieval_mode={cfg['retrieval_mode']}")
    online_cfg = {**cfg, "chunk_neighbors": runtime["chunk_neighbors"]}
    results = [
        run_query(
            query_row=row,
            graph=runtime["graph"],
            chunk_store=runtime["chunk_store"],
            chunk_retriever=runtime["chunk_retriever"],
            chunk_to_nodes=runtime["chunk_to_nodes"],
            chunk_to_edges=runtime["chunk_to_edges"],
            node_to_chunks=runtime["node_to_chunks"],
            edge_to_chunks=runtime["edge_to_chunks"],
            keyword_index=runtime["keyword_index"],
            cfg=online_cfg,
            query_index=index,
            total_queries=len(query_rows),
        )
        for index, row in enumerate(query_rows, start=1)
    ]
    payload, paths = _write_retrieval_outputs(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        cfg=cfg,
        limit_groups=limit_groups,
        results=results,
    )
    return payload, paths["retrieval_path"]


def run_corpus_queries_online(
    corpus_dir,
    llm_client,
    rewrites_file=None,
    questions_file=None,
    limit_groups=None,
    output_dir="associative_rag_project/runs",
    answer_output_file=None,
    max_workers=4,
    **cfg,
):
    """Run the online per-query path: retrieve -> answer.

    参数:
        corpus_dir: 语料库根目录。
        llm_client: LLM 客户端实例。
        rewrites_file: 可选的重写组文件路径。
        questions_file: 可选的问题文件路径。
        limit_groups: 限制检索的 query 数量。
        output_dir: 保存检索输出的目录。
        answer_output_file: 可选的答案输出文件路径。
        max_workers: 并行 query 处理线程数。
        **cfg: 其他运行配置。

    返回:
        (payload, retrieval_path, answer_records, answer_output_path)。

    并行执行在线检索与生成流程，适用于端到端实验运行。
    """
    corpus_dir = Path(corpus_dir)
    runtime = _load_runtime_resources(corpus_dir, cfg)
    query_rows = load_query_rows(
        rewrites_file=Path(rewrites_file) if rewrites_file else None,
        questions_file=Path(questions_file) if questions_file else None,
        limit_groups=limit_groups,
    )
    log(
        f"[run-online] queries={len(query_rows)} retrieval_mode={cfg['retrieval_mode']} "
        f"workers={max_workers}"
    )
    online_cfg = {**cfg, "chunk_neighbors": runtime["chunk_neighbors"]}
    retrieval_records = [None] * len(query_rows)
    answer_records = [None] * len(query_rows)

    def _one(index_and_row):
        index, row = index_and_row
        return run_query_online(
            query_row=row,
            graph=runtime["graph"],
            chunk_store=runtime["chunk_store"],
            chunk_retriever=runtime["chunk_retriever"],
            chunk_to_nodes=runtime["chunk_to_nodes"],
            chunk_to_edges=runtime["chunk_to_edges"],
            node_to_chunks=runtime["node_to_chunks"],
            edge_to_chunks=runtime["edge_to_chunks"],
            keyword_index=runtime["keyword_index"],
            cfg=online_cfg,
            llm_client=llm_client,
            query_index=index + 1,
            total_queries=len(query_rows),
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_one, (index, row)): index
            for index, row in enumerate(query_rows)
        }
        for future in as_completed(future_map):
            index = future_map[future]
            retrieval_record, answer_record = future.result()
            retrieval_records[index] = retrieval_record
            answer_records[index] = answer_record

    payload, paths = _write_retrieval_outputs(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        cfg=cfg,
        limit_groups=limit_groups,
        results=retrieval_records,
    )
    answer_output_path = Path(answer_output_file) if answer_output_file else paths["answer_path"]
    answer_output_path.write_text(json.dumps(answer_records, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[run-online] wrote {answer_output_path}")
    return payload, paths["retrieval_path"], answer_records, answer_output_path
