"""End-to-end retrieval pipeline for associative QFS."""

import json
from collections import defaultdict
from pathlib import Path

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
from .organization import (
    build_answer_facet_groups,
    build_candidate_points_from_groups,
    detect_query_contract,
)
from .retrieval import (
    BM25Index,
    DenseChunkIndex,
    HybridChunkRetriever,
    build_graph_keyword_index,
    merge_candidate_hits_with_graph,
    search_graph_focus_chunks,
    search_graph_keyword_chunks,
    score_root_edges,
    score_root_nodes,
    select_diverse_root_chunks,
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
            "peripheral": "peripheral-chunk",
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
            f"  structural-chunk: bridge={float(chunk_item.get('bridge_gain', chunk_item.get('selection_score', 0.0))):.3f} "
            f"touch={chunk_item.get('frontier_touch', 0)} "
            f"new_sources={chunk_item.get('new_source_count', 0)} "
            f"qrel={chunk_item.get('introduced_query_rel', 0.0):.3f} {chunk_item['chunk_id']}"
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
            f"  coverage-chunk: nodes={chunk_item.get('new_node_count', len(chunk_item.get('node_ids', [])))} "
            f"edges={chunk_item.get('new_edge_count', len(chunk_item.get('edge_ids', [])))} "
            f"new_sources={chunk_item.get('new_source_count', 0)} {chunk_item['chunk_id']}"
        )
    return lines


def _contract_candidate_top_k(cfg, query_contract):
    base = max(
        cfg["top_chunks"],
        cfg["top_chunks"] * max(cfg["chunk_candidate_multiplier"], 1),
        cfg.get("candidate_pool_size", 30),
    )
    if query_contract == "theme-grounded":
        return max(base * 3, cfg.get("candidate_pool_size", 30) * 2, 80)
    if query_contract == "comparison-grounded":
        return max(base, cfg.get("candidate_pool_size", 30) + cfg["top_chunks"] * 4)
    return base


def _contract_root_top_k(cfg, query_contract):
    if query_contract == "theme-grounded":
        return cfg["top_chunks"] + 3
    if query_contract == "comparison-grounded":
        return cfg["top_chunks"] + 1
    return cfg["top_chunks"]


def _contract_max_hop(cfg, query_contract):
    if query_contract == "theme-grounded":
        return max(2, cfg["max_hop"] - 1)
    return cfg["max_hop"]


def _contract_graph_merge_weights(query_contract):
    if query_contract == "section-grounded":
        return 0.22, 0.12
    if query_contract == "mechanism-grounded":
        return 0.28, 0.16
    if query_contract == "comparison-grounded":
        return 0.30, 0.18
    return 0.32, 0.18


def _pick_section_doc_ids(root_chunk_hits, chunk_store):
    doc_ids = [
        chunk_store.get(item["chunk_id"], {}).get("full_doc_id")
        for item in root_chunk_hits
        if chunk_store.get(item["chunk_id"], {}).get("full_doc_id")
    ]
    if not doc_ids:
        return None
    return {doc_ids[0]}


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

    Flow:
    chunk retrieval -> diverse root selection -> grounded graph association ->
    query-aware organization -> final prompt context assembly
    """
    prefix = f"[query {query_index}/{total_queries}] " if query_index is not None and total_queries is not None else ""
    log(f"{prefix}start {query_row['group_id']} :: {query_row['query']}")
    query_contract = detect_query_contract(query_row["query"])
    root_top_k = _contract_root_top_k(cfg, query_contract)
    max_hop = _contract_max_hop(cfg, query_contract)
    # Keep a wider pre-rerank candidate pool so retrieval-cliff estimation can
    # see whether the query really has a steep score drop or a broad frontier.
    candidate_top_k = _contract_candidate_top_k(cfg, query_contract)
    retrieval_mode_label = "dense+graph_focus"
    graph_focus_weight, graph_keyword_weight = _contract_graph_merge_weights(query_contract)
    candidate_chunk_hits = chunk_retriever.search(query_row["query"], top_k=candidate_top_k)
    graph_focus_hits = search_graph_focus_chunks(
        query=query_row["query"],
        graph=graph,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        top_chunk_k=max(candidate_top_k * 2, cfg["top_chunks"] * 10),
    )
    candidate_chunk_hits = merge_candidate_hits_with_graph(
        primary_hits=candidate_chunk_hits,
        graph_hits=graph_focus_hits,
        graph_weight=graph_focus_weight,
    )[:candidate_top_k]
    graph_keyword_hits = search_graph_keyword_chunks(
        query=query_row["query"],
        graph=graph,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        top_chunk_k=max(candidate_top_k * 2, cfg["top_chunks"] * 10),
    )
    candidate_chunk_hits = merge_candidate_hits_with_graph(
        primary_hits=candidate_chunk_hits,
        graph_hits=graph_keyword_hits,
        graph_weight=graph_keyword_weight,
    )[:candidate_top_k]
    log(f"{prefix}retrieval candidates={len(candidate_chunk_hits)} mode={retrieval_mode_label}")
    query_chunk_score_lookup = {
        item["chunk_id"]: item.get(
            "graph_focus_score_norm",
            item.get(
                "graph_keyword_score_norm",
                item.get("dense_score_norm", item.get("retrieval_score", item.get("score_norm", 0.0))),
            ),
        )
        for item in candidate_chunk_hits
    }
    root_chunk_hits = select_diverse_root_chunks(
        query=query_row["query"],
        candidate_hits=candidate_chunk_hits,
        chunk_store=chunk_store,
        chunk_to_nodes=chunk_to_nodes,
        chunk_to_edges=chunk_to_edges,
        top_k=root_top_k,
        query_contract=query_contract,
        graph=graph,
        keyword_index=keyword_index,
    )
    allowed_doc_ids = None
    if query_contract == "section-grounded":
        allowed_doc_ids = _pick_section_doc_ids(root_chunk_hits, chunk_store)
        if allowed_doc_ids:
            section_candidate_hits = [
                item
                for item in candidate_chunk_hits
                if chunk_store.get(item["chunk_id"], {}).get("full_doc_id") in allowed_doc_ids
            ]
            section_root_hits = select_diverse_root_chunks(
                query=query_row["query"],
                candidate_hits=section_candidate_hits,
                chunk_store=chunk_store,
                chunk_to_nodes=chunk_to_nodes,
                chunk_to_edges=chunk_to_edges,
                top_k=root_top_k,
                query_contract=query_contract,
                graph=graph,
                keyword_index=keyword_index,
            )
            if section_root_hits:
                root_chunk_hits = section_root_hits
    root_chunk_hits = [_serialize_root_chunk_hit(item) for item in root_chunk_hits]
    for line in _format_root_chunk_preview(root_chunk_hits, chunk_store):
        log(f"{prefix}root {line}")
    # Root chunk scores are reused later as the project's main notion of
    # evidence grounding when ranking nodes, edges, and semantic expansions.
    root_chunk_score_lookup = {item["chunk_id"]: item["score_norm"] for item in root_chunk_hits}
    root_chunk_ids = list(root_chunk_score_lookup)
    if query_contract == "section-grounded" and not allowed_doc_ids:
        allowed_doc_ids = _pick_section_doc_ids(root_chunk_hits, chunk_store)
    root_nodes = set()
    root_edges = set()
    for chunk_id in root_chunk_ids:
        root_nodes.update(chunk_to_nodes.get(chunk_id, set()))
        root_edges.update(chunk_to_edges.get(chunk_id, set()))
    budgets = {
        "association_rounds": cfg["association_rounds"],
        "path_budget": cfg["path_budget"],
        "semantic_edge_budget": cfg["semantic_edge_budget"],
        "semantic_node_budget": cfg["semantic_node_budget"],
        "semantic_edge_min_score": cfg["semantic_edge_min_score"],
        "semantic_node_min_score": cfg["semantic_node_min_score"],
        "group_limit": cfg["group_limit"],
        "max_source_chunks": cfg["max_source_chunks"],
        "max_source_word_budget": cfg["max_source_word_budget"],
    }

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

    expansion = expand_associative_graph(
        query=query_row["query"],
        query_contract=query_contract,
        graph=graph,
        root_nodes=root_nodes,
        root_edges=root_edges,
        root_chunk_ids=root_chunk_ids,
        root_chunk_score_lookup=root_chunk_score_lookup,
        query_chunk_score_lookup=query_chunk_score_lookup,
        chunk_to_nodes=chunk_to_nodes,
        chunk_to_edges=chunk_to_edges,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        chunk_neighbors=cfg["chunk_neighbors"],
        chunk_store=chunk_store,
        top_root_nodes=cfg["top_root_nodes"],
        top_root_edges=cfg["top_root_edges"],
        max_hop=max_hop,
        path_budget=budgets["path_budget"],
        semantic_edge_budget=budgets["semantic_edge_budget"],
        semantic_node_budget=budgets["semantic_node_budget"],
        association_rounds=budgets["association_rounds"],
        semantic_edge_min_score=budgets["semantic_edge_min_score"],
        semantic_node_min_score=budgets["semantic_node_min_score"],
        allowed_doc_ids=allowed_doc_ids,
    )
    effective_root_chunk_ids = expansion.get("effective_root_chunk_ids", root_chunk_ids)
    promoted_root_chunks = [_serialize_root_chunk_hit(item) for item in expansion.get("promoted_root_chunks", [])]
    effective_root_chunk_hits = root_chunk_hits + promoted_root_chunks
    for round_info in expansion["rounds"]:
        for line in _format_round_preview(round_info):
            log(f"{prefix}{line}")

    node_roles = build_node_role_sets(
        root_nodes=root_nodes,
        structural_nodes=set(expansion["all_structural_nodes"]),
        semantic_nodes=set(expansion["all_semantic_nodes"]),
    )
    edge_roles = build_edge_role_sets(
        root_edges=root_edges,
        structural_edges=set(expansion["all_structural_edges"]),
        semantic_edges=expansion["all_semantic_edges"],
    )
    facet_groups, query_contract = build_answer_facet_groups(
        query=query_row["query"],
        graph=graph,
        final_nodes=expansion["final_nodes"],
        final_edges=expansion["final_edges"],
        root_chunk_ids=effective_root_chunk_ids,
        last_structural_output=expansion["last_structural_output"],
        chunk_neighbors=cfg["chunk_neighbors"],
        chunk_to_nodes=chunk_to_nodes,
        chunk_to_edges=chunk_to_edges,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        chunk_store=chunk_store,
        group_limit=budgets["group_limit"],
        query_contract=query_contract,
        allowed_doc_ids=allowed_doc_ids,
    )
    candidate_points = build_candidate_points_from_groups(facet_groups, top_k=8)
    theme_selected_chunks = expansion.get("theme_selected_chunks", {})
    chunk_roles = _build_chunk_roles(
        effective_root_chunk_hits,
        promoted_root_chunks,
        facet_groups,
        theme_selected_chunks=theme_selected_chunks,
    )
    prompt_payload = build_prompt_context(
        query_row=query_row,
        query_contract=query_contract,
        root_chunk_hits=effective_root_chunk_hits,
        top_root_nodes=top_root_nodes,
        top_root_edges=top_root_edges,
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
        max_source_chunks=budgets["max_source_chunks"],
        max_source_word_budget=budgets["max_source_word_budget"],
    )
    log(
        f"{prefix}done roots=({len(root_nodes)}n/{len(root_edges)}e) "
        f"final=({len(expansion['final_nodes'])}n/{len(expansion['final_edges'])}e) "
        f"groups={len(facet_groups)} sources={prompt_payload['selected_source_chunk_count']} "
        f"words={prompt_payload['selected_source_word_count']}"
    )
    primary_root_chunks = [item for item in effective_root_chunk_hits if item.get("root_role") == "primary"]
    diversity_root_chunks = [item for item in effective_root_chunk_hits if item.get("root_role") == "diversity"]
    root_basin_count = len({tuple(item.get("basin_key", (item["chunk_id"],))) for item in effective_root_chunk_hits})
    initial_root_basin_count = len({tuple(item.get("basin_key", (item["chunk_id"],))) for item in root_chunk_hits})
    return {
        **query_row,
        "candidate_root_chunks": candidate_chunk_hits,
        "root_chunks": effective_root_chunk_hits,
        "primary_root_chunks": primary_root_chunks,
        "diversity_root_chunks": diversity_root_chunks,
        "promoted_root_chunks": promoted_root_chunks,
        "theme_selected_chunks": theme_selected_chunks,
        "chunk_roles": chunk_roles,
        "query_contract": query_contract,
        "stats": {
            "initial_root_chunk_count": len(root_chunk_hits),
            "promoted_root_chunk_count": len(promoted_root_chunks),
            "root_chunk_count": len(effective_root_chunk_hits),
            "primary_root_chunk_count": len(primary_root_chunks),
            "diversity_root_chunk_count": len(diversity_root_chunks),
            "initial_root_basin_count": initial_root_basin_count,
            "root_basin_count": root_basin_count,
            "contract_root_top_k": root_top_k,
            "contract_max_hop": max_hop,
            "root_node_count": len(root_nodes),
            "root_edge_count": len(root_edges),
            "top_root_node_count": len(top_root_nodes),
            "top_root_edge_count": len(top_root_edges),
            "association_round_count": len(expansion["rounds"]),
            "structural_path_count": sum(item["structural_path_count"] for item in expansion["rounds"]),
            "structural_chunk_bridge_count": sum(
                item.get("structural_chunk_bridge_count", 0) for item in expansion["rounds"]
            ),
            "structural_added_node_count": len(expansion["all_structural_nodes"]),
            "structural_added_edge_count": len(expansion["all_structural_edges"]),
            "semantic_added_node_count": len(expansion["all_semantic_nodes"]),
            "semantic_added_edge_count": len({item["edge"] for item in expansion["all_semantic_edges"]}),
            "semantic_chunk_coverage_count": sum(
                item.get("semantic_chunk_coverage_count", 0) for item in expansion["rounds"]
            ),
            "final_node_count": len(expansion["final_nodes"]),
            "final_edge_count": len(expansion["final_edges"]),
            "facet_group_count": len(facet_groups),
            "knowledge_group_count": len(facet_groups),
            "candidate_point_count": len(candidate_points),
            "selected_source_chunk_count": prompt_payload["selected_source_chunk_count"],
            "selected_source_word_count": prompt_payload["selected_source_word_count"],
        },
        "top_root_nodes": top_root_nodes,
        "top_root_edges": top_root_edges,
        "rounds": expansion["rounds"],
        "structural_association": expansion["last_structural_output"],
        "semantic_association": {
            "selected_edges": expansion["all_semantic_edges"],
            "selected_nodes": [{"id": node_id} for node_id in expansion["all_semantic_nodes"]],
        },
        "candidate_points": candidate_points,
        "facet_groups": facet_groups,
        "knowledge_groups": facet_groups,
        "prompt_context": prompt_payload["context"],
    }


def retrieve_corpus_queries(
    corpus_dir,
    rewrites_file=None,
    questions_file=None,
    limit_groups=None,
    output_dir="associative_rag_project/runs",
    **cfg,
):
    """Run retrieval for every question in a corpus and save one retrieval JSON."""
    corpus_dir = Path(corpus_dir)
    index_dir = resolve_corpus_index_dir(corpus_dir)
    log(f"[retrieve] loading corpus={corpus_dir} index={index_dir}")
    graph, chunk_store, index_dir = load_graph_corpus(corpus_dir)
    log(f"[retrieve] graph nodes={graph.number_of_nodes()} edges={graph.number_of_edges()} chunks={len(chunk_store)}")
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
            "[retrieve] dense index loaded "
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
        "[retrieve] keyword-index "
        f"vocab={len(keyword_index.vocabulary)} ignored={len(keyword_index.ignored_terms)} "
        f"df_threshold={keyword_index.ignored_threshold}"
    )
    query_rows = load_query_rows(
        rewrites_file=Path(rewrites_file) if rewrites_file else None,
        questions_file=Path(questions_file) if questions_file else None,
        limit_groups=limit_groups,
    )
    log(f"[retrieve] queries={len(query_rows)} retrieval_mode={cfg['retrieval_mode']}")
    results = [
        run_query(
            query_row=row,
            graph=graph,
            chunk_store=chunk_store,
            chunk_retriever=chunk_retriever,
            chunk_to_nodes=chunk_to_nodes,
            chunk_to_edges=chunk_to_edges,
            node_to_chunks=node_to_chunks,
            edge_to_chunks=edge_to_chunks,
            keyword_index=keyword_index,
            cfg={**cfg, "chunk_neighbors": chunk_neighbors},
            query_index=index,
            total_queries=len(query_rows),
        )
        for index, row in enumerate(query_rows, start=1)
    ]
    payload = {
        "corpus_dir": str(corpus_dir),
        "config": cfg,
        "summary": summarize_results(results),
        "results": results,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_name = infer_corpus_name(corpus_dir)
    limit_suffix = f"_limit{limit_groups}" if limit_groups is not None else ""
    stem = f"{corpus_name}_top{cfg['top_chunks']}_hop{cfg['max_hop']}_assoc_project{limit_suffix}"
    output_path = output_dir / f"{stem}_retrieval.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if results:
        (output_dir / f"{stem}_sample_context.txt").write_text(results[0]["prompt_context"], encoding="utf-8")
    log(f"[retrieve] wrote {output_path}")
    return payload, output_path
