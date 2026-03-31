"""End-to-end retrieval pipeline for associative QFS."""

import json
from collections import defaultdict
from pathlib import Path

from .adaptive_control import build_adaptive_controller
from .association import (
    build_edge_role_sets,
    build_node_role_sets,
    expand_associative_graph,
)
from .context import build_knowledge_groups, build_prompt_context
from .data import build_chunk_mappings, load_graph_corpus, load_query_rows
from .embedding_client import OpenAICompatibleEmbeddingClient
from .logging_utils import log, shorten
from .retrieval import BM25Index, DenseChunkIndex, HybridChunkRetriever, rerank_root_chunks, score_root_edges, score_root_nodes


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
            f"{item['chunk_id']} score={item['score_norm']:.3f} qrel={item.get('query_rel', 0.0):.3f} :: {preview}"
        )
    return previews


def _format_round_preview(round_info):
    lines = [
        f"round={round_info['round']} structural_paths={round_info['structural_path_count']} "
        f"+nodes={round_info['structural_added_node_count']} +edges={round_info['structural_added_edge_count']} "
        f"semantic +nodes={round_info['semantic_added_node_count']} +edges={round_info['semantic_added_edge_count']} "
        f"current=({round_info['current_node_count']}n/{round_info['current_edge_count']}e)"
    ]
    for path_item in round_info.get("structural_path_preview", []):
        lines.append(
            f"  structural: score={path_item['score']:.3f} span={path_item['support_span']:.2f} "
            f"path={' -> '.join(path_item['path'])}"
        )
    for edge_item in round_info.get("semantic_edge_preview", []):
        lines.append(
            f"  semantic-edge: score={edge_item['score']:.3f} rel={edge_item['query_rel']:.3f} "
            f"{edge_item['edge'][0]} -> {edge_item['edge'][1]} [{edge_item['category']}]"
        )
    for node_item in round_info.get("semantic_node_preview", []):
        lines.append(
            f"  semantic-node: score={node_item['score']:.3f} bridge={node_item['bridge_strength']:.2f} {node_item['id']}"
        )
    return lines


def _format_adaptive_preview(adaptive_profile):
    intent = adaptive_profile["intent"]
    graph = adaptive_profile["graph_features"]
    candidate = adaptive_profile["candidate_features"]
    retrieval = adaptive_profile["retrieval_features"]
    components = adaptive_profile.get("alpha_components", {})
    budgets = adaptive_profile["adapted_budgets"]
    return (
        f"adaptive style={adaptive_profile['query_style']} alpha={adaptive_profile['association_strength']:.3f} "
        f"overview_hits={intent['overview_hits']} focused_hits={intent['focused_hits']} "
        f"density={graph['root_density']:.3f} fragmentation={graph['fragmentation']:.3f} "
        f"density_signal={components.get('density_signal', 0.0):.3f} "
        f"frag_signal={components.get('fragmentation_signal', 0.0):.3f} "
        f"style_prior={components.get('style_prior', 0.0):.3f} "
        f"candidate_dispersion={candidate['candidate_chunk_dispersion']:.3f} "
        f"effective_candidates={candidate['effective_candidate_chunk_count']:.2f} "
        f"retrieval_cliff@{retrieval['retrieval_cliff_topn']}={retrieval['retrieval_cliff']:.3f} "
        f"cliff_brake={components.get('cliff_brake', 0.0):.3f} "
        f"budgets(rounds={budgets['association_rounds']}, path={budgets['path_budget']}, "
        f"sem_edges={budgets['semantic_edge_budget']}, sem_nodes={budgets['semantic_node_budget']}, "
        f"group_limit={budgets['group_limit']}, sources={budgets['max_source_chunks']}, "
        f"words={budgets['max_source_word_budget']})"
    )


def run_query(
    query_row,
    graph,
    chunk_store,
    chunk_retriever,
    chunk_to_nodes,
    chunk_to_edges,
    node_to_chunks,
    edge_to_chunks,
    cfg,
    query_index=None,
    total_queries=None,
):
    """Run the full retrieval pipeline for a single query.

    Flow:
    dense/bm25 retrieval -> root rerank -> adaptive budget selection ->
    alternating graph association -> knowledge group construction ->
    final prompt context assembly
    """
    prefix = f"[query {query_index}/{total_queries}] " if query_index is not None and total_queries is not None else ""
    log(f"{prefix}start {query_row['group_id']} :: {query_row['query']}")
    # Keep a wider pre-rerank candidate pool so retrieval-cliff estimation can
    # see whether the query really has a steep score drop or a broad frontier.
    candidate_top_k = max(
        cfg["top_chunks"],
        cfg["top_chunks"] * max(cfg["chunk_candidate_multiplier"], 1),
        cfg.get("adaptive_candidate_pool_size", 30),
    )
    candidate_chunk_hits = chunk_retriever.search(query_row["query"], top_k=candidate_top_k)
    log(f"{prefix}retrieval candidates={len(candidate_chunk_hits)} mode={cfg['retrieval_mode']}")
    root_chunk_hits = rerank_root_chunks(
        query=query_row["query"],
        candidate_hits=candidate_chunk_hits,
        chunk_store=chunk_store,
        chunk_to_nodes=chunk_to_nodes,
        chunk_to_edges=chunk_to_edges,
        top_k=cfg["top_chunks"],
    )
    for line in _format_root_chunk_preview(root_chunk_hits, chunk_store):
        log(f"{prefix}root {line}")
    # Root chunk scores are reused later as the project's main notion of
    # evidence grounding when ranking nodes, edges, and semantic expansions.
    root_chunk_score_lookup = {item["chunk_id"]: item["score_norm"] for item in root_chunk_hits}
    root_chunk_ids = list(root_chunk_score_lookup)
    root_nodes = set()
    root_edges = set()
    for chunk_id in root_chunk_ids:
        root_nodes.update(chunk_to_nodes.get(chunk_id, set()))
        root_edges.update(chunk_to_edges.get(chunk_id, set()))
    # Always compute adaptive signals so later analysis can compare priors
    # against outcomes, even when the actual budgets are fixed.
    adaptive_profile = build_adaptive_controller(
        query=query_row["query"],
        root_nodes=root_nodes,
        root_edges=root_edges,
        root_chunk_hits=root_chunk_hits,
        candidate_chunk_hits=candidate_chunk_hits,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        base_cfg=cfg,
    )
    # Keep the adaptive controller optional so fixed-budget ablations remain easy.
    if not cfg.get("adaptive_control", True):
        adaptive_profile = {
            **adaptive_profile,
            "adapted_budgets": {
                "association_rounds": cfg["association_rounds"],
                "path_budget": cfg["path_budget"],
                "semantic_edge_budget": cfg["semantic_edge_budget"],
                "semantic_node_budget": cfg["semantic_node_budget"],
                "semantic_edge_min_score": cfg["semantic_edge_min_score"],
                "semantic_node_min_score": cfg["semantic_node_min_score"],
                "group_limit": cfg["group_limit"],
                "max_source_chunks": cfg["max_source_chunks"],
                "max_source_word_budget": cfg["max_source_word_budget"],
            },
        }
    adapted = adaptive_profile["adapted_budgets"]
    log(f"{prefix}{_format_adaptive_preview(adaptive_profile)}")

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
        graph=graph,
        root_nodes=root_nodes,
        root_edges=root_edges,
        root_chunk_ids=root_chunk_ids,
        root_chunk_score_lookup=root_chunk_score_lookup,
        chunk_to_nodes=chunk_to_nodes,
        chunk_to_edges=chunk_to_edges,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        top_root_nodes=cfg["top_root_nodes"],
        top_root_edges=cfg["top_root_edges"],
        max_hop=cfg["max_hop"],
        path_budget=adapted["path_budget"],
        semantic_edge_budget=adapted["semantic_edge_budget"],
        semantic_node_budget=adapted["semantic_node_budget"],
        association_rounds=adapted["association_rounds"],
        semantic_edge_min_score=adapted["semantic_edge_min_score"],
        semantic_node_min_score=adapted["semantic_node_min_score"],
    )
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
    knowledge_groups = build_knowledge_groups(
        query=query_row["query"],
        graph=graph,
        final_nodes=expansion["final_nodes"],
        final_edges=expansion["final_edges"],
        node_roles=node_roles,
        edge_roles=edge_roles,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        chunk_store=chunk_store,
        group_limit=adapted["group_limit"],
    )
    prompt_payload = build_prompt_context(
        query_row=query_row,
        root_chunk_hits=root_chunk_hits,
        top_root_nodes=top_root_nodes,
        top_root_edges=top_root_edges,
        node_roles=node_roles,
        edge_roles=edge_roles,
        semantic_nodes=[{"id": node_id} for node_id in expansion["all_semantic_nodes"]],
        semantic_edges=expansion["all_semantic_edges"],
        final_nodes=expansion["final_nodes"],
        final_edges=expansion["final_edges"],
        knowledge_groups=knowledge_groups,
        chunk_store=chunk_store,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        max_source_chunks=adapted["max_source_chunks"],
        max_source_word_budget=adapted["max_source_word_budget"],
    )
    log(
        f"{prefix}done roots=({len(root_nodes)}n/{len(root_edges)}e) "
        f"final=({len(expansion['final_nodes'])}n/{len(expansion['final_edges'])}e) "
        f"groups={len(knowledge_groups)} sources={prompt_payload['selected_source_chunk_count']} "
        f"words={prompt_payload['selected_source_word_count']}"
    )
    return {
        **query_row,
        "candidate_root_chunks": candidate_chunk_hits,
        "root_chunks": root_chunk_hits,
        "query_style": adaptive_profile["query_style"],
        "adaptive_profile": adaptive_profile,
        "stats": {
            "root_chunk_count": len(root_chunk_hits),
            "root_node_count": len(root_nodes),
            "root_edge_count": len(root_edges),
            "top_root_node_count": len(top_root_nodes),
            "top_root_edge_count": len(top_root_edges),
            "association_round_count": len(expansion["rounds"]),
            "structural_path_count": sum(item["structural_path_count"] for item in expansion["rounds"]),
            "structural_added_node_count": len(expansion["all_structural_nodes"]),
            "structural_added_edge_count": len(expansion["all_structural_edges"]),
            "semantic_added_node_count": len(expansion["all_semantic_nodes"]),
            "semantic_added_edge_count": len({item["edge"] for item in expansion["all_semantic_edges"]}),
            "final_node_count": len(expansion["final_nodes"]),
            "final_edge_count": len(expansion["final_edges"]),
            "knowledge_group_count": len(knowledge_groups),
            "selected_source_chunk_count": prompt_payload["selected_source_chunk_count"],
            "selected_source_word_count": prompt_payload["selected_source_word_count"],
            "association_strength": adaptive_profile["association_strength"],
            "root_density": adaptive_profile["graph_features"]["root_density"],
            "root_fragmentation": adaptive_profile["graph_features"]["fragmentation"],
            "candidate_chunk_dispersion": adaptive_profile["candidate_features"]["candidate_chunk_dispersion"],
            "candidate_chunk_concentration": adaptive_profile["candidate_features"]["candidate_chunk_concentration"],
            "effective_candidate_chunk_count": adaptive_profile["candidate_features"]["effective_candidate_chunk_count"],
            "retrieval_cliff": adaptive_profile["retrieval_features"]["retrieval_cliff"],
            "retrieval_cliff_topn": adaptive_profile["retrieval_features"]["retrieval_cliff_topn"],
        },
        "top_root_nodes": top_root_nodes,
        "top_root_edges": top_root_edges,
        "rounds": expansion["rounds"],
        "structural_association": expansion["last_structural_output"],
        "semantic_association": {
            "selected_edges": expansion["all_semantic_edges"],
            "selected_nodes": [{"id": node_id} for node_id in expansion["all_semantic_nodes"]],
        },
        "knowledge_groups": knowledge_groups,
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
    log(f"[retrieve] loading corpus={corpus_dir}")
    graph, chunk_store = load_graph_corpus(corpus_dir)
    log(f"[retrieve] graph nodes={graph.number_of_nodes()} edges={graph.number_of_edges()} chunks={len(chunk_store)}")
    bm25_index = BM25Index.build(chunk_store)
    dense_index = DenseChunkIndex.load(corpus_dir / "vdb_chunks.json") if (corpus_dir / "vdb_chunks.json").exists() else None
    embedding_client = None
    if cfg["retrieval_mode"] in {"dense", "hybrid"}:
        embedding_client = OpenAICompatibleEmbeddingClient()
        log(f"[retrieve] dense index loaded rows={len(dense_index.chunk_ids) if dense_index else 0}")
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
    query_rows = load_query_rows(
        rewrites_file=Path(rewrites_file) if rewrites_file else None,
        questions_file=Path(questions_file) if questions_file else None,
        limit_groups=limit_groups,
    )
    log(f"[retrieve] queries={len(query_rows)} retrieval_mode={cfg['retrieval_mode']} adaptive={cfg.get('adaptive_control', True)}")
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
            cfg=cfg,
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
    stem = f"{corpus_dir.name}_top{cfg['top_chunks']}_hop{cfg['max_hop']}_assoc_project"
    output_path = output_dir / f"{stem}_retrieval.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if results:
        (output_dir / f"{stem}_sample_context.txt").write_text(results[0]["prompt_context"], encoding="utf-8")
    log(f"[retrieve] wrote {output_path}")
    return payload, output_path
