"""CLI entrypoint for retrieval, answer generation, judging, and full runs.

提供一套命令行接口，用于检索、生成、评判和端到端实验运行。
"""

import argparse
import json
from time import perf_counter
from pathlib import Path

from .config import load_judge_config, load_llm_config
from .data import (
    extract_questions,
    infer_corpus_name,
    load_baseline_answers,
    resolve_baseline_file,
    resolve_questions_file,
)
from .judge import load_judge_corpus_resources, render_winrate_markdown_table, run_winrate_judgement
from .llm_client import OpenAICompatibleClient, generate_answers
from .logging_utils import log
from .pipeline import retrieve_corpus_queries, run_corpus_queries_online


def build_parser():
    """Expose the project as a small experiment-runner CLI."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--corpus-dir", required=True)
    common.add_argument("--questions-file")
    common.add_argument("--rewrites-file")
    common.add_argument("--limit-groups", "--limit", dest="limit_groups", type=int)
    common.add_argument("--output-dir", default="associative_rag_project/runs")
    common.add_argument("--top-chunks", type=int, default=6)
    common.add_argument("--chunk-candidate-multiplier", type=int, default=3)
    common.add_argument("--candidate-pool-size", type=int, default=30)
    common.add_argument("--retrieval-mode", choices=["bm25", "dense", "hybrid"], default="dense")
    common.add_argument(
        "--retrieval-strategy",
        choices=["association", "evidence_trace"],
        default="evidence_trace",
        help="association runs the legacy bridge/support pipeline; evidence_trace runs the deep evidence tracing prototype.",
    )
    common.add_argument("--dense-weight", type=float, default=0.75)
    common.add_argument("--bm25-weight", type=float, default=0.25)
    common.add_argument(
        "--embedding-provider",
        choices=["openai_compatible", "bge_m3_server"],
        help="Dense query embedding backend. Defaults to config/env value.",
    )
    common.add_argument("--embedding-model", help="OpenAI-compatible embedding model override.")
    common.add_argument("--embedding-base-url", help="OpenAI-compatible embedding base URL override.")
    common.add_argument("--embedding-api-key", help="OpenAI-compatible embedding API key override.")
    common.add_argument("--top-root-nodes", type=int, default=12)
    common.add_argument("--top-root-edges", type=int, default=16)
    common.add_argument("--max-hop", type=int, default=4)
    common.add_argument("--path-budget", type=int, default=12)
    common.add_argument("--semantic-edge-budget", type=int, default=20)
    common.add_argument("--semantic-node-budget", type=int, default=12)
    common.add_argument("--semantic-edge-min-score", type=float, default=0.03)
    common.add_argument("--semantic-node-min-score", type=float, default=0.03)
    common.add_argument("--association-rounds", type=int, default=2)
    common.add_argument(
        "--frontier-edge-top-k",
        type=int,
        default=20,
        help="For evidence_trace, prune each trace frontier to the top shared-node links.",
    )
    common.add_argument("--group-limit", type=int, default=8)
    common.add_argument("--max-source-chunks", type=int, default=18)
    common.add_argument("--max-source-word-budget", type=int, default=10000)
    common.add_argument("--max-workers", type=int, default=12)
    common.add_argument(
        "--task-mode",
        choices=["qfs", "multihop_qa"],
        default="qfs",
        help="Answer task style. Use multihop_qa for short benchmark QA instead of QFS P1-P5.",
    )
    common.add_argument(
        "--context-constraint",
        choices=["none", "sample_context"],
        default="none",
        help="Use sample_context to restrict each query to its original benchmark contexts.",
    )

    subparsers.add_parser("retrieve", parents=[common])

    answer = subparsers.add_parser("answer")
    answer.add_argument("--retrieval-file", required=True)
    answer.add_argument("--output-file")
    answer.add_argument("--max-workers", type=int)

    judge = subparsers.add_parser("judge")
    judge.add_argument("--corpus-dir")
    judge.add_argument("--questions-file", required=True)
    judge.add_argument("--candidate-file", required=True)
    judge.add_argument("--baseline-file")
    judge.add_argument("--baseline-dir")
    judge.add_argument("--output-file")
    judge.add_argument("--summary-file")
    judge.add_argument("--candidate-label")
    judge.add_argument("--limit", type=int)
    judge.add_argument("--max-workers", type=int, default=12)
    judge.add_argument(
        "--judge-mode",
        choices=["quality", "source_compliance", "claim_diagnostics"],
        default="quality",
        help=(
            "quality restores the original QFS judge; source_compliance judges source-bounded contract adherence; "
            "claim_diagnostics runs the older claim-level corpus support analysis."
        ),
    )

    run = subparsers.add_parser("run", parents=[common])
    run.add_argument("--answer-output-file")

    run_all = subparsers.add_parser("run-all", parents=[common])
    run_all.add_argument("--baseline-file")
    run_all.add_argument("--answer-output-file")
    run_all.add_argument("--judge-output-file")
    run_all.add_argument(
        "--judge-mode",
        choices=["quality", "source_compliance", "claim_diagnostics"],
        default="quality",
        help="Judge mode used after run-all answer generation.",
    )
    return parser


def retrieval_config_from_args(args):
    """Translate CLI args into the retrieval config consumed by the pipeline."""
    return {
        "top_chunks": args.top_chunks,
        "chunk_candidate_multiplier": args.chunk_candidate_multiplier,
        "candidate_pool_size": args.candidate_pool_size,
        "retrieval_mode": args.retrieval_mode,
        "retrieval_strategy": args.retrieval_strategy,
        "dense_weight": args.dense_weight,
        "bm25_weight": args.bm25_weight,
        "embedding_provider": args.embedding_provider,
        "embedding_model": args.embedding_model,
        "embedding_base_url": args.embedding_base_url,
        "embedding_api_key": args.embedding_api_key,
        "top_root_nodes": args.top_root_nodes,
        "top_root_edges": args.top_root_edges,
        "max_hop": args.max_hop,
        "path_budget": args.path_budget,
        "semantic_edge_budget": args.semantic_edge_budget,
        "semantic_node_budget": args.semantic_node_budget,
        "semantic_edge_min_score": args.semantic_edge_min_score,
        "semantic_node_min_score": args.semantic_node_min_score,
        "association_rounds": args.association_rounds,
        "frontier_edge_top_k": args.frontier_edge_top_k,
        "group_limit": args.group_limit,
        "max_source_chunks": args.max_source_chunks,
        "max_source_word_budget": args.max_source_word_budget,
        "task_mode": args.task_mode,
        "context_constraint": args.context_constraint,
    }


def command_retrieve(args):
    """Run retrieval only and save the evidence package.

    仅运行检索阶段，输出检索结果和性能摘要。
    """
    corpus_name = infer_corpus_name(args.corpus_dir)
    questions_file = resolve_questions_file(corpus_name, args.questions_file)
    log(f"[main] retrieve corpus={args.corpus_dir} questions={questions_file}")
    payload, output_path = retrieve_corpus_queries(
        corpus_dir=args.corpus_dir,
        rewrites_file=args.rewrites_file,
        questions_file=str(questions_file) if questions_file else None,
        limit_groups=args.limit_groups,
        output_dir=args.output_dir,
        **retrieval_config_from_args(args),
    )
    print(output_path)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


def command_answer(args):
    """Turn a retrieval JSON into model answers.

    读取检索输出文件，调用生成模型并保存答案结果。
    """
    retrieval_file = Path(args.retrieval_file)
    log(f"[main] answer retrieval_file={retrieval_file}")
    payload = json.loads(retrieval_file.read_text(encoding="utf-8"))
    records = payload["results"]
    llm_client = OpenAICompatibleClient(load_llm_config())
    output_path = Path(args.output_file) if args.output_file else retrieval_file.with_name(retrieval_file.stem.replace("_retrieval", "_answers") + ".json")
    results = generate_answers(records, output_path=output_path, llm_client=llm_client, max_workers=args.max_workers)
    print(output_path)
    print(f"generated_answers={len(results)}")


def command_judge(args):
    """Compare candidate answers against a baseline with the FG-RAG-style judge.

    对候选答案与基线答案进行成对评判，并生成胜率报告。
    """
    candidate_answers = json.loads(Path(args.candidate_file).read_text(encoding="utf-8"))
    questions = extract_questions(Path(args.questions_file))
    llm_client = OpenAICompatibleClient(load_judge_config())
    candidate_label = args.candidate_label or Path(args.candidate_file).stem
    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else Path("Datasets") / infer_corpus_name(args.questions_file) / "index"
    corpus_resources = load_judge_corpus_resources(corpus_dir) if args.judge_mode == "claim_diagnostics" else None

    if bool(args.baseline_file) == bool(args.baseline_dir):
        raise ValueError("Please provide exactly one of --baseline-file or --baseline-dir")

    def _align_inputs(baseline_answers):
        aligned_total = min(len(questions), len(candidate_answers), len(baseline_answers))
        if len({len(questions), len(candidate_answers), len(baseline_answers)}) != 1:
            log(
                "[main] judge length mismatch "
                f"questions={len(questions)} candidate={len(candidate_answers)} baseline={len(baseline_answers)}; "
                f"auto-aligning to first {aligned_total}"
            )
        if args.limit is not None:
            aligned_total = min(aligned_total, args.limit)
        return (
            questions[:aligned_total],
            candidate_answers[:aligned_total],
            baseline_answers[:aligned_total],
        )

    if args.baseline_file:
        log(f"[main] judge candidate={args.candidate_file} baseline={args.baseline_file}")
        baseline_answers = load_baseline_answers(Path(args.baseline_file))
        aligned_questions, aligned_candidates, aligned_baselines = _align_inputs(baseline_answers)
        output_path = (
            Path(args.output_file)
            if args.output_file
            else Path(args.candidate_file).with_name(Path(args.candidate_file).stem + "_vs_baseline_winrate.json")
        )
        payload = run_winrate_judgement(
            aligned_questions,
            aligned_candidates,
            aligned_baselines,
            llm_client,
            output_path=output_path,
            max_workers=args.max_workers,
            corpus_resources=corpus_resources,
            judge_mode=args.judge_mode,
        )
        print(output_path)
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
        return

    baseline_dir = Path(args.baseline_dir)
    baseline_files = sorted(path for path in baseline_dir.glob("*.json") if path.is_file())
    if not baseline_files:
        raise ValueError(f"No baseline json files found under: {baseline_dir}")
    output_dir = Path(args.output_file).parent if args.output_file else Path(args.candidate_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = (
        Path(args.summary_file)
        if args.summary_file
        else output_dir / f"{Path(args.candidate_file).stem}_baseline_winrate_tables.md"
    )
    per_baseline_payloads = []
    markdown_sections = ["# Baseline Winrate Tables", ""]
    for baseline_file in baseline_files:
        log(f"[main] judge candidate={args.candidate_file} baseline={baseline_file}")
        baseline_answers = load_baseline_answers(baseline_file)
        aligned_questions, aligned_candidates, aligned_baselines = _align_inputs(baseline_answers)
        baseline_name = baseline_file.stem
        output_path = output_dir / f"{Path(args.candidate_file).stem}_vs_{baseline_name}_winrate.json"
        payload = run_winrate_judgement(
            aligned_questions,
            aligned_candidates,
            aligned_baselines,
            llm_client,
            output_path=output_path,
            max_workers=args.max_workers,
            corpus_resources=corpus_resources,
            judge_mode=args.judge_mode,
        )
        per_baseline_payloads.append(
            {
                "baseline_name": baseline_name,
                "baseline_file": str(baseline_file),
                "output_file": str(output_path),
                "summary": payload["summary"],
            }
        )
        markdown_sections.append(
            render_winrate_markdown_table(
                payload,
                candidate_label=candidate_label,
                baseline_label=baseline_name,
                title=baseline_name,
            )
        )
        markdown_sections.append("")

    summary_file.write_text("\n".join(markdown_sections), encoding="utf-8")
    manifest_path = summary_file.with_suffix(".json")
    manifest_path.write_text(json.dumps(per_baseline_payloads, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_file)
    print(manifest_path)
    print(json.dumps({"baselines": len(per_baseline_payloads)}, ensure_ascii=False, indent=2))


def command_run(args):
    """Convenience wrapper for retrieve -> answer, without judging.

    This is mainly useful for fast iteration and for measuring how retrieval
    settings affect the overall system runtime before running full evaluation.

    快速执行检索加生成流程，适合调参与验证效果。
    """
    corpus_name = infer_corpus_name(args.corpus_dir)
    log(f"[main] run corpus={corpus_name}")
    questions_file = resolve_questions_file(corpus_name, args.questions_file)
    if questions_file is None:
        raise ValueError(f"Could not resolve questions file for corpus: {corpus_name}")
    log(f"[main] questions={questions_file}")

    run_start = perf_counter()
    llm_client = OpenAICompatibleClient(load_llm_config())
    payload, retrieval_path, answers, answer_output = run_corpus_queries_online(
        corpus_dir=args.corpus_dir,
        llm_client=llm_client,
        rewrites_file=args.rewrites_file,
        questions_file=str(questions_file),
        limit_groups=args.limit_groups,
        output_dir=args.output_dir,
        answer_output_file=args.answer_output_file,
        max_workers=args.max_workers,
        **retrieval_config_from_args(args),
    )
    total_elapsed = perf_counter() - run_start
    log(
        f"[main] run done online_total={total_elapsed:.2f}s "
        f"queries={len(answers)}"
    )
    print(retrieval_path)
    print(answer_output)
    print(
        json.dumps(
            {
                "online_total_seconds": round(total_elapsed, 3),
                "generated_answers": len(answers),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def command_run_all(args):
    """Convenience wrapper for retrieve -> answer -> judge."""
    corpus_name = infer_corpus_name(args.corpus_dir)
    log(f"[main] run-all corpus={corpus_name}")
    questions_file = resolve_questions_file(corpus_name, args.questions_file)
    if questions_file is None:
        raise ValueError(f"Could not resolve questions file for corpus: {corpus_name}")
    baseline_file = resolve_baseline_file(corpus_name, args.baseline_file)
    log(f"[main] questions={questions_file} baseline={baseline_file}")
    llm_client = OpenAICompatibleClient(load_llm_config())
    payload, retrieval_path, answers, answer_output = run_corpus_queries_online(
        corpus_dir=args.corpus_dir,
        llm_client=llm_client,
        rewrites_file=args.rewrites_file,
        questions_file=str(questions_file),
        limit_groups=args.limit_groups,
        output_dir=args.output_dir,
        answer_output_file=args.answer_output_file,
        max_workers=args.max_workers,
        **retrieval_config_from_args(args),
    )
    output_dir = Path(args.output_dir)
    stem = retrieval_path.stem.replace("_retrieval", "")
    print(retrieval_path)
    print(answer_output)
    if baseline_file is not None:
        questions = extract_questions(questions_file)
        if args.limit_groups is not None:
            questions = questions[: args.limit_groups]
        baseline_answers = load_baseline_answers(baseline_file)[: len(answers)]
        judge_output = Path(args.judge_output_file) if args.judge_output_file else output_dir / f"{stem}_vs_{baseline_file.stem}_winrate.json"
        judge_client = OpenAICompatibleClient(load_judge_config())
        corpus_resources = load_judge_corpus_resources(args.corpus_dir) if args.judge_mode == "claim_diagnostics" else None
        judge_payload = run_winrate_judgement(
            questions[: len(answers)],
            answers,
            baseline_answers,
            judge_client,
            output_path=judge_output,
            max_workers=args.max_workers,
            corpus_resources=corpus_resources,
            judge_mode=args.judge_mode,
        )
        print(judge_output)
        print(json.dumps(judge_payload["summary"], ensure_ascii=False, indent=2))
    else:
        print("No baseline file resolved; skipped judging.")


def main():
    """Dispatch the selected CLI subcommand."""
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "retrieve":
        command_retrieve(args)
    elif args.command == "answer":
        command_answer(args)
    elif args.command == "judge":
        command_judge(args)
    elif args.command == "run":
        command_run(args)
    elif args.command == "run-all":
        command_run_all(args)


if __name__ == "__main__":
    main()
