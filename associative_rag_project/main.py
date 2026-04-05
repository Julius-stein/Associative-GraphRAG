"""CLI entrypoint for retrieval, answer generation, judging, and full runs."""

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
from .judge import run_winrate_judgement
from .llm_client import OpenAICompatibleClient, generate_answers
from .logging_utils import log
from .pipeline import retrieve_corpus_queries


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
    common.add_argument("--top-chunks", type=int, default=5)
    common.add_argument("--chunk-candidate-multiplier", type=int, default=3)
    common.add_argument("--candidate-pool-size", type=int, default=30)
    common.add_argument("--retrieval-mode", choices=["bm25", "dense", "hybrid"], default="dense")
    common.add_argument("--dense-weight", type=float, default=0.75)
    common.add_argument("--bm25-weight", type=float, default=0.25)
    common.add_argument("--top-root-nodes", type=int, default=12)
    common.add_argument("--top-root-edges", type=int, default=16)
    common.add_argument("--max-hop", type=int, default=4)
    common.add_argument("--path-budget", type=int, default=12)
    common.add_argument("--semantic-edge-budget", type=int, default=20)
    common.add_argument("--semantic-node-budget", type=int, default=12)
    common.add_argument("--semantic-edge-min-score", type=float, default=0.03)
    common.add_argument("--semantic-node-min-score", type=float, default=0.03)
    common.add_argument("--association-rounds", type=int, default=2)
    common.add_argument("--group-limit", type=int, default=8)
    common.add_argument("--max-source-chunks", type=int, default=14)
    common.add_argument("--max-source-word-budget", type=int, default=4500)
    common.add_argument("--max-workers", type=int, default=12)

    subparsers.add_parser("retrieve", parents=[common])

    answer = subparsers.add_parser("answer")
    answer.add_argument("--retrieval-file", required=True)
    answer.add_argument("--output-file")
    answer.add_argument("--max-workers", type=int)

    judge = subparsers.add_parser("judge")
    judge.add_argument("--questions-file", required=True)
    judge.add_argument("--candidate-file", required=True)
    judge.add_argument("--baseline-file", required=True)
    judge.add_argument("--output-file")
    judge.add_argument("--limit", type=int)
    judge.add_argument("--max-workers", type=int)

    run = subparsers.add_parser("run", parents=[common])
    run.add_argument("--answer-output-file")

    run_all = subparsers.add_parser("run-all", parents=[common])
    run_all.add_argument("--baseline-file")
    run_all.add_argument("--answer-output-file")
    run_all.add_argument("--judge-output-file")
    return parser


def retrieval_config_from_args(args):
    """Translate CLI args into the retrieval config consumed by the pipeline."""
    return {
        "top_chunks": args.top_chunks,
        "chunk_candidate_multiplier": args.chunk_candidate_multiplier,
        "candidate_pool_size": args.candidate_pool_size,
        "retrieval_mode": args.retrieval_mode,
        "dense_weight": args.dense_weight,
        "bm25_weight": args.bm25_weight,
        "top_root_nodes": args.top_root_nodes,
        "top_root_edges": args.top_root_edges,
        "max_hop": args.max_hop,
        "path_budget": args.path_budget,
        "semantic_edge_budget": args.semantic_edge_budget,
        "semantic_node_budget": args.semantic_node_budget,
        "semantic_edge_min_score": args.semantic_edge_min_score,
        "semantic_node_min_score": args.semantic_node_min_score,
        "association_rounds": args.association_rounds,
        "group_limit": args.group_limit,
        "max_source_chunks": args.max_source_chunks,
        "max_source_word_budget": args.max_source_word_budget,
    }


def command_retrieve(args):
    """Run retrieval only and save the evidence package."""
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
    """Turn a retrieval JSON into model answers."""
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
    """Compare candidate answers against a baseline with the FG-RAG-style judge."""
    log(f"[main] judge candidate={args.candidate_file} baseline={args.baseline_file}")
    candidate_answers = json.loads(Path(args.candidate_file).read_text(encoding="utf-8"))
    baseline_answers = load_baseline_answers(Path(args.baseline_file))
    questions = extract_questions(Path(args.questions_file))
    aligned_total = min(len(questions), len(candidate_answers), len(baseline_answers))
    if len({len(questions), len(candidate_answers), len(baseline_answers)}) != 1:
        log(
            "[main] judge length mismatch "
            f"questions={len(questions)} candidate={len(candidate_answers)} baseline={len(baseline_answers)}; "
            f"auto-aligning to first {aligned_total}"
        )
    if args.limit is not None:
        aligned_total = min(aligned_total, args.limit)
    questions = questions[:aligned_total]
    candidate_answers = candidate_answers[:aligned_total]
    baseline_answers = baseline_answers[:aligned_total]
    llm_client = OpenAICompatibleClient(load_judge_config())
    output_path = Path(args.output_file) if args.output_file else Path(args.candidate_file).with_name(Path(args.candidate_file).stem + "_vs_baseline_winrate.json")
    payload = run_winrate_judgement(
        questions,
        candidate_answers,
        baseline_answers,
        llm_client,
        output_path=output_path,
        max_workers=args.max_workers,
    )
    print(output_path)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


def command_run(args):
    """Convenience wrapper for retrieve -> answer, without judging.

    This is mainly useful for fast iteration and for measuring how retrieval
    settings affect the overall system runtime before running full evaluation.
    """
    corpus_name = infer_corpus_name(args.corpus_dir)
    log(f"[main] run corpus={corpus_name}")
    questions_file = resolve_questions_file(corpus_name, args.questions_file)
    if questions_file is None:
        raise ValueError(f"Could not resolve questions file for corpus: {corpus_name}")
    log(f"[main] questions={questions_file}")

    retrieve_start = perf_counter()
    payload, retrieval_path = retrieve_corpus_queries(
        corpus_dir=args.corpus_dir,
        rewrites_file=args.rewrites_file,
        questions_file=str(questions_file),
        limit_groups=args.limit_groups,
        output_dir=args.output_dir,
        **retrieval_config_from_args(args),
    )
    retrieve_elapsed = perf_counter() - retrieve_start
    log(f"[main] retrieve finished in {retrieve_elapsed:.2f}s -> {retrieval_path}")

    llm_client = OpenAICompatibleClient(load_llm_config())
    output_dir = Path(args.output_dir)
    stem = retrieval_path.stem.replace("_retrieval", "")
    answer_output = Path(args.answer_output_file) if args.answer_output_file else output_dir / f"{stem}_answers.json"

    answer_start = perf_counter()
    answers = generate_answers(
        payload["results"],
        output_path=answer_output,
        llm_client=llm_client,
        max_workers=args.max_workers,
    )
    answer_elapsed = perf_counter() - answer_start
    total_elapsed = retrieve_elapsed + answer_elapsed
    log(
        f"[main] run done retrieve={retrieve_elapsed:.2f}s "
        f"answer={answer_elapsed:.2f}s total={total_elapsed:.2f}s"
    )
    print(retrieval_path)
    print(answer_output)
    print(
        json.dumps(
            {
                "retrieval_seconds": round(retrieve_elapsed, 3),
                "answer_seconds": round(answer_elapsed, 3),
                "total_seconds": round(total_elapsed, 3),
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
    payload, retrieval_path = retrieve_corpus_queries(
        corpus_dir=args.corpus_dir,
        rewrites_file=args.rewrites_file,
        questions_file=str(questions_file),
        limit_groups=args.limit_groups,
        output_dir=args.output_dir,
        **retrieval_config_from_args(args),
    )
    llm_client = OpenAICompatibleClient(load_llm_config())
    output_dir = Path(args.output_dir)
    stem = retrieval_path.stem.replace("_retrieval", "")
    answer_output = Path(args.answer_output_file) if args.answer_output_file else output_dir / f"{stem}_answers.json"
    answers = generate_answers(
        payload["results"],
        output_path=answer_output,
        llm_client=llm_client,
        max_workers=args.max_workers,
    )
    print(answer_output)
    if baseline_file is not None:
        questions = extract_questions(questions_file)
        if args.limit_groups is not None:
            questions = questions[: args.limit_groups]
        baseline_answers = load_baseline_answers(baseline_file)[: len(answers)]
        judge_output = Path(args.judge_output_file) if args.judge_output_file else output_dir / f"{stem}_vs_{baseline_file.stem}_winrate.json"
        judge_client = OpenAICompatibleClient(load_judge_config())
        judge_payload = run_winrate_judgement(
            questions[: len(answers)],
            answers,
            baseline_answers,
            judge_client,
            output_path=judge_output,
            max_workers=args.max_workers,
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
