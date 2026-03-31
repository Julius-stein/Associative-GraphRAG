"""Run small, interpretable retrieval-budget profile sweeps.

The goal is not to optimize one monolithic alpha, but to discover which
parameter profile works best for which kind of query. Profiles are intentionally
small and human-readable so later adaptive-control rules can be justified.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from copy import deepcopy
from pathlib import Path


DEFAULT_BASE = {
    "top_chunks": 5,
    "top_root_nodes": 12,
    "top_root_edges": 16,
    "max_hop": 4,
    "path_budget": 12,
    "semantic_edge_budget": 20,
    "semantic_node_budget": 12,
    "semantic_edge_min_score": 0.03,
    "semantic_node_min_score": 0.03,
    "association_rounds": 2,
    "group_limit": 8,
    "max_source_chunks": 14,
    "max_source_word_budget": 4500,
    "disable_adaptive_control": True,
}


PROFILE_OVERRIDES = {
    "fixed_base": {},
    "precision": {
        "top_root_nodes": 10,
        "top_root_edges": 12,
        "path_budget": 8,
        "semantic_edge_budget": 14,
        "semantic_node_budget": 8,
        "semantic_edge_min_score": 0.05,
        "semantic_node_min_score": 0.05,
    },
    "coverage": {
        "top_root_nodes": 14,
        "top_root_edges": 20,
        "path_budget": 12,
        "semantic_edge_budget": 28,
        "semantic_node_budget": 18,
        "semantic_edge_min_score": 0.02,
        "semantic_node_min_score": 0.02,
    },
    "bridge": {
        "top_root_nodes": 14,
        "top_root_edges": 18,
        "path_budget": 20,
        "semantic_edge_budget": 16,
        "semantic_node_budget": 10,
        "semantic_edge_min_score": 0.03,
        "semantic_node_min_score": 0.035,
    },
    "rich": {
        "top_root_nodes": 14,
        "top_root_edges": 20,
        "path_budget": 18,
        "semantic_edge_budget": 28,
        "semantic_node_budget": 18,
        "semantic_edge_min_score": 0.02,
        "semantic_node_min_score": 0.02,
    },
}


def build_profile_config(profile_name: str) -> dict:
    if profile_name not in PROFILE_OVERRIDES:
        raise KeyError(f"Unknown profile: {profile_name}")
    cfg = deepcopy(DEFAULT_BASE)
    cfg.update(PROFILE_OVERRIDES[profile_name])
    cfg["profile_name"] = profile_name
    return cfg


def output_stem(corpus: str, profile_name: str, limit: int | None) -> str:
    suffix = f"_{limit}" if limit is not None else ""
    return f"{corpus}_{profile_name}{suffix}"


def build_run_all_command(corpus: str, profile_name: str, output_root: Path, max_workers: int | None, limit: int | None):
    cfg = build_profile_config(profile_name)
    run_dir = output_root / profile_name / corpus
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(corpus, profile_name, limit)

    command = [
        "python",
        "-m",
        "associative_rag_project.main",
        "run-all",
        "--corpus-dir",
        corpus,
        "--output-dir",
        str(run_dir),
        "--answer-output-file",
        str(run_dir / f"{stem}_answers.json"),
        "--judge-output-file",
        str(run_dir / f"{stem}_winrate.json"),
        "--top-chunks",
        str(cfg["top_chunks"]),
        "--top-root-nodes",
        str(cfg["top_root_nodes"]),
        "--top-root-edges",
        str(cfg["top_root_edges"]),
        "--max-hop",
        str(cfg["max_hop"]),
        "--path-budget",
        str(cfg["path_budget"]),
        "--semantic-edge-budget",
        str(cfg["semantic_edge_budget"]),
        "--semantic-node-budget",
        str(cfg["semantic_node_budget"]),
        "--semantic-edge-min-score",
        str(cfg["semantic_edge_min_score"]),
        "--semantic-node-min-score",
        str(cfg["semantic_node_min_score"]),
        "--association-rounds",
        str(cfg["association_rounds"]),
        "--group-limit",
        str(cfg["group_limit"]),
        "--max-source-chunks",
        str(cfg["max_source_chunks"]),
        "--max-source-word-budget",
        str(cfg["max_source_word_budget"]),
    ]
    if cfg.get("disable_adaptive_control", False):
        command.append("--disable-adaptive-control")
    if limit is not None:
        command.extend(["--limit-groups", str(limit)])
    if max_workers is not None:
        command.extend(["--max-workers", str(max_workers)])
    return command, run_dir, stem, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpora", nargs="+", required=True)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["fixed_base", "precision", "coverage", "bridge", "rich"],
    )
    parser.add_argument("--output-root", default="associative_rag_project/runs_profile_sweep")
    parser.add_argument("--limit-groups", type=int)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    manifest_path = output_root / "manifest.json"
    previous_manifest = []
    previous_index = {}
    if args.resume and manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous_index = {
            (item["corpus"], item["profile_name"]): item for item in previous_manifest
        }
    manifest = []
    for corpus in args.corpora:
        for profile_name in args.profiles:
            command, run_dir, stem, cfg = build_run_all_command(
                corpus=corpus,
                profile_name=profile_name,
                output_root=output_root,
                max_workers=args.max_workers,
                limit=args.limit_groups,
            )
            record = {
                "corpus": corpus,
                "profile_name": profile_name,
                "config": cfg,
                "run_dir": str(run_dir),
                "command": command,
                "answer_file": str(run_dir / f"{stem}_answers.json"),
                "judge_file": str(run_dir / f"{stem}_winrate.json"),
                "retrieval_file": str(run_dir / f"{corpus}_top5_hop4_assoc_project_retrieval.json"),
                "status": "pending",
            }
            prev = previous_index.get((corpus, profile_name))
            if prev and args.resume:
                record.update(
                    {
                        "status": prev.get("status", "unknown"),
                        "returncode": prev.get("returncode"),
                        "error": prev.get("error"),
                    }
                )
                if prev.get("status") == "completed" and Path(record["judge_file"]).exists():
                    manifest.append(record)
                    print(f"[resume] skip completed {corpus}/{profile_name}")
                    continue
            manifest.append(record)
            if args.dry_run:
                print(" ".join(command))
            else:
                completed = subprocess.run(command, check=False)
                record["returncode"] = completed.returncode
                if completed.returncode == 0:
                    record["status"] = "completed"
                else:
                    record["status"] = "failed"
                    record["error"] = f"subprocess returned {completed.returncode}"
                    manifest_path.parent.mkdir(parents=True, exist_ok=True)
                    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
                    if not args.continue_on_error:
                        raise subprocess.CalledProcessError(completed.returncode, command)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
