"""Align judge explanations with retrieval-side metrics.

This script answers questions like:

- When the candidate wins for "broader coverage", what do the retrieval-side
  metrics look like?
- When the baseline wins for "more useful for judgment", do those queries have
  denser or more fragmented root graphs?

It uses lightweight phrase buckets over judge explanations and joins them with
the per-query stats stored in retrieval json files.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from analyze_judge_losses import DIMENSIONS, REASON_PATTERNS


METRICS = [
    "root_density",
    "root_fragmentation",
    "retrieval_cliff",
    "candidate_chunk_dispersion",
    "effective_candidate_chunk_count",
    "association_strength",
    "final_node_count",
    "final_edge_count",
    "selected_source_word_count",
]


def _extract_side_explanations(verdict: dict, side: str) -> dict[str, list[str]]:
    """Collect dimension explanations attributed to the requested side."""
    out = defaultdict(list)
    expected = {
        "candidate": {"order_ab": "Answer 1", "order_ba": "Answer 2"},
        "baseline": {"order_ab": "Answer 2", "order_ba": "Answer 1"},
    }[side]
    for order_key, expected_winner in expected.items():
        order_payload = verdict.get(order_key, {})
        for dimension in DIMENSIONS:
            item = order_payload.get(dimension, {})
            if item.get("Winner") != expected_winner:
                continue
            explanation = (item.get("Explanation") or "").strip()
            if explanation:
                out[dimension].append(explanation)
    return out


def _top_bucket_for_text(text: str) -> str:
    text_lower = text.lower()
    scores = []
    for label, patterns in REASON_PATTERNS.items():
        score = sum(text_lower.count(pattern) for pattern in patterns)
        scores.append((label, score))
    scores.sort(key=lambda item: (-item[1], item[0]))
    if not scores or scores[0][1] <= 0:
        return "other"
    return scores[0][0]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def load_retrieval_stats(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {record["group_id"]: record["stats"] for record in payload["results"]}


def analyze_pair(winrate_path: Path, retrieval_path: Path) -> dict:
    verdict_payload = json.loads(winrate_path.read_text(encoding="utf-8"))
    stats_by_group = load_retrieval_stats(retrieval_path)

    rows = []
    for verdict in verdict_payload.get("verdicts", []):
        group_id = verdict.get("index")
        # winrate verdicts store an integer index, but query ids are also
        # present as q001-like ids in retrieval. Use the query order string.
        query_text = verdict.get("query", "")
        for side in ("candidate", "baseline"):
            side_explanations = _extract_side_explanations(verdict, side)
            for dimension, texts in side_explanations.items():
                for text in texts:
                    rows.append(
                        {
                            "query": query_text,
                            "group_id": None,
                            "side": side,
                            "dimension": dimension,
                            "reason_bucket": _top_bucket_for_text(text),
                            "explanation": text,
                        }
                    )

    retrieval_payload = json.loads(retrieval_path.read_text(encoding="utf-8"))
    query_to_group = {record["query"]: record["group_id"] for record in retrieval_payload["results"]}
    for row in rows:
        row["group_id"] = query_to_group.get(row["query"])
        stats = stats_by_group.get(row["group_id"], {})
        row["stats"] = stats

    summary = {
        "winrate_file": str(winrate_path),
        "retrieval_file": str(retrieval_path),
        "total_rows": len(rows),
        "by_side": {},
    }
    for side in ("candidate", "baseline"):
        side_rows = [row for row in rows if row["side"] == side]
        dimension_summary = {}
        for dimension in DIMENSIONS:
            dim_rows = [row for row in side_rows if row["dimension"] == dimension]
            bucket_counter = Counter(row["reason_bucket"] for row in dim_rows)
            bucket_metrics = {}
            for bucket, count in bucket_counter.most_common():
                bucket_rows = [row for row in dim_rows if row["reason_bucket"] == bucket]
                metric_summary = {}
                for metric in METRICS:
                    values = [row["stats"].get(metric) for row in bucket_rows if metric in row["stats"]]
                    values = [value for value in values if value is not None]
                    metric_summary[metric] = _mean(values)
                bucket_metrics[bucket] = {
                    "count": count,
                    "metric_means": metric_summary,
                    "sample_queries": [row["query"] for row in bucket_rows[:6]],
                }
            dimension_summary[dimension] = {
                "reason_buckets": bucket_counter.most_common(),
                "bucket_metrics": bucket_metrics,
            }
        overall_counter = Counter(row["reason_bucket"] for row in side_rows)
        summary["by_side"][side] = {
            "row_count": len(side_rows),
            "reason_buckets": overall_counter.most_common(),
            "dimensions": dimension_summary,
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--winrate-files", nargs="+", required=True)
    parser.add_argument("--retrieval-files", nargs="+", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    if len(args.winrate_files) != len(args.retrieval_files):
        raise ValueError("winrate-files and retrieval-files must have the same length")

    payload = []
    for winrate_file, retrieval_file in zip(args.winrate_files, args.retrieval_files):
        payload.append(analyze_pair(Path(winrate_file), Path(retrieval_file)))

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
