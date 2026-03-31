"""Inspect how retrieval-side signals relate to the perceived query shape.

This script is meant for manual analysis rather than model training. It helps
answer questions like:

- Which queries have the lowest / highest root density?
- Do high-fragmentation queries read like broader QFS prompts?
- Does retrieval cliff align with "focused" queries?

It can also export a CSV template for human annotation so we can compare
manual `broad / focused / mixed` judgments against current retrieval signals.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_METRICS = [
    "root_density",
    "root_fragmentation",
    "retrieval_cliff",
    "candidate_chunk_dispersion",
    "effective_candidate_chunk_count",
    "association_strength",
]


def load_results(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["results"]


def select_extremes(results: list[dict], metric: str, top_n: int) -> dict[str, list[dict]]:
    ranked = [record for record in results if metric in record.get("stats", {})]
    ranked.sort(key=lambda item: item["stats"][metric])
    return {
        "low": ranked[:top_n],
        "high": ranked[-top_n:],
    }


def summarize_metric(results: list[dict], metric: str) -> dict:
    values = [record["stats"][metric] for record in results if metric in record.get("stats", {})]
    if not values:
        return {"count": 0}
    values = sorted(values)
    count = len(values)
    return {
        "count": count,
        "mean": round(sum(values) / count, 6),
        "median": round(values[count // 2], 6),
        "min": round(values[0], 6),
        "max": round(values[-1], 6),
    }


def record_to_row(record: dict, metric: str, bucket: str, source_file: str) -> dict:
    stats = record["stats"]
    return {
        "source_file": source_file,
        "group_id": record["group_id"],
        "query": record["query"],
        "metric": metric,
        "bucket": bucket,
        "metric_value": stats[metric],
        "root_density": stats.get("root_density"),
        "root_fragmentation": stats.get("root_fragmentation"),
        "retrieval_cliff": stats.get("retrieval_cliff"),
        "candidate_chunk_dispersion": stats.get("candidate_chunk_dispersion"),
        "effective_candidate_chunk_count": stats.get("effective_candidate_chunk_count"),
        "association_strength": stats.get("association_strength"),
        "manual_shape_label": "",
        "notes": "",
    }


def export_annotation_csv(rows: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_file",
        "group_id",
        "query",
        "metric",
        "bucket",
        "metric_value",
        "root_density",
        "root_fragmentation",
        "retrieval_cliff",
        "candidate_chunk_dispersion",
        "effective_candidate_chunk_count",
        "association_strength",
        "manual_shape_label",
        "notes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("retrieval_files", nargs="+", help="retrieval json files")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--export-csv", help="optional annotation template csv")
    args = parser.parse_args()

    report = []
    csv_rows = []

    for file_path in args.retrieval_files:
        path = Path(file_path)
        results = load_results(path)
        file_report = {
            "file": str(path),
            "query_count": len(results),
            "metrics": {},
        }
        for metric in args.metrics:
            metric_summary = summarize_metric(results, metric)
            extremes = select_extremes(results, metric, args.top_n)
            file_report["metrics"][metric] = {
                "summary": metric_summary,
                "low": [
                    {
                        "group_id": record["group_id"],
                        "value": round(record["stats"][metric], 6),
                        "query": record["query"],
                    }
                    for record in extremes["low"]
                ],
                "high": [
                    {
                        "group_id": record["group_id"],
                        "value": round(record["stats"][metric], 6),
                        "query": record["query"],
                    }
                    for record in extremes["high"]
                ],
            }
            for bucket in ("low", "high"):
                for record in extremes[bucket]:
                    csv_rows.append(record_to_row(record, metric, bucket, str(path)))
        report.append(file_report)

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.export_csv:
        export_annotation_csv(csv_rows, Path(args.export_csv))


if __name__ == "__main__":
    main()
