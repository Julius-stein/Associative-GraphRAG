"""Summarize profile-sweep results and relate best profiles to query priors."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


OUTCOME_SCORE = {
    "candidate": 1.0,
    "tie": 0.5,
    "baseline": 0.0,
}

METRICS = [
    "root_density",
    "root_fragmentation",
    "retrieval_cliff",
    "candidate_chunk_dispersion",
    "effective_candidate_chunk_count",
    "association_strength",
]


def _mean(values):
    return round(sum(values) / len(values), 6) if values else None


def load_manifest(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_retrieval_stats(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {record["group_id"]: record for record in payload["results"]}


def load_judge_outcomes(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        verdict["query"]: {
            "final_winner": verdict["final_winner"],
            "query": verdict["query"],
        }
        for verdict in payload["verdicts"]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    by_corpus = defaultdict(list)
    for item in manifest:
        by_corpus[item["corpus"]].append(item)

    payload = []
    for corpus, records in sorted(by_corpus.items()):
        query_rows = defaultdict(list)
        profile_summaries = []
        for item in records:
            judge = load_judge_outcomes(Path(item["judge_file"]))
            retrieval = load_retrieval_stats(Path(item["retrieval_file"]))
            summary = json.loads(Path(item["judge_file"]).read_text(encoding="utf-8"))["summary"]
            profile_summaries.append(
                {
                    "profile_name": item["profile_name"],
                    "summary": summary,
                    "config": item["config"],
                }
            )
            for query, outcome in judge.items():
                group_id = next((record["group_id"] for record in retrieval.values() if record["query"] == query), None)
                stats = retrieval[group_id]["stats"] if group_id in retrieval else {}
                query_rows[query].append(
                    {
                        "profile_name": item["profile_name"],
                        "outcome": outcome["final_winner"],
                        "score": OUTCOME_SCORE[outcome["final_winner"]],
                        "stats": stats,
                    }
                )

        best_profile_counter = Counter()
        best_profile_examples = defaultdict(list)
        profile_metric_means = defaultdict(lambda: defaultdict(list))
        pairwise_preferences = defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0, "metric_means": defaultdict(list), "examples": []})

        for query, rows in query_rows.items():
            rows.sort(key=lambda row: (-row["score"], row["profile_name"]))
            best = rows[0]
            best_profile_counter[best["profile_name"]] += 1
            if len(best_profile_examples[best["profile_name"]]) < 8:
                best_profile_examples[best["profile_name"]].append(query)
            for metric in METRICS:
                if metric in best["stats"]:
                    profile_metric_means[best["profile_name"]][metric].append(best["stats"][metric])

            for left in rows:
                for right in rows:
                    if left["profile_name"] >= right["profile_name"]:
                        continue
                    key = f"{left['profile_name']}__vs__{right['profile_name']}"
                    if left["score"] > right["score"]:
                        pairwise_preferences[key]["wins"] += 1
                        chosen = left
                    elif left["score"] < right["score"]:
                        pairwise_preferences[key]["losses"] += 1
                        chosen = right
                    else:
                        pairwise_preferences[key]["ties"] += 1
                        chosen = None
                    if chosen is not None:
                        if len(pairwise_preferences[key]["examples"]) < 8:
                            pairwise_preferences[key]["examples"].append(query)
                        for metric in METRICS:
                            if metric in chosen["stats"]:
                                pairwise_preferences[key]["metric_means"][metric].append(chosen["stats"][metric])

        payload.append(
            {
                "corpus": corpus,
                "profile_summaries": profile_summaries,
                "best_profile_counts": best_profile_counter.most_common(),
                "best_profile_metric_means": {
                    profile: {metric: _mean(values) for metric, values in metric_map.items()}
                    for profile, metric_map in profile_metric_means.items()
                },
                "best_profile_examples": dict(best_profile_examples),
                "pairwise_preferences": {
                    key: {
                        "wins": value["wins"],
                        "losses": value["losses"],
                        "ties": value["ties"],
                        "preferred_metric_means": {
                            metric: _mean(values) for metric, values in value["metric_means"].items()
                        },
                        "examples": value["examples"],
                    }
                    for key, value in pairwise_preferences.items()
                },
            }
        )

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
