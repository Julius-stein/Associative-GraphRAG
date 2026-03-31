"""Summarize why the baseline beat the candidate in judge verdict files.

This helper focuses on baseline-win cases and aggregates the judge's
explanations by evaluation dimension. It is intentionally lightweight and
heuristic-driven so it can run locally without another model call.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


DIMENSIONS = ["Comprehensiveness", "Diversity", "Empowerment", "Overall Winner"]

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "by",
    "is",
    "are",
    "was",
    "were",
    "be",
    "being",
    "it",
    "its",
    "that",
    "this",
    "as",
    "from",
    "more",
    "less",
    "than",
    "because",
    "answer",
    "reader",
    "response",
}

REASON_PATTERNS = {
    "broader_coverage": [
        "more comprehensive",
        "broader",
        "wider range",
        "more complete",
        "covers more",
        "fuller",
        "scope",
    ],
    "more_specific_detail": [
        "more specific",
        "more precise",
        "more concrete",
        "specific examples",
        "details",
        "mechanisms",
        "causal",
    ],
    "better_structure": [
        "clearer framework",
        "better organized",
        "more coherent",
        "analytical framework",
        "structured",
        "clearer",
    ],
    "more_diverse_perspectives": [
        "more diverse",
        "varied",
        "multiple perspectives",
        "range of contexts",
        "range of artistic",
        "different ways",
    ],
    "better_support": [
        "better supported",
        "stronger examples",
        "stronger historical",
        "specific contexts",
        "evidence",
        "grounded",
    ],
    "more_useful_for_judgment": [
        "informed judgment",
        "analytical leverage",
        "useful",
        "nuanced conclusions",
        "better helps",
        "understand how and why",
    ],
}


def _extract_baseline_explanations(verdict: dict) -> dict[str, list[str]]:
    """Map AB/BA explanations back to the baseline answer."""
    out = defaultdict(list)
    order_expectation = {"order_ab": "Answer 2", "order_ba": "Answer 1"}
    for order_key, expected_winner in order_expectation.items():
        order_payload = verdict.get(order_key, {})
        for dimension in DIMENSIONS:
            item = order_payload.get(dimension, {})
            if item.get("Winner") != expected_winner:
                continue
            explanation = (item.get("Explanation") or "").strip()
            if explanation:
                out[dimension].append(explanation)
    return out


def _tokenize(text: str) -> list[str]:
    return [tok for tok in re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", text.lower()) if tok not in STOPWORDS]


def _top_terms(texts: list[str], limit: int = 12) -> list[tuple[str, int]]:
    counter = Counter()
    for text in texts:
        counter.update(_tokenize(text))
    return counter.most_common(limit)


def _reason_buckets(texts: list[str]) -> list[tuple[str, int]]:
    joined = "\n".join(texts).lower()
    counts = []
    for label, patterns in REASON_PATTERNS.items():
        score = sum(joined.count(pattern) for pattern in patterns)
        counts.append((label, score))
    counts.sort(key=lambda item: (-item[1], item[0]))
    return [item for item in counts if item[1] > 0]


def summarize_loss_file(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    verdicts = data.get("verdicts", [])

    dimension_texts = defaultdict(list)
    dimension_queries = defaultdict(list)
    baseline_loss_count = 0

    for verdict in verdicts:
        if verdict.get("final_winner") != "baseline":
            continue
        baseline_loss_count += 1
        extracted = _extract_baseline_explanations(verdict)
        for dimension, texts in extracted.items():
            dimension_texts[dimension].extend(texts)
            for _ in texts:
                dimension_queries[dimension].append(verdict.get("query", ""))

    summary = {
        "file": str(path),
        "baseline_win_cases": baseline_loss_count,
        "dimensions": {},
    }
    for dimension in DIMENSIONS:
        texts = dimension_texts.get(dimension, [])
        summary["dimensions"][dimension] = {
            "explanation_count": len(texts),
            "top_reason_buckets": _reason_buckets(texts),
            "top_terms": _top_terms(texts),
            "sample_queries": [query for query in dimension_queries.get(dimension, [])[:8] if query],
            "sample_explanations": texts[:5],
        }
    all_texts = [text for texts in dimension_texts.values() for text in texts]
    summary["overall"] = {
        "top_reason_buckets": _reason_buckets(all_texts),
        "top_terms": _top_terms(all_texts, limit=20),
        "sample_explanations": all_texts[:10],
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--output")
    args = parser.parse_args()

    payload = [summarize_loss_file(Path(file_path)) for file_path in args.files]
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
