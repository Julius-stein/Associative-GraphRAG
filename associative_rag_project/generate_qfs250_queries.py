"""Generate 250 QFS queries per dataset with four explicit contracts.

Contracts:
- theme_grounded
- section_grounded
- comparison_grounded
- mechanism_grounded

Hard constraint used during generation:
- reject questions whose anchor entities can all be supported by one chunk
  (to reduce single-chunk-answerable questions).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from associative_rag_project.common import GRAPH_FIELD_SEP, parse_source_ids, tokenize
else:
    from associative_rag_project.common import GRAPH_FIELD_SEP, parse_source_ids, tokenize


DEFAULT_DATASETS = ["agriculture", "art", "legal", "mix", "news"]
CONTRACTS = [
    "theme_grounded",
    "section_grounded",
    "comparison_grounded",
    "mechanism_grounded",
]

GENERIC_LABELS = {
    "company",
    "the company",
    "agreement",
    "borrower",
    "seller",
    "purchaser",
    "tenant",
    "landlord",
    "lender",
    "party",
    "parties",
    "document",
    "section",
    "chapter",
    "article",
}

NOISY_LABEL_PATTERNS = [
    re.compile(r"^(?:bibref|inlineform|fig|figure|table)\d*$", re.I),
    re.compile(r"^(?:doi|isbn|www|http)$", re.I),
]

NOISY_LABEL_SUBSTRINGS = {
    "bibref",
    "inlineform",
    "copyright",
    "doi",
    "isbn",
    "http",
    "www",
}


def normalize_label(raw):
    text = str(raw or "").replace('"', " ").replace("'", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    lower = text.lower()
    if any(pat.match(lower) for pat in NOISY_LABEL_PATTERNS):
        return ""
    if any(part in lower for part in NOISY_LABEL_SUBSTRINGS):
        return ""
    if lower in GENERIC_LABELS:
        return ""
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]*", text)
    if not tokens:
        return ""
    if len(tokens) > 6:
        return ""
    joined = "".join(tokens)
    if len(joined) < 4:
        return ""
    if len(tokens) == 1 and len(tokens[0]) <= 2:
        return ""
    return " ".join(tokens)


def normalize_relation(raw):
    text = str(raw or "")
    if GRAPH_FIELD_SEP in text:
        text = text.split(GRAPH_FIELD_SEP, 1)[0]
    text = text.strip().strip('"').strip("'").lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "related"
    if len(text.split()) > 6:
        return "related"
    return text


def chunk_doc_id(chunk_store, chunk_id):
    chunk = chunk_store.get(chunk_id) or {}
    return chunk.get("full_doc_id")


def build_anchor_pool(graph, chunk_store):
    total_docs = len({row.get("full_doc_id") for row in chunk_store.values() if row.get("full_doc_id")})
    anchors = {}
    for node_id, node_data in graph.nodes(data=True):
        label = normalize_label(node_id)
        if not label:
            continue
        src_chunks = set(parse_source_ids(node_data.get("source_id", "")))
        src_chunks = {cid for cid in src_chunks if cid in chunk_store}
        if len(src_chunks) < 2:
            continue
        src_docs = {chunk_doc_id(chunk_store, cid) for cid in src_chunks}
        src_docs.discard(None)
        if len(src_docs) < 2:
            continue
        doc_ratio = len(src_docs) / max(total_docs, 1)
        if doc_ratio > 0.35:
            continue
        deg = graph.degree(node_id)
        score = (0.65 * math.log1p(len(src_docs))) + (0.35 * math.log1p(deg))
        anchors[node_id] = {
            "node_id": node_id,
            "label": label,
            "chunks": src_chunks,
            "docs": src_docs,
            "degree": deg,
            "score": score,
        }
    ranked = sorted(anchors.values(), key=lambda item: (-item["score"], -item["degree"], item["label"]))
    return ranked, {row["node_id"]: row for row in ranked}


def build_pair_pool(graph, anchor_ranked, max_anchor_candidates=800, max_pairs=60000):
    subset = anchor_ranked[:max_anchor_candidates]
    pair_pool = []
    for left_index in range(len(subset)):
        left_row = subset[left_index]
        for right_index in range(left_index + 1, len(subset)):
            right_row = subset[right_index]
            # Hard filter: a single chunk should not support both anchor entities.
            if not left_row["chunks"].isdisjoint(right_row["chunks"]):
                continue
            union_docs = left_row["docs"] | right_row["docs"]
            if len(union_docs) < 3:
                continue
            relation = "related"
            strength = 0.0
            edge_data = graph.get_edge_data(left_row["node_id"], right_row["node_id"])
            if edge_data:
                relation = normalize_relation(edge_data.get("keywords") or edge_data.get("description"))
                strength = float(edge_data.get("weight", 1.0))
            pair_pool.append(
                {
                    "left": left_row,
                    "right": right_row,
                    "relation": relation,
                    "strength": strength,
                    "doc_coverage": len(union_docs),
                }
            )
    pair_pool.sort(
        key=lambda item: (
            -item["doc_coverage"],
            -item["strength"],
            item["relation"] == "related",
            item["left"]["label"],
            item["right"]["label"],
        )
    )
    return pair_pool[:max_pairs]


def no_single_chunk_support(anchor_rows):
    if not anchor_rows:
        return False
    intersect = set(anchor_rows[0]["chunks"])
    for row in anchor_rows[1:]:
        intersect &= row["chunks"]
        if not intersect:
            return True
    return len(intersect) == 0


def contains_all_anchor_tokens(query, anchors):
    query_tokens = set(tokenize(query))
    for row in anchors:
        anchor_tokens = set(tokenize(row["label"]))
        if not anchor_tokens:
            return False
        if not (anchor_tokens & query_tokens):
            return False
    return True


def question_templates(contract):
    if contract == "theme_grounded":
        return [
            "Across the corpus, what broader themes connect {a}, {b}, and {c}, and where do those themes converge or diverge?",
            "What recurring high-level narratives link discussions of {a}, {b}, and {c} across different parts of the corpus?",
            "If we synthesize evidence across the corpus, what shared themes emerge around {a}, {b}, and {c}, and which tensions remain unresolved?",
            "How are {a}, {b}, and {c} jointly framed at a thematic level when evidence is aggregated across the whole corpus?",
        ]
    if contract == "section_grounded":
        return [
            "Which sections or parts of the corpus discuss both {a} and {b}, and how does the emphasis shift between those sections?",
            "Across different sections, how do treatments of {a} versus {b} change in scope, evidence style, or conclusions?",
            "What section-level progression can be observed when tracking {a} and {b} across the corpus?",
            "If we map the corpus by section, where are {a} and {b} co-developed, and where are they separated?",
        ]
    if contract == "comparison_grounded":
        return [
            "Compared with {b}, how is {a} characterized across the corpus in terms of goals, constraints, and outcomes?",
            "What are the most consistent similarities and differences between {a} and {b} when synthesizing evidence corpus-wide?",
            "Across the corpus, how does the role of {a} differ from {b}, and in which contexts do they overlap?",
            "If we compare corpus-wide discussions of {a} and {b}, what trade-offs or contrasts are repeatedly highlighted?",
        ]
    if contract == "mechanism_grounded":
        return [
            "Through what mechanisms does {a} influence {b} across the corpus, and what mediating factors are repeatedly mentioned?",
            "What causal pathways are described from {a} to {b} when evidence is integrated across multiple sections?",
            "How does the corpus explain the process linking {a} and {b}, including key conditions that enable or block the effect?",
            "When synthesizing the full corpus, what mechanism-level explanation best connects {a} with {b}?",
        ]
    raise ValueError(f"Unknown contract: {contract}")


def maybe_third_anchor(edge_row, anchor_ranked, rnd):
    neighborhood = []
    left = edge_row["left"]["node_id"]
    right = edge_row["right"]["node_id"]
    blocked = {left, right}
    for row in anchor_ranked:
        if row["node_id"] in blocked:
            continue
        if no_single_chunk_support([edge_row["left"], edge_row["right"], row]):
            neighborhood.append(row)
    if not neighborhood:
        return None
    return rnd.choice(neighborhood[:80])


def synthesize_questions_for_contract(contract, pair_pool, anchor_ranked, target_count, rnd):
    templates = question_templates(contract)
    rows = []
    seen = set()
    cursor = 0

    while len(rows) < target_count and cursor < max(1, len(pair_pool) * 4):
        edge_row = pair_pool[cursor % len(pair_pool)]
        cursor += 1
        left = edge_row["left"]
        right = edge_row["right"]
        anchors = [left, right]
        if contract == "theme_grounded":
            third = maybe_third_anchor(edge_row, anchor_ranked, rnd)
            if third is None:
                continue
            anchors = [left, right, third]
        if not no_single_chunk_support(anchors):
            continue
        template = rnd.choice(templates)
        if len(anchors) == 3:
            query = template.format(a=anchors[0]["label"], b=anchors[1]["label"], c=anchors[2]["label"])
        else:
            query = template.format(a=anchors[0]["label"], b=anchors[1]["label"])
        query = re.sub(r"\s+", " ", query).strip()
        key = query.lower()
        if key in seen:
            continue
        if not contains_all_anchor_tokens(query, anchors):
            continue
        seen.add(key)
        rows.append(
            {
                "query": query,
                "contract": contract,
                "anchors": [row["label"] for row in anchors],
            }
        )
    return rows


def generate_dataset_queries(dataset_dir, target_total=250, seed=42):
    dataset_dir = Path(dataset_dir)
    dataset_name = dataset_dir.name
    index_dir = dataset_dir / "index"
    graph = nx.read_graphml(index_dir / "graph_chunk_entity_relation.graphml")
    chunk_store = json.loads((index_dir / "kv_store_text_chunks.json").read_text(encoding="utf-8"))

    anchor_ranked, anchor_lookup = build_anchor_pool(graph, chunk_store)
    pair_pool = build_pair_pool(graph, anchor_ranked)
    if len(anchor_ranked) < 200 or len(pair_pool) < 1000:
        raise ValueError(
            f"{dataset_name}: insufficient anchor/edge pool for robust QFS generation "
            f"(anchors={len(anchor_ranked)}, pairs={len(pair_pool)})."
        )

    # 250 -> 63,63,62,62
    contract_targets = {
        "theme_grounded": (target_total + 3) // 4,
        "section_grounded": (target_total + 2) // 4,
        "comparison_grounded": target_total // 4,
        "mechanism_grounded": target_total // 4,
    }
    # fix exact total
    while sum(contract_targets.values()) > target_total:
        for key in ["theme_grounded", "section_grounded", "comparison_grounded", "mechanism_grounded"]:
            if contract_targets[key] > 0 and sum(contract_targets.values()) > target_total:
                contract_targets[key] -= 1

    rnd = random.Random(seed + sum(ord(ch) for ch in dataset_name))
    by_contract = {}
    for contract in CONTRACTS:
        by_contract[contract] = synthesize_questions_for_contract(
            contract=contract,
            pair_pool=pair_pool,
            anchor_ranked=anchor_ranked,
            target_count=contract_targets[contract],
            rnd=rnd,
        )
        if len(by_contract[contract]) < contract_targets[contract]:
            raise ValueError(
                f"{dataset_name}: contract {contract} only generated "
                f"{len(by_contract[contract])}/{contract_targets[contract]} questions."
            )

    merged = []
    for contract in CONTRACTS:
        merged.extend(by_contract[contract])
    rnd.shuffle(merged)
    return merged, {key: len(value) for key, value in by_contract.items()}


def write_dataset_queries(dataset_dir, rows, output_suffix):
    dataset_dir = Path(dataset_dir)
    dataset_name = dataset_dir.name
    output_path = dataset_dir / "query" / f"{dataset_name}_{output_suffix}.json"
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 250 QFS queries per dataset with four contracts.")
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated dataset names under Datasets/, e.g. agriculture,art,legal,mix,news",
    )
    parser.add_argument("--root", default="Datasets", help="Dataset root directory.")
    parser.add_argument("--target-total", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-suffix", default="qfs250")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    dataset_names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    summary = []
    for dataset_name in dataset_names:
        dataset_dir = root / dataset_name
        rows, counts = generate_dataset_queries(
            dataset_dir=dataset_dir,
            target_total=args.target_total,
            seed=args.seed,
        )
        out = write_dataset_queries(dataset_dir, rows, args.output_suffix)
        summary.append(
            {
                "dataset": dataset_name,
                "output": str(out),
                "total": len(rows),
                "counts": counts,
            }
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
