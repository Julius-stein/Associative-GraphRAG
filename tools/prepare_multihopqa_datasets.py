#!/usr/bin/env python3
"""Download and normalize multi-hop QA datasets without dropping benchmark structure.

The script writes the same directory layout used by the existing corpora:

    Datasets/<dataset>/corpus/<dataset>_unique_contexts.json
    Datasets/<dataset>/corpus/<dataset>_documents.json
    Datasets/<dataset>/query/<dataset>.json
    Datasets/<dataset>/eval/<dataset>_official_gold.json/jsonl

The *_unique_contexts.json file is kept as a plain text list for LightRAG
indexing. The *_documents.json file keeps benchmark document ids, titles,
sentence/paragraph identities, sample-local context membership, and support
links so answer EM/F1 and evidence-aware analysis can be computed faithfully.
The eval gold file keeps the official evaluator input shape.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable


DATASET_SPECS = {
    "hotpotqa": {
        "hf_id": "hotpotqa/hotpot_qa",
        "hf_config": "distractor",
        "default_split": "validation",
        "schema": "hotpot_like",
    },
    "2wikimultihopqa": {
        "hf_id": "framolfese/2WikiMultihopQA",
        "hf_config": None,
        "default_split": "validation",
        "schema": "hotpot_like",
    },
    "musique": {
        "hf_id": "bdsaglam/musique",
        "hf_config": None,
        "default_split": "validation",
        "schema": "musique",
    },
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _stable_id(*parts: str) -> str:
    digest = hashlib.md5("||".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def _lightrag_doc_id(text: str) -> str:
    return "doc-" + hashlib.md5(text.strip().encode("utf-8")).hexdigest()


def _doc_text(title: str, sentences_or_text: Any) -> str:
    if isinstance(sentences_or_text, list):
        body = " ".join(str(sentence).strip() for sentence in sentences_or_text if str(sentence).strip())
    else:
        body = str(sentences_or_text or "").strip()
    title = str(title or "").strip()
    if title:
        return f"Title: {title}\n\n{body}".strip()
    return body.strip()


def _extract_context_dict(context: Any) -> list[tuple[str, str, list[str]]]:
    """Return (title, text, sentences) triples from HotpotQA/2Wiki context fields."""
    if not isinstance(context, dict):
        return []
    titles = context.get("title") or []
    sentences = context.get("sentences") or []
    pairs = []
    for title, sent_list in zip(titles, sentences):
        sentence_list = [str(sentence) for sentence in sent_list] if isinstance(sent_list, list) else []
        text = _doc_text(str(title), sent_list)
        if text:
            pairs.append((str(title), text, sentence_list))
    return pairs


def _extract_musique_paragraphs(paragraphs: Any) -> list[tuple[str, str, int | None, bool, dict[str, Any]]]:
    """Return paragraph metadata from MuSiQue rows."""
    pairs = []
    if not isinstance(paragraphs, list):
        return pairs
    for offset, paragraph in enumerate(paragraphs):
        if not isinstance(paragraph, dict):
            continue
        idx = paragraph.get("idx", offset)
        title = str(paragraph.get("title") or f"paragraph-{idx}")
        text = _doc_text(title, paragraph.get("paragraph_text") or paragraph.get("text") or "")
        if text:
            pairs.append((title, text, idx, bool(paragraph.get("is_supporting", False)), paragraph))
    return pairs


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _support_pairs(supporting_facts: dict[str, Any]) -> list[list[Any]]:
    titles = [str(title) for title in _as_list(supporting_facts.get("title"))]
    sent_ids = supporting_facts.get("sent_id") or []
    return [[title, sent_id] for title, sent_id in zip(titles, sent_ids)]


def _load_hf_dataset(spec: dict[str, Any], split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install with `python -m pip install datasets`."
        ) from exc

    if spec["hf_config"]:
        return load_dataset(spec["hf_id"], spec["hf_config"], split=split)
    return load_dataset(spec["hf_id"], split=split)


def _iter_limited(dataset: Iterable[dict[str, Any]], limit_questions: int | None):
    for index, row in enumerate(dataset):
        if limit_questions is not None and index >= limit_questions:
            break
        yield index, row


def _upsert_doc(
    docs: OrderedDict[str, dict[str, Any]],
    *,
    doc_id: str,
    text: str,
    title: str,
    source_dataset: str,
    source_qid: str,
    split: str,
    sentences: list[str] | None = None,
    paragraph_idx: int | None = None,
    is_supporting: bool = False,
    support_detail: dict[str, Any] | None = None,
):
    if doc_id not in docs:
        docs[doc_id] = {
            "doc_id": doc_id,
            "lightrag_doc_id": _lightrag_doc_id(text),
            "source_dataset": source_dataset,
            "split": split,
            "title": title,
            "text": text,
            "sentences": sentences or [],
            "paragraph_idx": paragraph_idx,
            "source_qids": [],
            "supporting_for": [],
        }
    doc = docs[doc_id]
    if source_qid not in doc["source_qids"]:
        doc["source_qids"].append(source_qid)
    if is_supporting:
        detail = {"qid": source_qid}
        if support_detail:
            detail.update(support_detail)
        if detail not in doc["supporting_for"]:
            doc["supporting_for"].append(detail)


def _prepare_hotpot_like(name: str, dataset: Iterable[dict[str, Any]], limit_questions: int | None):
    docs: OrderedDict[str, dict[str, Any]] = OrderedDict()
    queries = []
    official_gold = []
    for index, row in _iter_limited(dataset, limit_questions):
        query_id = str(row.get("id") or row.get("_id") or f"{name}-{index:06d}")
        context_pairs = _extract_context_dict(row.get("context"))
        support = row.get("supporting_facts") or {}
        support_titles = [str(title) for title in _as_list(support.get("title"))]
        support_sent_ids = support.get("sent_id") or []
        support_by_title: dict[str, list[Any]] = {}
        for title, sent_id in zip(support_titles, support_sent_ids):
            support_by_title.setdefault(title, []).append(sent_id)
        support_doc_ids = []
        support_lightrag_doc_ids = []
        context_doc_ids = []
        context_lightrag_doc_ids = []

        for title, text, sentences in context_pairs:
            doc_id = f"{name}-{_stable_id(title, _normalize_text(text))}"
            is_supporting = title in support_by_title
            _upsert_doc(
                docs,
                doc_id=doc_id,
                text=text,
                title=title,
                source_dataset=name,
                source_qid=query_id,
                split="",
                sentences=sentences,
                is_supporting=is_supporting,
                support_detail={"title": title, "sent_id": support_by_title.get(title, [])},
            )
            context_doc_ids.append(doc_id)
            context_lightrag_doc_ids.append(docs[doc_id]["lightrag_doc_id"])
            if title in support_titles:
                support_doc_ids.append(doc_id)
                support_lightrag_doc_ids.append(docs[doc_id]["lightrag_doc_id"])

        queries.append(
            {
                "group_id": query_id,
                "variant_id": "base",
                "query": str(row.get("question") or "").strip(),
                "base_query": str(row.get("question") or "").strip(),
                "answer": row.get("answer"),
                "answer_aliases": [],
                "source_dataset": name,
                "source_id": query_id,
                "question_type": row.get("type"),
                "level": row.get("level"),
                "supporting_facts": {
                    "title": support_titles,
                    "sent_id": support_sent_ids,
                },
                "context_doc_ids": context_doc_ids,
                "context_lightrag_doc_ids": sorted(set(context_lightrag_doc_ids)),
                "support_doc_ids": sorted(set(support_doc_ids)),
                "support_lightrag_doc_ids": sorted(set(support_lightrag_doc_ids)),
            }
        )

        gold_item = {
            "_id": query_id,
            "question": str(row.get("question") or "").strip(),
            "answer": row.get("answer"),
            "supporting_facts": _support_pairs(support),
        }
        if name == "2wikimultihopqa":
            gold_item.update(
                {
                    "answer_id": row.get("answer_id", ""),
                    "evidences": row.get("evidences", []),
                    "evidences_id": row.get("evidences_id", []),
                    "type": row.get("type"),
                    "entity_ids": row.get("entity_ids"),
                }
            )
        else:
            gold_item.update({"type": row.get("type"), "level": row.get("level")})
        official_gold.append(gold_item)
    return docs, queries, official_gold


def _prepare_musique(name: str, dataset: Iterable[dict[str, Any]], limit_questions: int | None):
    docs: OrderedDict[str, dict[str, Any]] = OrderedDict()
    queries = []
    official_gold = []
    for index, row in _iter_limited(dataset, limit_questions):
        query_id = str(row.get("id") or f"{name}-{index:06d}")
        paragraph_pairs = _extract_musique_paragraphs(row.get("paragraphs"))
        support_doc_ids = []
        support_lightrag_doc_ids = []
        context_doc_ids = []
        context_lightrag_doc_ids = []
        supporting_paragraphs = []

        for title, text, paragraph_idx, is_supporting, paragraph in paragraph_pairs:
            doc_id = f"{name}-{_stable_id(str(paragraph_idx), title, _normalize_text(text))}"
            _upsert_doc(
                docs,
                doc_id=doc_id,
                text=text,
                title=title,
                source_dataset=name,
                source_qid=query_id,
                split="",
                paragraph_idx=paragraph_idx,
                is_supporting=is_supporting,
                support_detail={"idx": paragraph_idx, "title": title},
            )
            context_doc_ids.append(doc_id)
            context_lightrag_doc_ids.append(docs[doc_id]["lightrag_doc_id"])
            if is_supporting:
                support_doc_ids.append(doc_id)
                support_lightrag_doc_ids.append(docs[doc_id]["lightrag_doc_id"])
                supporting_paragraphs.append({"idx": paragraph_idx, "title": title})

        queries.append(
            {
                "group_id": query_id,
                "variant_id": "base",
                "query": str(row.get("question") or "").strip(),
                "base_query": str(row.get("question") or "").strip(),
                "answer": row.get("answer"),
                "answer_aliases": row.get("answer_aliases") or [],
                "source_dataset": name,
                "source_id": query_id,
                "question_decomposition": row.get("question_decomposition") or [],
                "answerable": row.get("answerable"),
                "supporting_paragraphs": supporting_paragraphs,
                "context_doc_ids": context_doc_ids,
                "context_lightrag_doc_ids": sorted(set(context_lightrag_doc_ids)),
                "support_doc_ids": sorted(set(support_doc_ids)),
                "support_lightrag_doc_ids": sorted(set(support_lightrag_doc_ids)),
            }
        )
        official_gold.append(
            {
                "id": query_id,
                "question": str(row.get("question") or "").strip(),
                "answer": row.get("answer"),
                "answer_aliases": row.get("answer_aliases") or [],
                "answerable": row.get("answerable"),
                "paragraphs": row.get("paragraphs") or [],
                "question_decomposition": row.get("question_decomposition") or [],
            }
        )
    return docs, queries, official_gold


def prepare_dataset(name: str, output_root: Path, split: str | None, limit_questions: int | None):
    if name not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset: {name}. Choose from {sorted(DATASET_SPECS)}")
    spec = DATASET_SPECS[name]
    selected_split = split or spec["default_split"]
    dataset = _load_hf_dataset(spec, selected_split)

    if spec["schema"] == "hotpot_like":
        docs, queries, official_gold = _prepare_hotpot_like(name, dataset, limit_questions)
    elif spec["schema"] == "musique":
        docs, queries, official_gold = _prepare_musique(name, dataset, limit_questions)
    else:
        raise ValueError(f"Unsupported schema: {spec['schema']}")
    for doc in docs.values():
        doc["split"] = selected_split
    corpus = [doc["text"] for doc in docs.values()]
    documents = list(docs.values())

    dataset_dir = output_root / name
    corpus_dir = dataset_dir / "corpus"
    query_dir = dataset_dir / "query"
    index_dir = dataset_dir / "index"
    eval_dir = dataset_dir / "eval"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    query_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    corpus_file = corpus_dir / f"{name}_unique_contexts.json"
    documents_file = corpus_dir / f"{name}_documents.json"
    query_file = query_dir / f"{name}.json"
    official_gold_file = eval_dir / f"{name}_official_gold.json"
    corpus_file.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    documents_file.write_text(json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8")
    query_file.write_text(json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8")
    if name == "musique":
        official_gold_file = eval_dir / f"{name}_official_gold.jsonl"
        official_gold_file.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in official_gold) + "\n",
            encoding="utf-8",
        )
    else:
        official_gold_file.write_text(json.dumps(official_gold, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "dataset": name,
                "hf_id": spec["hf_id"],
                "split": selected_split,
                "corpus_file": str(corpus_file),
                "documents_file": str(documents_file),
                "query_file": str(query_file),
                "official_gold_file": str(official_gold_file),
                "doc_count": len(corpus),
                "query_count": len(queries),
            },
            ensure_ascii=False,
        )
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["hotpotqa", "2wikimultihopqa", "musique"],
        choices=sorted(DATASET_SPECS),
        help="Datasets to prepare. Defaults to all three.",
    )
    parser.add_argument("--output-root", default="Datasets")
    parser.add_argument("--split", help="Override split for all datasets; defaults to each dataset's validation split.")
    parser.add_argument("--limit-questions", type=int, help="Optional small pilot subset before full graph construction.")
    return parser


def main():
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    for dataset_name in args.datasets:
        prepare_dataset(dataset_name, output_root, args.split, args.limit_questions)


if __name__ == "__main__":
    main()
