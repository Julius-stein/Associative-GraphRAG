#!/usr/bin/env python3
"""Convert Association outputs and run official multi-hop QA evaluators.

This wrapper does not reimplement HotpotQA/2Wiki/MuSiQue metrics. It only
converts our answer/retrieval files into the prediction formats expected by
the official evaluator scripts, then executes those scripts as subprocesses.

Official scripts:
- HotpotQA: hotpotqa/hotpot/hotpot_evaluate_v1.py
- 2WikiMultiHopQA: evaluated with the same HotpotQA-style answer/support
  protocol by default, without optional relation-evidence aliases.
- MuSiQue: StonyBrookNLP/musique/evaluate_v1.0.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _answer_text(answer_record: dict[str, Any]) -> str:
    raw = str(answer_record.get("model_answer") or answer_record.get("answer") or "").strip()
    if not raw:
        return ""
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^(answer|final answer|short answer)\s*[:：]\s*", "", line, flags=re.I).strip()
        line = re.sub(r"^\*\*(answer|final answer|short answer)\*\*\s*[:：]\s*", "", line, flags=re.I).strip()
        if line:
            return line
    return raw


def _selected_chunk_ids(record: dict[str, Any]) -> list[str]:
    ordered = []

    def add(chunk_id):
        if isinstance(chunk_id, dict):
            chunk_id = chunk_id.get("chunk_id")
        if not isinstance(chunk_id, str):
            return
        if chunk_id and chunk_id not in ordered:
            ordered.append(chunk_id)

    chunk_roles = record.get("chunk_roles") or []
    if isinstance(chunk_roles, dict):
        for chunk_id in chunk_roles:
            add(chunk_id)
    else:
        for item in chunk_roles:
            add(item)
    for item in record.get("root_chunks", []) or []:
        add(item.get("chunk_id"))
    for item in record.get("promoted_root_chunks", []) or []:
        add(item.get("chunk_id"))
    for value in (record.get("theme_selected_chunks") or {}).values():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    add(item)
                elif isinstance(item, dict):
                    add(item.get("chunk_id"))
    for group in record.get("knowledge_groups", []) or record.get("facet_groups", []) or []:
        for chunk_id in group.get("supporting_chunk_ids", []) or []:
            add(chunk_id)
        for chunk_id in group.get("source_chunk_ids", []) or []:
            add(chunk_id)
    return ordered


def _load_chunk_doc_map(index_dir: Path) -> dict[str, str]:
    chunk_store = _load_json(index_dir / "kv_store_text_chunks.json")
    return {
        chunk_id: chunk.get("full_doc_id")
        for chunk_id, chunk in chunk_store.items()
        if chunk.get("full_doc_id")
    }


def _doc_maps(documents_file: Path):
    documents = _load_json(documents_file)
    by_lightrag_doc_id = {}
    for doc in documents:
        lightrag_doc_id = doc.get("lightrag_doc_id")
        if lightrag_doc_id:
            by_lightrag_doc_id[lightrag_doc_id] = doc
    return by_lightrag_doc_id


def _selected_docs(record, chunk_doc_map, docs_by_lightrag):
    docs = []
    seen = set()
    for chunk_id in _selected_chunk_ids(record):
        doc_id = chunk_doc_map.get(chunk_id)
        if not doc_id or doc_id in seen:
            continue
        doc = docs_by_lightrag.get(doc_id)
        if not doc:
            continue
        docs.append(doc)
        seen.add(doc_id)
    return docs


def _sentence_support_from_docs(docs):
    support = []
    seen = set()
    for doc in docs:
        title = doc.get("title")
        if not title:
            continue
        for sent_id, _ in enumerate(doc.get("sentences") or []):
            key = (title, sent_id)
            if key not in seen:
                support.append([title, sent_id])
                seen.add(key)
    return support


def _paragraph_support_from_docs(docs):
    support = []
    seen = set()
    for doc in docs:
        idx = doc.get("paragraph_idx")
        if idx is None or idx in seen:
            continue
        support.append(idx)
        seen.add(idx)
    return support


def _align_records_by_id(answers, retrieval):
    retrieval_by_id = {item.get("group_id"): item for item in retrieval}
    aligned = []
    for index, answer in enumerate(answers):
        group_id = answer.get("group_id")
        record = retrieval_by_id.get(group_id) if group_id else None
        if record is None and index < len(retrieval):
            record = retrieval[index]
        aligned.append((answer, record or {}))
    return aligned


def build_hotpot_or_2wiki_predictions(dataset, answers, retrieval, index_dir, documents_file, output_file):
    chunk_doc_map = _load_chunk_doc_map(index_dir)
    docs_by_lightrag = _doc_maps(documents_file)
    prediction = {"answer": {}, "sp": {}}

    for answer, record in _align_records_by_id(answers, retrieval):
        qid = str(answer.get("group_id") or record.get("group_id") or "")
        if not qid:
            continue
        docs = _selected_docs(record, chunk_doc_map, docs_by_lightrag)
        prediction["answer"][qid] = _answer_text(answer)
        prediction["sp"][qid] = _sentence_support_from_docs(docs)

    _write_json(output_file, prediction)
    return output_file


def build_musique_predictions(answers, retrieval, index_dir, documents_file, output_file):
    chunk_doc_map = _load_chunk_doc_map(index_dir)
    docs_by_lightrag = _doc_maps(documents_file)
    rows = []
    for answer, record in _align_records_by_id(answers, retrieval):
        qid = str(answer.get("group_id") or record.get("group_id") or "")
        if not qid:
            continue
        docs = _selected_docs(record, chunk_doc_map, docs_by_lightrag)
        rows.append(
            {
                "id": qid,
                "predicted_answer": _answer_text(answer),
                "predicted_support_idxs": _paragraph_support_from_docs(docs),
                "predicted_answerable": _answer_text(answer).strip().lower() not in {"", "insufficient evidence", "noanswer"},
            }
        )
    _write_jsonl(output_file, rows)
    return output_file


def _run_official(dataset, evaluator_script, prediction_file, gold_file, output_dir, aliases_file=None):
    if dataset == "musique":
        cmd = [sys.executable, str(evaluator_script), str(prediction_file), str(gold_file)]
    else:
        cmd = [sys.executable, str(evaluator_script), str(prediction_file), str(gold_file)]

    env = os.environ.copy()
    pythonpath_parts = [
        str(Path.cwd()),
        str(evaluator_script.parent),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = os.pathsep.join(part for part in pythonpath_parts if part)
    completed = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, env=env)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{dataset}_official_eval.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output_dir / f"{dataset}_official_eval.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise SystemExit(
            f"Official evaluator failed with code {completed.returncode}. "
            f"See {output_dir / f'{dataset}_official_eval.stderr.txt'}"
        )
    print(completed.stdout)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["hotpotqa", "2wikimultihopqa", "musique"])
    parser.add_argument("--answers-file", required=True)
    parser.add_argument("--retrieval-file", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--documents-file", required=True)
    parser.add_argument("--official-gold-file", required=True)
    parser.add_argument("--evaluator-script", required=True)
    parser.add_argument(
        "--aliases-file",
        help="Unused by the default unified protocol. Kept only for manual experiments with 2Wiki v1.1.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prediction-file")
    parser.add_argument("--no-run", action="store_true", help="Only write official prediction file; do not execute evaluator.")
    return parser


def main():
    args = build_parser().parse_args()
    dataset = args.dataset
    output_dir = Path(args.output_dir)
    answers = _load_json(Path(args.answers_file))
    retrieval_payload = _load_json(Path(args.retrieval_file))
    retrieval = retrieval_payload.get("results", retrieval_payload)
    prediction_file = Path(args.prediction_file) if args.prediction_file else output_dir / f"{dataset}_official_predictions"

    if dataset == "musique":
        prediction_file = prediction_file.with_suffix(".jsonl")
        build_musique_predictions(
            answers,
            retrieval,
            Path(args.index_dir),
            Path(args.documents_file),
            prediction_file,
        )
    else:
        prediction_file = prediction_file.with_suffix(".json")
        build_hotpot_or_2wiki_predictions(
            dataset,
            answers,
            retrieval,
            Path(args.index_dir),
            Path(args.documents_file),
            prediction_file,
        )
    print(prediction_file)
    if not args.no_run:
        _run_official(
            dataset,
            Path(args.evaluator_script),
            prediction_file,
            Path(args.official_gold_file),
            output_dir,
            aliases_file=Path(args.aliases_file) if args.aliases_file else None,
        )


if __name__ == "__main__":
    main()
