"""Project-native corpus index builder.

This module replaces the previous dependency on the ignored `lightrag/`
directory for preprocessing. It writes the subset of LightRAG-compatible files
used by the Association pipeline:

- kv_store_full_docs.json
- kv_store_text_chunks.json
- graph_chunk_entity_relation.graphml
- vdb_chunks.json

The online retriever only needs chunk text, chunk embeddings, and a graph whose
nodes/edges carry `source_id` provenance. Entity extraction still uses the same
record format as LightRAG, but the implementation lives in this project.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from hashlib import md5
from pathlib import Path

import networkx as nx
import numpy as np
from openai import OpenAI

from .config import load_llm_config
from .embedding_client import build_embedding_client
from .logging_utils import log


GRAPH_FIELD_SEP = "<SEP>"
TUPLE_DELIMITER = "<|>"
RECORD_DELIMITER = "##"
COMPLETION_DELIMITER = "<|COMPLETE|>"
DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event", "location", "concept"]


ENTITY_EXTRACTION_PROMPT = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are clearly related to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why the source entity and the target entity are related
- relationship_keywords: one or more high-level keywords that summarize the relationship
- relationship_strength: a numeric score indicating relationship strength
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use {record_delimiter} as the list delimiter.

4. When finished, output {completion_delimiter}

-Real Data-
Entity_types: {entity_types}
Text:
{input_text}
Output:
"""

CONTINUE_EXTRACTION_PROMPT = "MANY entities were missed in the last extraction. Add them below using the same format:"
IF_LOOP_EXTRACTION_PROMPT = "It appears some entities may have still been missed. Answer YES | NO if there are still entities that need to be added."


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + md5(content.encode("utf-8")).hexdigest()


def _read_corpus(corpus_file: Path) -> list[str]:
    payload = json.loads(corpus_file.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise ValueError(f"Expected corpus JSON to be a list of strings: {corpus_file}")
    docs = [item.strip() for item in payload if item and item.strip()]
    if not docs:
        raise ValueError(f"Corpus file contains no non-empty documents: {corpus_file}")
    return docs


def _encode_tokens(text: str, model_name: str):
    try:
        import tiktoken
    except ImportError as exc:
        raise ImportError("Index building requires `tiktoken` for LightRAG-compatible chunking.") from exc
    return tiktoken.encoding_for_model(model_name).encode(text)


def _decode_tokens(tokens: list[int], model_name: str) -> str:
    import tiktoken

    return tiktoken.encoding_for_model(model_name).decode(tokens)


def chunk_text(content: str, *, max_tokens: int, overlap_tokens: int, model_name: str) -> list[dict]:
    if max_tokens <= overlap_tokens:
        raise ValueError("--chunk-token-size must be larger than --chunk-overlap-token-size")
    tokens = _encode_tokens(content, model_name)
    chunks = []
    stride = max_tokens - overlap_tokens
    for index, start in enumerate(range(0, len(tokens), stride)):
        token_slice = tokens[start : start + max_tokens]
        if not token_slice:
            continue
        chunks.append(
            {
                "tokens": len(token_slice),
                "content": _decode_tokens(token_slice, model_name).strip(),
                "chunk_order_index": index,
            }
        )
    return [chunk for chunk in chunks if chunk["content"]]


def build_doc_and_chunk_store(
    corpus: list[str],
    *,
    chunk_token_size: int,
    chunk_overlap_token_size: int,
    tiktoken_model_name: str,
) -> tuple[dict, dict]:
    full_docs = {
        compute_mdhash_id(doc, prefix="doc-"): {"content": doc}
        for doc in corpus
    }
    text_chunks = {}
    for doc_id, doc in full_docs.items():
        for chunk in chunk_text(
            doc["content"],
            max_tokens=chunk_token_size,
            overlap_tokens=chunk_overlap_token_size,
            model_name=tiktoken_model_name,
        ):
            chunk_id = compute_mdhash_id(chunk["content"], prefix="chunk-")
            text_chunks.setdefault(chunk_id, {**chunk, "full_doc_id": doc_id})
    if not text_chunks:
        raise ValueError("Chunking produced no chunks.")
    return full_docs, text_chunks


def _chat_completion(config: dict, messages: list[dict], *, max_tokens: int | None = None) -> str:
    client = OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
        timeout=config.get("timeout", 120),
        max_retries=config.get("max_retries", 3),
    )
    response = client.chat.completions.create(
        model=config["model"],
        temperature=0.0,
        max_tokens=max_tokens or config.get("index_llm_max_tokens", 2200),
        messages=messages,
    )
    return response.choices[0].message.content or ""


def _format_endpoint(config: dict) -> str:
    return f"model={config.get('model')} base_url={config.get('base_url')}"


def _preflight_chat_completion(config: dict):
    log(f"[index] chat preflight start {_format_endpoint(config)} timeout={config.get('timeout', 120)}")
    started = time.perf_counter()
    reply = _chat_completion(
        config,
        [{"role": "user", "content": "Reply with exactly: OK"}],
        max_tokens=8,
    ).strip()
    elapsed = time.perf_counter() - started
    if "ok" not in reply.lower():
        raise RuntimeError(f"Chat preflight returned an unexpected response: {reply!r}")
    log(f"[index] chat preflight ok seconds={elapsed:.2f}")


def _clean_str(value) -> str:
    text = str(value or "").strip()
    text = text.replace("\u200b", "")
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_multi(text: str, markers: list[str]) -> list[str]:
    if not markers:
        return [text]
    pattern = "|".join(re.escape(marker) for marker in markers)
    return [part.strip() for part in re.split(pattern, text) if part.strip()]


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _parse_extraction_records(raw: str, chunk_id: str) -> tuple[dict[str, list], dict[tuple[str, str], list]]:
    nodes = defaultdict(list)
    edges = defaultdict(list)
    records = _split_multi(raw, [RECORD_DELIMITER, COMPLETION_DELIMITER])
    for item in records:
        match = re.search(r"\((.*)\)", item, flags=re.DOTALL)
        if not match:
            continue
        fields = _split_multi(match.group(1), [TUPLE_DELIMITER])
        if not fields:
            continue
        record_type = _clean_str(fields[0]).lower()
        if record_type == "entity" and len(fields) >= 4:
            entity_name = _clean_str(fields[1]).upper()
            if not entity_name:
                continue
            nodes[entity_name].append(
                {
                    "entity_name": entity_name,
                    "entity_type": _clean_str(fields[2]).upper() or "UNKNOWN",
                    "description": _clean_str(fields[3]),
                    "source_id": chunk_id,
                }
            )
        elif record_type == "relationship" and len(fields) >= 6:
            src_id = _clean_str(fields[1]).upper()
            tgt_id = _clean_str(fields[2]).upper()
            if not src_id or not tgt_id or src_id == tgt_id:
                continue
            weight_raw = _clean_str(fields[-1])
            edges[tuple(sorted((src_id, tgt_id)))].append(
                {
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "description": _clean_str(fields[3]),
                    "keywords": _clean_str(fields[4]),
                    "weight": float(weight_raw) if _is_float(weight_raw) else 1.0,
                    "source_id": chunk_id,
                }
            )
    return dict(nodes), dict(edges)


def _extract_chunk_entities(
    chunk_id: str,
    chunk: dict,
    *,
    config: dict,
    max_gleaning: int,
) -> dict:
    started = time.perf_counter()
    prompt = ENTITY_EXTRACTION_PROMPT.format(
        tuple_delimiter=TUPLE_DELIMITER,
        record_delimiter=RECORD_DELIMITER,
        completion_delimiter=COMPLETION_DELIMITER,
        entity_types=", ".join(DEFAULT_ENTITY_TYPES),
        input_text=chunk["content"],
    )
    messages = [{"role": "user", "content": prompt}]
    final_result = _chat_completion(config, messages)
    messages.append({"role": "assistant", "content": final_result})

    for glean_index in range(max(max_gleaning, 0)):
        messages.append({"role": "user", "content": CONTINUE_EXTRACTION_PROMPT})
        glean_result = _chat_completion(config, messages)
        messages.append({"role": "assistant", "content": glean_result})
        final_result += "\n" + glean_result
        if glean_index == max_gleaning - 1:
            break
        messages.append({"role": "user", "content": IF_LOOP_EXTRACTION_PROMPT})
        should_continue = _chat_completion(config, messages, max_tokens=8).strip().strip("\"'").lower()
        messages.append({"role": "assistant", "content": should_continue})
        if should_continue != "yes":
            break

    nodes, edges = _parse_extraction_records(final_result, chunk_id)
    return {
        "chunk_id": chunk_id,
        "raw": final_result,
        "nodes": nodes,
        "edges": {f"{key[0]}{GRAPH_FIELD_SEP}{key[1]}": value for key, value in edges.items()},
        "seconds": round(time.perf_counter() - started, 3),
    }


def _load_extraction_cache(cache_file: Path) -> dict:
    if not cache_file.exists():
        return {}
    cache = {}
    for line in cache_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        cache[item["chunk_id"]] = item
    return cache


def extract_graph_records(
    text_chunks: dict,
    *,
    config: dict,
    index_dir: Path,
    max_workers: int,
    max_gleaning: int,
    force_reextract: bool,
) -> list[dict]:
    cache_file = index_dir / "chunk_extractions.jsonl"
    cache = {} if force_reextract else _load_extraction_cache(cache_file)
    missing = [(chunk_id, chunk) for chunk_id, chunk in text_chunks.items() if chunk_id not in cache]
    log(f"[index] entity extraction chunks={len(text_chunks)} cached={len(cache)} missing={len(missing)}")

    if missing:
        log(
            f"[index] entity extraction run {_format_endpoint(config)} "
            f"workers={max(max_workers, 1)} max_gleaning={max_gleaning} "
            f"timeout={config.get('timeout', 120)}"
        )
        with cache_file.open("a", encoding="utf-8") as handle:
            with ThreadPoolExecutor(max_workers=max(max_workers, 1)) as executor:
                missing_iter = iter(missing)
                futures = {}
                submitted = 0

                def submit_next():
                    nonlocal submitted
                    try:
                        chunk_id, chunk = next(missing_iter)
                    except StopIteration:
                        return False
                    future = executor.submit(
                        _extract_chunk_entities,
                        chunk_id,
                        chunk,
                        config=config,
                        max_gleaning=max_gleaning,
                    )
                    submitted += 1
                    futures[future] = chunk_id
                    if submitted <= max(max_workers, 1):
                        log(f"[index] extraction submitted {submitted}/{len(missing)} chunk={chunk_id}")
                    return True

                for _ in range(max(max_workers, 1)):
                    if not submit_next():
                        break

                completed = 0
                while futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        chunk_id = futures.pop(future)
                        try:
                            item = future.result()
                        except Exception as exc:
                            log(
                                f"[index] extraction failed chunk={chunk_id} "
                                f"{_format_endpoint(config)} error={exc.__class__.__name__}: {exc}"
                            )
                            raise RuntimeError(f"Entity extraction failed for chunk {chunk_id}") from exc
                        cache[chunk_id] = item
                        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                        handle.flush()
                        completed += 1
                        log(
                            f"[index] extracted {completed}/{len(missing)} chunk={chunk_id} "
                            f"nodes={len(item['nodes'])} edges={len(item['edges'])} "
                            f"seconds={item.get('seconds')}"
                        )
                        submit_next()
    return [cache[chunk_id] for chunk_id in text_chunks if chunk_id in cache]


def _merge_graph(extraction_records: list[dict]) -> nx.Graph:
    node_buckets = defaultdict(list)
    edge_buckets = defaultdict(list)
    for item in extraction_records:
        for node_id, rows in item.get("nodes", {}).items():
            node_buckets[node_id].extend(rows)
        for edge_key, rows in item.get("edges", {}).items():
            src_id, tgt_id = edge_key.split(GRAPH_FIELD_SEP, 1)
            edge_buckets[tuple(sorted((src_id, tgt_id)))].extend(rows)

    graph = nx.Graph()
    for node_id, rows in node_buckets.items():
        entity_type = Counter(row.get("entity_type", "UNKNOWN") for row in rows).most_common(1)[0][0]
        descriptions = sorted({row.get("description", "") for row in rows if row.get("description")})
        source_ids = sorted({row.get("source_id", "") for row in rows if row.get("source_id")})
        graph.add_node(
            node_id,
            entity_type=f'"{entity_type}"',
            description=GRAPH_FIELD_SEP.join(f'"{item}"' for item in descriptions),
            source_id=GRAPH_FIELD_SEP.join(source_ids),
        )

    for (src_id, tgt_id), rows in edge_buckets.items():
        descriptions = sorted({row.get("description", "") for row in rows if row.get("description")})
        keywords = sorted({row.get("keywords", "") for row in rows if row.get("keywords")})
        source_ids = sorted({row.get("source_id", "") for row in rows if row.get("source_id")})
        if not graph.has_node(src_id):
            graph.add_node(src_id, entity_type='"UNKNOWN"', description="", source_id=GRAPH_FIELD_SEP.join(source_ids))
        if not graph.has_node(tgt_id):
            graph.add_node(tgt_id, entity_type='"UNKNOWN"', description="", source_id=GRAPH_FIELD_SEP.join(source_ids))
        graph.add_edge(
            src_id,
            tgt_id,
            weight=float(sum(float(row.get("weight", 1.0)) for row in rows)),
            description=GRAPH_FIELD_SEP.join(f'"{item}"' for item in descriptions),
            keywords=GRAPH_FIELD_SEP.join(f'"{item}"' for item in keywords),
            source_id=GRAPH_FIELD_SEP.join(source_ids),
        )
    if graph.number_of_nodes() == 0:
        raise ValueError("Entity extraction produced an empty graph.")
    return graph


def _write_chunk_vectors(text_chunks: dict, index_dir: Path, config: dict, *, force_reembed: bool = False):
    output_file = index_dir / "vdb_chunks.json"
    if output_file.exists() and not force_reembed:
        log(f"[index] reuse existing chunk embeddings file={output_file}")
        return

    embedding_client = build_embedding_client(config)
    chunk_ids = list(text_chunks)
    contents = [text_chunks[chunk_id]["content"] for chunk_id in chunk_ids]
    batch_size = int(config.get("index_embedding_batch_size", config.get("embedding_batch_size", 16)))
    vectors = []
    for start in range(0, len(contents), batch_size):
        batch = contents[start : start + batch_size]
        vectors.extend(embedding_client.embed_texts(batch))
        log(f"[index] embedded chunks {min(start + len(batch), len(contents))}/{len(contents)}")
    matrix = np.asarray(vectors, dtype=np.float32)
    expected_dim = int(config.get("embedding_dim", matrix.shape[1]))
    if matrix.ndim != 2 or matrix.shape[1] != expected_dim:
        raise ValueError(
            "Chunk embedding dimension does not match configured embedding_dim: "
            f"matrix_shape={matrix.shape}, embedding_dim={expected_dim}"
        )
    payload = {
        "embedding_dim": expected_dim,
        "data": [
            {
                "__id__": chunk_id,
                "__vector__": matrix[index].tolist(),
            }
            for index, chunk_id in enumerate(chunk_ids)
        ],
    }
    output_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def build_index(args) -> dict:
    started = time.time()
    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    config = load_llm_config()
    if args.llm_model:
        config["model"] = args.llm_model
    if args.llm_base_url:
        config["base_url"] = args.llm_base_url
    if args.llm_api_key:
        config["api_key"] = args.llm_api_key
    if args.llm_timeout:
        config["timeout"] = args.llm_timeout
    if args.llm_max_retries is not None:
        config["max_retries"] = args.llm_max_retries
    if args.embedding_model:
        config["embedding_model"] = args.embedding_model
    if args.embedding_base_url:
        config["embedding_base_url"] = args.embedding_base_url
    if args.embedding_api_key:
        config["embedding_api_key"] = args.embedding_api_key
    if args.embedding_dim:
        config["embedding_dim"] = args.embedding_dim

    corpus = _read_corpus(Path(args.corpus_file))
    full_docs, text_chunks = build_doc_and_chunk_store(
        corpus,
        chunk_token_size=args.chunk_token_size,
        chunk_overlap_token_size=args.chunk_overlap_token_size,
        tiktoken_model_name=args.tiktoken_model_name,
    )
    log(f"[index] docs={len(full_docs)} chunks={len(text_chunks)}")

    (index_dir / "kv_store_full_docs.json").write_text(json.dumps(full_docs, ensure_ascii=False, indent=2), encoding="utf-8")
    (index_dir / "kv_store_text_chunks.json").write_text(json.dumps(text_chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    _preflight_chat_completion(config)
    _write_chunk_vectors(text_chunks, index_dir, config, force_reembed=args.force_reembed)
    extraction_records = extract_graph_records(
        text_chunks,
        config=config,
        index_dir=index_dir,
        max_workers=args.max_workers,
        max_gleaning=args.entity_extract_max_gleaning,
        force_reextract=args.force_reextract,
    )
    graph = _merge_graph(extraction_records)
    nx.write_graphml(graph, index_dir / "graph_chunk_entity_relation.graphml")

    manifest = {
        "corpus_file": str(args.corpus_file),
        "index_dir": str(index_dir),
        "doc_count": len(full_docs),
        "chunk_count": len(text_chunks),
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "llm_model": config.get("model"),
        "embedding_model": config.get("embedding_model"),
        "embedding_base_url": config.get("embedding_base_url"),
        "embedding_dim": int(config.get("embedding_dim", 0)),
        "chunk_token_size": args.chunk_token_size,
        "chunk_overlap_token_size": args.chunk_overlap_token_size,
        "seconds": round(time.time() - started, 3),
        "builder": "associative_rag_project.index_builder",
    }
    (index_dir / "index_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-file", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-api-key")
    parser.add_argument("--llm-timeout", type=float)
    parser.add_argument("--llm-max-retries", type=int)
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-base-url")
    parser.add_argument("--embedding-api-key")
    parser.add_argument("--embedding-dim", type=int)
    parser.add_argument("--chunk-token-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap-token-size", type=int, default=100)
    parser.add_argument("--tiktoken-model-name", default="gpt-4o-mini")
    parser.add_argument("--entity-extract-max-gleaning", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--force-reembed", action="store_true")
    parser.add_argument("--force-reextract", action="store_true")
    return parser


def main():
    manifest = build_index(build_parser().parse_args())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
