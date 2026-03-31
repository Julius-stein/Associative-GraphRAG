"""Data loading helpers for corpora, questions, baselines, and provenance maps."""

import json
import re
from pathlib import Path

import networkx as nx

from .common import edge_key, parse_source_ids


def extract_questions(file_path: Path) -> list[str]:
    """Read the benchmark question file format used in this repo."""
    data = file_path.read_text(encoding="utf-8").replace("**", "")
    matches = re.findall(r"- Question \d+: (.+)", data)
    if matches:
        return matches
    return [line.strip() for line in data.splitlines() if line.strip()]


def load_query_rows(
    rewrites_file: Path | None,
    questions_file: Path | None,
    limit_groups: int | None,
):
    """Normalize either rewrite-groups or plain question lists into query rows."""
    if rewrites_file is not None:
        groups = json.loads(rewrites_file.read_text(encoding="utf-8"))
        selected_groups = groups if limit_groups is None else groups[:limit_groups]
        rows = []
        for group in selected_groups:
            base_variant = next(
                (variant for variant in group["variants"] if variant["variant_id"] == "base"),
                group["variants"][0],
            )
            rows.append(
                {
                    "group_id": group["group_id"],
                    "variant_id": base_variant["variant_id"],
                    "query": base_variant["query"],
                    "base_query": group["base_query"],
                }
            )
        return rows
    if questions_file is None:
        raise ValueError("Either rewrites_file or questions_file must be provided")
    questions = extract_questions(questions_file)
    if limit_groups is not None:
        questions = questions[:limit_groups]
    return [
        {
            "group_id": f"q{index:03d}",
            "variant_id": "base",
            "query": query,
            "base_query": query,
        }
        for index, query in enumerate(questions, start=1)
    ]


def resolve_questions_file(corpus_name: str, explicit_path: str | None):
    """Find the default question file for a corpus unless the user overrides it."""
    if explicit_path:
        return Path(explicit_path)
    candidate = Path("datasets/questions") / f"{corpus_name}_questions.txt"
    if candidate.exists():
        return candidate
    return None


def resolve_baseline_file(corpus_name: str, explicit_path: str | None):
    """Default to the FG-RAG 4o-mini outputs for head-to-head comparison."""
    if explicit_path:
        return Path(explicit_path)
    candidates = [
        Path("FG-RAG") / f"{corpus_name}" / "output" / f"FG-RAG-4o-mini.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_graph_corpus(corpus_dir: Path):
    """Load the graph and chunk store produced by the LightRAG preprocessing stage."""
    graph = nx.read_graphml(corpus_dir / "graph_chunk_entity_relation.graphml")
    chunk_store = json.loads((corpus_dir / "kv_store_text_chunks.json").read_text(encoding="utf-8"))
    return graph, chunk_store


def build_chunk_mappings(graph, chunk_ids):
    """Build the provenance maps that connect chunks to nodes/edges and back."""
    chunk_id_set = set(chunk_ids)
    chunk_to_nodes = {chunk_id: set() for chunk_id in chunk_ids}
    chunk_to_edges = {chunk_id: set() for chunk_id in chunk_ids}
    node_to_chunks = {}
    edge_to_chunks = {}

    for node_id, node_data in graph.nodes(data=True):
        source_ids = set(parse_source_ids(node_data.get("source_id", "")))
        node_to_chunks[node_id] = source_ids
        for chunk_id in source_ids:
            if chunk_id in chunk_id_set:
                chunk_to_nodes[chunk_id].add(node_id)

    for src_id, tgt_id, edge_data in graph.edges(data=True):
        ek = edge_key(src_id, tgt_id)
        source_ids = set(parse_source_ids(edge_data.get("source_id", "")))
        edge_to_chunks[ek] = source_ids
        for chunk_id in source_ids:
            if chunk_id in chunk_id_set:
                chunk_to_edges[chunk_id].add(ek)
    return chunk_to_nodes, chunk_to_edges, node_to_chunks, edge_to_chunks


def load_baseline_answers(result_path: Path):
    """Load baseline outputs in the same list format used by our answers."""
    data = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in baseline file: {result_path}")
    return data
