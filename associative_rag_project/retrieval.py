"""Chunk retrieval and root-scoring utilities.

This module first retrieves a small pool of root chunks, then reranks chunk-,
node-, and edge-level candidates with cheap query-aware signals.
"""

import base64
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import (
    GRAPH_FIELD_SEP,
    lexical_overlap_score,
    normalize_text,
    query_prefers_technical_content,
    safe_mean,
    technical_density,
    tokenize,
)


@dataclass
class BM25Index:
    postings: dict
    doc_lengths: dict
    avgdl: float
    num_docs: int
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, chunk_store):
        """Build an in-memory BM25 index from the chunk text store."""
        postings = defaultdict(dict)
        doc_lengths = {}
        for chunk_id, chunk_data in chunk_store.items():
            tokens = tokenize(chunk_data.get("content", ""))
            doc_lengths[chunk_id] = len(tokens)
            tf = Counter(tokens)
            for term, freq in tf.items():
                postings[term][chunk_id] = freq
        avgdl = sum(doc_lengths.values()) / max(len(doc_lengths), 1)
        return cls(
            postings=dict(postings),
            doc_lengths=doc_lengths,
            avgdl=avgdl,
            num_docs=len(doc_lengths),
        )

    def search(self, query, top_k):
        """Return the top-k lexical matches with normalized scores."""
        query_terms = tokenize(query)
        if not query_terms:
            return []
        scores = defaultdict(float)
        query_tf = Counter(query_terms)
        for term, qtf in query_tf.items():
            postings = self.postings.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5))
            for chunk_id, tf in postings.items():
                dl = self.doc_lengths[chunk_id]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                scores[chunk_id] += qtf * idf * ((tf * (self.k1 + 1)) / max(denom, 1e-9))
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if not ranked:
            return []
        top_score = ranked[0][1]
        return [
            {
                "chunk_id": chunk_id,
                "score": score,
                "score_norm": score / max(top_score, 1e-9),
            }
            for chunk_id, score in ranked[:top_k]
        ]


@dataclass
class DenseChunkIndex:
    chunk_ids: list[str]
    matrix: np.ndarray
    normalized_matrix: np.ndarray

    @classmethod
    def load(cls, vdb_file):
        """Load a precomputed dense chunk matrix from the LightRAG vector store."""
        payload = json.loads(Path(vdb_file).read_text(encoding="utf-8"))
        chunk_ids = [item["__id__"] for item in payload.get("data", [])]
        dim = int(payload["embedding_dim"])
        raw = base64.b64decode(payload.get("matrix", ""))
        if not chunk_ids or not raw:
            matrix = np.zeros((0, dim), dtype=np.float32)
            return cls(chunk_ids=chunk_ids, matrix=matrix, normalized_matrix=matrix)
        matrix = np.frombuffer(raw, dtype=np.float32).reshape(len(chunk_ids), dim)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalized_matrix = matrix / np.clip(norms, 1e-12, None)
        return cls(chunk_ids=chunk_ids, matrix=matrix, normalized_matrix=normalized_matrix)

    def search(self, query_vector, top_k):
        """Cosine similarity search over the normalized chunk matrix."""
        if self.normalized_matrix.size == 0:
            return []
        query_vector = np.asarray(query_vector, dtype=np.float32)
        query_vector = query_vector / max(np.linalg.norm(query_vector), 1e-12)
        scores = self.normalized_matrix @ query_vector
        top_indices = np.argsort(-scores)[:top_k]
        top_score = float(scores[top_indices[0]]) if len(top_indices) else 1.0
        return [
            {
                "chunk_id": self.chunk_ids[index],
                "dense_score": float(scores[index]),
                "dense_score_norm": float(scores[index] / max(top_score, 1e-9)),
            }
            for index in top_indices
        ]


@dataclass
class HybridChunkRetriever:
    bm25_index: BM25Index
    dense_index: DenseChunkIndex | None
    embedding_client: object | None
    mode: str = "hybrid"
    dense_weight: float = 0.75
    bm25_weight: float = 0.25

    def search(self, query, top_k):
        """Fuse bm25 and dense signals into one ranked candidate list."""
        if self.mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError(f"Unsupported retrieval mode: {self.mode}")
        bm25_hits = self.bm25_index.search(query, top_k=top_k) if self.mode in {"bm25", "hybrid"} else []
        dense_hits = []
        if self.mode in {"dense", "hybrid"}:
            if self.dense_index is None or self.embedding_client is None:
                raise ValueError("Dense retrieval requested but dense index or embedding client is unavailable")
            query_vector = self.embedding_client.embed_text(query)
            dense_hits = self.dense_index.search(query_vector, top_k=top_k)
        merged = {}
        for item in bm25_hits:
            merged.setdefault(item["chunk_id"], {}).update(item)
        for item in dense_hits:
            merged.setdefault(item["chunk_id"], {}).update(item)
        if not merged:
            return []
        scored = []
        for chunk_id, item in merged.items():
            bm25_score_norm = item.get("score_norm", 0.0)
            dense_score_norm = item.get("dense_score_norm", 0.0)
            if self.mode == "bm25":
                retrieval_score = bm25_score_norm
            elif self.mode == "dense":
                retrieval_score = dense_score_norm
            else:
                retrieval_score = self.dense_weight * dense_score_norm + self.bm25_weight * bm25_score_norm
            scored.append(
                {
                    "chunk_id": chunk_id,
                    "score": retrieval_score,
                    "score_norm": retrieval_score,
                    "bm25_score": item.get("score", 0.0),
                    "bm25_score_norm": bm25_score_norm,
                    "dense_score": item.get("dense_score", 0.0),
                    "dense_score_norm": dense_score_norm,
                    "retrieval_score": retrieval_score,
                }
            )
        scored.sort(
            key=lambda item: (
                -item["retrieval_score"],
                -item["dense_score_norm"],
                -item["bm25_score_norm"],
                item["chunk_id"],
            )
        )
        top_score = scored[0]["retrieval_score"] if scored else 1.0
        return [{**item, "score_norm": item["retrieval_score"] / max(top_score, 1e-9)} for item in scored[:top_k]]


def support_score(source_chunks, root_chunk_score_lookup):
    """How well does a candidate align with the current root evidence pool?"""
    if not source_chunks:
        return 0.0, 0.0
    overlapping = [root_chunk_score_lookup[cid] for cid in source_chunks if cid in root_chunk_score_lookup]
    overlap_count = len(overlapping)
    support = overlap_count / max(len(root_chunk_score_lookup), 1)
    chunk_alignment = safe_mean(overlapping)
    return support, chunk_alignment


def normalize_relation_category(edge_data):
    """Collapse noisy edge metadata into a coarse relation label."""
    raw = normalize_text(edge_data.get("keywords") or edge_data.get("description") or "unknown_relation")
    parts = [part.strip().lower() for part in raw.split(GRAPH_FIELD_SEP) if part.strip()]
    if not parts:
        return "unknown_relation"
    first = parts[0]
    first = re.sub(r"[^a-z0-9 ]+", " ", first)
    first = " ".join(first.split())
    return first or "unknown_relation"


def relation_entropy(categories):
    """A cheap diversity proxy used by semantic association."""
    if not categories:
        return 0.0
    counter = Counter(categories)
    total = sum(counter.values())
    entropy = 0.0
    for count in counter.values():
        prob = count / total
        entropy -= prob * math.log(prob + 1e-12)
    return entropy


def rerank_root_chunks(query, candidate_hits, chunk_store, chunk_to_nodes, chunk_to_edges, top_k):
    """Rerank retrieved chunks before they become graph roots.

    Dense/bm25 retrieval gets us in the right neighborhood. This second stage
    rewards chunks that are both textually relevant and graph-rich enough to
    seed useful association later.
    """
    if not candidate_hits:
        return []
    query_is_technical = query_prefers_technical_content(query)
    graph_sizes = []
    for item in candidate_hits:
        chunk_id = item["chunk_id"]
        graph_sizes.append(len(chunk_to_nodes.get(chunk_id, set())) + 2 * len(chunk_to_edges.get(chunk_id, set())))
    max_graph_size = max(graph_sizes) if graph_sizes else 1
    scored = []
    for item in candidate_hits:
        chunk_id = item["chunk_id"]
        chunk = chunk_store.get(chunk_id, {})
        query_rel = lexical_overlap_score(query, chunk.get("content", ""))
        graph_size = len(chunk_to_nodes.get(chunk_id, set())) + 2 * len(chunk_to_edges.get(chunk_id, set()))
        graph_yield = graph_size / max(max_graph_size, 1)
        dense_term = item.get("dense_score_norm", 0.0)
        bm25_term = item.get("bm25_score_norm", 0.0)
        technical_penalty = 0.0 if query_is_technical else technical_density(chunk.get("content", "")) * 0.20
        rerank_score = (
            0.35 * item["score_norm"]
            + 0.25 * query_rel
            + 0.15 * graph_yield
            + 0.15 * dense_term
            + 0.10 * bm25_term
            - technical_penalty
        )
        scored.append(
            {
                **item,
                "query_rel": round(query_rel, 6),
                "graph_yield": round(graph_yield, 6),
                "technical_penalty": round(technical_penalty, 6),
                "rerank_score": round(rerank_score, 6),
            }
        )
    scored.sort(key=lambda item: (-item["rerank_score"], -item["score_norm"], item["chunk_id"]))
    top_score = scored[0]["rerank_score"] if scored else 1.0
    reranked = []
    for item in scored[:top_k]:
        reranked.append({**item, "score_norm": item["rerank_score"] / max(top_score, 1e-9)})
    return reranked


def score_root_nodes(query, root_nodes, graph, node_to_chunks, root_chunk_score_lookup):
    """Rank root nodes so structural association starts from strong anchors."""
    scored = []
    for node_id in root_nodes:
        node_data = graph.nodes[node_id]
        query_rel = lexical_overlap_score(
            query,
            " ".join(
                [
                    normalize_text(node_id),
                    normalize_text(node_data.get("entity_type", "")),
                    normalize_text(node_data.get("description", "")),
                ]
            ),
        )
        support, chunk_alignment = support_score(node_to_chunks.get(node_id, set()), root_chunk_score_lookup)
        score = 0.55 * query_rel + 0.25 * support + 0.20 * chunk_alignment
        scored.append(
            {
                "id": node_id,
                "score": round(score, 6),
                "query_rel": round(query_rel, 6),
                "support": round(support, 6),
                "chunk_alignment": round(chunk_alignment, 6),
                "entity_type": node_data.get("entity_type", "UNKNOWN"),
                "description": node_data.get("description", ""),
                "source_chunk_ids": sorted(node_to_chunks.get(node_id, set())),
            }
        )
    scored.sort(key=lambda item: (-item["score"], item["id"]))
    return scored


def score_root_edges(query, root_edges, graph, edge_to_chunks, root_chunk_score_lookup):
    """Rank root edges as relation-first anchors for later expansion."""
    scored = []
    for edge_id in root_edges:
        edge_data = graph.get_edge_data(edge_id[0], edge_id[1]) or {}
        edge_text = " ".join(
            [
                normalize_text(edge_id[0]),
                normalize_text(edge_id[1]),
                normalize_text(edge_data.get("keywords", "")),
                normalize_text(edge_data.get("description", "")),
            ]
        )
        query_rel = lexical_overlap_score(query, edge_text)
        support, chunk_alignment = support_score(edge_to_chunks.get(edge_id, set()), root_chunk_score_lookup)
        weight_term = math.log1p(float(edge_data.get("weight", 0.0) or 0.0)) / 5.0
        score = 0.50 * query_rel + 0.20 * support + 0.15 * chunk_alignment + 0.15 * weight_term
        scored.append(
            {
                "edge": edge_id,
                "score": round(score, 6),
                "query_rel": round(query_rel, 6),
                "support": round(support, 6),
                "chunk_alignment": round(chunk_alignment, 6),
                "weight_term": round(weight_term, 6),
                "keywords": edge_data.get("keywords", ""),
                "description": edge_data.get("description", ""),
                "weight": edge_data.get("weight", 0.0),
                "source_chunk_ids": sorted(edge_to_chunks.get(edge_id, set())),
            }
        )
    scored.sort(key=lambda item: (-item["score"], item["edge"]))
    return scored
