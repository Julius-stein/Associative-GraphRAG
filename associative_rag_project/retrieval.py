"""Chunk retrieval and root-scoring utilities.

Current refactor direction:
- keep chunk retrieval simple and grounded
- select diverse root chunks instead of linearly reranking them by many factors
- keep root node/edge scoring lightweight so later stages remain interpretable
"""

import base64
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import GRAPH_FIELD_SEP, lexical_overlap_score, normalize_text, safe_mean, tokenize


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


def _root_base_score(item):
    """Prefer dense grounding when available; fall back to retrieval score."""
    if item.get("dense_score_norm", 0.0) > 0:
        return float(item["dense_score_norm"])
    return float(item.get("score_norm", item.get("retrieval_score", 0.0)))


def _chunk_graph_signature(chunk_id, chunk_to_nodes, chunk_to_edges):
    nodes = set(chunk_to_nodes.get(chunk_id, set()))
    edges = set(chunk_to_edges.get(chunk_id, set()))
    return nodes, edges


def _same_doc_band(chunk_a, chunk_b, window):
    if chunk_a.get("full_doc_id") != chunk_b.get("full_doc_id"):
        return False
    order_a = chunk_a.get("chunk_order_index", -10**9)
    order_b = chunk_b.get("chunk_order_index", 10**9)
    return abs(order_a - order_b) <= window


def _provenance_overlap(nodes_a, edges_a, nodes_b, edges_b):
    left = set(nodes_a) | set(edges_a)
    right = set(nodes_b) | set(edges_b)
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left | right), 1)


def select_diverse_root_chunks(
    candidate_hits,
    chunk_store,
    chunk_to_nodes,
    chunk_to_edges,
    top_k,
    same_doc_window=1,
    max_same_doc_roots=1,
    relaxed_max_same_doc_roots=2,
    max_provenance_overlap=0.55,
    relaxed_max_provenance_overlap=0.85,
):
    """Select root chunks as diverse grounded starting points.

    The rule is intentionally simple:
    - keep the strongest dense/root retrieval anchor
    - prefer one root per document in the first pass
    - never select adjacent chunks from the same document band
    - defer graph-near-duplicate roots until the relaxed pass

    This gives us a small set of multi-start roots without reintroducing a
    large weighted rerank score.
    """
    if not candidate_hits:
        return []

    candidates = []
    for item in candidate_hits:
        chunk_id = item["chunk_id"]
        chunk = chunk_store.get(chunk_id, {})
        nodes, edges = _chunk_graph_signature(chunk_id, chunk_to_nodes, chunk_to_edges)
        candidates.append(
            {
                **item,
                "base_score": round(_root_base_score(item), 6),
                "full_doc_id": chunk.get("full_doc_id"),
                "chunk_order_index": chunk.get("chunk_order_index", -1),
                "graph_nodes": nodes,
                "graph_edges": edges,
            }
        )
    candidates.sort(key=lambda item: (-item["base_score"], -len(item["graph_nodes"]), item["chunk_id"]))

    selected = []
    deferred = []
    doc_counts = Counter()

    for candidate in candidates:
        if len(selected) >= top_k:
            break
        if not selected:
            selected.append(
                {
                    **candidate,
                    "novelty_gain": len(candidate["graph_nodes"]) + len(candidate["graph_edges"]),
                    "max_selected_overlap": 0.0,
                }
            )
            if candidate.get("full_doc_id"):
                doc_counts[candidate["full_doc_id"]] += 1
            continue

        same_doc_count = doc_counts.get(candidate.get("full_doc_id"), 0)
        same_band = any(
            _same_doc_band(
                chunk_store.get(existing["chunk_id"], {}),
                chunk_store.get(candidate["chunk_id"], {}),
                same_doc_window,
            )
            for existing in selected
        )
        overlaps = [
            _provenance_overlap(
                candidate["graph_nodes"],
                candidate["graph_edges"],
                existing["graph_nodes"],
                existing["graph_edges"],
            )
            for existing in selected
        ]
        max_overlap = max(overlaps) if overlaps else 0.0
        if same_doc_count >= max_same_doc_roots or same_band or max_overlap > max_provenance_overlap:
            deferred.append(candidate)
            continue

        selected.append(
            {
                **candidate,
                "novelty_gain": len(candidate["graph_nodes"]) + len(candidate["graph_edges"]),
                "max_selected_overlap": round(max_overlap, 6),
            }
        )
        if candidate.get("full_doc_id"):
            doc_counts[candidate["full_doc_id"]] += 1

    if len(selected) < top_k:
        deferred.sort(
            key=lambda item: (
                doc_counts.get(item.get("full_doc_id"), 0),
                max(
                    [
                        _provenance_overlap(
                            item["graph_nodes"],
                            item["graph_edges"],
                            existing["graph_nodes"],
                            existing["graph_edges"],
                        )
                        for existing in selected
                    ]
                    or [0.0]
                ),
                -item["base_score"],
                -(len(item["graph_nodes"]) + len(item["graph_edges"])),
                item["chunk_id"],
            )
        )
        for candidate in deferred:
            if len(selected) >= top_k:
                break
            same_doc_count = doc_counts.get(candidate.get("full_doc_id"), 0)
            same_band = any(
                _same_doc_band(
                    chunk_store.get(existing["chunk_id"], {}),
                    chunk_store.get(candidate["chunk_id"], {}),
                    same_doc_window,
                )
                for existing in selected
            )
            # Even in the relaxed pass, do not allow adjacent chunk roots.
            if same_band or same_doc_count >= relaxed_max_same_doc_roots:
                continue
            overlaps = [
                _provenance_overlap(
                    candidate["graph_nodes"],
                    candidate["graph_edges"],
                    existing["graph_nodes"],
                    existing["graph_edges"],
                )
                for existing in selected
            ]
            max_overlap = max(overlaps) if overlaps else 0.0
            if max_overlap > relaxed_max_provenance_overlap:
                continue
            selected.append(
                {
                    **candidate,
                    "novelty_gain": len(candidate["graph_nodes"]) + len(candidate["graph_edges"]),
                    "max_selected_overlap": round(max_overlap, 6),
                }
            )
            if candidate.get("full_doc_id"):
                doc_counts[candidate["full_doc_id"]] += 1

    top_score = max((item["base_score"] for item in selected), default=1.0)
    return [
        {
            key: value
            for key, value in {
                **item,
                "score_norm": item["base_score"] / max(top_score, 1e-9),
                "selection_score": item["base_score"],
            }.items()
            if key not in {"graph_nodes", "graph_edges"}
        }
        for item in selected
    ]


def score_root_nodes(query, root_nodes, graph, node_to_chunks, root_chunk_score_lookup):
    """Rank root nodes by support first, using query relevance only as tie-break."""
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
        scored.append(
            {
                "id": node_id,
                "score": round(support, 6),
                "query_rel": round(query_rel, 6),
                "support": round(support, 6),
                "chunk_alignment": round(chunk_alignment, 6),
                "entity_type": node_data.get("entity_type", "UNKNOWN"),
                "description": node_data.get("description", ""),
                "source_chunk_ids": sorted(node_to_chunks.get(node_id, set())),
            }
        )
    scored.sort(key=lambda item: (-item["support"], -item["chunk_alignment"], -item["query_rel"], item["id"]))
    return scored


def score_root_edges(query, root_edges, graph, edge_to_chunks, root_chunk_score_lookup):
    """Rank root edges by support first, with edge weight as secondary signal."""
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
        scored.append(
            {
                "edge": edge_id,
                "score": round(support, 6),
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
    scored.sort(key=lambda item: (-item["support"], -item["weight_term"], -item["query_rel"], item["edge"]))
    return scored
