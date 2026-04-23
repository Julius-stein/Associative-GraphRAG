"""Chunk retrieval and root-scoring utilities.

Current refactor direction:
- keep chunk retrieval simple and grounded
- select diverse root chunks instead of linearly reranking them by many factors
- keep root node/edge scoring lightweight so later stages remain interpretable

检索与根 chunk 打分模块，负责多通道候选采集与多样性锚点选择。
"""

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import GRAPH_FIELD_SEP, STOPWORDS, lexical_overlap_score, normalize_text, safe_mean, tokenize


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
        """Build an in-memory BM25 index from the chunk text store.

        从 chunk 文本构建 BM25 倒排索引，用于快速词项检索。
        """
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

    def search(self, query, top_k, allowed_chunk_ids=None):
        """Return the top-k lexical matches with normalized scores."""
        query_terms = tokenize(query)
        if not query_terms:
            return []
        allowed_chunk_ids = set(allowed_chunk_ids or [])
        scores = defaultdict(float)
        query_tf = Counter(query_terms)
        for term, qtf in query_tf.items():
            postings = self.postings.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5))
            for chunk_id, tf in postings.items():
                if allowed_chunk_ids and chunk_id not in allowed_chunk_ids:
                    continue
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
        """Load a precomputed dense chunk matrix from the LightRAG vector store.

        从向量数据库文件加载预先计算的 chunk 嵌入矩阵。
        """
        payload = json.loads(Path(vdb_file).read_text(encoding="utf-8"))
        rows = payload.get("data", [])
        chunk_ids = [item["__id__"] for item in rows]
        dim = int(payload["embedding_dim"])
        if not chunk_ids:
            matrix = np.zeros((0, dim), dtype=np.float32)
            return cls(chunk_ids=chunk_ids, matrix=matrix, normalized_matrix=matrix)
        if not all("__vector__" in item for item in rows):
            raise ValueError(
                "Unsupported vdb_chunks.json format. Rebuild the index with "
                "`python -m associative_rag_project.index_builder` so chunk embeddings "
                "are stored as OpenAI-compatible float vectors under data[*].__vector__."
            )
        matrix = np.asarray([item["__vector__"] for item in rows], dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != dim:
            raise ValueError(
                "Dense chunk vector shape does not match vdb_chunks.json embedding_dim: "
                f"matrix_shape={matrix.shape}, embedding_dim={dim}"
            )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalized_matrix = matrix / np.clip(norms, 1e-12, None)
        return cls(chunk_ids=chunk_ids, matrix=matrix, normalized_matrix=normalized_matrix)

    def search(self, query_vector, top_k, allowed_chunk_ids=None):
        """Cosine similarity search over the normalized chunk matrix."""
        if self.normalized_matrix.size == 0:
            return []
        allowed_chunk_ids = set(allowed_chunk_ids or [])
        query_vector = np.asarray(query_vector, dtype=np.float32)
        expected_dim = int(self.normalized_matrix.shape[1])
        actual_dim = int(query_vector.shape[-1]) if query_vector.ndim else 0
        if actual_dim != expected_dim:
            raise ValueError(
                "Dense query embedding dimension does not match the chunk index: "
                f"query_dim={actual_dim}, index_dim={expected_dim}. "
                "Rebuild the LightRAG index with the same embedding model and embedding_dim."
            )
        query_vector = query_vector / max(np.linalg.norm(query_vector), 1e-12)
        if allowed_chunk_ids:
            candidate_indices = [index for index, chunk_id in enumerate(self.chunk_ids) if chunk_id in allowed_chunk_ids]
            if not candidate_indices:
                return []
            candidate_scores = self.normalized_matrix[candidate_indices] @ query_vector
            local_top_indices = np.argsort(-candidate_scores)[:top_k]
            top_indices = [candidate_indices[index] for index in local_top_indices]
            scores = self.normalized_matrix @ query_vector
        else:
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
class GraphKeywordIndex:
    """Precomputed graph keyword view for chunk-level root selection.

    图结构关键词索引用于增强根 chunk 的语义对齐和多样性。
    """

    chunk_term_weights: dict[str, dict[str, float]]
    chunk_anchor_terms: dict[str, list[str]]
    vocabulary: set[str]
    ignored_terms: set[str]
    ignored_threshold: int


def _keyword_tokens(text):
    return [
        token
        for token in tokenize(normalize_text(text))
        if len(token) >= 3 and not token.isdigit()
    ]


def build_graph_keyword_index(graph, chunk_to_nodes, chunk_to_edges, chunk_store, max_terms_per_chunk=18):
    """Build a corpus-adaptive keyword index from graph nodes/edges.

    The index is intentionally corpus-specific:
    - terms frequent in too many chunks are auto-pruned
    - no dataset hardcoded keyword list is required
    """
    if not chunk_store:
        return GraphKeywordIndex({}, {}, set(), set(), 0)

    raw_chunk_terms = {}
    term_df = Counter()
    chunk_count = len(chunk_store)
    high_df_threshold = max(6, int(chunk_count * 0.12))
    blocked_terms = set(STOPWORDS) | {
        "unknown",
        "relation",
        "entity",
        "entities",
        "section",
        "theme",
        "themes",
        "comparison",
        "mechanism",
        "process",
        "effect",
        "effects",
        "factor",
        "factors",
    }

    for chunk_id in chunk_store:
        weighted_terms = defaultdict(float)
        for node_id in chunk_to_nodes.get(chunk_id, set()):
            degree = graph.degree(node_id) if graph is not None and graph.has_node(node_id) else 0
            degree_weight = 1.0 / max(1.0, 1.0 + math.log1p(max(degree, 1)))
            node_data = graph.nodes[node_id] if graph is not None and graph.has_node(node_id) else {}
            for token in _keyword_tokens(node_id):
                weighted_terms[token] += 1.0 * degree_weight
            for token in _keyword_tokens(node_data.get("entity_type", "")):
                weighted_terms[token] += 0.55 * degree_weight
            for token in _keyword_tokens(node_data.get("description", "")):
                weighted_terms[token] += 0.35 * degree_weight

        for edge_id in chunk_to_edges.get(chunk_id, set()):
            edge_data = graph.get_edge_data(edge_id[0], edge_id[1]) if graph is not None else {}
            edge_text = " ".join(
                [
                    str(edge_id[0]),
                    str(edge_id[1]),
                    normalize_relation_category(edge_data or {}),
                    normalize_text((edge_data or {}).get("keywords", "")),
                    normalize_text((edge_data or {}).get("description", "")),
                ]
            )
            for token in _keyword_tokens(edge_text):
                weighted_terms[token] += 0.22

        # Safety fallback so chunks with sparse graph links still get lexical keys.
        if not weighted_terms:
            for token in _keyword_tokens(chunk_store.get(chunk_id, {}).get("content", "")):
                weighted_terms[token] += 0.08

        cleaned = {
            term: weight
            for term, weight in weighted_terms.items()
            if term not in blocked_terms and len(term) >= 3
        }
        if not cleaned:
            cleaned = dict(weighted_terms)
        ranked_terms = sorted(cleaned.items(), key=lambda item: (-item[1], item[0]))[:max_terms_per_chunk]
        raw_chunk_terms[chunk_id] = dict(ranked_terms)
        for term in raw_chunk_terms[chunk_id]:
            term_df[term] += 1

    ignored_terms = {term for term, df in term_df.items() if df >= high_df_threshold}
    chunk_term_weights = {}
    chunk_anchor_terms = {}
    vocabulary = set()
    for chunk_id, term_weights in raw_chunk_terms.items():
        filtered = {term: weight for term, weight in term_weights.items() if term not in ignored_terms}
        if not filtered:
            filtered = term_weights
        ranked_terms = sorted(filtered.items(), key=lambda item: (-item[1], item[0]))[:max_terms_per_chunk]
        final = dict(ranked_terms)
        chunk_term_weights[chunk_id] = final
        anchors = [term for term, _ in ranked_terms[:3]]
        chunk_anchor_terms[chunk_id] = anchors
        vocabulary.update(final)

    return GraphKeywordIndex(
        chunk_term_weights=chunk_term_weights,
        chunk_anchor_terms=chunk_anchor_terms,
        vocabulary=vocabulary,
        ignored_terms=ignored_terms,
        ignored_threshold=high_df_threshold,
    )


@dataclass
class HybridChunkRetriever:
    bm25_index: BM25Index
    dense_index: DenseChunkIndex | None
    embedding_client: object | None
    mode: str = "hybrid"
    dense_weight: float = 0.75
    bm25_weight: float = 0.25

    def search(self, query, top_k, allowed_doc_ids=None, chunk_store=None):
        """Fuse bm25 and dense signals into one ranked candidate list.

        参数:
            query: 查询文本。
            top_k: 返回候选 chunk 数量上限。

        返回:
            排序后的候选 chunk 列表，包含融合后的 score_norm。
        """
        if self.mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError(f"Unsupported retrieval mode: {self.mode}")
        allowed_chunk_ids = None
        if allowed_doc_ids is not None:
            if chunk_store is None:
                raise ValueError("allowed_doc_ids requires chunk_store for constrained retrieval")
            allowed_doc_ids = set(allowed_doc_ids)
            allowed_chunk_ids = {
                chunk_id
                for chunk_id, chunk in chunk_store.items()
                if chunk.get("full_doc_id") in allowed_doc_ids
            }
            if not allowed_chunk_ids:
                return []
        bm25_hits = self.bm25_index.search(query, top_k=top_k, allowed_chunk_ids=allowed_chunk_ids) if self.mode in {"bm25", "hybrid"} else []
        dense_hits = []
        if self.mode in {"dense", "hybrid"}:
            if self.dense_index is None or self.embedding_client is None:
                raise ValueError("Dense retrieval requested but dense index or embedding client is unavailable")
            query_vector = self.embedding_client.embed_text(query)
            if self.dense_index.normalized_matrix.ndim == 2 and self.dense_index.normalized_matrix.shape[1] != len(query_vector):
                raise ValueError(
                    "Dense embedding dimension mismatch: "
                    f"index_dim={self.dense_index.normalized_matrix.shape[1]} vs query_dim={len(query_vector)}. "
                    "Please rebuild vdb_chunks.json with the same embedding model used at retrieval time."
                )
            dense_hits = self.dense_index.search(query_vector, top_k=top_k, allowed_chunk_ids=allowed_chunk_ids)
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


def search_graph_keyword_chunks(
    query,
    graph,
    node_to_chunks,
    edge_to_chunks,
    top_chunk_k,
    top_node_k=180,
    top_edge_k=180,
):
    """Direct query->(node/edge)->chunk lexical retrieval.

    This is a recall-side channel that does not depend on dense chunk retrieval.
    """
    if graph is None:
        return []
    chunk_scores = defaultdict(float)
    node_hits = []
    for node_id, node_data in graph.nodes(data=True):
        node_text = " ".join(
            [
                normalize_text(node_id),
                normalize_text(node_data.get("entity_type", "")),
                normalize_text(node_data.get("description", "")),
            ]
        )
        rel = lexical_overlap_score(query, node_text)
        if rel <= 0:
            continue
        degree = graph.degree(node_id) if graph.has_node(node_id) else 0
        specificity = 1.0 / max(1.0, 1.0 + math.log1p(max(degree, 1)))
        node_hits.append((node_id, rel, specificity))
    node_hits.sort(key=lambda item: (-item[1], -item[2], item[0]))
    for node_id, rel, specificity in node_hits[:top_node_k]:
        contribution = rel * (0.7 + 0.3 * specificity)
        for chunk_id in node_to_chunks.get(node_id, set()):
            chunk_scores[chunk_id] += contribution

    edge_hits = []
    for edge_id in edge_to_chunks:
        edge_data = graph.get_edge_data(edge_id[0], edge_id[1]) or {}
        edge_text = " ".join(
            [
                normalize_text(edge_id[0]),
                normalize_text(edge_id[1]),
                normalize_text(edge_data.get("relation", "")),
                normalize_text(edge_data.get("keywords", "")),
                normalize_text(edge_data.get("description", "")),
            ]
        )
        rel = lexical_overlap_score(query, edge_text)
        if rel <= 0:
            continue
        edge_hits.append((edge_id, rel))
    edge_hits.sort(key=lambda item: (-item[1], item[0]))
    for edge_id, rel in edge_hits[:top_edge_k]:
        contribution = rel * 0.9
        for chunk_id in edge_to_chunks.get(edge_id, set()):
            chunk_scores[chunk_id] += contribution

    if not chunk_scores:
        return []
    ranked = sorted(chunk_scores.items(), key=lambda item: (-item[1], item[0]))[:top_chunk_k]
    top_score = ranked[0][1] if ranked else 1.0
    return [
        {
            "chunk_id": chunk_id,
            "graph_keyword_score": float(score),
            "graph_keyword_score_norm": float(score / max(top_score, 1e-9)),
        }
        for chunk_id, score in ranked
    ]


def search_graph_focus_chunks(
    query,
    graph,
    node_to_chunks,
    edge_to_chunks,
    top_chunk_k,
    top_node_k=220,
    top_edge_k=220,
):
    """Theme-focused retrieval using non-syntactic query terms on graph text.

    Pipeline:
    - extract query focus terms (content terms only)
    - match these terms against node/edge texts
    - project top node/edge hits back to chunks
    - aggregate weighted chunk scores
    """
    if graph is None:
        return []
    query_terms = _query_focus_terms(query)
    if not query_terms:
        return []

    chunk_scores = defaultdict(float)
    chunk_hit_terms = defaultdict(set)

    node_hits = []
    for node_id, node_data in graph.nodes(data=True):
        node_text = " ".join(
            [
                normalize_text(node_id),
                normalize_text(node_data.get("entity_type", "")),
                normalize_text(node_data.get("description", "")),
            ]
        )
        node_terms = set(_keyword_tokens(node_text))
        hit_terms = sorted(query_terms & node_terms)
        if not hit_terms:
            continue
        hit_count = len(hit_terms)
        coverage = hit_count / max(len(query_terms), 1)
        degree = graph.degree(node_id) if graph.has_node(node_id) else 0
        specificity = 1.0 / max(1.0, 1.0 + math.log1p(max(degree, 1)))
        score = coverage * (0.72 + 0.28 * specificity) * (1.0 + 0.18 * min(hit_count, 4))
        node_hits.append((node_id, score, hit_terms))

    node_hits.sort(key=lambda item: (-item[1], -len(item[2]), item[0]))
    for node_id, score, hit_terms in node_hits[:top_node_k]:
        contribution = score
        for chunk_id in node_to_chunks.get(node_id, set()):
            chunk_scores[chunk_id] += contribution
            chunk_hit_terms[chunk_id].update(hit_terms)

    edge_hits = []
    for edge_id in edge_to_chunks:
        edge_data = graph.get_edge_data(edge_id[0], edge_id[1]) or {}
        edge_text = " ".join(
            [
                normalize_text(edge_id[0]),
                normalize_text(edge_id[1]),
                normalize_text(edge_data.get("relation", "")),
                normalize_text(edge_data.get("keywords", "")),
                normalize_text(edge_data.get("description", "")),
            ]
        )
        edge_terms = set(_keyword_tokens(edge_text))
        hit_terms = sorted(query_terms & edge_terms)
        if not hit_terms:
            continue
        hit_count = len(hit_terms)
        coverage = hit_count / max(len(query_terms), 1)
        score = coverage * 0.9 * (1.0 + 0.12 * min(hit_count, 4))
        edge_hits.append((edge_id, score, hit_terms))

    edge_hits.sort(key=lambda item: (-item[1], -len(item[2]), item[0]))
    for edge_id, score, hit_terms in edge_hits[:top_edge_k]:
        contribution = score
        for chunk_id in edge_to_chunks.get(edge_id, set()):
            chunk_scores[chunk_id] += contribution
            chunk_hit_terms[chunk_id].update(hit_terms)

    if not chunk_scores:
        return []

    ranked = sorted(
        chunk_scores.items(),
        key=lambda item: (-item[1], -len(chunk_hit_terms.get(item[0], set())), item[0]),
    )[:top_chunk_k]
    top_score = ranked[0][1] if ranked else 1.0
    return [
        {
            "chunk_id": chunk_id,
            "graph_focus_score": float(score),
            "graph_focus_score_norm": float(score / max(top_score, 1e-9)),
            "graph_focus_hit_count": len(chunk_hit_terms.get(chunk_id, set())),
            "graph_focus_hit_terms": sorted(chunk_hit_terms.get(chunk_id, set()))[:8],
            "retrieval_score": float(score / max(top_score, 1e-9)),
            "score_norm": float(score / max(top_score, 1e-9)),
            "dense_score_norm": 0.0,
            "bm25_score_norm": 0.0,
        }
        for chunk_id, score in ranked
    ]


def search_graph_evidence_chunks(
    query,
    graph,
    node_to_chunks,
    edge_to_chunks,
    top_chunk_k,
    top_node_k=220,
    top_edge_k=220,
    focus_weight=0.65,
    keyword_weight=0.35,
):
    """Unified graph-to-chunk recall for the anchor stage.

    参数:
        query: 查询文本。
        graph: 实体关系图对象。
        node_to_chunks: 节点到 chunk 的映射。
        edge_to_chunks: 边到 chunk 的映射。
        top_chunk_k: 返回的 chunk 数量上限。
        top_node_k: 节点检索的 top-k 限制。
        top_edge_k: 边检索的 top-k 限制。
        focus_weight: graph_focus 信号的融合权重。
        keyword_weight: graph_keyword 信号的融合权重。

    返回:
        包含 graph_evidence_score 和归一化分数的 chunk 命中列表。

    The public abstraction is intentionally simple: graph-side evidence recall
    projects graph relevance back onto chunk candidates.
    """
    focus_hits = search_graph_focus_chunks(
        query=query,
        graph=graph,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        top_chunk_k=top_chunk_k,
        top_node_k=top_node_k,
        top_edge_k=top_edge_k,
    )
    keyword_hits = search_graph_keyword_chunks(
        query=query,
        graph=graph,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        top_chunk_k=top_chunk_k,
        top_node_k=max(1, top_node_k - 40),
        top_edge_k=max(1, top_edge_k - 40),
    )

    merged = {}
    for item in focus_hits:
        merged[item["chunk_id"]] = {
            **item,
            "graph_evidence_score": focus_weight * float(item.get("graph_focus_score_norm", item.get("score_norm", 0.0))),
            "graph_evidence_hit_terms": sorted(item.get("graph_focus_hit_terms", [])),
        }

    for item in keyword_hits:
        chunk_id = item["chunk_id"]
        keyword_score = float(item.get("graph_keyword_score_norm", item.get("score_norm", 0.0)))
        if chunk_id not in merged:
            merged[chunk_id] = {
                "chunk_id": chunk_id,
                "graph_keyword_score": item.get("graph_keyword_score", 0.0),
                "graph_keyword_score_norm": keyword_score,
                "graph_focus_score": 0.0,
                "graph_focus_score_norm": 0.0,
                "graph_focus_hit_count": 0,
                "graph_focus_hit_terms": [],
                "graph_evidence_hit_terms": [],
                "graph_evidence_score": 0.0,
            }
        payload = merged[chunk_id]
        payload["graph_keyword_score"] = item.get("graph_keyword_score", 0.0)
        payload["graph_keyword_score_norm"] = keyword_score
        payload["graph_evidence_score"] += keyword_weight * keyword_score

    if not merged:
        return []

    ranked = sorted(
        merged.values(),
        key=lambda item: (
            -float(item.get("graph_evidence_score", 0.0)),
            -len(item.get("graph_evidence_hit_terms", [])),
            item["chunk_id"],
        ),
    )[:top_chunk_k]
    top_score = float(ranked[0].get("graph_evidence_score", 0.0)) if ranked else 1.0
    for item in ranked:
        norm = float(item.get("graph_evidence_score", 0.0)) / max(top_score, 1e-9)
        item["graph_evidence_score_norm"] = norm
        item["retrieval_score"] = norm
        item["score_norm"] = norm
        item["dense_score_norm"] = 0.0
        item["bm25_score_norm"] = 0.0
    return ranked


def merge_candidate_hits_with_graph(primary_hits, graph_hits, graph_weight=0.25):
    """Merge dense/bm25 candidates with graph-side candidates.

    参数:
        primary_hits: 来自 BM25/dense 检索的候选 chunk 列表。
        graph_hits: 来自图检索的候选 chunk 列表。
        graph_weight: 图检索信号在最终分数中的权重。

    返回:
        合并后的 chunk 列表，包含统一 retrieval_score 和 score_norm。
    """
    if not graph_hits:
        return primary_hits
    merged = {}
    for item in primary_hits:
        merged[item["chunk_id"]] = dict(item)
    for item in graph_hits:
        chunk_id = item["chunk_id"]
        graph_norm = float(
            item.get(
                "graph_keyword_score_norm",
                item.get("graph_focus_score_norm", item.get("score_norm", 0.0)),
            )
        )
        if chunk_id not in merged:
            payload = {
                "chunk_id": chunk_id,
                "score": graph_weight * graph_norm,
                "score_norm": graph_weight * graph_norm,
                "bm25_score": 0.0,
                "bm25_score_norm": 0.0,
                "dense_score": 0.0,
                "dense_score_norm": 0.0,
                "retrieval_score": graph_weight * graph_norm,
            }
            for key in (
                "graph_keyword_score",
                "graph_keyword_score_norm",
                "graph_focus_score",
                "graph_focus_score_norm",
                "graph_focus_hit_count",
                "graph_focus_hit_terms",
                "graph_evidence_score",
                "graph_evidence_score_norm",
                "graph_evidence_hit_terms",
            ):
                if key in item:
                    payload[key] = item[key]
            if "graph_keyword_score_norm" not in payload and "graph_keyword_score" in payload:
                payload["graph_keyword_score_norm"] = graph_norm
            merged[chunk_id] = payload
            continue
        current = merged[chunk_id]
        for key in (
            "graph_keyword_score",
            "graph_keyword_score_norm",
            "graph_focus_score",
            "graph_focus_score_norm",
            "graph_focus_hit_count",
            "graph_focus_hit_terms",
            "graph_evidence_score",
            "graph_evidence_score_norm",
            "graph_evidence_hit_terms",
        ):
            if key in item:
                current[key] = item[key]
        retrieval_score = float(current.get("retrieval_score", current.get("score_norm", 0.0)))
        boosted = (1.0 - graph_weight) * retrieval_score + graph_weight * graph_norm
        current["retrieval_score"] = boosted
        current["score"] = boosted

    scored = list(merged.values())
    scored.sort(
        key=lambda item: (
            -item.get("retrieval_score", item.get("score_norm", 0.0)),
            -item.get("dense_score_norm", 0.0),
            -item.get("bm25_score_norm", 0.0),
            -item.get("graph_focus_score_norm", 0.0),
            -item.get("graph_keyword_score_norm", 0.0),
            item["chunk_id"],
        )
    )
    top = scored[0].get("retrieval_score", scored[0].get("score_norm", 1.0)) if scored else 1.0
    return [
        {
            **item,
            "score_norm": float(item.get("retrieval_score", item.get("score_norm", 0.0)) / max(top, 1e-9)),
        }
        for item in scored
    ]


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


def _chunk_relation_entropy(chunk_id, chunk_to_edges, graph):
    """Estimate how relation-diverse a chunk looks through its attached edges."""
    if graph is None:
        return 0.0
    categories = []
    for edge_id in chunk_to_edges.get(chunk_id, set()):
        edge_data = graph.get_edge_data(edge_id[0], edge_id[1]) or {}
        categories.append(normalize_relation_category(edge_data))
    return relation_entropy([item for item in categories if item not in {"unknown_relation", "unknown relation"}])


def _query_alignment(item):
    """Reuse dense similarity when available; otherwise fall back to retrieval score."""
    if item.get("dense_score_norm", 0.0) > 0:
        return float(item["dense_score_norm"])
    return float(item.get("retrieval_score", item.get("score_norm", 0.0)))


def _root_sort_key(item):
    return (
        -item["base_score"],
        -item.get("structure_penalty", 1.0),
        -item.get("keyword_match_score", 0.0),
        -item["query_alignment"],
        -item["relation_entropy"],
        item.get("graph_mass", 0),
        item["chunk_id"],
    )


def _root_primary_quota(top_k):
    return min(top_k, max(2, (top_k // 2) + 1))


def _root_base_score(item):
    """Prefer dense grounding when available; fall back to retrieval score."""
    if item.get("dense_score_norm", 0.0) > 0:
        return float(item["dense_score_norm"])
    return float(item.get("score_norm", item.get("retrieval_score", 0.0)))


def _inverse_sqrt_degree(size):
    return 1.0 / math.sqrt(max(float(size), 1.0))


def _chunk_structure_penalty(nodes, edges, node_to_chunks=None, edge_to_chunks=None):
    """Down-weight chunks that win mainly because they touch too many/popular graph items."""
    raw_mass = len(nodes) + len(edges)
    if raw_mass <= 0:
        return 1.0
    weighted_terms = []
    for node_id in nodes:
        degree = len((node_to_chunks or {}).get(node_id, set()))
        weighted_terms.append(_inverse_sqrt_degree(degree))
    for edge_id in edges:
        degree = len((edge_to_chunks or {}).get(edge_id, set()))
        weighted_terms.append(_inverse_sqrt_degree(degree))
    degree_term = sum(weighted_terms) / max(len(weighted_terms), 1)
    size_term = 1.0 / math.sqrt(1.0 + (raw_mass / 6.0))
    return max(0.18, min(1.0, degree_term * size_term))


def _chunk_graph_signature(chunk_id, chunk_to_nodes, chunk_to_edges):
    nodes = set(chunk_to_nodes.get(chunk_id, set()))
    edges = set(chunk_to_edges.get(chunk_id, set()))
    return nodes, edges


_QUERY_STOPWORDS = {
    "the",
    "a",
    "an",
    "in",
    "on",
    "of",
    "to",
    "for",
    "and",
    "or",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "that",
    "this",
    "those",
    "these",
    "it",
    "its",
    "as",
    "by",
    "with",
    "from",
    "at",
    "into",
    "about",
    "how",
    "what",
    "which",
    "who",
    "whom",
    "when",
    "where",
    "why",
    "can",
    "could",
    "would",
    "should",
    "do",
    "does",
    "did",
    "we",
    "our",
    "you",
    "your",
    "their",
    "there",
    "than",
    "then",
    "infer",
    "significant",
    "pattern",
    "patterns",
    "lead",
    "led",
} | set(STOPWORDS)


def _query_focus_terms(query, keyword_index=None):
    tokens = {
        token
        for token in tokenize(normalize_text(query))
        if len(token) >= 3 and token not in _QUERY_STOPWORDS
    }
    if not tokens:
        return set()
    if keyword_index is None or not getattr(keyword_index, "vocabulary", None):
        return tokens
    in_vocab = {
        token
        for token in tokens
        if token in keyword_index.vocabulary and token not in keyword_index.ignored_terms
    }
    return in_vocab or tokens


def _anchor_query_overlap(anchor_terms, focus_terms):
    if not anchor_terms or not focus_terms:
        return 0.0
    anchor_tokens = set()
    for term in anchor_terms:
        anchor_tokens.update(tokenize(normalize_text(term)))
    if not anchor_tokens:
        return 0.0
    return len(anchor_tokens & focus_terms) / max(len(focus_terms), 1)


def _focus_term_hit_count(text, focus_terms):
    if not text or not focus_terms:
        return 0
    text_tokens = set(tokenize(normalize_text(text)))
    return len(text_tokens & focus_terms)


def _chunk_anchor_terms(chunk_id, chunk_to_nodes, chunk_to_edges, graph, keyword_index=None):
    if keyword_index is not None:
        anchors = list(keyword_index.chunk_anchor_terms.get(chunk_id, []))
        if anchors:
            return anchors
    counter = Counter()
    for node_id in chunk_to_nodes.get(chunk_id, set()):
        label = normalize_text(node_id).strip()
        if not label:
            continue
        degree = graph.degree(node_id) if graph is not None and graph.has_node(node_id) else 0
        weight = 1.0 / max(1.0, 1.0 + math.log1p(degree))
        counter[label] += weight
    for edge_id in chunk_to_edges.get(chunk_id, set()):
        edge_data = graph.get_edge_data(edge_id[0], edge_id[1]) if graph is not None else {}
        category = normalize_relation_category(edge_data or {})
        if category and category not in {"unknown relation", "unknown_relation"}:
            counter[category] += 0.35
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return [label for label, _ in ranked[:3]]


def _chunk_basin_signature(chunk_id, chunk_store, chunk_to_nodes, chunk_to_edges, graph, keyword_index=None):
    chunk = chunk_store.get(chunk_id, {})
    anchors = _chunk_anchor_terms(
        chunk_id,
        chunk_to_nodes,
        chunk_to_edges,
        graph,
        keyword_index=keyword_index,
    )
    doc_id = chunk.get("full_doc_id") or "no-doc"
    if anchors:
        basin_key = tuple(anchors[:2])
        basin_signature = " | ".join(anchors[:2])
    else:
        basin_key = (doc_id, chunk.get("chunk_order_index", -1))
        basin_signature = doc_id
    return basin_key, basin_signature, anchors


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


def _candidate_query_terms(candidate):
    terms = candidate.get("keyword_hit_terms") or []
    if not terms:
        terms = candidate.get("graph_focus_hit_terms") or []
    if not terms:
        terms = candidate.get("graph_keyword_hit_terms") or []
    return set(terms)


def _chunk_keyword_match(chunk_id, query_terms, keyword_index):
    if keyword_index is None or not query_terms:
        return 0.0, 0, []
    term_weights = keyword_index.chunk_term_weights.get(chunk_id, {})
    if not term_weights:
        return 0.0, 0, []
    hit_terms = [term for term in sorted(query_terms) if term in term_weights]
    if not hit_terms:
        return 0.0, 0, []
    total_weight = sum(term_weights.values())
    hit_weight = sum(term_weights[term] for term in hit_terms)
    score = hit_weight / max(total_weight, 1e-9)
    return float(score), len(hit_terms), hit_terms[:6]


def _select_anchor_root_chunks(candidates, chunk_store, top_k, same_doc_window, primary_quota):
    basin_groups = defaultdict(list)
    for candidate in candidates:
        basin_groups[candidate["basin_key"]].append(candidate)

    ordered_groups = []
    for basin_key, items in basin_groups.items():
        items.sort(key=_root_sort_key)
        ordered_groups.append((basin_key, items))
    ordered_groups.sort(key=lambda item: _root_sort_key(item[1][0]))

    selected = []
    doc_counts = Counter()
    covered_query_terms = set()
    max_group_len = max((len(items) for _, items in ordered_groups), default=0)

    for pass_index in range(max_group_len):
        if len(selected) >= top_k:
            break
        any_added = False
        for _, items in ordered_groups:
            if len(selected) >= top_k or pass_index >= len(items):
                continue
            candidate = items[pass_index]
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
            same_doc_limit = 1 if pass_index <= 1 else 2
            overlap_limit = 0.42 if pass_index == 0 else 0.58
            term_gain = len(_candidate_query_terms(candidate) - covered_query_terms)
            if same_doc_count >= same_doc_limit or same_band or max_overlap > overlap_limit:
                continue
            if pass_index <= 1 and selected and term_gain <= 0:
                continue
            selected.append(
                {
                    **candidate,
                    "root_role": "primary" if len(selected) < primary_quota else "diversity",
                    "novelty_gain": round(
                        (len(candidate["graph_nodes"]) + len(candidate["graph_edges"]))
                        * candidate.get("structure_penalty", 1.0),
                        6,
                    ),
                    "query_term_gain": term_gain,
                    "max_selected_overlap": round(max_overlap, 6),
                }
            )
            covered_query_terms.update(_candidate_query_terms(candidate))
            if candidate.get("full_doc_id"):
                doc_counts[candidate["full_doc_id"]] += 1
            any_added = True
        if not any_added:
            break

    if len(selected) < top_k:
        leftovers = [item for _, items in ordered_groups for item in items if item["chunk_id"] not in {s["chunk_id"] for s in selected}]
        leftovers.sort(
            key=lambda item: (
                doc_counts.get(item.get("full_doc_id"), 0),
                -len(_candidate_query_terms(item) - covered_query_terms),
                -item["query_alignment"],
                -item["relation_entropy"],
                -item["base_score"],
                item["chunk_id"],
            )
        )
        for candidate in leftovers:
            if len(selected) >= top_k:
                break
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
            if max_overlap > 0.72:
                continue
            selected.append(
                {
                    **candidate,
                    "root_role": "primary" if len(selected) < primary_quota else "diversity",
                    "novelty_gain": round(
                        (len(candidate["graph_nodes"]) + len(candidate["graph_edges"]))
                        * candidate.get("structure_penalty", 1.0),
                        6,
                    ),
                    "query_term_gain": len(_candidate_query_terms(candidate) - covered_query_terms),
                    "max_selected_overlap": round(max_overlap, 6),
                }
            )
            covered_query_terms.update(_candidate_query_terms(candidate))
            if candidate.get("full_doc_id"):
                doc_counts[candidate["full_doc_id"]] += 1
    return selected[:top_k]


def select_diverse_root_chunks(
    query,
    candidate_hits,
    chunk_store,
    chunk_to_nodes,
    chunk_to_edges,
    top_k,
    node_to_chunks=None,
    edge_to_chunks=None,
    graph=None,
    keyword_index=None,
    same_doc_window=1,
    max_same_doc_roots=1,
    relaxed_max_same_doc_roots=2,
    max_provenance_overlap=0.55,
    relaxed_max_provenance_overlap=0.85,
):
    """Select chunk anchors for the anchor stage.

    The selector keeps three principles:
    - keep strong query-grounded anchors
    - spread anchors across different evidence basins
    - avoid local-band and provenance-near duplicates

    在 anchor 阶段选择多样化根 chunk，以提高检索覆盖与图结构连贯性。
    """
    if not candidate_hits:
        return []

    candidates = []
    query_focus_terms = _query_focus_terms(query, keyword_index=keyword_index)
    for item in candidate_hits:
        chunk_id = item["chunk_id"]
        chunk = chunk_store.get(chunk_id, {})
        nodes, edges = _chunk_graph_signature(chunk_id, chunk_to_nodes, chunk_to_edges)
        structure_penalty = _chunk_structure_penalty(
            nodes,
            edges,
            node_to_chunks=node_to_chunks,
            edge_to_chunks=edge_to_chunks,
        )
        basin_key, basin_signature, anchor_terms = _chunk_basin_signature(
            chunk_id,
            chunk_store,
            chunk_to_nodes,
            chunk_to_edges,
            graph,
            keyword_index=keyword_index,
        )
        keyword_match_score, keyword_hit_count, keyword_hit_terms = _chunk_keyword_match(
            chunk_id,
            query_focus_terms,
            keyword_index,
        )
        candidates.append(
            {
                **item,
                "raw_base_score": round(_root_base_score(item), 6),
                "base_score": round(_root_base_score(item) * structure_penalty, 6),
                "query_alignment": round(_query_alignment(item), 6),
                "relation_entropy": round(_chunk_relation_entropy(chunk_id, chunk_to_edges, graph), 6),
                "full_doc_id": chunk.get("full_doc_id"),
                "chunk_order_index": chunk.get("chunk_order_index", -1),
                "graph_nodes": nodes,
                "graph_edges": edges,
                "graph_mass": len(nodes) + len(edges),
                "structure_penalty": round(structure_penalty, 6),
                "basin_key": basin_key,
                "basin_signature": basin_signature,
                "anchor_terms": anchor_terms,
                "query_lexical": round(lexical_overlap_score(query, chunk.get("content", "")), 6),
                "anchor_query_overlap": round(_anchor_query_overlap(anchor_terms, query_focus_terms), 6),
                "focus_term_hit_count": _focus_term_hit_count(chunk.get("content", ""), query_focus_terms),
                "keyword_match_score": round(keyword_match_score, 6),
                "keyword_hit_count": int(keyword_hit_count),
                "keyword_hit_terms": keyword_hit_terms,
            }
        )
    candidates.sort(key=_root_sort_key)
    if not candidates:
        return []
    selected = _select_anchor_root_chunks(
        candidates=candidates,
        chunk_store=chunk_store,
        top_k=top_k,
        same_doc_window=same_doc_window,
        primary_quota=_root_primary_quota(top_k),
    )

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


def select_anchor_root_chunks(*args, **kwargs):
    """Public alias used by the anchor stage."""
    return select_diverse_root_chunks(*args, **kwargs)


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
