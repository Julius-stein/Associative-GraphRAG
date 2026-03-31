"""Shared text and graph helpers used across the project.

These helpers are intentionally lightweight because they sit on hot paths
inside retrieval, association, and context construction.
"""

import math
import re
from collections import Counter


GRAPH_FIELD_SEP = "<SEP>"
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "them",
    "these",
    "this",
    "those",
    "through",
    "to",
    "what",
    "which",
    "with",
    "within",
    "across",
}

TECHNICAL_TERMS = {
    "algorithm",
    "algorithms",
    "annotation",
    "arxiv",
    "benchmark",
    "bert",
    "chatbert",
    "classifier",
    "classifiers",
    "corpus",
    "data",
    "dataset",
    "datasets",
    "decoder",
    "embedding",
    "embeddings",
    "evaluation",
    "experiments",
    "friendsbert",
    "gru",
    "infobox",
    "language",
    "languages",
    "linguistic",
    "metrics",
    "model",
    "models",
    "multilingual",
    "neural",
    "prediction",
    "predictions",
    "pretrained",
    "research",
    "sample",
    "samples",
    "script",
    "scripts",
    "sentiment",
    "system",
    "systems",
    "tagging",
    "task",
    "tasks",
    "technology",
    "training",
    "twitter",
    "wiki",
    "wikibio",
}

QUERY_TECHNICAL_TRIGGERS = {
    "algorithm",
    "algorithms",
    "annotation",
    "arxiv",
    "benchmark",
    "benchmarks",
    "bert",
    "chatbert",
    "classifier",
    "classifiers",
    "decoder",
    "embedding",
    "embeddings",
    "evaluation",
    "friendsbert",
    "gru",
    "metrics",
    "model",
    "models",
    "multilingual",
    "neural",
    "prediction",
    "predictions",
    "pretrained",
    "sentiment",
    "system",
    "systems",
    "tagging",
    "technology",
    "training",
}


def parse_source_ids(raw_value):
    """Decode LightRAG-style `<SEP>`-joined provenance ids into a Python list."""
    if not isinstance(raw_value, str):
        return []
    return [item for item in raw_value.split(GRAPH_FIELD_SEP) if item]


def edge_key(left, right):
    """Represent an undirected graph edge with a canonical tuple order."""
    return tuple(sorted((left, right)))


def tokenize(text):
    """Very lightweight tokenizer for ranking features and lexical overlap."""
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [token for token in tokens if len(token) > 1 and token not in STOPWORDS]


def normalize_text(value):
    if value is None:
        return ""
    return " ".join(str(value).replace("\n", " ").split())


def approx_word_count(text):
    return len(re.findall(r"\S+", text or ""))


def safe_mean(values):
    return sum(values) / len(values) if values else 0.0


def build_csv(rows):
    return "\n".join([",\t".join(map(str, row)) for row in rows])


def lexical_overlap_score(query, text):
    """Cosine-like token overlap used as a cheap, stable relevance feature."""
    query_tokens = tokenize(query)
    text_tokens = tokenize(text)
    if not query_tokens or not text_tokens:
        return 0.0
    q_counter = Counter(query_tokens)
    t_counter = Counter(text_tokens)
    overlap = sum(min(q_counter[token], t_counter[token]) for token in q_counter)
    norm = math.sqrt(sum(v * v for v in q_counter.values()) * sum(v * v for v in t_counter.values()))
    if norm <= 0:
        return 0.0
    return overlap / norm


def technical_density(text):
    """Estimate how much a span looks like dataset/model/method metadata."""
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    hits = sum(1 for token in tokens if token in TECHNICAL_TERMS)
    return hits / len(tokens)

# ？
def query_prefers_technical_content(query):
    """Allow technical answers only when the query itself clearly asks for them."""
    query_tokens = set(tokenize(query))
    if query_tokens & QUERY_TECHNICAL_TRIGGERS:
        return True
    return technical_density(query) >= 0.18
