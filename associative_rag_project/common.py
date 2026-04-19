"""Shared text and graph helpers used across the project.

These helpers are intentionally lightweight because they sit on hot paths
inside retrieval, association, and context construction.

公共文本与图结构辅助函数，提供轻量级的分词、规范化、重用等工具。
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
    """Decode LightRAG-style `<SEP>`-joined provenance ids into a Python list.

    参数:
        raw_value: 原始 provenance 字符串，多个 chunk id 用 `<SEP>` 连接。

    返回:
        解析后的 chunk id 列表，若输入不是字符串则返回空列表。

    将图节点/边 provenance 字符串拆成 chunk id 列表。
    """
    if not isinstance(raw_value, str):
        return []
    return [item for item in raw_value.split(GRAPH_FIELD_SEP) if item]


def edge_key(left, right):
    """Represent an undirected graph edge with a canonical tuple order.

    参数:
        left: 边的一端节点 id。
        right: 边的另一端节点 id。

    返回:
        规范化后的边元组，保证 (a,b) 和 (b,a) 表示相同无向边。

    对无向边进行规范化，使边表示在两个方向下都一致。
    """
    return tuple(sorted((left, right)))


def tokenize(text):
    """Very lightweight tokenizer for ranking features and lexical overlap.

    提取小写字母数字词元，过滤停用词，适合快速相似度计算。
    """
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [token for token in tokens if len(token) > 1 and token not in STOPWORDS]


def normalize_text(value):
    """Normalize text into one-line whitespace-cleaned string.

    参数:
        value: 任何可转换为字符串的值。

    返回:
        去除换行、重复空白后的单行字符串。

    统一文本格式，适用于摘要和短语比较。
    """
    if value is None:
        return ""
    return " ".join(str(value).replace("\n", " ").split())


def approx_word_count(text):
    """Approximate word count by splitting on whitespace.

    参数:
        text: 文本字符串。

    返回:
        词数估计值，用于简易预算控制。
    """
    return len(re.findall(r"\S+", text or ""))


def safe_mean(values):
    """Compute mean safely, returning 0.0 for empty lists.

    参数:
        values: 数值序列。

    返回:
        平均值，若输入为空则返回 0.0。
    """
    return sum(values) / len(values) if values else 0.0


def build_csv(rows):
    """Build a simple CSV string from rows of values.

    参数:
        rows: 二维列表，每行是一组值。

    返回:
        以换行分隔、制表符分隔的 CSV 字符串。
    """
    return "\n".join([",\t".join(map(str, row)) for row in rows])


def lexical_overlap_score(query, text):
    """Cosine-like token overlap used as a cheap, stable relevance feature.

    计算查询与文本之间的词元重合度，作为简单相关性度量。
    """
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
    """Estimate how much a span looks like dataset/model/method metadata.

    检测文本中是否包含技术术语，用于区分元信息与事实内容。
    """
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    hits = sum(1 for token in tokens if token in TECHNICAL_TERMS)
    return hits / len(tokens)

# ？
def query_prefers_technical_content(query):
    """Allow technical answers only when the query itself clearly asks for them.

    参数:
        query: 用户查询文本。

    返回:
        如果查询包含技术触发词或本身技术密度较高，则返回 True，否则 False。
    """
    query_tokens = set(tokenize(query))
    if query_tokens & QUERY_TECHNICAL_TRIGGERS:
        return True
    return technical_density(query) >= 0.18
