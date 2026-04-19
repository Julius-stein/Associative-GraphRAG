"""Data loading helpers for corpora, questions, baselines, and provenance maps.

数据加载与路径解析工具，支持语料、问题、基线输出和 chunk provenance。
"""

import json
import re
from pathlib import Path

import networkx as nx

from .common import edge_key, parse_source_ids


def infer_corpus_name(corpus_path: str | Path) -> str:
    """Infer a dataset name from either a corpus name or a dataset path.

    参数:
        corpus_path: 语料库目录或文件路径。

    返回:
        推断出的语料库名称。

    Supported examples:
    - `agriculture` -> `agriculture`
    - `Datasets/agriculture` -> `agriculture`
    - `Datasets/agriculture/corpus` -> `agriculture`
    - `Datasets/agriculture/corpus/agriculture_unique_contexts.json` -> `agriculture`

    从路径中提取语料库名称，兼容旧目录结构与文件路径。
    """
    path = Path(corpus_path)
    parts = [part for part in path.parts if part not in {".", ""}]
    if not parts:
        return str(corpus_path)

    leaf = parts[-1]
    if leaf in {"corpus", "query", "output", "index"} and len(parts) >= 2:
        return parts[-2]

    if leaf.endswith(".json") and len(parts) >= 3 and parts[-2] == "corpus":
        return parts[-3]

    if len(parts) >= 2 and parts[-2] in {"corpus", "query", "output", "index"}:
        return parts[-1]

    return path.stem if path.suffix else leaf


def extract_questions(file_path: Path) -> list[str]:
    """Read benchmark questions from either text or JSON files.

    参数:
        file_path: 问题文件路径，支持 .json 或文本格式。

    返回:
        提取后的查询列表。

    支持纯文本或 JSON 格式的问题列表输入。
    """
    if file_path.suffix.lower() == ".json":
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            questions = []
            for item in payload:
                if isinstance(item, str):
                    questions.append(item.strip())
                elif isinstance(item, dict):
                    query = item.get("query")
                    if query:
                        questions.append(str(query).strip())
            if questions:
                return questions
        raise ValueError(f"Unsupported JSON question format: {file_path}")

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
    """Normalize either rewrite-groups or plain question lists into query rows.

    参数:
        rewrites_file: 可选的重写组 JSON 文件路径。
        questions_file: 可选的原始问题文件路径。
        limit_groups: 如果不为 None，则限制加载的查询数量。

    返回:
        标准化后的 query row 列表，每项包含 group_id、variant_id、query 和 base_query。

    统一加载问题行，支持重写组和原始问题文件格式。
    """
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
    if questions_file.suffix.lower() == ".json":
        payload = json.loads(questions_file.read_text(encoding="utf-8"))
        if isinstance(payload, list) and payload and all(isinstance(item, dict) for item in payload):
            rows = []
            selected_items = payload if limit_groups is None else payload[:limit_groups]
            for index, item in enumerate(selected_items, start=1):
                query = str(item.get("query", "")).strip()
                if not query:
                    continue
                row = {
                    "group_id": item.get("group_id", f"q{index:03d}"),
                    "variant_id": item.get("variant_id", "base"),
                    "query": query,
                    "base_query": item.get("base_query", query),
                }
                for key in ("query_contract", "organization_contract", "contract", "layout"):
                    if item.get(key):
                        row[key] = item[key]
                rows.append(row)
            if rows:
                return rows
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
    """Find the default question file for a corpus unless the user overrides it.

    参数:
        corpus_name: 语料库名称。
        explicit_path: 如果指定则直接使用该路径。

    返回:
        解析后的问题文件路径或 None。
    """
    if explicit_path:
        return Path(explicit_path)

    query_dir = Path("Datasets") / corpus_name / "query"
    if not query_dir.exists():
        return None

    preferred_candidates = [
        query_dir / f"{corpus_name}.json",
        query_dir / f"{corpus_name}.txt",
        query_dir / f"{corpus_name}_questions.json",
        query_dir / f"{corpus_name}_questions.txt",
        query_dir / "questions.json",
        query_dir / "questions.txt",
        query_dir / "query.json",
        query_dir / "query.txt",
    ]
    for candidate in preferred_candidates:
        if candidate.exists():
            return candidate

    fallback_files = sorted(
        [
            path
            for path in query_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".json", ".txt"}
        ]
    )
    if len(fallback_files) == 1:
        return fallback_files[0]
    if fallback_files:
        return fallback_files[0]
    return None


def resolve_baseline_file(corpus_name: str, explicit_path: str | None):
    """Default to the FG-RAG 4o-mini outputs for head-to-head comparison.

    参数:
        corpus_name: 语料库名称。
        explicit_path: 如果指定则直接使用该路径。

    返回:
        基线答案文件路径或 None。
    """
    if explicit_path:
        return Path(explicit_path)
    candidates = [
        Path("Datasets") / f"{corpus_name}" / "output" / f"FG-RAG-4o-mini.json",
        Path("FG-RAG") / f"{corpus_name}" / "output" / f"FG-RAG-4o-mini.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_corpus_index_dir(corpus_dir: Path):
    """Resolve the actual indexed corpus directory.

    参数:
        corpus_dir: 语料库根目录，可能是 corpus 或 index 子目录。

    返回:
        实际包含 graphml 和 kv_store 文档的索引目录。

    兼容旧的 corpus 目录和新的 index 目录结构。
    """
    corpus_dir = Path(corpus_dir)
    candidates = [corpus_dir]
    if corpus_dir.name == "corpus":
        candidates.append(corpus_dir.parent / "index")
    if corpus_dir.name == "index":
        candidates.append(corpus_dir.parent / "corpus")
    candidates.append(corpus_dir / "index")
    for candidate in candidates:
        if (candidate / "graph_chunk_entity_relation.graphml").exists() and (candidate / "kv_store_text_chunks.json").exists():
            return candidate
    return corpus_dir


def load_graph_corpus(corpus_dir: Path):
    """Load the graph and chunk store produced by the LightRAG preprocessing stage.

    参数:
        corpus_dir: 语料库目录，支持自动解析 index 子目录。

    返回:
        三元组 (graph, chunk_store, index_dir)。
    """
    index_dir = resolve_corpus_index_dir(corpus_dir)
    graph = nx.read_graphml(index_dir / "graph_chunk_entity_relation.graphml")
    chunk_store = json.loads((index_dir / "kv_store_text_chunks.json").read_text(encoding="utf-8"))
    return graph, chunk_store, index_dir


def build_chunk_mappings(graph, chunk_ids):
    """Build the provenance maps that connect chunks to nodes/edges and back.

    参数:
        graph: NetworkX 图对象，其中节点/边含 source_id provenance。
        chunk_ids: 目标 chunk id 列表。

    返回:
        chunk_to_nodes, chunk_to_edges, node_to_chunks, edge_to_chunks 四个映射。
    """
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


def build_chunk_neighborhoods(chunk_store, radius=1):
    """Build local same-document chunk neighborhoods.

    参数:
        chunk_store: chunk 数据字典。
        radius: 以同一文档中 chunk 顺序为半径构建邻居。

    返回:
        每个 chunk 与其同文档邻居 chunk id 集合。
    """
    by_doc = {}
    for chunk_id, chunk in chunk_store.items():
        full_doc_id = chunk.get("full_doc_id")
        if not full_doc_id:
            continue
        by_doc.setdefault(full_doc_id, []).append((chunk.get("chunk_order_index", -1), chunk_id))

    neighborhoods = {chunk_id: set() for chunk_id in chunk_store}
    for _, items in by_doc.items():
        items.sort()
        ordered_ids = [chunk_id for _, chunk_id in items]
        for index, chunk_id in enumerate(ordered_ids):
            left = max(0, index - radius)
            right = min(len(ordered_ids), index + radius + 1)
            neighborhoods[chunk_id].update(ordered_ids[left:right])
            neighborhoods[chunk_id].discard(chunk_id)
    return neighborhoods


def load_baseline_answers(result_path: Path):
    """Load baseline outputs in the same list format used by our answers.

    参数:
        result_path: 基线结果 JSON 文件路径。

    返回:
        基线答案列表，若格式不对则抛出异常。
    """
    data = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in baseline file: {result_path}")
    return data
