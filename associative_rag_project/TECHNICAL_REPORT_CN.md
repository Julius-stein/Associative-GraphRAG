# Associative RAG 技术报告（算法实现版）

本文只描述当前代码怎样运行，不讨论方案取舍。重点是：

- 输入数据如何进入系统
- 每一步生成什么中间结构
- 每个关键函数的候选构造、过滤、排序和输出

相关代码：

- [main.py](/Users/Admin/projects/Association/associative_rag_project/main.py)
- [pipeline.py](/Users/Admin/projects/Association/associative_rag_project/pipeline.py)
- [retrieval.py](/Users/Admin/projects/Association/associative_rag_project/retrieval.py)
- [association.py](/Users/Admin/projects/Association/associative_rag_project/association.py)
- [organization.py](/Users/Admin/projects/Association/associative_rag_project/organization.py)
- [context.py](/Users/Admin/projects/Association/associative_rag_project/context.py)
- [llm_client.py](/Users/Admin/projects/Association/associative_rag_project/llm_client.py)
- [data.py](/Users/Admin/projects/Association/associative_rag_project/data.py)

## 1. 主调用链

CLI 在 [main.py](/Users/Admin/projects/Association/associative_rag_project/main.py)。

`run-all` 的实际调用顺序：

1. `command_run_all(...)`
2. `retrieve_corpus_queries(...)`
3. `run_query(...)`
4. `generate_answers(...)`
5. `run_winrate_judgement(...)`

其中 retrieval 主路径全部在 `run_query(...)` 中完成。

`run_query(...)` 的顺序：

1. `chunk_retriever.search(query, top_k)`
2. `detect_query_contract(query)`
3. `_select_contract_root_chunks(...)`
4. `score_root_nodes(...)`
5. `score_root_edges(...)`
6. `expand_associative_graph(...)`
7. `build_answer_facet_groups(...)`
8. `build_prompt_context(...)`

## 2. 输入结构

### 2.1 corpus

[data.py](/Users/Admin/projects/Association/associative_rag_project/data.py) 中：

- `load_graph_corpus(corpus_dir)` 读取 `graph_chunk_entity_relation.graphml`
- 同时读取 `kv_store_text_chunks.json`

graph 中用到的字段：

- node: `source_id`, `entity_type`, `description`
- edge: `source_id`, `keywords`, `description`, `weight`

chunk store 中用到的字段：

- `content`
- `full_doc_id`
- `chunk_order_index`

### 2.2 provenance 映射

`build_chunk_mappings(...)` 会构建四个双向映射：

- `chunk_to_nodes`
- `chunk_to_edges`
- `node_to_chunks`
- `edge_to_chunks`

其中：

- node 的 `source_id` 决定它来自哪些 chunks
- edge 的 `source_id` 决定它来自哪些 chunks

### 2.3 chunk 邻接

`build_chunk_neighborhoods(chunk_store, radius=1)`：

- 对每个 `full_doc_id` 内的 chunks 按 `chunk_order_index` 排序
- 每个 chunk 与前后 `radius` 个 chunk 建邻接

所以 `chunk_neighbors` 表示“同一文档中的局部 band 邻接”。

## 3. query 标准化

`load_query_rows(...)` 支持两种输入：

- rewrites file
- questions file

最终每条 query row 的结构是：

```python
{
    "group_id": "...",
    "variant_id": "base",
    "query": "...",
    "base_query": "..."
}
```

## 4. 检索层

实现文件：[retrieval.py](/Users/Admin/projects/Association/associative_rag_project/retrieval.py)

### 4.1 BM25

`BM25Index.build(chunk_store)`：

1. 对每个 chunk 的 `content` 做 tokenize
2. 记录 term frequency
3. 建立 postings
4. 记录 `doc_lengths`
5. 计算 `avgdl`

`BM25Index.search(query, top_k)`：

1. tokenize query
2. 对每个 term 取 postings
3. 用 BM25 公式累积分数
4. 按分数降序截断
5. 用 top1 分数归一化为 `score_norm`

输出：

```python
{
    "chunk_id": ...,
    "score": ...,
    "score_norm": ...,
}
```

### 4.2 Dense

`DenseChunkIndex.load(vdb_file)`：

1. 读取 `vdb_chunks.json`
2. 解码 embedding matrix
3. 计算每行向量范数
4. 构建 `normalized_matrix`

`DenseChunkIndex.search(query_vector, top_k)`：

1. 对 query vector 归一化
2. 与 `normalized_matrix` 做点积
3. 取 top-k
4. 归一化为 `dense_score_norm`

### 4.3 Hybrid

`HybridChunkRetriever.search(query, top_k)`：

1. 视 `mode` 决定是否调用 bm25 / dense
2. 按 `chunk_id` 合并结果
3. 如果是 hybrid，使用

```python
retrieval_score = dense_weight * dense_score_norm + bm25_weight * bm25_score_norm
```

4. 按 `retrieval_score` 排序

## 5. query contract 检测

实现文件：[organization.py](/Users/Admin/projects/Association/associative_rag_project/organization.py)

函数：

- `detect_query_contract(query)`

输出只可能是：

- `section-grounded`
- `mechanism-grounded`
- `comparison-grounded`
- `theme-grounded`

### 5.1 判定规则

判定顺序是硬编码的：

1. 如果 query 命中 `SECTION_EXPLICIT_PHRASES`，直接判 `section-grounded`
2. 否则如果命中 `COMPARISON_PHRASES`，判 `comparison-grounded`
3. 否则如果命中 section list pattern，再判 `section-grounded`
4. 否则检查 mechanism 显式 cue
5. 否则检查 `how...` / `in what ways...` 与 `MECHANISM_LINK_VERBS` 的组合
6. 否则检查 `THEME_REASON_CUES`
7. 最后默认 `theme-grounded`

这里没有训练分类器，也没有调用 LLM。

## 6. root 选择

实现文件：[retrieval.py](/Users/Admin/projects/Association/associative_rag_project/retrieval.py)

主函数：

- `select_diverse_root_chunks(...)`

### 6.1 候选构造

对于每个 `candidate_hit`，函数先补一组派生字段：

- `base_score = _root_base_score(item)`
- `full_doc_id`
- `chunk_order_index`
- `graph_nodes`
- `graph_edges`
- `relation_categories`

其中：

- `_root_base_score(...)` 优先使用 `dense_score_norm`，没有 dense 时回退到 `score_norm`
- `_chunk_graph_signature(...)` 返回该 chunk 覆盖到的 nodes 和 edges
- `_chunk_relation_categories(...)` 从 chunk 覆盖到的 edges 上提取 relation category

### 6.2 初始排序

候选先按以下 key 排序：

```python
(-base_score, -len(graph_nodes), chunk_id)
```

即：

1. retrieval stronger first
2. 节点覆盖更多者优先
3. 最后按 chunk_id 稳定排序

### 6.3 第一阶段选择

`selected` 初始为空，`deferred` 初始为空。

第一阶段逐个扫描候选：

1. 第一个候选直接进入 `selected`
2. 之后对每个候选计算：
   - `same_doc_count`
   - `same_band`
   - `overlaps`
   - `max_overlap`

其中：

- `same_doc_count`：当前文档已经选了几个 roots
- `same_band`：是否与已选 root 在同文档且 `chunk_order_index` 距离不超过 `same_doc_window`
- `max_overlap`：与任何已选 root 的 provenance overlap 最大值

provenance overlap 定义为：

```python
len((nodes_a ∪ edges_a) ∩ (nodes_b ∪ edges_b)) / len((nodes_a ∪ edges_a) ∪ (nodes_b ∪ edges_b))
```

第一阶段拒绝条件：

- `same_doc_count >= max_same_doc_roots`
- 或 `same_band == True`
- 或 `max_overlap > max_provenance_overlap`

被拒绝的进入 `deferred`。

### 6.4 信息熵增益

系统在 root selection 内保留了 relation entropy。

函数：

- `relation_entropy(categories)`

对于 deferred 候选和 relaxed 阶段选中的候选，都会计算：

```python
before = relation_entropy(selected_relation_categories)
after = relation_entropy(selected_relation_categories + candidate["relation_categories"])
entropy_gain = max(after - before, 0.0)
```

也就是说，熵不是直接对 query 打分，而是衡量“把这个 chunk 加进 roots 后，relation category 分布是否变得更丰富”。

### 6.5 第二阶段 relaxed pass

如果第一阶段后 `len(selected) < top_k`，则进入第二阶段。

`deferred` 的排序 key：

```python
(
    doc_counts.get(full_doc_id, 0),
    -entropy_gain,
    current_max_overlap_to_selected,
    -base_score,
    -(len(graph_nodes) + len(graph_edges)),
    chunk_id,
)
```

即：

1. 优先文档占用更少的候选
2. 再优先 entropy gain 更高的候选
3. 再优先与已选集合重叠更小的候选
4. 再看 retrieval score
5. 再看图覆盖量

第二阶段仍然保留约束：

- 禁止 `same_band`
- 限制 `same_doc_count < relaxed_max_same_doc_roots`
- 限制 `max_overlap <= relaxed_max_provenance_overlap`

### 6.6 输出

最后返回的 root chunk 条目中，会保留：

- `novelty_gain`
- `entropy_gain`
- `max_selected_overlap`
- `selection_score`

这些字段会进入 retrieval JSON。

## 7. section 题的 root 特化

实现文件：[pipeline.py](/Users/Admin/projects/Association/associative_rag_project/pipeline.py)

函数：

- `_select_contract_root_chunks(...)`

当 `query_contract != "section-grounded"` 时：

- 直接调用 `select_diverse_root_chunks(...)`

当 `query_contract == "section-grounded"` 时：

1. 在前 `max(top_k * 4, 12)` 个候选上统计每个 `full_doc_id` 的累计 `score_norm`
2. 选择得分最高的文档 `dominant_doc_id`
3. 仅保留该文档内的候选
4. 在这个子集上再次调用 `select_diverse_root_chunks(...)`

这样 section 题的 roots 会从一开始就被压到单文档带内。

## 8. root nodes / root edges 打分

实现文件：[retrieval.py](/Users/Admin/projects/Association/associative_rag_project/retrieval.py)

### 8.1 `score_root_nodes(...)`

对每个 root node 计算：

- `query_rel`
- `support`
- `chunk_alignment`

其中：

- `query_rel` 是 query 和 `node_id + entity_type + description` 的 lexical overlap
- `support, chunk_alignment = support_score(node_to_chunks[node_id], root_chunk_score_lookup)`

排序 key：

```python
(-support, -chunk_alignment, -query_rel, node_id)
```

### 8.2 `score_root_edges(...)`

对每个 root edge 计算：

- `query_rel`
- `support`
- `chunk_alignment`
- `weight_term = log1p(weight) / 5`

排序 key：

```python
(-support, -weight_term, -query_rel, edge_id)
```

它们的作用不是最终答案排序，而是为 association 提供 seed nodes / edges。

## 9. association 总体结构

实现文件：[association.py](/Users/Admin/projects/Association/associative_rag_project/association.py)

主函数：

- `expand_associative_graph(...)`

association 每一轮都执行两段：

1. `bridge_association(...)`
2. `coverage_association(...)`

而两段内部又同时使用 graph-side 和 chunk-side 信号，因此形成 2x2 结构：

- graph-bridge
- chunk-bridge
- graph-coverage
- chunk-coverage

## 10. association 中的 contract-aware 过滤

### 10.1 `_candidate_contract_features(...)`

任何 path / chunk / edge / node 候选都会先计算：

- `source_docs`
- `root_docs`
- `root_alignment`
- `query_overlap`
- `same_root_doc`
- `section_consistent`

定义如下：

`source_docs`

- 候选来源 chunks 的 `full_doc_id` 集合

`root_docs`

- roots 对应 chunks 的 `full_doc_id` 集合

`root_alignment`

- `support_score(source_chunk_ids, root_chunk_score_lookup)` 的第二个输出

`query_overlap`

- query 与候选 preview text 的 lexical overlap

`same_root_doc`

- `source_docs` 与 `root_docs` 是否有交集

`section_consistent`

- 只有在 `section-grounded` 时有效
- 检查 `source_docs ⊆ root_docs`

### 10.2 允许条件

不同 contract 的 allow 规则不同：

`section-grounded`

```python
allowed = bool(section_consistent)
```

`mechanism-grounded`

```python
allowed = same_root_doc or root_alignment >= 0.08 or query_overlap >= 0.035
```

`comparison-grounded`

```python
allowed = same_root_doc or root_alignment >= 0.05 or query_overlap >= 0.025
```

`theme-grounded`

```python
allowed = same_root_doc or root_alignment >= 0.03 or query_overlap >= 0.02
```

只有 `allowed == True` 的候选会进入排序。

### 10.3 `_contract_sort_key(...)`

contract-aware 排序并不替代 primary rank，而是把 contract 信号和 primary rank 组合。

`section-grounded`

```python
(
    -section_consistent,
    -same_root_doc,
    -root_alignment,
    -query_overlap,
    *primary_fields
)
```

`mechanism-grounded`

```python
(
    *primary_fields,
    -query_overlap,
    -same_root_doc,
    -root_alignment
)
```

`comparison-grounded`

```python
(
    *primary_fields,
    -same_root_doc,
    -query_overlap,
    -root_alignment
)
```

`theme-grounded`

```python
(
    *primary_fields,
    -query_overlap,
    -same_root_doc,
    -root_alignment
)
```

## 11. graph-bridge association

函数：

- `_graph_bridge_association(...)`

### 11.1 当前连通分量

先用 `build_root_components(current_nodes, current_edges)` 构造当前子图的连通分量：

- `components`
- `node_to_component`

### 11.2 seed nodes

seed 来自：

1. `top_nodes` 中的 node ids
2. `top_edges` 中两端的 node ids

合并去重后得到 `seed_nodes`

### 11.3 候选路径构造

对每个 `source_id in seed_nodes`：

1. 调用 `nx.single_source_shortest_path(graph, source_id, cutoff=max_hop)`
2. 遍历所有 `target_id in current_nodes`
3. 只保留跨 component 的路径
4. 去掉长度 <= 1 的路径
5. 用正向/逆向较小的 tuple 做 canonical path 去重
6. 如果该路径上的 edge 全都已经在 `current_edges` 中，则跳过

### 11.4 路径特征

每条 path 计算：

- `path_chunks`
- `bridge_gain`
- `new_source_count`
- `path_length`
- contract features

其中：

`bridge_gain`

- `_path_bridge_signature(path, node_to_component)`
- 本质上只看 path 首尾是否来自不同 component

`new_source_count`

- `len(path_chunks - covered_chunks)`，再经 `_bounded_gain(...)` 截断

### 11.5 排序

primary rank 为：

```python
(-bridge_gain, -new_source_count, path_length, path)
```

然后通过 `_contract_sort_key(...)` 注入 contract 优先级。

### 11.6 输出

输出：

- `selected_paths`
- `selected_path_nodes`
- `selected_path_edges`

## 12. chunk-bridge association

函数：

- `_chunk_bridge_association(...)`

### 12.1 候选 chunk 集合

1. 先取当前覆盖 chunks：

```python
covered_chunks = _covered_chunk_ids(current_nodes, current_edges, ...)
```

2. 再扩一层 local band：

```python
covered_band = _expand_chunk_band(covered_chunks, chunk_neighbors)
```

3. 再把 `covered_band` 的邻居并进来

4. 最后去掉已经覆盖的 chunks

### 12.2 每个 chunk 的特征

对每个候选 chunk：

- `chunk_nodes`
- `chunk_edges`
- `new_nodes`
- `new_edges`
- `component_ids`
- `bridge_gain`
- `frontier_touch`
- `new_source_count`
- `root_overlap`
- `root_band_alignment`
- contract features

其中：

`bridge_gain`

- 如果 chunk 同时碰到两个及以上当前 component，则取 1，否则 0

`frontier_touch`

- `_chunk_bridge_touch(...)`
- 由三部分相加：
  - 与当前 nodes 的交
  - 与当前 edges 的交
  - 与 `covered_band` 的 chunk 邻接触碰数

`root_band_alignment`

- 取 `_expand_chunk_band({chunk_id})` 内各 chunk 在 `root_chunk_score_lookup` 中的最大值

### 12.3 排序

primary rank：

```python
(
    -bridge_gain,
    -frontier_touch,
    -new_source_count,
    -(new_node_count + new_edge_count),
    -root_overlap,
    -root_band_alignment,
    chunk_id
)
```

再通过 `_contract_sort_key(...)` 注入 contract 优先级。

## 13. coverage association

函数：

- `coverage_association(...)`

它同时处理 edge candidates、node candidates，以及 chunk-side coverage。

### 13.1 当前状态量

先构造：

- `current_categories = build_current_relation_categories(current_edges, graph)`
- `covered_chunks`
- `covered_chunk_band`
- `root_chunk_id_set`

### 13.2 candidate edges

来源有两类：

1. 所有 `current_nodes` 的图邻边
2. `covered_chunk_band` 中 chunk 所覆盖但尚未进入 `current_edges` 的 edges

对于每条 candidate edge，计算：

- `source_chunks`
- `local_source_chunks`
- `expanded_band`
- `new_source_count`
- `category`
- `new_relation`
- `chunk_alignment`
- `root_overlap`
- `coverage_gain`
- contract features

其中：

```python
coverage_gain = new_source_count if new_source_count > 0 else new_relation
```

也就是说：

- 如果能带来新 source band，优先使用这个增益
- 否则退回到“是否引入新 relation type”

candidate edge 只有满足以下条件才保留：

1. contract allow
2. `coverage_gain >= semantic_edge_min_score`
3. `coverage_gain > 0`

排序 primary key：

```python
(-coverage_gain, -new_relation, -root_overlap, -chunk_alignment, edge)
```

### 13.3 candidate nodes

来源有两类：

1. `current_nodes` 的图邻接点
2. `covered_chunk_band` 中 chunk 覆盖但尚未进入 `current_nodes` 的点

对于每个 candidate node，计算：

- `source_chunks`
- `local_source_chunks`
- `expanded_band`
- `new_source_count`
- `relation_categories`
- `bridge_strength`
- `new_relation_count`
- `chunk_alignment`
- `root_overlap`
- `coverage_gain`
- contract features

其中：

`bridge_strength`

- 该 node 有多少邻居已经在 `current_nodes` 或 `expanded_nodes` 中

`coverage_gain`

```python
coverage_gain = new_source_count if new_source_count > 0 else new_relation_count
```

candidate node 保留条件：

1. contract allow
2. `coverage_gain >= semantic_node_min_score`
3. 或者 `coverage_gain <= 0` 但 `bridge_strength > 0`

排序 primary key：

```python
(-coverage_gain, -bridge_strength, -root_overlap, -chunk_alignment, node_id)
```

### 13.4 chunk-side coverage

`coverage_association(...)` 最后还会调用：

- `_chunk_coverage_association(...)`

它的候选来自 `covered_chunk_band` 的邻居 chunks。

对每个 chunk 计算：

- `new_node_count`
- `new_edge_count`
- `new_relation_count`
- `new_source_count`
- `root_overlap`
- `root_band_alignment`
- contract features

排序 primary key：

```python
(
    -(new_node_count + new_edge_count),
    -new_relation_count,
    -new_source_count,
    -root_overlap,
    -root_band_alignment,
    chunk_id
)
```

### 13.5 输出

`coverage_association(...)` 返回：

- `selected_edges`
- `selected_nodes`
- `selected_chunks`
- `final_nodes`
- `final_edges`

其中 `final_nodes/final_edges` 是把当前子图与本轮新增内容合并后的结果。

## 14. 多轮 expansion

函数：

- `expand_associative_graph(...)`

循环逻辑：

1. 对当前 `current_nodes/current_edges` 重新执行 `score_root_nodes/edges`
2. 调用 `bridge_association(...)`
3. 把 structural 结果并入当前子图
4. 调用 `coverage_association(...)`
5. 用 semantic 输出覆盖 `current_nodes/current_edges`
6. 记录本轮 round summary
7. 如果本轮 structural 和 semantic 都没有新增内容，则提前结束

最终输出：

- `final_nodes`
- `final_edges`
- `all_structural_nodes`
- `all_structural_edges`
- `all_semantic_nodes`
- `all_semantic_edges`
- `rounds`
- `last_structural_output`

`last_structural_output` 后面会被 organization 用来切 bridge regions。

## 15. 从 final graph 划分 evidence regions

实现文件：[organization.py](/Users/Admin/projects/Association/associative_rag_project/organization.py)

总入口：

- `collect_overlapping_regions(...)`

它会生成三类 region：

- root regions
- bridge regions
- theme regions

### 15.1 `_chunk_local_index(...)`

先把 final subgraph 投影回 chunk：

- `chunk_to_final_nodes`
- `chunk_to_final_edges`

### 15.2 `_root_chunk_band(...)`

对所有 roots 扩一层 chunk 邻接：

```python
root_chunk_band = roots ∪ neighbors(roots)
```

这个 band 后面用于判断 region 是否仍与 roots 保持局部连接。

### 15.3 `_make_region(...)`

这是所有 region 的统一构造器。

输入：

- `region_kind`
- `root_chunk_ids`
- `node_ids`
- `edge_ids`
- `root_chunk_band`

内部步骤：

1. 调用 `_supporting_chunks_for_region(node_ids, edge_ids, ...)`
   从 nodes / edges 反推 supporting chunks
2. 调用 `_rank_chunks(...)`
   先按是否 root，再按 query overlap 排 chunk
3. 取前 5 个作为 `anchor_chunk_ids`
4. 用 `_relation_themes(...)` 汇总 relation themes
5. 用 `_focus_entities(...)` 选 query 最相关的实体
6. 用 `_evidence_descriptor_text(...)` 构建 `descriptor_text`
7. 计算 `root_connected`

`root_connected` 的定义：

```python
bool(
    anchor_chunk_ids 与 root_chunk_ids 有交
    or supporting_chunk_ids 与 root_chunk_band 有交
)
```

### 15.4 root regions

函数：

- `_collect_root_regions(...)`

对每个 root chunk：

1. 从 `chunk_to_final_nodes/chunk_to_final_edges` 找到 seed nodes / edges
2. 找这些 seed 覆盖到的 supporting chunks
3. 按 query overlap 对 supporting chunks 排序
4. 取前 4 个 supporting chunks，把它们的 final nodes / edges 再并进局部集合
5. 额外补上“节点都已在局部集合中的 final edges”
6. 交给 `_make_region(...)`

所以 root region 不是单 chunk，而是“以 root chunk 为中心扩一圈局部证据”。

### 15.5 bridge regions

函数：

- `_collect_bridge_regions(...)`

bridge region 有两种来源。

第一种：graph bridge path

1. 读取 `last_structural_output["selected_paths"]`
2. 每条 path 先把路径上的 nodes / edges 放进局部集合
3. 再取 path 覆盖到的 chunks，按 query overlap 排序
4. 取前 3 个 chunks，把它们覆盖到的 nodes / edges 并入
5. 再补齐局部闭包内的 final edges
6. 交给 `_make_region(...)`

第二种：chunk-side bridge

1. 读取 `last_structural_output["selected_chunks"]`
2. 用 chunk 自带的 `node_ids/edge_ids` 作为局部集合
3. 交给 `_make_region(...)`

两种 bridge region 最后都要求：

- `region is not None`
- `region.root_connected == True`

### 15.6 theme regions

函数：

- `_collect_theme_regions(...)`

步骤：

1. 遍历全部 `final_edges`
2. 按 `normalize_relation_category(...)` 分组
3. 排序 key：

```python
(-lexical_overlap_score(query, theme), -len(edges), theme)
```

4. 只取前 `max_themes=5`
5. 对每个 theme：
   - 先把 theme 对应的 edges 和两端 nodes 放进局部集合
   - 反推 supporting chunks
   - 要求 supporting chunks 与 `root_chunk_band` 有交
   - 对 supporting chunks 排序，取前 4 个，再把这些 chunk 覆盖到的 final nodes / edges 并入
   - 交给 `_make_region(...)`

theme region 的作用是把 final graph 中最强的 relation themes 单独抽出来。

## 16. 从 regions 组装 facet groups

统一 group 构造函数：

- `_build_group(...)`

它会把一组 regions 合并成一个 facet group。

### 16.1 `_build_group(...)` 的聚合逻辑

给定一组 regions：

1. 合并所有 `root_chunk_ids`
2. 合并 `node_ids`
3. 合并 `edge_ids`
4. 合并 `supporting_chunk_ids`
5. 统计 `region_kinds`
6. 统计 `relation_themes`
7. 统计 `focus_entities`
8. 合并 `doc_ids`
9. 合并 `growth_traces`
10. 对聚合后的 `anchor_chunk_ids` 再用 `_rank_chunks(...)` 排一次

然后构造：

- `group_score = lexical_overlap_score(query, descriptor_text)`
- `query_rel`
- `anchor_support`
- `root_anchor_count`
- `group_summary`

### 16.2 机制题 group

函数：

- `_build_mechanism_groups(...)`

逻辑：

1. 建 `root_by_chunk`，把 root regions 按 root chunk 反查
2. 遍历每个 bridge region
3. 把该 bridge region 相关的 root regions 合并进来
4. 再尝试把与它共享 roots 或 supporting chunks 的 theme regions 合并进来
5. 标签优先取 bridge region 的 primary theme，若有次主题则写成 `primary -> secondary`
6. 如果没有 bridge candidates，则退化为用 root/theme regions 按 primary theme 分组
7. 最后过滤掉：
   - 不含 bridge kind
   - 且 `edge_count < 3`
   的 groups

### 16.3 对比题 group

函数：

- `_build_comparison_groups(...)`

逻辑：

1. 每个 root region 先单独形成一组
2. 如果 bridge region 覆盖两个及以上 roots，再额外形成一个 `contrast around ...` 组

### 16.4 主题题 group

函数：

- `_build_theme_groups(...)`

逻辑：

1. 遍历 root/theme/bridge regions
2. 过滤：
   - bridge region 要求 `query_rel >= 0.12`
   - root/theme region 要求 `query_rel >= 0.05`
3. 分组键：

```python
(_primary_theme(region), root_key)
```

其中 `root_key` 优先取第一个 `root_chunk_id`，没有时退化到第一个 `doc_id`

### 16.5 section 题 group

函数：

- `_build_section_groups(...)`

逻辑：

1. 对每个 root region，取前 3 个 anchor chunks 作为 seed
2. 对每个 seed 在同文档内扩 `radius=2` 的 chunk band
3. 把这个 band 覆盖到的 final nodes / edges 收集起来
4. 再构成一个 `section` region
5. label 形如：

```python
section band {doc_suffix}:{start_order}-{end_order}
```

## 17. group 选择算法

函数：

- `_select_groups(groups, limit, distinct_key_fn, coverage_key_fn=None)`

这是当前 organization 中最重要的选择器。

### 17.1 group 排序基线

所有 groups 先按 `_group_rank_key(...)` 排序：

```python
(
    -root_anchor_count,
    -query_rel,
    -len(region_kinds),
    -anchor_support,
    -node_count,
    facet_label
)
```

### 17.2 greedy coverage-first

第一阶段是 coverage-first greedy：

1. 维护：
   - `selected`
   - `used`
   - `covered`
2. 每轮扫描所有未选 group
3. 计算该 group 带来的新增 coverage：

```python
gain = len(set(coverage_key_fn(group)) - covered)
```

4. 只考虑 `gain > 0` 的候选
5. 选择：
   - `gain` 最大
   - 若相同，则 `_group_rank_key` 更优

6. 选中后更新：
   - `selected`
   - `used`
   - `covered`

### 17.3 fallback 填充

如果 greedy 后还没满：

1. 再按排好序的 groups 顺序扫描
2. 过滤：
   - 已经选中
   - `distinct_key_fn(group)` 已在 `used`
   - `anchor_chunk_ids` 与已选 group 完全相同

3. 通过的直接补入

### 17.4 不同 contract 的 coverage 键

不同 `_build_*_groups(...)` 给 `_select_groups(...)` 传入不同的 `coverage_key_fn`。

常见 coverage 键包括：

- `label`
- `theme`
- `doc`
- `root`
- `kinds`
- `band`

因此 `_select_groups(...)` 是一个统一 greedy 框架，而“覆盖什么”由不同 contract 的调用方决定。

## 18. context 组装

实现文件：[context.py](/Users/Admin/projects/Association/associative_rag_project/context.py)

### 18.1 supporting chunk 排序

`rank_supporting_chunks(...)`：

1. 被 final nodes 覆盖到的 chunk 加分
2. 被 final edges 覆盖到的 chunk 加分
3. root chunks 再加 1

最后按分数降序得到全局 chunk 排名。

### 18.2 source chunk 选择

`choose_diverse_source_chunks(...)` 分三轮：

第一轮：

- 每个 facet group 尝试贡献 1 个 chunk

第二轮：

- 每个 facet group 再尝试贡献 1 个 chunk

第三轮：

- 按全局 `ranked_chunk_ids` 补齐

每次加入 chunk 前都会经过 `try_add(...)`，检查：

- 是否已选
- 是否超过 `max_source_chunks`
- 是否违反 `_violates_local_band_cap(...)`
- 是否超过 `max_source_word_budget`

### 18.3 最终 prompt_context

`build_prompt_context(...)` 输出一个固定模板字符串，包含：

- Root Chunks
- Focused Entities
- Focused Relations
- Facet Groups
- Coverage Checklist
- Facet Group Dossiers
- Sources

同时返回：

```python
{
    "context": str,
    "selected_source_word_count": int,
    "selected_source_chunk_count": int,
}
```

## 19. 最终 retrieval record

`run_query(...)` 返回一条记录，核心字段：

- `candidate_root_chunks`
- `root_chunks`
- `query_contract`
- `stats`
- `top_root_nodes`
- `top_root_edges`
- `rounds`
- `structural_association`
- `semantic_association`
- `facet_groups`
- `knowledge_groups`
- `prompt_context`

其中：

- `structural_association = expansion["last_structural_output"]`
- `semantic_association` 只保留被选中的 semantic edges/nodes 预览

## 20. 生成层

实现文件：[llm_client.py](/Users/Admin/projects/Association/associative_rag_project/llm_client.py)

### 20.1 prompt 构造

`build_generation_prompt(query, prompt_context, query_contract)`：

1. 先写固定 answer requirements
2. 插入 contract-specific hints
3. 如果 `_is_broad_theme_query(...)` 为真，再增加 broad theme 覆盖提示
4. 把 `query` 和 `prompt_context` 拼进去

### 20.2 LLM 调用

`generate_answers(...)` 会对每条 retrieval record：

1. 构造 prompt
2. 调 `llm_client.generate(...)`
3. 收集：
   - `group_id`
   - `query`
   - `query_contract`
   - `model_answer`
   - `stats`

## 21. 总结

当前系统可以被精确描述为：

1. 用 lexical / dense retrieval 找候选 chunks
2. 用多样性、局部 band 约束和 relation entropy 选 roots
3. 用 query contract 决定联想扩展的 allow rule 和 sort key
4. 用 2x2 association 结构扩展 final graph
5. 从 final graph 中切出 root / bridge / theme evidence regions
6. 再把 regions 合并成 facet groups，并用 greedy coverage-first 选最终 groups
7. 把 groups 和 source chunks 打包成 evidence package 交给 LLM

如果要读代码实现细节，推荐顺序：

1. [pipeline.py](/Users/Admin/projects/Association/associative_rag_project/pipeline.py)
2. [retrieval.py](/Users/Admin/projects/Association/associative_rag_project/retrieval.py)
3. [association.py](/Users/Admin/projects/Association/associative_rag_project/association.py)
4. [organization.py](/Users/Admin/projects/Association/associative_rag_project/organization.py)
5. [context.py](/Users/Admin/projects/Association/associative_rag_project/context.py)
6. [llm_client.py](/Users/Admin/projects/Association/associative_rag_project/llm_client.py)
