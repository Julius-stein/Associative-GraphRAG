# Associative RAG 技术报告（当前代码实现）

本文档只描述当前代码实际运行的实现，不讨论旧版本路线，也不描述已经删除的控制分支。主代码文件如下：

- [main.py](/Users/Admin/projects/Association/associative_rag_project/main.py)
- [pipeline.py](/Users/Admin/projects/Association/associative_rag_project/pipeline.py)
- [data.py](/Users/Admin/projects/Association/associative_rag_project/data.py)
- [retrieval.py](/Users/Admin/projects/Association/associative_rag_project/retrieval.py)
- [association.py](/Users/Admin/projects/Association/associative_rag_project/association.py)
- [organization.py](/Users/Admin/projects/Association/associative_rag_project/organization.py)
- [context.py](/Users/Admin/projects/Association/associative_rag_project/context.py)
- [llm_client.py](/Users/Admin/projects/Association/associative_rag_project/llm_client.py)
- [judge.py](/Users/Admin/projects/Association/associative_rag_project/judge.py)

## 1. 系统总览

当前系统是一条固定流水线：

1. 从 chunk corpus 中检索 query 的候选 chunk。
2. 从候选 chunk 中选出少量 root chunks。
3. 把 root chunks 投影到图上得到 root nodes 和 root edges。
4. 在图结构和 chunk 邻域上做联想扩展，得到 final graph。
5. 从 final graph 中抽取 evidence regions，并按 query contract 组织成 facet groups。
6. 从 facet groups 和 supporting chunks 组装 evidence package。
7. 用 LLM 基于 evidence package 生成答案。
8. 用 LLM judge 对 candidate 与 baseline 做双顺序比较，并按 query contract 聚合总体结论。

这条链路里，联想和组织是分开的：

- [association.py](/Users/Admin/projects/Association/associative_rag_project/association.py) 只负责扩图。
- [organization.py](/Users/Admin/projects/Association/associative_rag_project/organization.py) 只负责把 final graph 划分为可回答问题的 evidence groups。

## 2. CLI 与主入口

CLI 定义在 [main.py](/Users/Admin/projects/Association/associative_rag_project/main.py)。

支持四个主要命令：

- `retrieve`
- `answer`
- `judge`
- `run-all`

`run-all` 的执行顺序是：

1. `retrieve_corpus_queries(...)`
2. `generate_answers(...)`
3. `run_winrate_judgement(...)`

当前主路径没有 adaptive controller，也没有 `query_style` 控制变量。命令行中的核心检索参数是：

- `--top-chunks`
- `--chunk-candidate-multiplier`
- `--candidate-pool-size`
- `--retrieval-mode`
- `--top-root-nodes`
- `--top-root-edges`
- `--max-hop`
- `--path-budget`
- `--semantic-edge-budget`
- `--semantic-node-budget`
- `--association-rounds`
- `--group-limit`
- `--max-source-chunks`
- `--max-source-word-budget`

`--limit-groups` 与 `--limit` 是同一参数；输出目录由 `--output-dir` 控制。

## 3. 数据加载与基础结构

数据读取在 [data.py](/Users/Admin/projects/Association/associative_rag_project/data.py)。

### 3.1 语料输入

每个 corpus 目录需要：

- `graph_chunk_entity_relation.graphml`
- `kv_store_text_chunks.json`

若使用 dense 或 hybrid retrieval，还需要：

- `vdb_chunks.json`

`load_graph_corpus(...)` 负责读入：

- `graph`：实体-关系图
- `chunk_store`：chunk 文本字典

### 3.2 provenance 映射

`build_chunk_mappings(...)` 构造四个双向映射：

- `chunk_to_nodes`
- `chunk_to_edges`
- `node_to_chunks`
- `edge_to_chunks`

这四个映射是整个系统的核心桥梁。因为后续所有步骤都在反复做两类转换：

- chunk -> graph
- graph -> supporting chunks

### 3.3 chunk 邻域

`build_chunk_neighborhoods(...)` 按 `full_doc_id + chunk_order_index` 为每个 chunk 建局部邻域。默认半径是 `1`。这个邻域在后面有三种用途：

- root 阶段避免同 band 重复起点
- association 阶段做 chunk-side bridge / coverage
- organization 阶段构造 root chunk band

## 4. 检索层

检索代码在 [retrieval.py](/Users/Admin/projects/Association/associative_rag_project/retrieval.py)。

### 4.1 三种检索器

系统支持三种模式：

- `BM25Index`
- `DenseChunkIndex`
- `HybridChunkRetriever`

`HybridChunkRetriever.search(...)` 的逻辑是：

1. 先分别取 BM25 与 dense top-k。
2. 以 `chunk_id` 合并结果。
3. 在 hybrid 模式下计算：

`retrieval_score = dense_weight * dense_score_norm + bm25_weight * bm25_score_norm`

4. 依据 `(retrieval_score, dense_score_norm, bm25_score_norm)` 排序。

当前主实验默认可用 dense，也支持 hybrid 与 bm25。

### 4.2 candidate pool

在 [pipeline.py](/Users/Admin/projects/Association/associative_rag_project/pipeline.py) 的 `run_query(...)` 里，实际送入根选择的候选池大小为：

`max(top_chunks, top_chunks * chunk_candidate_multiplier, candidate_pool_size)`

这一步的作用是给 root 选择留出冗余候选空间，使 root 选择阶段可以在高分候选内部再做去重与分散化。

## 5. Root Chunk 选择

核心函数是 [select_diverse_root_chunks(...)](/Users/Admin/projects/Association/associative_rag_project/retrieval.py#L247)。

### 5.1 输入与目标

输入：

- `candidate_hits`
- `chunk_store`
- `chunk_to_nodes`
- `chunk_to_edges`
- `top_k`

输出：

- 一个长度不超过 `top_k` 的 root chunk 列表

目标是从高分候选中挑出少量、多起点、低重复的 roots。

### 5.2 预处理特征

每个 candidate 会先被补齐下列字段：

- `base_score`
- `full_doc_id`
- `chunk_order_index`
- `graph_nodes`
- `graph_edges`

其中：

- `base_score` 由 `_root_base_score(...)` 决定，优先使用 `dense_score_norm`，否则回退到 `score_norm`
- `graph_nodes / graph_edges` 由 `_chunk_graph_signature(...)` 给出，用于估计 provenance 重叠

### 5.3 第一阶段选择

第一阶段按 `(base_score, graph_nodes 数量, chunk_id)` 排序后依次遍历。

第一个 candidate 直接入选。

后续 candidate 需要同时满足：

- 同文档已选 root 数量小于 `max_same_doc_roots`
- 与已选 root 不在同一 local band
- 与已选 root 的最大 provenance overlap 不超过 `max_provenance_overlap`

这里的关键判断函数是：

- `_same_doc_band(...)`
- `_provenance_overlap(...)`

不满足者进入 `deferred`。

### 5.4 第二阶段 relaxed pass

若第一阶段选不满 `top_k`，对 `deferred` 再做一次放宽筛选。

放宽后仍保留两条硬约束：

- 不允许同 band roots
- 不允许超过 `relaxed_max_same_doc_roots`

同时 provenance overlap 门槛放宽到 `relaxed_max_provenance_overlap`。

### 5.5 输出字段

最终每个 root chunk 记录包含：

- `chunk_id`
- `score_norm`
- `selection_score`
- `base_score`
- `full_doc_id`
- `chunk_order_index`
- `novelty_gain`
- `max_selected_overlap`

这里没有多因素加权总分。它是规则筛选加字典序排序。

## 6. Root Node / Root Edge 打分

roots 投影到图上后，系统会分别对 root nodes 和 root edges 轻量打分。

### 6.1 root node

函数：[score_root_nodes(...)](/Users/Admin/projects/Association/associative_rag_project/retrieval.py#L415)

为每个 node 计算：

- `support`
- `chunk_alignment`
- `query_rel`

其中：

- `support, chunk_alignment = support_score(node_to_chunks[node], root_chunk_score_lookup)`
- `query_rel = lexical_overlap_score(query, node 文本描述)`

排序键是：

`(-support, -chunk_alignment, -query_rel, id)`

### 6.2 root edge

函数：[score_root_edges(...)](/Users/Admin/projects/Association/associative_rag_project/retrieval.py#L447)

为每条 edge 计算：

- `support`
- `chunk_alignment`
- `query_rel`
- `weight_term = log1p(weight) / 5`

排序键是：

`(-support, -weight_term, -query_rel, edge)`

这一步的作用是为联想阶段提供高质量的 structural seeds。

## 7. 联想层

联想代码在 [association.py](/Users/Admin/projects/Association/associative_rag_project/association.py)。

### 7.1 设计结构

当前联想是一个明确的 2 x 2：

- graph bridge
- chunk bridge
- graph coverage
- chunk coverage

联想的输入是：

- 当前子图 `current_nodes/current_edges`
- root chunk 集合
- provenance 映射
- chunk 邻域

联想的输出是：

- structural 新增节点和边
- semantic 新增节点和边
- 最终 `final_nodes/final_edges`

### 7.2 structural association

入口函数是 `bridge_association(...)`。

它把 `path_budget` 一分为二：

- `graph_bridge_budget = path_budget // 2`
- `chunk_bridge_budget = path_budget - graph_bridge_budget`

#### 7.2.1 graph bridge

函数：`_graph_bridge_association(...)`

算法步骤：

1. 用 `build_root_components(...)` 构造当前图的连通分量。
2. 从 `top_nodes` 和 `top_edges` 提取 seed nodes。
3. 对每个 seed node 做 `nx.single_source_shortest_path(..., cutoff=max_hop)`。
4. 只保留连接不同 current components 的 path。
5. 过滤掉 edge 全在当前图中的 path。
6. 为每条 path 计算：
   - `bridge_gain`
   - `new_source_count`
   - `path_length`
7. 排序键：

`(-bridge_gain, -new_source_count, path_length, path)`

8. 取前 `path_budget` 条。

这里的 `bridge_gain` 由 `_path_bridge_signature(...)` 计算，本质上衡量 path 是否连接了不同连通分量。

#### 7.2.2 chunk bridge

函数：`_chunk_bridge_association(...)`

算法步骤：

1. 先计算 `covered_chunks`。
2. 再扩展成 `covered_band = covered_chunks + chunk_neighbors`。
3. 从 `covered_band` 再向外扩一圈得到 candidate chunks。
4. 对每个 candidate chunk 计算：
   - `bridge_gain`：是否触达多个 current components
   - `frontier_touch`：与当前 frontier 的接触强度
   - `new_source_count`
   - `root_overlap`
   - `root_band_alignment`
5. 排序后取前 `chunk_budget` 个。

这里的重点是：chunk bridge 通过 chunk 邻域和 provenance 直接把局部证据带进 current graph。

### 7.3 semantic association

入口函数是 `coverage_association(...)`。

它的目标是扩充“新证据覆盖”，侧重把未覆盖的证据类型和 supporting chunks 加入当前子图。

#### 7.3.1 edge coverage

对 candidate edges，系统计算：

- `new_source_count`
- `new_relation`
- `root_overlap`
- `chunk_alignment`

其中覆盖增益定义为：

- 优先用 `new_source_count`
- 若没有新 source，再退到 `new_relation`

筛选条件：

- `coverage_gain >= semantic_edge_min_score`
- `coverage_gain > 0`

排序键：

`(-coverage_gain, -new_relation, -root_overlap, -chunk_alignment, edge)`

#### 7.3.2 node coverage

对 candidate nodes，系统计算：

- `new_source_count`
- `new_relation_count`
- `bridge_strength`
- `root_overlap`
- `chunk_alignment`

同样定义：

- 优先 `new_source_count`
- 否则退到 `new_relation_count`

筛选条件：

- `coverage_gain >= semantic_node_min_score`
- 并且 `coverage_gain > 0` 或 `bridge_strength > 0`

排序键：

`(-coverage_gain, -bridge_strength, -root_overlap, -chunk_alignment, id)`

#### 7.3.3 chunk coverage

函数：`_chunk_coverage_association(...)`

该函数从 `covered_band` 外围的相邻 chunk 中挑出能引入：

- 新节点
- 新边
- 新关系类别
- 新 source chunks

的 chunk bands。

排序键：

`(-(new_node_count + new_edge_count), -new_relation_count, -new_source_count, -root_overlap, -root_band_alignment, chunk_id)`

### 7.4 多轮执行

最终由 [expand_associative_graph(...)](/Users/Admin/projects/Association/associative_rag_project/association.py#L662) 统一调度。

每一轮做：

1. 当前图重新计算 `score_root_nodes / score_root_edges`
2. 做一次 `bridge_association`
3. 把 structural 结果并入 current graph
4. 做一次 `coverage_association`
5. 得到新一轮 current graph

系统会累计：

- `all_structural_nodes`
- `all_structural_edges`
- `all_semantic_nodes`
- `all_semantic_edges`

同时保留每轮的 `round_outputs`，用于日志和后续展示。

## 8. 组织层

组织代码在 [organization.py](/Users/Admin/projects/Association/associative_rag_project/organization.py)。

组织阶段的输入是 final graph，输出是 facet groups。

### 8.1 query contract 判定

入口函数是 [detect_query_contract(...)](/Users/Admin/projects/Association/associative_rag_project/organization.py#L222)。

当前系统只选择一个 contract：

- `section-grounded`
- `mechanism-grounded`
- `comparison-grounded`
- `theme-grounded`

判定依据是 query 的启发式短语匹配。例如：

- 明确出现 section / part / period / stage 等提示时走 `section-grounded`
- 出现 compare / difference / similarity 等提示时走 `comparison-grounded`
- “how / in what ways” 且带 causation / process cue 时走 `mechanism-grounded`
- 其他默认走 `theme-grounded`

### 8.2 EvidenceRegion

组织阶段先把 final graph 切成较小的 `EvidenceRegion`。定义在 [organization.py](/Users/Admin/projects/Association/associative_rag_project/organization.py#L190)。

每个 region 记录：

- `region_kind`
- `root_chunk_ids`
- `anchor_chunk_ids`
- `supporting_chunk_ids`
- `node_ids`
- `edge_ids`
- `relation_themes`
- `focus_entities`
- `descriptor_text`
- `root_connected`
- `doc_ids`
- `growth_traces`

region 的统一构造函数是 `_make_region(...)`。它的输入是：

- `region_id`
- `region_kind`
- `query`
- `root_chunk_ids`
- `node_ids`
- `edge_ids`
- `graph`
- `node_to_chunks`
- `edge_to_chunks`
- `chunk_store`
- `root_chunk_band`
- `growth_traces`

算法步骤如下。

1. 对 `node_ids` 和 `edge_ids` 去重、排序。
2. 若 `node_ids` 与 `edge_ids` 同时为空，直接返回 `None`。
3. 调 `_supporting_chunks_for_region(...)`：
   - 遍历 region 内所有 nodes，对每个 node 回收 `node_to_chunks[node_id]`
   - 遍历 region 内所有 edges，对每条 edge 回收 `edge_to_chunks[edge_id]`
   - 合并后得到 `supporting_chunk_ids`
4. 若 `supporting_chunk_ids` 为空，返回 `None`。
5. 令 `root_chunk_id_set = set(root_chunk_ids)`。
6. 调 `_rank_chunks(query, supporting_chunk_ids, chunk_store, root_chunk_id_set)` 生成 anchor 候选排序：
   - 第一排序键：该 chunk 是否是 root chunk
   - 第二排序键：`lexical_overlap_score(query, chunk_content)`
   - 第三排序键：`chunk_id`
7. 取前 `5` 个作为 `anchor_chunk_ids`。
8. 若 `graph is not None`：
   - 用 `_relation_themes(edge_ids, graph)` 提取 relation themes
   - 用 `_focus_entities(query, node_ids, graph)` 提取 focus entities
   否则：
   - `relation_themes = []`
   - `focus_entities = sorted(node_ids)[:6]`
9. 用 `_evidence_descriptor_text(...)` 生成 `descriptor_text`。其内容由以下部分串接：
   - label
   - 前若干 relation themes
   - 前若干 focus entities
   - 前两个 anchor chunks 的文本预览
10. 计算 `root_connected`：
   - 若 `anchor_chunk_ids` 与 `root_chunk_id_set` 有交集，则为真
   - 否则若 `supporting_chunk_ids` 与 `root_chunk_band` 有交集，则为真
   - 否则为假
11. 用 `_doc_ids(supporting_chunk_ids, chunk_store)` 生成 `doc_ids`
12. 返回 `EvidenceRegion(...)`

因此 `_make_region(...)` 的职责是把一块局部 final graph 子结构标准化成一个统一的证据单元，后续 root/bridge/theme 三类 region 全部复用这一构造过程。

### 8.3 三类 region 的收集

#### 8.3.1 root regions

函数：`_collect_root_regions(...)`

输入中最关键的两个局部索引是：

- `chunk_to_final_nodes`
- `chunk_to_final_edges`

它们由 `_chunk_local_index(...)` 从 `final_nodes/final_edges` 反推得到，表示“某个 chunk 在最终子图中支持了哪些 nodes/edges”。

对每个 `root_chunk_id`，算法如下：

1. 取 `seed_nodes = chunk_to_final_nodes[root_chunk_id]`
2. 取 `seed_edges = chunk_to_final_edges[root_chunk_id]`
3. 若两者都为空，跳过该 root
4. 计算 `support_chunks`：
   - 先用 `_supporting_chunks_for_region(seed_nodes, seed_edges, ...)` 回收直接 support chunks
   - 再删去 `root_chunk_id` 本身
   - 最后按 `(query overlap, chunk_id)` 排序
5. 初始化：
   - `local_nodes = seed_nodes`
   - `local_edges = seed_edges`
6. 取前 `4` 个 `support_chunks`，把这些 chunk 在最终图中支持的 nodes 和 edges 合并回 `local_nodes/local_edges`
7. 再补一层闭包：
   - 对所有 `final_edges`
   - 若该 edge 的两端都已经在 `local_nodes` 中，则把它加入 `local_edges`
8. 调 `_make_region(...)` 构造 root region
9. 在 `growth_traces` 中记录 `Root-anchored around {root_chunk_id}`

root region 的特点是：它总是从一个 root chunk 出发，保留强 root 锚点，并在该 root 周围收一小圈 final-graph 局部上下文。

#### 8.3.2 bridge regions

函数：`_collect_bridge_regions(...)`

数据来源是 `last_structural_output`，包括两类：

- graph bridge paths
- chunk bridge items

算法分成两支。

第一支处理 graph bridge paths：

1. 遍历 `last_structural_output["selected_paths"]`
2. 对每条 path：
   - `node_ids = path 中所有节点`
   - `edge_ids = path 中相邻节点形成的边`
3. 调 `_path_chunk_ids(path, node_to_chunks, edge_to_chunks)` 回收 path 相关 chunks
4. 用 `_rank_chunks(...)` 对 path chunks 排序
5. 取前 `3` 个 path-support chunks，把这些 chunk 支持的 nodes/edges 补回当前 bridge region
6. 再补一层 edge 闭包：
   - 若某条 `final_edge` 的两端都已在 `node_ids` 中，则把它并入 `edge_ids`
7. 调 `_make_region(...)`
8. 仅保留 `region.root_connected == True` 的 region
9. `growth_traces` 里写入：
   - path 本身
   - `new_source_count`

第二支处理 chunk bridge items：

1. 遍历 `last_structural_output["selected_chunks"]`
2. 对每个 chunk bridge item：
   - 直接读取 `node_ids`
   - 直接读取 `edge_ids`
3. 调 `_make_region(...)`
4. 同样只保留 `root_connected == True` 的 region
5. `growth_traces` 里写入：
   - bridge chunk id
   - `frontier_touch`
   - `new_source_count`

bridge region 的特点是：它们来自联想扩展阶段的 structural growth，本身携带“怎样从 root graph 走到这里”的结构信息。

#### 8.3.3 theme regions

函数：`_collect_theme_regions(...)`

theme region 的目标不是复原路径，而是把 final graph 中“关系主题相似”的局部证据收拢出来。

算法如下：

1. 遍历 `final_edges`
2. 对每条 edge 计算：
   - `theme = normalize_relation_category(graph.get_edge_data(...))`
3. 忽略 `unknown_relation`
4. 得到 `theme_to_edges: theme -> [edge_ids]`
5. 对所有 theme 按以下键排序：
   - `-lexical_overlap_score(query, theme)`
   - `-len(edges)`
   - `theme`
6. 取前 `max_themes`
7. 对每个 theme：
   - 先把该 theme 下所有 edge 的两端节点收进 `node_ids`
   - `edge_ids = 该 theme 下所有 edges`
   - 回收 `support_chunks`
8. 若 `support_chunks` 与 `root_chunk_band` 没有交集，则丢弃
9. 对 `support_chunks` 调 `_rank_chunks(...)`
10. 取前 `4` 个 ranked chunks，把这些 chunk 支持的 final nodes/final edges 补进 region
11. 调 `_make_region(...)`
12. 只保留 `root_connected == True` 的 region

theme region 的特点是：它以 relation theme 为中心聚合 final graph 子结构，因此比 root/bridge region 更适合做高层主题覆盖。

### 8.4 从 regions 到 facet groups

入口函数是 [build_answer_facet_groups(...)](/Users/Admin/projects/Association/associative_rag_project/organization.py#L1041)。

它的流程是：

1. 先判 `contract`
2. 调 `collect_overlapping_regions(...)`
3. 根据 contract 走不同的组装器：
   - `_build_section_groups(...)`
   - `_build_mechanism_groups(...)`
   - `_build_comparison_groups(...)`
   - `_build_theme_groups(...)`

这里的分工是：

- `collect_overlapping_regions(...)` 负责把 final graph 切成标准化证据单元
- contract-specific builder 负责决定“哪些单元应该合并成一个 facet”

### 8.5 group 构造

统一构造函数是 `_build_group(...)`。

它把多个 region 合并成一个 facet group，并执行以下步骤：

1. 初始化以下聚合容器：
   - `root_chunk_ids`
   - `anchor_chunk_ids`
   - `supporting_chunk_ids`
   - `node_ids`
   - `edge_ids`
   - `relation_counter`
   - `focus_counter`
   - `region_kind_counter`
   - `doc_ids`
   - `growth_traces`
2. 遍历输入 regions，把每个 region 的字段合并进上述容器
3. 对合并后的 `anchor_chunk_ids` 重新用 `_rank_chunks(...)` 排序，截到前 `5`
4. 从 `relation_counter` 取前 `5` 个作为 `relation_themes`
5. 从 `focus_counter` 取前 `8` 个作为 `focus_entities`
6. 用 `_evidence_descriptor_text(...)` 生成 group 级 `descriptor_text`
7. 计算：
   - `group_score = lexical_overlap_score(query, descriptor_text)`
   - `query_rel = lexical_overlap_score(query, descriptor_text)`
8. 生成 `group_summary`
9. 产出 group 字典

当前 group 输出字段包括：

- `facet_label`
- `organization_contract`
- `facet_prompt`
- `group_score`
- `query_rel`
- `anchor_support`
- `root_anchor_count`
- `node_count`
- `edge_count`
- `region_count`
- `unique_doc_count`
- `unique_root_count`
- `root_chunk_ids`
- `doc_ids`
- `region_kinds`
- `region_kind_counts`
- `root_region_count`
- `bridge_region_count`
- `theme_region_count`
- `section_region_count`
- `relation_themes`
- `focus_entities`
- `supporting_chunk_ids`
- `anchor_chunk_ids`
- `source_previews`
- `growth_traces`
- `group_summary`
- `nodes`
- `edges`

### 8.6 四类 group 组装算法

#### 8.6.1 section groups

`_build_section_groups(...)` 面向“按文档局部段落/阶段回答”的 query。

算法如下：

1. 先用 `_build_doc_chunk_index(chunk_store)` 为每个 doc 建 `(order, chunk_id)` 有序表
2. 对每个 root region：
   - 取 `seed_chunk_ids = region.anchor_chunk_ids[:3]`
   - 调 `_section_band_chunk_ids(seed_chunk_ids, chunk_store, doc_to_chunks, radius=2)`
   - 得到同文档、以这些 anchor 为中心、前后各 2 个 order 的 band
3. 对 band 内每个 chunk：
   - 回收它支持的 final nodes
   - 回收它支持的 final edges
4. 调 `_make_region(...)` 构造一个 `section` region
5. 用 `_chunk_order_bounds(...)` 计算该 band 的起止 order，拼成 label
6. 用 `_build_group(...)` 把这个 section region 变成候选 group
7. 调 `_select_groups(...)` 做 coverage-first 去冗余

它的 distinct key 是：

- `(doc_id, 前两个 anchor_chunk_ids)`

因此 section 组的核心约束是“一个 facet 对应一个相对局部、相对连续的文档带状区域”。

label 会优先写成：

`section band {doc后缀}:{start_order}-{end_order}`

#### 8.6.2 mechanism groups

`_build_mechanism_groups(...)` 面向“how / process / what drove ...”这类问题。

算法如下：

1. 先建立 `root_by_chunk`：
   - `root_chunk_id -> [root_regions]`
2. 以每个 `bridge_region` 为主轴：
   - 初始化 `related = [bridge_region]`
   - 把与其共享 `root_chunk_ids` 的 root regions 合并进来
   - 再遍历 theme regions，若满足以下任一条件则合并：
     - lexical overlap 达到阈值
     - 与 bridge region 共享 root chunk
     - 与 bridge region 共享 supporting chunks
3. 对 `related` 去重，得到 `deduped`
4. 用 bridge region 的 `relation_themes` 生成 label：
   - 若只有一个主 theme，则 label = `primary_theme`
   - 否则 label = `primary_theme -> secondary_theme`
5. 调 `_build_group(...)`
6. 若没有任何 bridge candidates，则 fallback：
   - 直接把 `root_regions + theme_regions` 按 `primary_theme` 分桶
   - 每桶生成一个候选 group
7. 对候选 group 做弱机制过滤：
   - 若 group 不含 `bridge`
   - 且 `edge_count < 3`
   - 则跳过
8. 调 `_select_groups(...)`

并且会优先过滤掉：

- 没有 `bridge` 角色
- 且 `edge_count < 3`

的弱机制组

#### 8.6.3 comparison groups

`_build_comparison_groups(...)` 面向“区别 / 对比 / patterns across cases”类问题。

算法如下：

1. 每个 root region 先单独生成一个候选 group
2. 再遍历 bridge regions：
   - 若该 region 关联至少两个 `root_chunk_ids`
   - 则生成一个 `contrast around {primary_theme}` 候选 group
3. 调 `_select_groups(...)`

其 distinct key 优先使用：

- `前两个 root_chunk_ids`

这使 comparison 组天然偏向“用 root 锚点区分比较对象”。

#### 8.6.4 theme groups

`_build_theme_groups(...)` 是当前组织层里最复杂的一支，因为它要同时做到：

- 覆盖多个方面
- 保留 root-connected anchor
- 不把扩展证据直接变成杂乱细节

当前算法分两层。

第一层：coverage pooling

1. 遍历 `root_regions + theme_regions + bridge_regions`
2. 若 `region.root_connected == False`，跳过
3. 计算 `region_rel = lexical_overlap_score(query, region.descriptor_text)`
4. 过滤阈值：
   - `bridge` region 要求 `region_rel >= 0.06`
   - `root/theme` region 要求 `region_rel >= 0.04`
5. 按 `primary_theme(region)` 分桶，得到：
   - `theme_label -> [regions]`

第二层：representative anchoring

对每个 theme 桶：

1. 调 `_theme_representative_regions(query, regions)`：
   - 先取 `root/theme` regions，按以下键排序：
     - `-lexical_overlap_score(query, region.descriptor_text)`
     - `-len(region.anchor_chunk_ids)`
     - `-len(region.supporting_chunk_ids)`
     - `region_id`
   - 再取 `bridge` regions，按以下键排序：
     - `-lexical_overlap_score(query, region.descriptor_text)`
     - `-len(region.root_chunk_ids)`
     - `-len(region.supporting_chunk_ids)`
     - `region_id`
   - 默认保留：
     - 最多 `3` 个 root/theme representative regions
     - 最多 `2` 个 bridge representative regions
2. 用这些 representative regions 调 `_build_group(...)`
3. 但随后把 group 的覆盖字段回填成整桶 `regions` 的并集：
   - `supporting_chunk_ids`
   - `root_chunk_ids`
   - `doc_ids`
   - `region_kinds`
   - `region_count`
   - `bridge_region_count`
   - `theme_region_count`
   - `root_region_count`
   - `unique_doc_count`
   - `unique_root_count`
4. 再调 `_retune_theme_group(...)` 对写作锚点做第二次校正：
   - 从 representative regions 中收集 `representative_anchor_chunks`
   - 若其中有 root chunks，优先保 root chunks
   - 再用 `_rank_chunks(...)` 选前 `5` 个稳定 anchor
   - 重写：
     - `anchor_chunk_ids`
     - `anchor_support`
     - `root_anchor_count`
     - `source_previews`
     - `group_summary`
5. 计算 `selection_priority`：
   - `bridge_region_count * 2`
   - `theme_region_count * 2`
   - `min(unique_doc_count, 3)`
   - `min(unique_root_count, 3)`
   - `min(region_count, 4)`
6. 调 `_select_groups(...)`

这意味着当前 theme 组装是“宽覆盖 + 稳锚点”结构：

- coverage 来自整桶 region
- anchor 来自代表性 root-connected evidence
- bridge/theme expansion 只作为 coverage 和方面发现的来源，不直接主导 anchor 选择

### 8.7 group 选择

最终所有 contract 都通过 `_select_groups(...)` 做 greedy 选择。

算法：

1. 先按 `_group_rank_key(...)` 排序。
2. 维护 `selected / used / covered`。
3. 每轮优先选择能带来最多新 coverage keys 的 group。
4. 若 gain 相同，再用 `_group_rank_key(...)` 破平。
5. coverage-first 选完后，再按排序补足到 `limit`。

其中 `_group_rank_key(...)` 当前使用的主要字段有：

- `selection_priority`
- `root_anchor_count`
- `query_rel`
- `region_kinds` 数量
- `bridge_region_count`
- `theme_region_count`
- `unique_doc_count`
- `anchor_support`
- `node_count`

`coverage_key_fn` 则由每类 builder 自己提供。例如：

- `section` 更看 doc / band / theme
- `mechanism` 更看 label / root / kinds
- `comparison` 更看 root / theme / doc
- `theme` 更看 label / root / doc / theme / kinds

因此 `_select_groups(...)` 不是简单取 top-k，而是一个“先覆盖、后补强”的贪心选择过程。

## 9. 上下文组包

上下文构造代码在 [context.py](/Users/Admin/projects/Association/associative_rag_project/context.py)。

### 9.1 supporting chunk 全局排序

`rank_supporting_chunks(...)` 会统计每个 chunk 对 final nodes 和 final edges 的支持次数。

每出现一次：

- node provenance 加 1
- edge provenance 加 1
- root chunk 再额外加 1

排序后得到全局 chunk rank。

### 9.2 source chunk 选择

真正送进 prompt 的 sources 由 `choose_diverse_source_chunks(...)` 选择。

选择过程分三轮：

1. 每个 facet group 先争取一个 chunk
2. 每个 facet group 再争取一个 chunk
3. 用全局 rank 补齐剩余预算

同时施加局部约束：

- `max_source_chunks`
- `max_source_word_budget`
- 同文档 local band cap

该 band cap 由 `_violates_local_band_cap(...)` 实现，默认不让同文档相邻 band 过度挤占 source budget。

### 9.3 dossier 组织

[build_prompt_context(...)](/Users/Admin/projects/Association/associative_rag_project/context.py#L396) 会生成一个多段 evidence package，包含：

- `Root Chunks`
- `Focused Entities`
- `Focused Relations`
- `Facet Groups`
- `Coverage Checklist`
- `Facet Group Dossiers`
- `Sources`

每个 facet dossier 内含：

- `Facet`
- `Facet Prompt`
- `Summary`
- `Themes`
- `Evidence Roles`
- `Key Entities`
- `Structural Growth Trace`
- `Anchor Evidence`
- `Key Relations`
- `Linked Sources`

这一步的目标是把结构化图证据翻译成 LLM 能消费的证据包。

## 10. 生成层

生成代码在 [llm_client.py](/Users/Admin/projects/Association/associative_rag_project/llm_client.py)。

### 10.1 prompt 结构

`build_generation_prompt(...)` 的输入只有：

- `query`
- `prompt_context`
- `query_contract`

当前生成层没有 `query_style`。

prompt 里包含三部分约束：

1. 通用 grounded QA 约束
2. contract-specific hints
3. query 触发的额外约束

例如：

- `section-grounded` 提示更贴近局部 section / band
- `mechanism-grounded` 提示明确因果链
- `comparison-grounded` 提示显式比较维度
- `theme-grounded` 提示只围绕强主题组织

此外，若 query 命中 `_is_broad_theme_query(...)` 的 cue，prompt 会放宽 coverage，但仍明确要求不能超出 query。

### 10.2 生成输出

`generate_answers(...)` 会并行调用 LLM，并为每条答案保留：

- `group_id`
- `query`
- `query_contract`
- `model_answer`
- `stats`

## 11. Judge 评测

judge 代码在 [judge.py](/Users/Admin/projects/Association/associative_rag_project/judge.py)。

### 11.1 当前评测维度

当前 judge 同时输出八项：

- `Comprehensiveness`
- `Diversity`
- `Empowerment`
- `Focus Match`
- `Evidence Anchoring`
- `Scope Discipline`
- `Scenario Fidelity`
- `Overall Winner`

其中新增的两个维度含义是：

- `Scope Discipline`：答案是否受 query 边界约束
- `Scenario Fidelity`：答案是否避免自设情景、动机、流程、场景

### 11.2 双顺序判定

`judge_pair(...)` 会调用 judge 两次：

- `candidate vs baseline`
- `baseline vs candidate`

然后用 `_map_swapped_winner(...)` 把第二次结果映射回原坐标，减少位置偏置。

### 11.3 organization analysis

judge 同时要求模型输出三项 contract 标签：

- `Query Organization Need`
- `Answer 1 Organization`
- `Answer 2 Organization`

`_extract_organization_analysis(...)` 再把它们统一映射成：

- `query_contract`
- `candidate_answer_contract`
- `baseline_answer_contract`
- 是否 contract match

### 11.4 contract-conditioned overall

当前最终胜负不直接等于 LLM 原始 `Overall Winner`。

系统会先读取 `query_contract`，再用 `_resolve_contract_conditioned_decision(...)` 做条件化聚合。

四类 query 的主指标集合由 `CONTRACT_PRIMARY_METRICS` 固定定义：

- `theme-grounded`
  - `Comprehensiveness`
  - `Diversity`
  - `Focus Match`
  - `Scope Discipline`
  - `Scenario Fidelity`
- `mechanism-grounded`
  - `Focus Match`
  - `Empowerment`
  - `Evidence Anchoring`
  - `Scope Discipline`
  - `Scenario Fidelity`
- `section-grounded`
  - `Focus Match`
  - `Evidence Anchoring`
  - `Scope Discipline`
  - `Scenario Fidelity`
- `comparison-grounded`
  - `Focus Match`
  - `Comprehensiveness`
  - `Empowerment`
  - `Scope Discipline`
  - `Scenario Fidelity`

聚合步骤：

1. 先把主指标的双顺序投票加总。
2. 若主指标能分出胜负，直接作为 `final_winner`。
3. 若主指标打平，再看次级指标。
4. 若仍打平，再回退到 LLM 原始 `Overall Winner`。

因此当前 judge 输出里有三个层次：

- `llm_overall_winner`
- `contract_conditioned_decision`
- `final_winner`

其中正式 summary 使用的是 `final_winner`。

### 11.5 judge 输出结构

`run_winrate_judgement(...)` 最终输出：

- `summary`
- `llm_overall_summary`
- `criteria_summary`
- `criteria_summary_by_contract`
- `contract_conditioned_summary_by_contract`
- `organization_summary`
- `verdicts`

这使得分析可以同时从三种角度展开：

- 全局维度统计
- 按 contract 的维度统计
- 按 contract 的最终胜率统计

## 12. 当前实现边界

为了避免误读，这里明确列出当前主路径没有做的事情：

- 没有 adaptive controller
- 没有 query-style routing
- 没有把 association 和 organization 合并成单一阶段
- 没有把 judge 的总体结论完全交给模型自由裁定

当前实现的实际结构是：

- 固定预算检索
- 多起点 root 选择
- 2 x 2 联想扩展
- contract-aware 组织
- facet dossier 组包
- contract-conditioned judge

这就是当前代码版 Associative RAG 的完整工作方式。

## 13. v4 典型案例

本节从 `runs_v4` 的评测结果中抽取代表性样本，用于说明当前系统在什么类型的问题上明显占优，以及在什么类型的问题上明显失分。这里不直接粘贴完整判词，只保留问题、胜负结果和判词主因，控制单例长度，方便回看。

### 13.1 样本筛选方法

筛选口径如下：

- 以 `final_winner` 作为最终胜负标签。
- 以七个非 overall 维度的净胜数 `wins - losses` 作为强弱程度指标。
- `大赢` 指 candidate 在七个维度里形成显著净胜。
- `大输` 指 baseline 在七个维度里形成显著净胜。
- 样本同时覆盖 `art` 和 `agriculture` 两个数据集，并优先覆盖 `theme-grounded`，因为该 contract 是当前系统的主要波动来源。

### 13.2 大赢案例

#### 案例 1：art-40

- Query: `How does the dataset depict the tension between national identity and international art movements?`
- Contract: `theme-grounded`
- 结果: `candidate`，七维全胜。
- 判词主因:
  - candidate 直接围绕“张力如何在数据集中被呈现”组织答案。
  - candidate 同时覆盖制度、区域、艺术家与展览层面的证据。
  - judge 明确认为 baseline 更像泛化的艺术史综述，dataset 贴合度更弱。
- 启示:
  - 当 query 明确要求“依据数据集概括某种 tension”时，当前系统的强项会被放大。

#### 案例 2：art-43

- Query: `How did the commodification of art alter its perception and value in society?`
- Contract: `theme-grounded`
- 结果: `candidate`，`Comprehensiveness` 之外几乎全胜。
- 判词主因:
  - candidate 用较少但更直接的机制说明“商品化如何改变价值判断、生产方式和社会认知”。
  - candidate 的例子更具体，judge 点名了 auction、Warhol、Hirst、collectors 等锚点。
  - baseline 虽然更散、更广，但被认为加入了不必要的历史背景和技术背景。
- 启示:
  - 当 theme 问题本身允许围绕一个清晰中心机制展开时，稳锚点组织是有效的。

#### 案例 3：art-76

- Query: `What themes from the dataset offer the richest material for interdisciplinary studies?`
- Contract: `theme-grounded`
- 结果: `candidate`
- 判词主因:
  - candidate 明确列出若干“最值得跨学科展开的主题”，并解释每个主题为何重要。
  - baseline 被 judge 认为扩展到了 methodology、applications、mental health 等次生话题。
  - candidate 在 `Focus Match`、`Evidence Anchoring`、`Scope Discipline` 上都占优。
- 启示:
  - 当题目需要“挑最有价值的主题”而不是“尽量列全所有主题”时，当前方法更容易赢。

#### 案例 4：agriculture-31

- Query: `What management tools are suggested for improving farm efficiency?`
- Contract: `theme-grounded`
- 结果: `candidate`，七维全胜。
- 判词主因:
  - candidate 给出的不是泛化原则，而是具体工具和决策要点，例如 machinery replacement、rotation、cover crops、financial tools。
  - baseline 被认为更像管理口号，缺少真正的“tool-level”内容。
  - judge 同时给了 candidate 更高的 `Empowerment` 和 `Evidence Anchoring`。
- 启示:
  - 在农业 domain 中，只要问题本身偏操作性，theme 答案也可以靠“具体条目 + 稳证据”取得大胜。

#### 案例 5：agriculture-33

- Query: `What strategies for managing labor are discussed in the book?`
- Contract: `theme-grounded`
- 结果: `candidate`，七维全胜。
- 判词主因:
  - candidate 更像“把书里的 labor 策略摘出来并排好”，包括 labor budgeting、family division、payroll services、workflow redesign。
  - baseline 被认为混入了更泛的 sector-level technology 叙述。
  - judge 特别认可 candidate 的 actionability。
- 启示:
  - 当主题问句实际指向“书里具体写了哪些策略”时，当前 evidence-group 组织与生成方式非常合拍。

#### 案例 6：agriculture-51

- Query: `What policy recommendations does the book offer for supporting sustainable farming?`
- Contract: `theme-grounded`
- 结果: `candidate`，七维全胜。
- 判词主因:
  - candidate 给出了多个可执行的 policy lever，如 subsidy reform、carbon credits、new-farmer support、processing infrastructure。
  - baseline 更像高层政策愿景，具体 recommendation 数量和颗粒度都更弱。
- 启示:
  - 在以“book recommendations”为中心的 query 上，当前系统可以把 theme 组织做得既广又具体。

### 13.3 大输案例

#### 案例 7：art-3

- Query: `Can we infer any patterns of political upheaval that led to significant art movements?`
- Contract: `theme-grounded`
- 结果: `baseline`，七维全败。
- 判词主因:
  - baseline 给出了更多 movement、更多国家与历史阶段，形成了典型的 broad survey。
  - candidate 过早收缩到少量概括，并加入 market/commercialization 线索，导致 scope 反而被 judge 视为偏移。
  - judge 认为这类“patterns”问题需要大量例子支撑 generalization。
- 暴露问题:
  - 对于 broad thematic synthesis，当前系统仍然缺少“方面覆盖计划”。

#### 案例 8：art-22

- Query: `What are examples of cross-cultural influences that led to new art styles or themes?`
- Contract: `theme-grounded`
- 结果: `baseline`，七维全败。
- 判词主因:
  - baseline 使用的是 judge 更熟悉、更典型的艺术史例子集合，如 Japanese prints、Gauguin/Polynesia 等。
  - candidate 提供的例子更少，而且混入 exhibition networks、market engagement 等旁支。
  - judge 认为该题需要“例子列表型的 broad theme answer”，不是 dataset 内部逻辑的窄概括。
- 暴露问题:
  - 当前 theme 组织缺少“典型例子优先级”，仍可能选出 graph 上连得通、但不够代表性的材料。

#### 案例 9：art-106

- Query: `How did the industrial revolution shift artist themes and methodologies?`
- Contract: `theme-grounded`
- 结果: `baseline`，七维全败。
- 判词主因:
  - baseline 同时覆盖主题、方法、技术、社会批评、受众变化、艺术流派。
  - candidate 仍然倾向把答案压缩到 mechanization / commercialization 这类窄轴。
  - judge 还指出 candidate 引入了偏后期的 contemporary references，削弱了历史对位。
- 暴露问题:
  - 当 query 含明确历史时期时，theme 组织需要更强的“时期内覆盖”约束。

#### 案例 10：agriculture-4

- Query: `Why is ongoing learning and staying connected with the beekeeping community important?`
- Contract: `theme-grounded`
- 结果: `baseline`
- 判词主因:
  - baseline 从 research、pest control、mentorship、regulations、resource sharing 等多个角度解释“为什么重要”。
  - candidate 的范围控制更好，但内容明显更短，导致 `Comprehensiveness`、`Diversity`、`Empowerment` 一起失分。
  - judge 明确给了 candidate 更紧的 scope/scenario 评价，但在 theme 题里这不足以扳回整体。
- 暴露问题:
  - 当前系统已经能减少自设情景，但在“why is X important”类 broad rationale 问题上仍然容易欠展开。

#### 案例 11：agriculture-14

- Query: `How do environmental factors influence bee activity and health?`
- Contract: `theme-grounded`
- 结果: `baseline`
- 判词主因:
  - baseline 按 temperature、humidity、floral availability、pollution、pesticides、seasonality 等标准环境因子展开。
  - candidate 虽然更像 evidence-based answer，但引入了 electromagnetism、solar eruptions 等不必要具体化。
  - judge 将其视为 scope 与 scenario 两边一起变差的例子。
- 暴露问题:
  - theme 生成阶段仍缺一个“标准因素优先，边缘因素降级”的控制。

#### 案例 12：agriculture-69

- Query: `How can policies encourage value-added production on small farms?`
- Contract: `theme-grounded`
- 结果: `baseline`
- 判词主因:
  - baseline 从 funding、training、regulation、market access、partnership 等政策工具展开，更像一份政策框架图。
  - candidate 更贴近小农经营实践，但 judge 认为它偏成了 farm strategy，而不是 policy design。
- 暴露问题:
  - 对于 policy 类 theme 问题，facet 组织需要明确区分“政策工具”与“经营做法”。

### 13.4 样本归纳

从以上案例可以归纳出 v4 的几个稳定现象：

- 当前系统的大赢通常出现在 query 对数据集依赖强、问题边界清晰、且可以由少量稳证据直接支撑的场景。
- 当前系统的大输主要集中在 `theme-grounded` 的 broad survey 问题，尤其是需要大量典型例子、标准维度覆盖、历史阶段展开的问题。
- `Scope Discipline` 与 `Scenario Fidelity` 的提升已经稳定可见，但它们本身不足以抵消 broad theme 问题上的覆盖不足。
- `art` 数据集对“例子密度、历史跨度、综述宽度”的要求显著高于 `agriculture`，因此同一套 `theme` 组织策略会出现明显域间差异。

这组案例说明，后续优化重点不在于继续强化 theme 的证据收缩，而在于为 `theme-grounded` 增加显式的 coverage planning，并把“方面覆盖”和“写作锚点”分成两个独立步骤。
