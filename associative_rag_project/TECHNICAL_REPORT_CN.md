# Associative RAG 中文技术报告

本文档只描述当前代码已经实现并正在运行的系统，不追溯已废弃路线，也不把实验性想法写成既成事实。核心实现位于：

- [main.py](/Users/Admin/projects/Association/associative_rag_project/main.py)
- [pipeline.py](/Users/Admin/projects/Association/associative_rag_project/pipeline.py)
- [data.py](/Users/Admin/projects/Association/associative_rag_project/data.py)
- [retrieval.py](/Users/Admin/projects/Association/associative_rag_project/retrieval.py)
- [association.py](/Users/Admin/projects/Association/associative_rag_project/association.py)
- [organization.py](/Users/Admin/projects/Association/associative_rag_project/organization.py)
- [context.py](/Users/Admin/projects/Association/associative_rag_project/context.py)
- [llm_client.py](/Users/Admin/projects/Association/associative_rag_project/llm_client.py)
- [judge.py](/Users/Admin/projects/Association/associative_rag_project/judge.py)

## 1. 任务定义与总体思路

本项目面向 `Query-Focused Summarization`。输入不是一篇完整文档，而是：

- 一个由 chunk 构成的文本语料
- 一个 chunk-entity-relation 图
- 一条 query

系统目标不是做普通检索式问答，而是：

1. 找到足够支撑 query 的 chunk
2. 通过图和 chunk 邻域做联想扩展
3. 把联想到的证据组织成适合回答该 query 的多点结构
4. 用 LLM 在这些证据上生成最终总结性回答

当前版本最重要的设计选择有两个：

1. 检索与联想主干统一  
   所有 query contract 都走同一套 theme-style 检索与多轮联想，不再保留旧的“theme 一套，其他 contract 一套”的扩图主干。

2. 差异主要放在组织层  
   `section / mechanism / comparison / theme` 的差异主要体现在最终 facet grouping，而不是前面完全不同的 retrieve 路线。

## 2. 系统输入输出

### 2.1 输入文件

每个语料目录 `Datasets/<name>/index` 需要提供：

- `graph_chunk_entity_relation.graphml`
- `kv_store_text_chunks.json`
- `vdb_chunks.json`，当使用 dense 或 hybrid 检索时需要

查询文件通常位于：

- `Datasets/<name>/query/<name>.json`

baseline 答案通常位于：

- `Datasets/<name>/output/FG-RAG-4o-mini.json`

### 2.2 输出文件

一轮 `run-all` 通常会产生：

- `*_retrieval.json`
- `*_answers.json`
- `*_vs_FG-RAG-4o-mini_winrate.json`
- `*_sample_context.txt`

其中 retrieval 文件是最关键的中间产物，里面保存了：

- 候选 chunk
- root chunks
- promoted roots
- final subgraph
- facet groups
- candidate points
- prompt context
- 各种 per-query 统计量

## 3. 主流程

主流程在 [pipeline.py](/Users/Admin/projects/Association/associative_rag_project/pipeline.py) 的 `run_query()` 中完成，可概括为：

1. `detect_query_contract(query)`
2. primary retriever 取 chunk 候选
3. `search_graph_focus_chunks()` 与 `search_graph_keyword_chunks()` 追加图侧召回
4. `merge_candidate_hits_with_graph()` 融合候选池
5. `select_diverse_root_chunks()` 选 roots
6. `expand_associative_graph()` 多轮联想扩图
7. `build_answer_facet_groups()` 组织 final graph
8. `build_candidate_points_from_groups()` 抽取候选要点
9. `build_prompt_context()` 组装提示上下文

## 4. 数据结构与索引层

### 4.1 基础图与 chunk store

[data.py](/Users/Admin/projects/Association/associative_rag_project/data.py) 负责加载：

- `graph`: entity-relation 图
- `chunk_store`: chunk 文本、doc id、chunk 顺序等元数据

### 4.2 四个核心双向映射

`build_chunk_mappings()` 构造：

- `chunk_to_nodes`
- `chunk_to_edges`
- `node_to_chunks`
- `edge_to_chunks`

这是整个系统的桥梁。后续几乎每一步都在做：

- chunk -> 图节点/边
- 图节点/边 -> supporting chunks

### 4.3 chunk 邻域

`build_chunk_neighborhoods()` 基于 `full_doc_id + chunk_order_index` 构造 chunk 邻域。它有三类用途：

- root 去重时避免同一 local band 重复起点
- association 时做 chunk-side structural neighbor 扩展
- section 题构造 section band

## 5. Query Contract 检测

contract 检测在 [organization.py](/Users/Admin/projects/Association/associative_rag_project/organization.py) 的 `detect_query_contract()` 中完成。

当前支持四类：

- `section-grounded`
- `mechanism-grounded`
- `comparison-grounded`
- `theme-grounded`

这是一个启发式分类器，不是训练模型。它主要依据 query 中的提示语判断：

- 是否更像 section/local-band 问题
- 是否更像 mechanism/causal question
- 是否明确要求 compare/contrast
- 是否属于 broad theme synthesis

当前版本明显偏向把开放综述题归入 `theme-grounded`，原因是现在的 theme 路线已经成为主干能力。

## 6. 候选检索算法

当前所有 contract 都走统一的候选检索主干，只是 merge 权重略有差别。

### 6.1 Primary Retriever

主检索器定义在 [retrieval.py](/Users/Admin/projects/Association/associative_rag_project/retrieval.py)：

- `BM25Index`
- `DenseChunkIndex`
- `HybridChunkRetriever`

`HybridChunkRetriever.search()` 会按配置使用：

- `bm25`
- `dense`
- `hybrid`

当为 `hybrid` 时，主检索分数由 dense 与 BM25 归一化分数组合而成。

### 6.2 Graph Keyword Retrieval

`search_graph_keyword_chunks()` 是直接的 query -> graph lexical recall：

1. 用 query 与 node 文本做 lexical overlap
2. 对 node 引入一个 degree-specificity 修正
3. 把 node hit 投票回 chunk
4. 再对 edge 文本做 lexical overlap
5. 把 edge hit 投票回 chunk
6. 汇总得到 `graph_keyword_score_norm`

它的特点是：

- 不依赖 dense index
- recall 较直接
- 更像一个图侧 lexical channel

### 6.3 Graph Focus Retrieval

`search_graph_focus_chunks()` 是当前更重要的图侧召回通道。

流程是：

1. 从 query 中抽取非语法性的 focus terms
2. 用这些 terms 去匹配 node/edge 文本中的关键词
3. 统计 hit terms 的覆盖率与命中数
4. 将命中的 node/edge 投票回 chunk
5. 输出 `graph_focus_score_norm` 与 `graph_focus_hit_terms`

与 `graph_keyword` 相比，它更强调：

- query 内容词覆盖
- 命中 term 的数量
- 图元素的 specificity

### 6.4 候选融合

`merge_candidate_hits_with_graph()` 将 primary hits 与 graph-side hits 融合。

统一框架是：

- 先保留 primary retrieval score
- 对已出现 chunk 做图侧 boost
- 对未出现在 primary 中的 chunk 允许以图侧分数进入候选池

当前 `run_query()` 中的 contract 差异主要体现在 graph 权重：

- `section-grounded`: 图侧权重更保守
- `mechanism-grounded`: 中等偏保守
- `comparison-grounded`: 略高
- `theme-grounded`: 最高

这样做的原因是：

- `theme` 更需要 broad aspect recall
- `section` 更需要局部稳定性

## 7. Root Chunk 选择算法

根选择由 [retrieval.py](/Users/Admin/projects/Association/associative_rag_project/retrieval.py) 的 `select_diverse_root_chunks()` 完成。

### 7.1 候选增强特征

每个候选 chunk 会被补充以下特征：

- `base_score`
- `query_alignment`
- `relation_entropy`
- `full_doc_id`
- `chunk_order_index`
- `graph_nodes`
- `graph_edges`
- `basin_key`
- `basin_signature`
- `anchor_terms`
- `query_lexical`
- `anchor_query_overlap`
- `focus_term_hit_count`
- `keyword_match_score`
- `keyword_hit_terms`

这些特征不一定直接线性加权，但会进入排序、约束和去重规则。

### 7.2 Theme Root 选择

`theme-grounded` 使用 `_select_theme_root_chunks()`。

该算法的核心思想是：

1. 先按 `basin_key` 将候选分组
2. 各 basin 内按 root sort key 排序
3. 跨 basin 轮转式抽样
4. 约束：
   - 避免同 doc 过多 roots
   - 避免同 local band
   - 避免 provenance overlap 太高
   - 早期优先带来新的 query terms

这使 root 不会过度集中在同一 query term basin 中。

### 7.3 Non-theme Root 选择

`section / mechanism / comparison` 仍保留 contract-aware root 排序和 relaxed pass，但现在后续联想阶段已经统一走 theme-style 扩图。

### 7.4 Section 特殊约束

当前实现中，`section-grounded` 会先根据最高优先级 root 锚定一个 `full_doc_id`，然后：

- 候选 roots 重新限制在该 doc 内
- 后续扩图也通过 `allowed_doc_ids` 保持在这个 doc 内

这保证了 section 题不会在多个文档之间漂移。

## 8. Root Node / Edge 打分

roots 投影到图上后，系统会用轻量打分为后续观察和上下文组织提供 ranked roots。

### 8.1 Node

`score_root_nodes()` 主要使用：

- `support_score(node_to_chunks[node], root_chunk_score_lookup)`
- query 与 node 文本的 lexical overlap

### 8.2 Edge

`score_root_edges()` 主要使用：

- `support_score(edge_to_chunks[edge], root_chunk_score_lookup)`
- query 与 edge 文本的 lexical overlap
- edge weight 的轻量修正

## 9. 统一的多轮联想扩图算法

这是当前系统最核心的算法变化。

`expand_associative_graph()` 已统一调用 `_expand_theme_chunk_graph()`，即：

- `theme`
- `mechanism`
- `comparison`
- `section`

都共享同一个多轮 chunk-level association 主干。

### 9.1 核心状态变量

扩图过程中维护四组关键集合：

- `selected_chunk_ids`: 当前已纳入证据池的 chunk
- `active_root_chunk_ids`: 当前轮真正对外爬行的 roots
- `effective_root_chunk_ids`: 历史上所有被当作 root 的 chunk
- `local_query_chunk_score_lookup`: 会在轮次间更新的 query relevance

### 9.2 初始候选池

初始 `candidate_chunk_ids` 来自：

- query 检索得到的 chunk candidates
- roots
- 这些 chunk 的 structural neighbors

因此联想不是单纯在 root 上做 hop，而是在一个“候选池 + 当前 frontier”的空间内做调度。

### 9.3 每轮三类 chunk

每一轮扩图都从三类 chunk 中选：

#### A. Bridge Chunk

`_rank_theme_bridge_chunks()` 负责 bridge ranking。

目标是：

- 连接当前 frontier 与已选证据
- 引入更多 query relevant nodes/edges
- 带来新的 source/doc

它更像结构主干。

#### B. Support Chunk

`_rank_theme_support_chunks()` 负责 support ranking，`_select_diverse_support_chunks()` 负责去重和多样性筛选。

目标是：

- 引入信息增益
- 提高新 aspect 的覆盖
- 避免 support 全部围绕原 roots 打转

它更像 breadth/comprehensive/diversity 的主要来源。

#### C. Peripheral Chunk

`_rank_theme_peripheral_chunks()` 负责补外围上下文。

目标是：

- 给 bridge/support 附近再补一层局部图上下文
- 在不引入太多噪声的前提下补足近邻证据

### 9.4 每轮后的 query 分数刷新

`_update_theme_query_scores_from_round()` 会用本轮新增的 bridge/support/peripheral 反向更新 `local_query_chunk_score_lookup`。

这一步意味着：

- query relevance 不再在第一轮固定
- 新证据会改变下一轮“什么更像相关 chunk”的判断

这是“多轮联想逐步爬行”的关键。

### 9.5 每轮后的 root 重选

`_reseed_theme_root_chunks()` 是当前系统的另一个关键算法。

流程是：

1. 以 `selected_chunk_ids + round_chunk_ids + 邻居 + candidate_chunk_ids` 构造 seed pool
2. 用 `_build_theme_reseed_candidate_hits()` 构造新的 root 候选
3. 再调用 `select_diverse_root_chunks()` 重新选 root
4. 生成新的 `active_root_chunk_ids`

这意味着 root 不再是“一开始选完就固定”，而是：

- 一轮联想
- 重打分
- 换一批 root
- 再联想

### 9.6 Contract 级预算微调

虽然主干统一，但每种 contract 的轮次配额略有差别：

- `section-grounded`: bridge/support 更保守，peripheral 最少
- `mechanism-grounded`: 保持路径解释能力，peripheral 偏少
- `comparison-grounded`: bridge/support 略强，以维持对照侧面
- `theme-grounded`: 使用最宽松的 broad expansion

### 9.7 Section 的单文档过滤

`allowed_doc_ids` 会在统一扩图器里持续生效：

- roots 会过滤
- frontier 会过滤
- reseed 候选也会过滤

所以 section 虽然共享 theme-style 扩图主干，但不会越出锚定文档。

## 10. Evidence Region 构造

在 [organization.py](/Users/Admin/projects/Association/associative_rag_project/organization.py) 中，`collect_overlapping_regions()` 将 final graph 划分为三类 region：

- `root regions`
- `bridge regions`
- `theme regions`

每个 region 含有：

- root chunks
- anchor chunks
- supporting chunks
- nodes
- edges
- focus entities
- relation themes
- descriptor text

region 是后续 facet grouping 的中间层。

## 11. Contract-aware Facet Grouping

这是统一联想主干之后，contract 差异真正落地的地方。

### 11.1 Section Group

`_build_section_groups()` 会：

1. 以 root region 的 seed chunks 为中心
2. 根据 chunk order 构造一个 section band
3. 在该 band 内收集 nodes/edges
4. 输出 `section band <doc>:<start>-<end>` 形式的 facet

section 的关键不是 broad aspect，而是局部段落带。

### 11.2 Mechanism Group

`_build_mechanism_groups()` 会围绕：

- root regions
- bridge regions
- theme regions

抽出能够形成因果链或作用链的区域，并输出：

- `pathway: A`
- `pathway: A -> B`

这种 label 的目的，是显式鼓励机制链条，而不是单纯 broad theme。

### 11.3 Comparison Group

`_build_comparison_groups()` 当前输出两类 facet：

- `comparison side N: ...`
- `contrast axis: ...`

前者强调可比的多个侧面，后者强调连接多个 side 的对照轴。

### 11.4 Theme Group

`_build_theme_groups()` 是当前 theme 组织的核心。

它不再只是 legacy 的按 primary theme 粗分，而是 slot-based 组织：

1. `_theme_slot_specs(query)` 先根据 query 生成 slots，例如：
   - examples
   - support
   - actions
   - drivers
   - outcomes
   - contexts
2. `_region_slot_affinity()` 计算 region 对 slot 的亲和度
3. `_select_regions_for_slot()` 为每个 slot 挑 3 到 4 个 region
4. `_slot_group_label()` 生成 facet label

这一步的目标，是把 final graph 中杂乱的证据压成多个 query-relevant aspects。

## 12. Candidate Points

`build_candidate_points_from_groups()` 将 facet groups 进一步压缩成候选要点。

它们不是最终答案，而是：

- 给 answer prompt 提供可选要点
- 帮助最终 LLM 从 facet 中挑重点

## 13. Prompt Context 组装算法

[context.py](/Users/Admin/projects/Association/associative_rag_project/context.py) 的 `build_prompt_context()` 将系统状态整理为 LLM 可以直接消费的 evidence package。

它会打包：

- root chunks
- top root nodes
- top root edges
- final nodes / final edges
- facet groups
- candidate points
- chunk roles
- source chunks

### 13.1 Theme 的 source packing

当 contract 为 theme 且存在 `theme_selected_chunks` 时，source chunk 优先级按：

1. `core`
2. `bridge`
3. `support`
4. `peripheral`

并且会：

- 放大 source chunk 上限
- 限制单 chunk 文本长度

这样做是为了让 broad QFS 有更多独立证据角度，而不是被一两个长 chunk 吃掉预算。

## 14. 最终回答 Prompt

[llm_client.py](/Users/Admin/projects/Association/associative_rag_project/llm_client.py) 的 `build_generation_prompt()` 负责拼接最终回答提示。

它包含三部分约束：

1. 通用 QFS 约束
2. contract-specific hints
3. broad theme 的额外覆盖约束

对于 `theme-grounded`，还会要求 Theme QFS 模板：

- `P1. Titles`
- `P2. Answer Outline`
- `P3. Queries, Summaries, and Evidence`
- `P4. Document Sections`
- `P5. Refinement`

当前实现里，theme 会更明确鼓励：

- 至少 5 个 aspect
- 每个 aspect 给 concrete named example
- 在证据允许时覆盖跨时期、跨场景的多样性

## 15. Judge 体系

评测由 [judge.py](/Users/Admin/projects/Association/associative_rag_project/judge.py) 完成。

### 15.1 Judge Prompt

judge 先判断：

- query 需要什么 organization contract
- candidate answer 实际使用了什么 organization
- baseline answer 实际使用了什么 organization

再比较七个维度：

- `Comprehensiveness`
- `Diversity`
- `Empowerment`
- `Focus Match`
- `Evidence Anchoring`
- `Scope Discipline`
- `Scenario Fidelity`

最后给出 `Overall Winner`。

### 15.2 双顺序判定

`judge_pair()` 会同时比较：

- candidate vs baseline
- baseline vs candidate

然后再把两次 judge 的 winner 映射回统一的 candidate/baseline 票数。

### 15.3 Contract-aware 汇总

`run_winrate_judgement()` 会输出：

- overall win rate
- per-criterion win rate
- per-contract criterion summary
- contract-conditioned summary

因此我们既可以看总体，也可以只盯某个 contract 的关键指标。

## 16. 当前系统的几个关键设计结论

### 16.1 检索与联想不再分裂

当前代码已经把所有 contract 统一到 theme-style retrieval + multi-round association 上。  
这使系统更容易讲清楚，也减少了“某类题一套逻辑、另一类题另一套逻辑”的维护成本。

### 16.2 Contract 差异主要后移到 organization

这意味着系统现在的叙事可以更清晰：

- 前面先尽量 recall 足够多的 chunk-side evidence
- 后面再决定以什么结构回答

### 16.3 Theme 路线是主干能力

目前 broad synthesis 的核心能力来自：

- graph focus recall
- 多轮 root reseeding
- slot-based theme grouping

### 16.4 仍然存在的数据集差异

尽管主干已经统一，不同数据集仍可能表现不同。主要原因通常是：

- 图的规模不同
- 图的稠密度不同
- chunk 粒度不同
- query 所需联想半径不同
- baseline 风格不同

这意味着后续还可以继续研究：

- 数据集级预算自适应
- query 级预算重分配
- contract-aware 但不分裂的细粒度 budget 调控

## 17. 一句话流程图版本

`Query -> Contract Detection -> Primary Retrieval -> Graph Focus / Graph Keyword Recall -> Diverse Root Selection -> Multi-round Chunk Association -> Root Reseeding -> Final Subgraph -> Contract-aware Facet Grouping -> Prompt Context Packing -> Answer LLM -> Contract-aware Judge`

## 18. 建议阅读顺序

如果要按代码理解系统，建议顺序如下：

1. [pipeline.py](/Users/Admin/projects/Association/associative_rag_project/pipeline.py)
2. [retrieval.py](/Users/Admin/projects/Association/associative_rag_project/retrieval.py)
3. [association.py](/Users/Admin/projects/Association/associative_rag_project/association.py)
4. [organization.py](/Users/Admin/projects/Association/associative_rag_project/organization.py)
5. [context.py](/Users/Admin/projects/Association/associative_rag_project/context.py)
6. [llm_client.py](/Users/Admin/projects/Association/associative_rag_project/llm_client.py)
7. [judge.py](/Users/Admin/projects/Association/associative_rag_project/judge.py)
