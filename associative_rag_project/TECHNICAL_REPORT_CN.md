# Associative RAG 项目技术报告（中文，当前代码版）

对应的主要文件如下：

- [main.py](/Users/Admin/projects/Association/associative_rag_project/main.py)
- [pipeline.py](/Users/Admin/projects/Association/associative_rag_project/pipeline.py)
- [retrieval.py](/Users/Admin/projects/Association/associative_rag_project/retrieval.py)
- [association.py](/Users/Admin/projects/Association/associative_rag_project/association.py)
- [organization.py](/Users/Admin/projects/Association/associative_rag_project/organization.py)
- [context.py](/Users/Admin/projects/Association/associative_rag_project/context.py)
- [llm_client.py](/Users/Admin/projects/Association/associative_rag_project/llm_client.py)
- [judge.py](/Users/Admin/projects/Association/associative_rag_project/judge.py)
- [data.py](/Users/Admin/projects/Association/associative_rag_project/data.py)
- [config.py](/Users/Admin/projects/Association/associative_rag_project/config.py)

## 1. 系统目标

当前系统面向 Query-Focused Summarization。核心目标不是做开放式长回答，而是：

1. 从 corpus 中找出与 query 最相关的原始 chunk。
2. 将这些 chunk 投影到图结构中，形成 grounded roots。
3. 在图和 provenance 两条结构上做联想扩展，尽量把相关证据带展开。
4. 将最终子图组织成适合回答当前 query 的 facet groups。
5. 将 facet groups 和 source chunks 打包成 evidence package。
6. 让生成模型基于 evidence package 写出 query-focused answer。
7. 用 LLM judge 从多个维度做双向对比评测。

项目当前的基本分工很明确：

- 检索与联想阶段主要追求 recall 和 evidence coverage。
- 组织阶段主要决定答案应当围绕哪些 facet 展开。
- 生成阶段主要决定“同一包证据如何被写出来”。

## 2. 端到端流程

完整流水线在 [pipeline.py](/Users/Admin/projects/Association/associative_rag_project/pipeline.py) 的 `run_query(...)` 中串起来，顺序如下：

1. `chunk_retriever.search(...)`
   从 chunk 索引中拿 query 的 candidate chunks。
2. `select_diverse_root_chunks(...)`
   从 candidate 中挑出少量但更分散的 root chunks。
3. `score_root_nodes(...)` / `score_root_edges(...)`
   对 roots 投影出的节点和边做轻量打分。
4. `expand_associative_graph(...)`
   在图和 chunk 侧做多轮联想扩展。
5. `build_answer_facet_groups(...)`
   识别 query contract，并把最终子图组织成 facet groups。
6. `build_prompt_context(...)`
   为 LLM 生成最终 evidence package。
7. `build_generation_prompt(...)`
   按 query 类型拼出最终生成 prompt。
8. `generate_answers(...)`
   调用 LLM 生成答案。

## 3. CLI 与默认参数

CLI 在 [main.py](/Users/Admin/projects/Association/associative_rag_project/main.py) 中定义，支持：

- `retrieve`
- `answer`
- `judge`
- `run`
- `run-all`

当前默认超参数如下。

### 3.1 检索层默认

- `retrieval_mode = dense`
- `top_chunks = 5`
- `chunk_candidate_multiplier = 3`
- `adaptive_candidate_pool_size = 30`
- `dense_weight = 0.75`
- `bm25_weight = 0.25`

解释：

- 默认第一跳是 dense retrieval。
- 实际最终 roots 默认是 `5` 个。
- 但第一轮候选池会放大到至少 `top_chunks * 3`，并且不少于 `30`，方便后面估计 retrieval cliff、candidate dispersion 等特征。
- `dense_weight / bm25_weight` 只有在 `hybrid` 模式下才真正生效。

### 3.2 root graph 默认

- `top_root_nodes = 12`
- `top_root_edges = 16`

解释：

- roots 投影到图以后，不是所有 root-induced nodes/edges 都平等进入后续联想。
- 系统会先对 root nodes 和 root edges 做轻量打分，再截断到这两个上限。

### 3.3 联想层默认

- `max_hop = 4`
- `path_budget = 12`
- `semantic_edge_budget = 20`
- `semantic_node_budget = 12`
- `semantic_edge_min_score = 0.03`
- `semantic_node_min_score = 0.03`
- `association_rounds = 2`

解释：

- 结构联想的 graph shortest path 最多看 4 跳。
- 每轮会做桥接型扩展和覆盖型扩展。
- 默认总共做 2 轮。

### 3.4 组织与上下文默认

- `group_limit = 8`
- `max_source_chunks = 14`
- `max_source_word_budget = 4500`

解释：

- 最终 facet groups 最多保留 8 个。
- 最终送进 prompt 的 source chunks 最多 14 个。
- source 文本预算上限是 4500 词左右。

### 3.5 adaptive control 默认

- `adaptive_control = False`

解释：

- 代码里已经有 adaptive controller，但默认关闭。
- 当前主实验默认仍然使用固定预算。
- 即便关闭，系统仍会计算 adaptive features，方便后续分析和日志记录。

### 3.6 `query_style` 默认状态

当前代码里 `query_style` 有三类：

- `balanced`
- `synthesis`
- `concrete`

但在默认配置下，最终写入结果文件的往往是：

- `query_style = "disabled"`

原因不是系统没有这条逻辑，而是：

- `query_style` 的判定属于 adaptive controller 的一部分
- 当前默认 `adaptive_control = False`
- 因此 pipeline 会保留 adaptive profile 用于分析，但在最终记录里把实际生效的 style 标为 `disabled`

换句话说：

- 系统内部仍然会估计 query 更像 `synthesis / concrete / balanced`
- 但默认实验中，这个 style 不作为正式启用的控制信号
- 只有开启 adaptive control 后，`query_style` 才会真正作为运行时策略的一部分传到生成层

## 4. 数据输入假设

相关逻辑在 [data.py](/Users/Admin/projects/Association/associative_rag_project/data.py)。

每个 corpus 目录至少需要：

- `graph_chunk_entity_relation.graphml`
- `kv_store_text_chunks.json`

若使用 dense 或 hybrid，还需要：

- `vdb_chunks.json`

问题文件支持：

- `.txt`
- `.json`

baseline 默认通过 corpus 名自动解析，也可以手动传入。

## 5. Chunk 检索

相关逻辑在 [retrieval.py](/Users/Admin/projects/Association/associative_rag_project/retrieval.py)。

### 5.1 BM25

[BM25Index](/Users/Admin/projects/Association/associative_rag_project/retrieval.py) 是标准 in-memory BM25：

- 预先 tokenize 每个 chunk
- 记录 postings
- 记录文档长度
- 查询时返回 `score` 和 `score_norm`

### 5.2 Dense

[DenseChunkIndex](/Users/Admin/projects/Association/associative_rag_project/retrieval.py)：

- 从 `vdb_chunks.json` 读取 embedding matrix
- 归一化后做 cosine similarity top-k
- 返回 `dense_score` 和 `dense_score_norm`

### 5.3 Hybrid

[HybridChunkRetriever](/Users/Admin/projects/Association/associative_rag_project/retrieval.py) 支持：

- `bm25`
- `dense`
- `hybrid`

`hybrid` 融合公式为：

`retrieval_score = dense_weight * dense_score_norm + bm25_weight * bm25_score_norm`

当前默认仍是 `dense`。

## 6. Root Chunk 选择

当前 root 选择逻辑在 [select_diverse_root_chunks(...)](/Users/Admin/projects/Association/associative_rag_project/retrieval.py)。

这一步是当前实现非常关键的设计点。系统不再单纯按 retrieval rank 取前 `k`，而是做一个简单但可解释的 diversity-first 策略。

### 6.1 当前默认规则

函数默认参数为：

- `same_doc_window = 1`
- `max_same_doc_roots = 1`
- `relaxed_max_same_doc_roots = 2`
- `max_provenance_overlap = 0.55`
- `relaxed_max_provenance_overlap = 0.85`

### 6.2 选择逻辑

第一阶段：

- 按 `base_score` 排序，优先保最强 anchor。
- 尽量一篇文档只保一个 root。
- 禁止选取同一文档相邻 band 的 chunk。
- 如果某个候选与已选 roots 的 provenance overlap 太高，则先 defer。

第二阶段：

- 若第一阶段还没选满 `top_k`，则对 deferred 候选做 relaxed pass。
- relaxed pass 允许有限度地补回同文档 root 和较高 overlap root。

### 6.3 设计目的

这一步的目的不是“均匀抽样”，而是：

- 避免最强文档连续占满起始预算。
- 尽量把起点分散到多个证据簇。
- 为后续 association 和 organization 留出更宽的 coverage 空间。

## 7. 图联想：当前是 2 x 2 结构

相关逻辑在 [association.py](/Users/Admin/projects/Association/associative_rag_project/association.py)。

当前实现把联想明确拆成两个结构、两种目标，因此是一个 2 x 2 方案：

- graph-bridge association
- chunk-bridge association
- graph-coverage association
- chunk-coverage association

也就是：

- 一边在 graph 结构上找桥。
- 一边在 chunk / provenance 邻接上找桥。
- 一边补结构连接。
- 一边补新证据带和新关系。

### 7.1 结构桥接

桥接型扩展的目标是把当前分散的证据区域连起来。

[bridge_association(...)](/Users/Admin/projects/Association/associative_rag_project/association.py) 会同时做两件事：

1. `_graph_bridge_association(...)`
   用 shortest paths 连接不同的 root components。
2. `_chunk_bridge_association(...)`
   从 chunk 邻域和 provenance 邻域里找能触达当前 frontier、并引入新图内容的 chunk。

这意味着系统不只把“图上的最短路”当桥，也把“chunk 侧的局部带状扩展”当桥。

### 7.2 覆盖扩展

覆盖型扩展的目标是引入新的证据维度，而不是只补结构。

这部分会综合考虑：

- 新引入的 supporting chunks 数量
- 关系类型的丰富度
- 与 root evidence 的对齐程度
- 当前 frontier 的连接性

### 7.3 多轮联想

默认 `association_rounds = 2`。

每一轮都不是只在最初 roots 上操作，而是在“当前已经扩展出来的子图”上继续做：

1. structural association
2. semantic / coverage association

这也是为什么当前系统的 association 更像逐轮扩展，而不是单次 hop-based 搜索。

## 8. 组织阶段：先选一个 query contract

相关逻辑在 [organization.py](/Users/Admin/projects/Association/associative_rag_project/organization.py)。

组织阶段的第一步是 `detect_query_contract(query)`。当前系统要求每个 query 只选一个 contract：

- `section-grounded`
- `mechanism-grounded`
- `comparison-grounded`
- `theme-grounded`

### 8.1 contract 识别方式

当前是词法启发式，不是训练分类器。

主要线索包括：

- 明确 section / parts / passages / periods 等表述
- compare / difference / versus 等对比表述
- how + affect/change/influence/lead to 等机制表述
- reasons / themes / patterns / role / examples 等 broad theme 表述

这一步的设计目标是：

- 先粗分 query 需要的答案组织方式。
- 再用该 contract 去组织 facet groups 和约束生成阶段。

## 9. Evidence Regions 与 Facet Groups

组织阶段不是直接把 final graph 的 connected components 原样塞给 LLM。

系统会先构建若干 `EvidenceRegion`，来源包括：

- root-anchored regions
- bridge-derived regions
- theme / relation regions

每个 region 会记录：

- `region_kind`
- `root_chunk_ids`
- `anchor_chunk_ids`
- `supporting_chunk_ids`
- `node_ids`
- `edge_ids`
- `relation_themes`
- `focus_entities`
- `doc_ids`
- `growth_traces`

这些 region 之后会被进一步合并和筛选为最终 facet groups。

## 10. 当前 facet 选择逻辑：coverage-first

这部分是当前系统相较早期版本变化最大的地方之一。

[organization.py](/Users/Admin/projects/Association/associative_rag_project/organization.py) 中的 `_select_groups(...)` 不再只是“按分数排序然后去重”，而是 coverage-first greedy selection。

### 10.1 当前选择原则

系统先看一个候选 group 能否补到新的 coverage 维度，再决定是否优先保留。当前覆盖键会根据 contract 使用不同组合，但常见维度包括：

- `facet_label`
- `relation theme`
- `doc`
- `root`
- `region kind`

### 10.2 这一步的效果

这会让最终 facet groups 更偏向：

- 先保不同方面
- 再保高分重复支线

而不是：

- 被一条很强但很窄的 dense support chain 占满

### 10.3 这一步不做什么

当前 coverage-first 仍然不是强模板系统。它不会：

- 强制每个 facet 都变成一个段落
- 强制每一类维度都必须出现
- 在证据不足时硬凑 coverage

它只是把 facet selection 的优先级从“先看 rank”改成了“先看能否补新 coverage”。

## 11. Prompt Context 构造

相关逻辑在 [context.py](/Users/Admin/projects/Association/associative_rag_project/context.py)。

这一层的作用是把 facet groups 和 source chunks 变成最终 LLM 能读的 evidence package。

### 11.1 source chunk 排名

`rank_supporting_chunks(...)` 会根据 final graph 对 chunk 做支持度排序：

- 被更多 final nodes 覆盖的 chunk 分数更高
- 被更多 final edges 覆盖的 chunk 分数更高
- root chunks 只获得轻量加分，不再拥有巨大常数 boost

这意味着 grounding 不再主要靠“root 永远压倒一切”，而更多依赖 facet 内部的 anchor selection。

### 11.2 当前 source packaging 策略

`choose_diverse_source_chunks(...)` 现在采用三段式：

1. 每个 facet 先争取 1 个 chunk
2. 每个 facet 再争取第 2 个 chunk
3. 最后才按全局 rank 补剩余预算

同时还有限制：

- 同一文档相邻 local band 不能无限重复入选
- `_violates_local_band_cap(...)` 会限制相近 chunk 的局部堆叠

### 11.3 Coverage Checklist

当前 evidence package 中还会加入一个轻量的 `Coverage Checklist`，来自 `_coverage_checklist_lines(...)`。

其作用不是强制答案逐项 checklist 化，而是提醒生成模型：

- 这些 facet labels 是当前已拿到的 major supported aspects
- 如果 query 属于 broad theme 类型，不要过早把它们压缩掉

## 12. 生成阶段：当前真正决定“写法”的一层

相关逻辑在 [llm_client.py](/Users/Admin/projects/Association/associative_rag_project/llm_client.py)。

### 12.1 system prompt

当前 `GENERATION_SYSTEM_PROMPT` 的核心约束是：

- 做 query-focused summarization
- 优先 multi-source aggregation
- 优先 theme organization
- 证据薄弱时显式承认不确定性
- 不要把回答写成“这是图检索系统给我的结果”
- 优先 breadth，而不是把一个窄线程过度展开

### 12.2 contract hints

`_contract_template_hints(...)` 会根据 `query_contract` 给出软提示：

- section-grounded
- mechanism-grounded
- comparison-grounded
- theme-grounded

这些只是 optional organization templates，不是硬模板。

### 12.3 `query_style` 在生成阶段的作用

`build_generation_prompt(...)` 还会接收一个 `query_style` 参数。当前它支持三种风格：

- `synthesis`
- `concrete`
- `balanced`

对应影响如下：

`synthesis`

- 额外提示模型强调 cross-source patterns、contrasts 和 thematic structure。
- 更适合 overview / pattern / broad-theme 类问题。

`concrete`

- 额外提示模型优先给出直接、实用、局部、明确支持的点。
- 尽量避免不必要的泛化扩展。

`balanced`

- 不额外向两端偏置，保持默认的 query-focused summary 风格。

### 12.4 broad theme query 检测

`_is_broad_theme_query(...)` 会对 broad `theme-grounded` query 给额外提示。

这一步使用的是通用 query-shape cues，例如：

- `what are the primary reasons`
- `what strategies`
- `which external resources`
- `in what ways`
- `what role`

### 12.5 当前 broad-theme 生成策略

这是当前版本最重要的生成改动之一。

对 broad theme query，prompt 会明确要求：

- 先覆盖 major supported aspects，再展开局部
- 若证据支持多个并列方面，优先覆盖约 `4-7` 个 aspect
- 使用 `Coverage Checklist` 作为提醒，但不能硬塞 unsupported items

同时，prompt 明确取消了早期那种过强的压缩倾向。当前会要求模型：

- 先识别“主要支持的 aspects”
- 不要只抓最强的 `1-2` 个 facet
- 也不要机械地把每个 facet 原样照抄出来

换句话说，当前生成层追求的是：

- 保持 grounded
- 但不要过早把多 facet coverage 压成 2 到 3 个大点

## 12A. `query_style` 是如何判出来的

这部分逻辑在 [adaptive_control.py](/Users/Admin/projects/Association/associative_rag_project/adaptive_control.py)。

### 12A.1 判定方式

`compute_query_intent_profile(query)` 会根据通用 discourse cues 做一个很轻量的 intent 估计。

它维护两组 cue：

- `OVERVIEW_CUES`
- `FOCUSED_CUES`

例如：

- overview cues 更偏 `overall / patterns / themes / across / various / role / influence`
- focused cues 更偏 `which / where / specific / section / steps / signs / costs / examples`

然后计算：

- `overview_hits`
- `focused_hits`

规则非常简单：

- 如果 `focused_hits >= overview_hits + 1`，判为 `concrete`
- 如果 `overview_hits >= focused_hits + 1`，判为 `synthesis`
- 否则判为 `balanced`

### 12A.2 它和 adaptive control 的关系

当前代码里，`query_style` 是 adaptive controller 输出的一部分，但它和 budget 调节并不是同一件事。

更准确地说：

- adaptive controller 会同时产出
  - `query_style`
  - `association_strength`
  - graph / retrieval / candidate features
  - adapted budgets
- 当 `adaptive_control = False` 时
  - 预算不会按这些特征动态调整
  - pipeline 会把最终 `query_style` 记为 `disabled`
  - 但底层 intent/profile 仍然会被计算并保存在 `adaptive_profile` 里，供日志和分析使用

### 12A.3 为什么会这样设计

当前实现故意把 adaptive control 设计得很轻：

- 默认系统行为应当是非自适应、可复现、易解释
- style 判定和各种先验特征先保留下来做分析
- 等这些先验被证明稳定有效后，再把它们逐步转成正式控制信号

因此，看到输出里是：

- `"query_style": "disabled"`

并不表示系统完全没有 query-style 概念，而是表示：

- 当前这次运行没有让 query-style 作为正式在线控制量生效
- 只是保留了这套判定机制和相关诊断信息

## 13. Judge 维度

相关逻辑在 [judge.py](/Users/Admin/projects/Association/associative_rag_project/judge.py)。

当前 judge 不是只给一个 overall winner，而是显式拆成以下维度：

- `Comprehensiveness`
- `Diversity`
- `Empowerment`
- `Focus Match`
- `Evidence Anchoring`
- `Overall Winner`

同时，judge 还会额外输出：

- `Query Organization Need`
- `Answer 1 Organization`
- `Answer 2 Organization`

### 13.1 这些维度的实际作用

`Comprehensiveness`

- 看答案覆盖面是否足够全。

`Diversity`

- 看答案是否提供了多个角度、多个方面，而不是重复一个主线。

`Empowerment`

- 看答案是否让读者更容易做出判断、理解结构和因果。

`Focus Match`

- 看答案的组织方式是否匹配 query 真正需要的组织方式。
- 这更像是“有没有答对题、答在点上”。

`Evidence Anchoring`

- 看答案是否稳定站在 corpus-specific evidence 上。
- 这更像是“是不是 grounded，而不是只听起来合理”。

### 13.2 双向评测

`judge_pair(...)` 会：

1. candidate vs baseline
2. baseline vs candidate

双向各判一次，再把票数映射回 candidate / baseline，以降低位置偏差。

## 14. 当前系统为什么会有 tree3 这类提升

从当前代码和最近实验现象看，系统提升的来源可以分成三层：

### 14.1 tree1：组织层开始显式保 coverage

通过 coverage-first facet selection 和 per-facet source packaging，系统不再只是保最强 support chain，而是更倾向于保不同方面。

### 14.2 tree2：root 选择更分散

通过更严格的 same-doc / same-band / provenance-overlap 限制，roots 起点更宽，部分 broad theme query 能拿到更分散的原始证据。

### 14.3 tree3：生成层不再过早压缩 facet

这是当前代码最关键的一步。

在 retrieval 统计基本不变的情况下，tree3 的明显提升说明：

- 证据包本身没有大幅变化
- 变化主要来自最终 prompt

也就是：

- 同一包 evidence，在旧 prompt 下容易被压成过窄答案
- 在当前 prompt 下，模型更愿意先覆盖 major aspects，再展开细节

因此当前代码版的一个核心结论是：

> 系统主瓶颈已经不只是“检到什么”，而是“同一包证据如何被组织和写出来”。

## 15. 是否存在针对 agriculture / art 的专门优化

在当前代码中，我额外检查了是否存在针对 `agriculture` 或 `art` 数据集的专门优化，包括：

- 按 corpus 名分支
- 特定数据集关键词触发特殊逻辑
- 针对农业或艺术题材的专门 prompt
- 针对具体 benchmark 题目的硬编码回答指导

结论是：

### 15.1 没有发现数据集名级别的专门优化

当前代码中没有发现如下模式：

- `if corpus_name == "agriculture": ...`
- `if corpus_name == "art": ...`
- 针对 `agriculture` 或 `art` 单独换 prompt
- 针对这两个数据集单独换检索、组织、打包、生成逻辑

### 15.2 但存在通用的 query-shape 词法启发式

系统中确实存在一些词法 cue，例如：

- `what are the primary reasons`
- `which external resources`
- `what strategies`
- `in what ways`
- `compare`
- `what sections`

这些 cue 出现在：

- [organization.py](/Users/Admin/projects/Association/associative_rag_project/organization.py) 的 contract 检测
- [llm_client.py](/Users/Admin/projects/Association/associative_rag_project/llm_client.py) 的 broad-theme 检测

它们的作用是识别 query 的组织需求，而不是识别数据集主题。

因此更准确的表述应当是：

- 当前系统没有针对 `agriculture` 或 `art` 的 corpus-specific tuning
- 但有针对 query form 的通用启发式
- 如果某个 benchmark 的题面经常使用这些表述，那么它会自然更容易触发对应的组织策略

这属于 query-shape adaptation，不属于 dataset-specific hardcoding。

## 16. 当前版本的优点与边界

### 16.1 当前优点

- root 选择更分散，起点不容易被单文档占满
- association 同时利用 graph 路径和 chunk 邻域
- organization 不再只是去重，而是显式保 coverage
- source packaging 不再只按全局 rank，而是先保每个 facet 的发声权
- broad theme query 的生成 prompt 更重视覆盖多个 supported aspects
- judge 维度更细，能定位是 focus 问题、grounding 问题还是 coverage 问题

### 16.2 当前边界

- contract 检测仍是词法启发式，不是学习式分类器
- broad-theme 检测同样是启发式
- adaptive control 已经接入，但默认关闭
- source budget 仍然比较紧，宽 query 上仍可能出现 coverage 不足
- 对某些需要更窄、更局部、更强 anchored 的 query，coverage-first 策略未必总是占优

## 17. 当前默认结论

如果只用一句话概括当前代码版系统：

> 这是一个以“多起点 root + 多轮图联想 + contract-aware 组织 + coverage-first 打包 + broad-theme 生成约束”为核心的 Associative RAG QFS 系统。

再更具体一点：

- 检索层负责把证据找出来。
- 联想层负责把证据带展开。
- 组织层负责把证据整理成 query-facing facets。
- 生成层负责避免把 facet 过早压缩成过窄答案。

而从当前代码和最近实验一起看，系统最重要的经验是：

> 对 broad thematic queries，真正的性能拐点往往不是再多检一点，而是不要把已经拿到的 supported facets 提前写没了。
