# 方法草稿（中文论文写法）

## 1. 任务定义

我们研究的任务是图增强的查询聚焦总结（Graph-Enhanced Query-Focused Summarization, QFS）。给定一个查询 `q`、一个由文本块构成的语料集合 `C={c_1,...,c_n}`，以及由语料诱导出的 chunk-entity-relation 图 `G=(V,E)`，系统需要生成一个既覆盖查询要求、又严格受证据约束的总结性回答 `a`。与传统单文档摘要或普通检索式问答不同，该任务的关键难点在于：一方面，系统需要围绕查询主动整合多个证据片段，形成多角度、可综合的回答；另一方面，系统又必须避免在跨文档联想过程中引入脱离问题范围的扩写、假想情景或不受支撑的叙事补全。

现有基于图的 RAG 方法虽然能够利用图结构连接分散证据，但在 QFS 场景下仍面临两个突出问题。第一，许多方法的联想过程本质上是围绕初始高分命中做局部扩展，容易反复停留在同一语义盆地中，导致最终回答虽然聚焦，却缺乏足够的广度和多样性。第二，不同类型的问题对知识组织方式有不同要求，例如 section-grounded 问题强调局部文段带，comparison-grounded 问题强调显式对照维度，而 theme-grounded 问题则更依赖多方面归纳；若检索与组织方式不匹配，即使召回了一部分相关证据，最终回答仍然可能在 `Comprehensiveness`、`Diversity` 或 `Focus Match` 上受损。

基于以上观察，我们提出一种统一的 Associative RAG 框架。该框架的核心思想是：先以 chunk 为基本调度单位，通过“检索-联想-重选根-再联想”的多轮过程在图上逐步爬行，从而获得更完整的证据覆盖；再根据查询契约（query contract）将同一批 final graph 组织为不同形态的回答结构。换言之，我们不再为不同问题类型设计彼此割裂的检索主干，而是使用统一的主题式联想骨干保证 recall 和 aspect coverage，再在组织层显式体现 section、mechanism、comparison 和 theme 之间的差异。

## 2. 方法总览

我们的方法由五个阶段组成：

1. `契约识别`：根据查询文本识别其主要组织需求，判断该问题更适合 section-grounded、mechanism-grounded、comparison-grounded 还是 theme-grounded 的回答结构。
2. `统一候选检索`：从 lexical/dense 主检索器和图侧 focus/keyword 检索器中召回候选 chunk，构造既保持 query relevance 又保留图侧扩展潜力的候选池。
3. `多样化根选择与多轮联想`：从候选池中选择少量多样化 root chunks，并在 chunk 图与实体关系图上交替扩展 bridge、support 与 peripheral 证据，同时在每一轮后重估新的根。
4. `契约感知的知识组织`：在 final graph 上构造 evidence regions，并按不同契约生成 section band、mechanism pathway、comparison side/axis 或 theme aspect。
5. `证据打包与答案生成`：将根证据、facet groups、支持 chunks 和候选要点压缩为一个 evidence package，交给生成模型产出最终回答。

图 1 可以概括为如下流程：

`Query -> Contract Detection -> Unified Candidate Retrieval -> Diverse Root Selection -> Multi-round Association with Root Reseeding -> Final Graph -> Contract-aware Facet Organization -> Evidence Packing -> Answer Generation`

我们的方法有三点核心创新。

第一，我们将联想的基本调度单位从“单个图节点/边”转移为“chunk”，把图理解为索引图而不是最终回答的直接表达对象。由于最终送入 LLM 的仍然是文本 chunk，而不是裸图结构，因此 chunk-level 调度更符合 QFS 的真实目标：我们真正关心的是哪些 chunk 共同支撑了一个方面、哪些 chunk 起到了桥接作用、哪些 chunk 带来了新的信息增益。

第二，我们提出多轮 root reseeding 的联想机制。传统图扩展容易围绕首次命中的核心词或核心局部区域不断重复扩展，而我们的策略在每一轮扩展后都重新构造候选根集合，并优先提升能够带来新方面、新文档或新 query term 覆盖的 chunk，从而把联想过程从“围绕单一中心旋转”改为“逐轮外爬并切换锚点”。

第三，我们将“检索”和“组织”明确分离：前者的目标是 recall 足够多、足够分散且仍受 query 约束的证据，后者才决定如何把这些证据组织成 theme、comparison 或 mechanism 结构。这样可以避免把“答案应该怎么写”的要求过早压到检索阶段，导致 recall 被不必要地收窄。

## 3. 查询契约识别

尽管本文最终采用统一的联想骨干，我们仍然保留查询契约识别模块，因为不同问题对最终知识组织的要求确实不同。形式上，我们定义一个映射函数

`f_contract(q) -> y, y in {section, mechanism, comparison, theme}`。

该模块是一个轻量级启发式识别器，其目标不是做细粒度语义理解，而是估计查询需要哪一类组织结构。对于包含 section、part、across different sections 等局部范围提示的查询，我们更倾向于 `section-grounded`；对于强调 how、what led to、how did X affect Y 这类机制链条的问题，我们标为 `mechanism-grounded`；对于出现 compare、contrast、difference、similarity 或显式双边对象的问题，我们标为 `comparison-grounded`；对于询问 overall influence、main themes、primary reasons、broad effects 的开放综述性问题，我们标为 `theme-grounded`。

需要强调的是，在我们的系统中，契约识别不再决定“是否走 completely different retrieval pipeline”，而是决定“在统一 final graph 上应如何组织答案”。这使得契约检测的风险显著下降：即便某条 query 的契约判定存在边界误差，系统也不会因此丢失整条检索主干，只会影响最后的组织形式。

## 4. 统一候选检索

给定查询 `q`，我们首先用主检索器获得候选 chunk 集合。主检索器可以是 BM25、dense retrieval 或 hybrid retrieval。设第 `i` 个 chunk 的主检索分数为 `s_ret(c_i|q)`。在 hybrid 设置下，该分数由 lexical 与 dense 两部分加权组合得到。

但仅依赖主检索会带来两个问题：一是容易过度依附 query 表面词；二是对图中潜在可扩展的结构性证据不敏感。为此，我们额外引入两条图侧召回通道。

其一是 `graph keyword retrieval`。我们将 query 直接与图中节点和边的文本描述进行 lexical 匹配，并把命中的节点、边投票回对应 chunk，得到图侧 lexical 分数 `s_kw(c_i|q)`。该通道的作用是提供一种不依赖 dense 向量的高召回图侧入口，尤其在图中已有较明确关键词标注时效果较稳。

其二是 `graph focus retrieval`。我们先从 query 中抽取非语法性内容词集合 `T_q`，再统计这些词在图节点和边文本中的覆盖率、命中数及其 specificity，最终得到 focus 分数 `s_focus(c_i|q)`。与简单的 lexical overlap 不同，该分数更强调“一个 chunk 是否通过其关联节点/边覆盖了 query 的关键内容词”。因此它不是泛泛地找相似文本，而是在图层面估计“该 chunk 是否可能成为一个 query-relevant aspect 的入口”。

最终，我们将三类信号合并为统一候选分数：

`s_cand(c_i|q) = Merge(s_ret(c_i|q), s_focus(c_i|q), s_kw(c_i|q))`

其中 `Merge` 在实现上采用保守加权的融合策略：首先保持主检索分数作为主锚点，再用 `s_focus` 和 `s_kw` 为已有候选做 boost，同时允许少量只在图侧命中的 chunk 进入候选池。对不同契约，我们只调整图侧权重，而不改变整体检索骨干。这样做的直觉是，theme 问题需要更高的图侧 recall，而 section 问题则更需要局部稳定性。

## 5. 多样化根选择

在候选池上，我们不直接取 top-k 最高分 chunk，而是进行多样化根选择。原因在于：QFS 问题特别是 broad theme 问题往往需要多个“起点方面”，如果根集合本身过度集中，后续联想即便走多轮，也极易在同一局部语义区反复游走。

具体地，我们首先为每个候选 chunk 构造一组辅助特征，包括：

- 主检索强度 `base_score`
- query 对齐强度 `query_alignment`
- chunk 文本与 query 的 lexical 对齐 `query_lexical`
- 由关联节点、边诱导的 provenance signature
- basin key，用于描述该 chunk 所处的语义/结构盆地
- relation entropy，用于描述该 chunk 引出的关系多样性

对于 theme 风格的根选择，我们进一步按 basin 将候选分组，并采用跨盆地轮转的方式选择 roots。每一次尝试加入新根时，我们都会检查：

1. 它是否与已选根来自同一文档局部带；
2. 它与已选根的 provenance overlap 是否过高；
3. 它是否能够带来新的 query term 覆盖；
4. 它是否帮助根集合进入新的文档或新的 basin。

由此得到的根集合不是简单的高分 chunk 排名，而是一组“彼此尽量分散、但仍受 query 约束”的起点。对于 section-grounded 问题，我们进一步引入单文档约束：首先由最高优先级根锚定一个 `full_doc_id`，然后仅在该文档内部保留并重选 roots，从而确保 section 回答始终围绕同一文档带展开。

## 6. 多轮联想扩图

根选择完成后，我们进入本文方法的核心步骤：多轮 chunk-level associative expansion。与传统图扩展只围绕初始根做一次固定 hop 不同，我们采用多轮循环，每一轮都在当前证据池基础上重新评估可扩展的 chunk，并允许根集合发生更新。

设第 `t` 轮时，已选 chunk 集合为 `S_t`，当前活动根集合为 `R_t`。我们首先从 `R_t` 出发构造 frontier chunk 集合 `F_t`，其来源包括：

- 当前根的结构邻居
- 与当前根共享节点或边的 chunk
- 局部 chunk 邻域中的相邻块

然后，我们从三个角色中挑选新增证据。

### 6.1 Bridge Chunk

bridge chunk 的作用是把当前证据区与新的图区域连接起来。我们用如下直觉对其打分：一个好的 bridge，不仅应当接触当前 frontier，还应当引入新的 query-relevant nodes/edges、新的 source chunks 甚至新的文档。用符号表示，可将其近似理解为：

`s_bridge(c) = alpha * frontier_touch(c) + beta * introduced_query_rel(c) + gamma * source_novelty(c)`

其中 `frontier_touch` 衡量它与当前 frontier 的连接程度，`introduced_query_rel` 衡量它新带入的节点和边与 query 的相关性，`source_novelty` 则鼓励它引入此前未覆盖的来源。

### 6.2 Support Chunk

support chunk 的作用不是结构桥接，而是信息增益。我们希望 support 能够把新的方面、新的实体、新的关系类型或新的文档信息带入证据池，而不是继续重复当前已知点。因此我们在 support 选择中更强调：

- 新节点/新边数量
- 新关系主题
- 新文档
- 信息熵与边际增益

也就是说，support 的目标更接近 breadth/comprehensiveness/diversity，而不是 bridge 的连接性目标。

### 6.3 Peripheral Chunk

peripheral chunk 是 bridge/support 周围的一层局部补充。它的作用是为刚刚引入的新证据提供近邻上下文，但其预算通常较小，以避免系统在外围噪声中失控扩张。

### 6.4 Query 分数刷新

如果联想过程始终使用初始 query relevance，系统很容易被最初命中的核心词“锁死”。为此，在每一轮完成 bridge/support/peripheral 选择后，我们用本轮新增证据去更新 chunk 的 query score lookup。直觉上讲，某个 chunk 即使在最初与 query 的表面匹配较弱，只要它在本轮通过新引入的相关节点、边证明了自己对当前证据图有价值，就应在下一轮得到更高优先级。

这一刷新机制使联想从“只看原始 query 相似度”变成“query relevance 与当前证据图共同决定下一轮扩展”。

### 6.5 Root Reseeding

这是我们方法区别于普通 theme expansion 的关键。每轮扩展后，我们不继续机械地围绕原始 roots 扩展，而是重新构造一个 seed pool，其中包括：

- 当前已选 chunks
- 本轮新增 chunks
- 这些 chunk 的邻居
- 原始候选池中尚未使用的高潜力 chunk

然后在该 pool 上重新进行多样化 root selection，得到下一轮活动根 `R_{t+1}`。新的根更容易来自：

- 新出现的方面
- 新文档
- 新 basin
- 在本轮中表现出较高 query gain 的 chunk

这一设计从根本上改变了联想行为：系统不再围绕同一个核心局部旋转，而是通过多轮“扩展-重评分-换根-再扩展”的循环，逐步向不同方面爬行。

### 6.6 统一主干与契约微调

我们将上述多轮联想主干统一用于所有 contract，但仍保留轻量的预算微调。例如：

- `section-grounded` 限制在单 `full_doc_id` 内，且 peripheral 预算更小；
- `mechanism-grounded` 适当保留 bridge 与 support 预算，以形成更清晰的路径链；
- `comparison-grounded` 会略提升 side/axis 所需的 bridge-support 组合；
- `theme-grounded` 使用最宽松的 broad expansion。

这种设计的好处在于：系统仍然只有一条主干方法，但可以通过很小的预算调节适应不同问题的组织需求。

## 7. 最终图上的证据区域构造

完成多轮扩展后，我们得到 final graph 及其覆盖的 chunk 集合。接下来，系统并不直接把全部图元素送给 LLM，而是先构造中间层的 `evidence regions`。每个 region 同时记录：

- 关联的 root chunks
- anchor chunks
- supporting chunks
- 覆盖的节点和边
- 焦点实体
- 关系主题
- 区域描述文本

region 的作用是把生硬的图结构转成更接近“可被总结的证据块”的对象。随后，我们再基于这些 region 做契约感知的 facet grouping。

## 8. 契约感知的知识组织

虽然前面的 retrieve 与 associate 主干统一了，但最终回答结构仍必须与 query 需求相匹配。

对于 `section-grounded` 问题，我们围绕 seed chunks 所在文档带构造 section band，并让 facet 明确对应文档中的局部范围。这类问题的重点不是尽量多角度，而是要稳稳锚定在相关 sections 上。

对于 `mechanism-grounded` 问题，我们围绕 bridge region 和 root region 组织 `pathway`。每个 facet 试图表达一条显式的作用链或因果链，而不是把所有相关信息平铺成主题列表。换言之，这类问题的组织目标是“机制路径”，不是“主题广度”。

对于 `comparison-grounded` 问题，我们显式组织出 `comparison side` 与 `contrast axis`。这样做的目的，是强迫最终答案具备清晰的对照维度，而不是把比较题写成一个泛化综述。

对于 `theme-grounded` 问题，我们采用 slot-based aspect grouping。系统先根据 query 自动生成若干方面槽位，例如 examples、drivers、outcomes、contexts、actions 等，再为每个槽位挑选最合适的 regions。最终得到的不是一组随意聚类的图片段，而是多个“与 query 直接相关的综述方面”。

这一步将统一的 final graph 映射为不同的回答结构，也是我们“统一检索主干，差异化组织输出”思想的核心体现。

## 9. 证据打包与答案生成

在得到 facet groups 之后，我们进一步抽取 candidate points，并将 roots、facet groups、节点边摘要、source chunks 一并压缩为一个 evidence package。这里最关键的设计原则是：facet groups 只是证据导航，而不是最终回答脚本。最终 LLM 被要求基于 evidence package 自主选择最合适的点来回答问题，而不是机械复述所有 group。

对于 theme 问题，我们在 prompt 中加入了更强的结构化约束，例如要求输出 aspect titles、outline、summary 与 evidence tuple，以鼓励模型真正把多角度证据转化为多方面综述，而不是把所有证据压成一条狭窄的叙事线。

## 10. 方法小结

综上，我们的方法可以概括为一种“统一 theme-style 联想骨干 + 契约感知组织层”的 QFS 框架。与传统图 RAG 相比，它的关键改进不在于简单增加 hop 或扩大图，而在于：

1. 用 chunk 而不是裸节点/边作为联想调度单位；
2. 用 graph focus recall 和 graph keyword recall 补足主检索的盲区；
3. 通过多轮 root reseeding 避免联想长期困在同一 query term basin；
4. 将 recall 与 organization 解耦，先追求足够好的证据覆盖，再根据 query contract 组织为不同回答形态。

从方法论上看，这一框架尤其适合 query-focused summarization 这类任务：系统既要在多文档、多 chunk 间做总结，又不能失去 focus。我们的设计正是在“广度”和“约束”之间建立一个可控的中间层，从而让最终回答既更全面，也更贴合问题。
