# 面向 QFS 的关系感知联想式图RAG（NOLLM）

### 1.1 Query-Focused Summarization 的任务本质

Query-Focused Summarization（QFS）任务的目标是围绕用户 query，从长文本或多文档语料中发现、筛选并组织分散信息，最终形成面向 query 的综合性总结。与传统事实型问答相比，QFS 更强调两类连续能力：其一，系统需要围绕 query 对语料进行多方面扩展，从不同主题侧面、相关实体与潜在关联中发现支持总结的候选证据；其二，系统还需要将这些分散证据进一步压缩、去重与归并，形成适合生成总结的中间知识结构。因而，QFS 的关键并不只是“找到相关内容”，而是“围绕 query 完成有控制的知识发散，并将发散结果组织为可总结的证据集合”。GraphRAG 也明确指出，面向整个语料的 global question 本质上属于 query-focused summarization，而非传统显式检索任务。

### 1.2 从传统 RAG 到 GraphRAG：从局部命中到结构化总结

传统基于 chunk 相似度的 RAG 通常采用平面化文本切分与向量检索机制，其优势在于能够快速命中局部相关片段，但对于 QFS 这类需要跨文档、跨主题整合信息的任务，往往只能返回若干语义相近却彼此割裂的局部证据，平面化表示会导致复杂依赖关系难以被捕获，从而产生上下文割裂和回答碎片化问题。正是在这一背景下，GraphRAG 路线逐步形成：研究者开始将原本分散在文本中的实体、关系、主题和语料片段显式组织为图结构，使系统不仅能够“检索到局部相关内容”，还能够借助图上的邻接关系、社区层级或关系路径完成多源知识的扩展与聚合，从而更适合支撑面向 query 的总结生成。

### 1.3 现有 GraphRAG 方法的共同范式：离线构图、在线发散与知识组织

现有 GraphRAG 方法虽然在实现细节上存在差异，但整体上大多遵循相近的三阶段范式，即**离线构图、在线检索发散以及在线知识组织**。其中，离线阶段的核心目标，是将原本以 chunk 形式分散存储的文本内容，转化为具有显式语义关系的图结构及其中间表示；在线阶段则围绕用户 query，在图结构上进行相关知识扩展，并进一步将扩展得到的多源证据组织为适合总结生成的输入。

在**离线构图**方面，GraphRAG首先利用大模型从原始语料中抽取实体及其关系，构建实体知识图谱，并进一步对图中紧密关联的实体群体进行社区划分与社区摘要预生成，从而形成层级化的图索引与中间总结结构。LightRAG则更强调图索引的轻量化与可更新性，在保留图增强检索能力的同时，构建兼顾低层实体知识与高层主题信息的双层检索结构。PathRAG采用了和LightRAG一致的构图方式。FG-RAG同样建立在图索引之上，但其离线表示更服务于后续细粒度、query-aware 的实体扩展与总结过程。

在**在线检索发散**方面，各方法都不再满足于传统 RAG 中“query 对 chunk 的一次性相似度命中”，而是尝试借助图结构围绕 query 进行多方面扩展。GraphRAG 在查询时主要通过社区级中间表示触发对全局相关主题的覆盖，更适合面向语料整体的全局性问题。LightRAG 则通过双层检索联合图扩展机制，在实体层和主题层同时召回相关内容，其中基于图邻域的局部扩展可以看作是围绕 query 进行的一种轻量发散。PathRAG 的在线检索过程进一步从“召回更多节点”转向“召回更关键的关系路径”，强调通过路径作为基本单元来约束扩展方向。FG-RAG 则更明确地将 query-awareness 注入发散阶段，通过 context-aware entity expansion 扩展与 query 紧密相关的实体及其上下文信息，以增强召回内容的覆盖度与针对性。

在**在线知识组织**方面，现有方法普遍认识到：仅靠发散式召回仍不足以支撑高质量 QFS，系统还必须将多源、异构且可能冗余的检索结果压缩为可总结的知识结构。GraphRAG 的做法是以社区摘要为中介，先基于各社区生成 partial responses，再对多个局部响应进行再次汇总，从而完成全局组织。LightRAG 虽然组织机制相对轻量，但本质上也在尝试将双层召回结果统一整理后交由模型生成。PathRAG 则将知识组织显式提升为核心问题，指出现有 graph-based RAG 的瓶颈更多在于检索结果冗余和 prompt 内部的扁平化组织，因此通过 flow-based pruning 和 path-based prompting，将图中的关系路径转化为更具逻辑结构的生成依据。FG-RAG 进一步提出 query-level fine-grained summarization，不再满足于粗粒度社区或局部片段汇总，而是将召回到的上下文信息压缩为与当前 query 更细粒度对齐的中间证据表示。

总体来看，这些方法共同表明，GraphRAG 面向 QFS 的关键不只是“把文本构造成图”，而是通过**离线构图提供结构基础，通过在线检索实现围绕 query 的知识发散，再通过在线组织将发散结果整理为支持总结的证据集合**。不同方法之间的主要差异，也恰恰体现在这三个环节的侧重点不同：GraphRAG 更强调社区级组织，LightRAG 更强调轻量图增强检索，PathRAG 更强调路径级收束与去冗余，FG-RAG 则更突出 query-aware 的细粒度扩展与总结。

### 1.4 现有方法的关键不足：发散过程仍然薄弱

尽管现有方法已经引入图结构，但其检索联想过程在很大程度上仍然依赖一种相似范式：首先由 LLM 从 query 中生成关键词、实体或主题锚点，再借助 embedding 匹配在图中命中节点、边或局部子图，随后通过邻域扩展、路径筛选或细粒度汇总获得候选知识。该范式相较于纯 chunk 检索已显著增强了系统的知识组织能力，但在“面向 query 的发散”这一更前置的环节上，仍存在明显局限。其一，系统的发散起点高度依赖于初始 query 解析的质量，一旦关键词或实体锚定偏窄，后续图扩展往往只能在有限局部内游走；其二，多数方法的扩展机制仍偏向局部邻域传播或关系链筛选，缺乏围绕 query 进行多维主题展开的主动发散能力；其三，现有方法更多关注召回后如何压缩、剪枝和组织证据，而较少研究如何从 query 出发稳定地产生覆盖更广、方向更合理的发散结果。对于 QFS 而言，系统真正需要的并不只是“想到一些相关内容”，而是能够围绕 query 在知识库中展开更有组织的多方面扩展，并据此形成支持总结的结构化证据集合。

### 1.5 关键问题不止是关键词摇摆，而是回答被“预设情景”限制

仅仅证明 LLM 生成的 keyword 或 entity 会漂移还不够。对于 QFS 而言，更关键的问题在于：**即使图上仍然存在足够相关的信息，只要系统的第一步把 query 解释成了一个过窄或过于常识化的情景，最终 answer 也会被限制在这个情景之内。** 因而，本文关注的并不是“LLM 生成的关键词是否完全错误”，而是更强的命题：

> **现有 LightRAG / FG-RAG 在 QFS 上都默认：query 可以先被 LLM 转成一组可靠的 keyword / entity，然后系统再从这些入口出发做图检索与总结。**
>
> **但对 QFS 来说，query 往往问的是方面、机制、模式与比较关系，而不是显式实体。此时如果第一步交给 LLM 自由联想，模型往往会先设定一个“想象中的回答情景”，再据此生成关键词或实体。这样得到的入口不一定完全错误，但会让后续检索与总结更容易停留在该情景中，从而导致回答具有较强的局限性。**

换言之，问题不能止步于“图入口不稳定”，而必须落实到：

> **LLM-first query association 会在 evidence grounding 之前先对 query 做情景化解释，而最终 answer 往往继承这种先验情景，因此表现为覆盖面不足、组织方式偏离，或者生成出看似合理但局限很强的总结。**

下面给出三个直接面向最终 answer 的 motivating examples。

#### Example 1: Art 数据集中，query 被预设为“艺术史常识叙事”，导致回答锁定在单一框架中

以 `art` 数据集中的 query 为例：

> *How did the political climate described in various sections influence the emergence of new art forms?*

在 LightRAG 的 query keyword cache 中，可以直接观察到第一步 LLM 产出的高层与低层关键词：

- high-level: `Political climate`, `Emergence`, `New art forms`, `Cultural influence`
- low-level: `Art movements`, `Historical context`, `Socio-political factors`, `Avant-garde`, `Expressionism`

这些词表面上与 query 相符，但它们实际上已经把问题预设成了一种较典型的艺术史叙事，即“政治动荡如何推动现代主义或前卫艺术产生”。问题在于，QFS query 真正需要的是围绕当前语料去综合不同方面的证据，而不是先接受一套通用的艺术史解释框架。

这一点最终体现在 answer 上。LightRAG 基线答案会自然落到如下结构中：

- 以一战、现代主义、Dada、先锋派等历史叙事为主轴；
- 用若干艺术史运动或宏观事件来串联回答；
- 回答虽然流畅、合理，但更像对“艺术与政治关系”的一般性常识总结，而不是严格围绕当前语料完成的多方面 QFS。

一个具有代表性的回答节选如下：

> “The political climate of the early 20th century significantly influenced the emergence of new art forms...”
>
> “Rise of Modernism in Response to World War I...”
>
> “Similarly, the American artist Patrick Dougherty was inspired by nature and bowerbirds...”
>
> “The Weimar Republic created a backdrop for the emergence of the New Sobriety movement...”

更重要的是，这种预设框架会引入一些并不该成为回答主轴的内容，使得回答在形式上看似完整，实质上却存在明显局限。**也就是说，问题不是系统完全答错，而是系统在回答前就被 LLM 带入了一个过于熟悉的解释情景，最终答案被这一情景收窄。**

与之对照，我们的 chunk-first + group-oriented answer 在相同 query 上更倾向于按“政治参与 / 意识形态批判 / 形式创新 / 传播与公共性”等方面组织内容。这里的提升不来自“图更大”，而来自：**系统没有在第一步让 LLM 先替 query 发明一个艺术史场景，而是先从 dense 命中的真实 chunks 出发，再让图辅助扩展与组织。**

#### Example 2: Legal 数据集中，query 本应是 section-grounded 的，但系统被导向了“泛合规框架”

再看 `legal` 数据集中的 query：

> *What sections document the procedures for ensuring compliance with new laws?*

这一问题的真正难点，不在于“合规”这个主题本身，而在于回答需要：

- **回到 section / clause 级证据锚点；**
- **指出“哪些部分”在记录这些 procedures；**
- **同时保持对 procedure 的组织，而不是泛泛而谈 compliance 常识。**

然而，在 LightRAG 的 keyword cache 中，第一步 LLM 给出的往往是：

- high-level: `Compliance procedures`, `New laws`, `Documentation`
- low-level: `Regulatory guidelines`, `Legal requirements`, `Compliance framework`, `Policy updates`, `Implementation steps`

这些关键词不是错的，但它们已经把 query 改写成了一个“合规流程框架”问题，而不是一个“section-grounded procedure retrieval”问题。结果就是，系统更容易从图中进入一组关于 compliance、guideline、framework 的泛化语义区域，而不是优先回到能支撑 section-level answer 的具体证据块。

FG-RAG 在这一点上也具有相同的结构前提。其代码流程中，`Extractor.extract_query()` 会先让 LLM 从 query 中抽取 `query_entity`，随后 `Pipeline.match_initial_entity()` 再利用这些由 LLM 生成的 query entities 去匹配图中的实体描述索引。**因此，FG-RAG 的第一步同样是“让 LLM 先决定 query 应该从哪些实体进入图”。**

这一结构性假设最终体现在 answer 上：FG-RAG 对该 query 的回答会先承认具体 sections 并不明确，然后转入对 compliance clauses、responsibilities、monitoring procedures 的一般性说明。这样的答案并不离题，但明显更像“关于法律文书合规条款的概括”，而不是“根据当前数据集中哪些 section 在记录 procedures”这一更精确的 QFS 任务。

一个具有代表性的回答节选如下：

> “While specific Sections addressing this matter are not directly provided...”
>
> “Compliance with Laws... Responsibilities... Monitoring and Reporting Procedures...”
>
> “Penalties for Non-Compliance... Update Mechanisms...”

因此，这个例子很好地说明：**即便图中存在足够多与合规相关的内容，只要第一步把 query 解释成“泛合规框架”，最终 answer 就会滑向一般性主题综述，而失去 section-grounded 的回答能力。**

#### Example 3: Agriculture 数据集中，query 被改写成通用 advice 场景，导致回答“合理但局限”

第三个例子来自 `agriculture` 数据集：

> *How does the book suggest new beekeepers to maintain their commitment to beekeeping?*

LightRAG 的第一步关键词抽取通常会给出：

- high-level: `Beekeeping`, `Commitment`, `New beekeepers`
- low-level: `Book suggestions`, `Maintenance tips`, `Beekeeping practices`, `Motivation`, `Time management`

这些词同样不是错误的，但它们已经把 query 预设成一个典型的“新手如何坚持爱好 / 如何保持投入”的 advice scenario。于是系统后续更容易去召回和组织：

- motivation
- community support
- time management
- practical tips

这类非常符合一般常识的内容。

FG-RAG 在同题上的输出就清楚地体现了这一症状：回答围绕 `start small`、`set clear goals`、`engage with the beekeeping community` 展开。这样的答案当然有帮助，但它读起来更像一个“合理的自助建议总结”，而不一定体现出当前书中多源证据的真实分布与重点。

一个具有代表性的回答节选如下：

> “Start Small with a Nucleus Colony”
>
> “Setting Clear and Achievable Goals”
>
> “Engaging with the Beekeeping Community”
>
> “Continuous Learning and Resource Utilization”

这个例子的问题需要说得更明确。这个 query 真正要求的，不是一般意义上的“如何坚持一个新爱好”，而是：

- 书中究竟把“commitment”理解成什么；
- 书中通过哪些机制来维持这种 commitment；
- 这些机制如何围绕 beekeeping 这一实践活动被组织起来。

因此，一个更正确的 QFS answer 至少应当做到：

- 回到 **book-grounded** 的建议逻辑，而不是泛化的成长建议；
- 明确指出“长期投入而非短期 hobby”这一前提；
- 说明“第一年的重点是学习和建立现实预期，而不是立即追求产出”；
- 把 community、ongoing learning、seasonal planning 这些点组织成 **维持 commitment 的机制**，而不是平铺几个看似合理的 tips。

换言之，理想回答不只是说“start small / set goals / join community”，而要解释这些点为什么在当前语料里构成了 commitment 的支撑结构。FG-RAG 式回答的问题恰恰在于：它虽然合理，却明显更像模型基于先验知识生成出来的 advice template。

这类 case 非常重要，因为它说明：**LLM-first 的问题并不总是导致明显错误；更常见的情况是，它让系统产出“合理但局限”的答案。** 这种答案往往看起来顺畅、内容也并非虚构，但它更接近模型基于先验知识生成的 advice template，而不是围绕当前 corpus 的 QFS aggregation。

#### 小结：问题不只是入口不稳，而是 answer 被第一步联想“预收窄”

以上三个例子共同表明，现有方法的问题不能简单表述为“keyword drift”或“图入口摇摆”。更准确地说：

> **现有 LightRAG / FG-RAG 在第一步都把 query 交给 LLM 做情景化解释；而对 QFS 来说，这种解释往往先行设定了回答的框架。后续图检索与总结即便在图上覆盖到了相关信息，最终 answer 仍然容易被这一预设框架限制。**

因此，本文真正要解决的不是单纯的关键词不稳定，而是：

> **LLM-first query association 会诱导系统在 evidence grounding 之前先承诺一个看似合理但局限的回答情景，最终导致 QFS answer 表现出覆盖不足、锚点不足或过度常识化的局限性。**

...

针对以上现状，本文希望解决的问题可以概括为两点。

第一，现有方法通常把 query 的初始发散过程建立在 **LLM 对 query 的先验解读** 之上。这一步既不稳定，也无法保证与当前 corpus 中真实可达的知识形态一致。对于 QFS 而言，这意味着系统很可能在 evidence grounding 之前就把回答限制在某个想象出的情景之中。

第二，现有方法即便在图上检索到了较丰富的信息，也往往缺少一个真正面向 QFS 的中间组织层。很多系统最终仍然把零散的 entities、relations 和 sources 直接交给生成模型，导致 answer 要么过于平铺，要么过于泛化，难以稳定形成“按方面组织、同时保留证据锚点”的综述答案。

因此，本文真正面对的核心挑战不是“如何让 LLM 生成更好的关键词”，而是：

- 如何在**不依赖 LLM 先验联想**的前提下，从 query 命中的真实文本证据出发，在图上进行高质量发散？
- 如何把发散得到的图证据进一步组织成一种**支持 QFS 的中间知识结构**，使最终 LLM 生成更全面、更有条理、同时又不过度脱离原始语料？

基于这一目标，本文提出一种面向 QFS 的 **chunk-first associative graph retrieval** 思路。其整体流程可以简要概括为：

1. 先直接对 query 做 dense retrieve，得到一小组高相关 chunks，作为初始语义锚点；
2. 将这些命中的 chunks 投影到图上，利用 chunk 与实体、关系之间的多对多映射构造 root graph；
3. 在 root graph 基础上执行结构联想与语义联想，使发散过程显式地发生在图上，而不是发生在 LLM 对 query 的空想中；
4. 再将扩展后的图证据组织成面向 query 的 knowledge groups，作为最终 QFS 生成的中间层输入。

换言之，本文的方法主线不是：

> `query -> LLM keyword/entity -> graph entry -> answer`

而是：

> `query -> dense chunks -> chunk-rooted graph association -> knowledge groups -> answer`

后续具体的联想控制策略和组织细节仍然可以继续调整，但总体方向始终保持不变：**用真实命中的 chunk 来约束联想，用图来承担跨块扩展与关系组织，再把最终证据整理成适合 QFS 的知识中间层。**

### 1.3 本文的切入点

本文不再把问题表述为“如何修正 LLM 关键词漂移”，而是重新定义为：

> **如何从 query 命中的 chunks 出发，在图上进行受控联想，并把联想结果组织成适合 QFS 的知识结构。**

这意味着图不再只是“在线检索时沿边扩展的结构”，而是一个同时承担：

- chunk 间桥接，
- 关系模式补全，
- query-focused knowledge grouping

的知识组织中间层。

---

## 二、从预实验得到的新认识

### 2.1 早期假设：更强联想应该更适合 QFS

一开始的直觉是：QFS 本来就允许一定噪声和冗余，因此如果从 top-k root chunks 出发，在图上进行更积极的结构联想与语义联想，就应该更容易形成“多方面、可总结”的知识空间。

因此最初方法设计强调：

- chunk anchoring
- root graph projection
- 多轮 structural association
- 多轮 semantic-gain association
- knowledge groups

这个方向本身并没有错。后续实验也证明，在 `legal`、`art`、`news` 这样的数据集上，这条路线是有效的。

### 2.2 `mix` 暴露出的系统问题

但 `mix` 预实验很快暴露出一个更本质的问题：

> **联想强度不能是固定的。**

在 `mix` 上，图联想经常把 query 从 passage 内容带向：

- dataset / benchmark / model 元信息，
- 图中语义上相关但总结价值弱的“边角主题”，
- 与 root topic 有桥接关系、但不属于 query 核心主题的扩展支路。

这说明一个关键事实：

> QFS 并不天然欢迎“更强联想”。  
> QFS 欢迎的是“更高价值、更可组织的联想”。

换句话说，问题不只是联想到多少，而是联想是否值得。

### 2.3 四个数据集带来的进一步修正

在与 FG-RAG baseline 的比较中，我们观察到：

- `legal`：大幅领先；
- `art`：大幅领先；
- `news`：明显领先；
- `agriculture`：明显落后。

这组结果说明两件事。

第一，本文方法不是偶然在一个数据集上有效，而是在真正偏 QFS 的数据集上都有效。尤其是：

- `legal` 需要跨 section 的综合归纳；
- `art` 需要从分散案例中抽象主题与因果链；
- `news` 需要把不同报道角度组织成趋势综述。

这些场景都天然适合“chunk-first + relation-aware association + knowledge groups”。

第二，`agriculture` 的失败说明：

> 并不是所有 QFS query 都应该用同样强的图联想。

`agriculture` 中很多问题更接近：

- 实操建议总结；
- 具体失败原因归纳；
- 单主题经验型问答。

这类 query 需要的是**具体性、操作性、局部高精度证据**，而不是进一步扩展到泛生态、泛社区、泛背景层面。

### 2.4 新的核心判断

因此，本文最终的系统判断不是：

> “图联想越强越好”

而是：

> **图联想强度应由 query 当前的图状态自适应决定。**

这就是本文后续方法设计的关键转向：

> 从固定预算的联想式 GraphRAG，转向**自适应联想强度控制的 Associative GraphRAG**。

---

## 三、核心思想

### 3.1 总体思想

本文提出一种面向 QFS 的**自适应关系感知联想式图检索框架**。其核心思想是：

1. 先用 query 定位一小组高相关 chunks，作为语义锚点；
2. 将这些 chunks 投影到图上，构造 root graph；
3. 在 root graph 基础上执行结构联想与语义增益联想；
4. 不再固定联想预算，而是根据 query-time graph features 自适应调节联想强度；
5. 将最终扩展后的知识空间切分为 query-focused knowledge groups，交给 LLM 做 QFS。

### 3.2 图在本文中的角色

本文认为图在 QFS 中的角色至少包含三层：

1. **Bridge**
   图帮助不同 root chunks 之间建立桥接关系。

2. **Organizer**
   图中的边、边属性和共享 source support 帮助系统把零散知识组织成局部主题组。

3. **Controller**
   图特征本身反过来决定系统是否应该继续联想，以及联想到什么程度。

因此，图不仅是“被 traversed 的对象”，也是“控制联想力度的状态空间”。

### 3.3 Relation 的重新定位

在传统图检索中，edge 往往只是连接实体的结构边。本文进一步强调：

> **relation 不只是连接器，而是知识组织的基本单元。**

边上的：

- `keywords`
- `description`
- `weight`
- `source_id`

共同决定了它是否：

- 与 query 有关；
- 被 root chunks 支持；
- 能补充当前知识集合中缺失的关系维度；
- 值得被纳入最终知识组织。

---

## 四、方法总览

### 4.1 整体流程

本文方法由六个阶段组成：

1. **Query-to-Chunk Anchoring**
2. **Root Graph Projection**
3. **Relation-Aware Root Scoring**
4. **Adaptive Association Control**
5. **Associative Knowledge Organization**
6. **LLM-based QFS Generation**

可以概括为：

> `query -> chunk anchors -> root graph -> adaptive association -> knowledge groups -> summary`

---

## 五、Query-to-Chunk Anchoring

### 5.1 目标

给定 query，首先在原始 chunk 集合中检索 top-k 个最相关文本片段，作为后续图联想的语义锚点。

### 5.2 实现选择

本文采用 **dense + lexical hybrid retrieval**。

具体而言：

- 用现有 chunk vector DB 计算 dense similarity；
- 用 lexical/BM25 信号补足 query 的显式词面约束；
- 将两者融合得到候选 chunk；
- 再用轻量 rerank 对 root chunks 进行二次排序；
- 对明显偏 dataset / benchmark / model metadata 的 chunks 施加轻量噪声惩罚。

### 5.3 Root Chunk Rerank

对候选 chunk $c$，我们综合以下因素：

- dense relevance
- lexical relevance
- chunk content 与 query 的词面相似度
- chunk 在图上的产出能力（graph yield）

可写为：

$$
S_{chunk}(c|q)=
\alpha_1 Dense(c,q)+
\alpha_2 Lex(c,q)+
\alpha_3 Rel(c,q)+
\alpha_4 Yield(c)
$$

其中：

- $Dense(c,q)$：dense score；
- $Lex(c,q)$：lexical score；
- $Rel(c,q)$：query 与 chunk 内容的直接相关性；
- $Yield(c)$：该 chunk 在图上可投影出的节点/边规模。

在 `mix` 预实验中，我们观察到：若不对技术元信息做轻量抑制，root chunks 容易偏向 dataset / benchmark / model 说明段落，而不是 query 真正关心的 passage 内容。因此，在不显式做 dataset-specific 规则的前提下，本文在 root rerank 中加入弱噪声惩罚：

$$
S'_{chunk}(c|q)=S_{chunk}(c|q)-\tau \cdot MetaNoise(c,q)
$$

其中 $MetaNoise(c,q)$ 用于衡量 chunk 是否过于偏向技术元信息，而 query 本身并未显式要求方法或数据集描述。

这样做的动机是避免 root chunks 只由文本相似度控制，而忽视其是否真能在图上提供有效组织线索。

---

## 六、Root Graph Projection

### 6.1 定义

设 root chunks 为：

$$
\mathcal{C}_q=\{c_1,\dots,c_k\}
$$

若节点 $v$ 的 `source_id` 与 $\mathcal{C}_q$ 有交集，则 $v$ 属于 root node：

$$
V_q^{root}=\{v \mid source(v)\cap \mathcal{C}_q\neq\emptyset\}
$$

若边 $e$ 的 `source_id` 与 $\mathcal{C}_q$ 有交集，则 $e$ 属于 root edge：

$$
E_q^{root}=\{e \mid source(e)\cap \mathcal{C}_q\neq\emptyset\}
$$

### 6.2 作用

这一阶段把文本级 anchor 投影到图结构中，得到 query 的局部图入口。

与“query 直接命中图节点”的路线相比，chunk anchoring 有三个优势：

1. 入口更稳定；
2. 保留原始文本 grounding；
3. 为后续知识组织提供 root evidence layer。

---

## 七、Relation-Aware Root Scoring

### 7.1 Node Scoring

节点在本文中主要承担“主题锚点”作用。对 root node $v$，定义：

$$
S_{node}(v|q)=
\beta_1 Rel(v,q)+
\beta_2 Support(v,\mathcal{C}_q)+
\beta_3 Align(v,\mathcal{C}_q)
$$

其中：

- $Rel(v,q)$：query 与节点名称、类型、描述的相关性；
- $Support(v,\mathcal{C}_q)$：有多少 root chunks 支持该节点；
- $Align(v,\mathcal{C}_q)$：支持该节点的 root chunks 与 query 的平均匹配强度。

节点得分的作用不是构造最终排序，而是筛出后续联想的 anchor nodes。

### 7.2 Edge Scoring

边在本文中是核心组织单元。对 root edge $e$，定义：

$$
S_{edge}(e|q,\mathcal{K})=
\lambda_1 Rel(e,q)+
\lambda_2 Support(e,\mathcal{C}_q)+
\lambda_3 Align(e,\mathcal{C}_q)+
\lambda_4 IG(e|\mathcal{K})
$$

其中：

- $Rel(e,q)$：query 与 edge keywords / description 的相关性；
- $Support(e,\mathcal{C}_q)$：多少 root chunks 支持该边；
- $Align(e,\mathcal{C}_q)$：这些 root chunks 对 query 的平均匹配强度；
- $IG(e|\mathcal{K})$：该边相对当前知识集合 $\mathcal{K}$ 的关系信息增益。

### 7.3 Relation Information Gain

若只依赖相关性，系统会反复加入大量关系模式相近的边。为此，引入关系类别熵：

$$
H(\mathcal{R}_{\mathcal{K}})= -\sum_{r\in \mathcal{R}} p(r)\log p(r)
$$

定义边 $e$ 的增益为：

$$
IG(e|\mathcal{K})=H(\mathcal{R}_{\mathcal{K}\cup\{e\}})-H(\mathcal{R}_{\mathcal{K}})
$$

这使系统更倾向于加入能够补充新关系维度的边，而不是简单重复已有模式。

---

## 八、Associative Knowledge Organization

### 8.1 两类联想

本文中的联想由两条并列机制组成：

1. **结构联想（Structural Association）**
2. **语义增益联想（Semantic-Gain Association）**

二者既可以交替执行，也可以被自适应控制模块削弱或提前停止。

### 8.2 结构联想

结构联想的目标是寻找能桥接不同 root 区域的路径。设最大跳数为 $L$，对候选路径 $\pi$ 定义：

$$
B(\pi)=
\mu_1 RootReach(\pi)+
\mu_2 SupportSpan(\pi)+
\mu_3 RelPath(\pi)-
\mu_4 Len(\pi)
$$

其中：

- $RootReach(\pi)$：路径是否连接不同 root components；
- $SupportSpan(\pi)$：路径涉及的 source chunk 覆盖范围；
- $RelPath(\pi)$：路径上边对 query 的平均关系相关性；
- $Len(\pi)$：长度惩罚。

每轮只保留桥接分数最高的前 $K_s$ 条路径。

### 8.3 语义增益联想

语义增益联想从当前节点、边及其支撑 chunks 出发，寻找值得纳入的新节点和新边。

对候选边 $e$，定义：

$$
G(e|q,\mathcal{K})=
\gamma_1 Rel(e,q)+
\gamma_2 Support(e,\mathcal{C}_q)+
\gamma_3 Align(e,\mathcal{C}_q)+
\gamma_4 IG(e|\mathcal{K})
$$

对候选节点 $v$，定义：

$$
G(v|q,\mathcal{K})=
\delta_1 Rel(v,q)+
\delta_2 Support(v,\mathcal{C}_q)+
\delta_3 Align(v,\mathcal{C}_q)+
\delta_4 Bridge(v,\mathcal{K})
$$

其中 $Bridge(v,\mathcal{K})$ 表示该节点与当前知识空间之间的桥接强度。

### 8.4 多轮更新

设初始知识空间为 $\mathcal{K}^{(0)}=(V_q^{root},E_q^{root})$，则第 $t$ 轮更新为：

$$
\mathcal{K}^{(t+1)}=
\mathcal{K}^{(t)} \cup
\mathcal{P}^{(t)} \cup
\mathcal{G}^{(t)}
$$

其中：

- $\mathcal{P}^{(t)}$：第 $t$ 轮结构联想结果；
- $\mathcal{G}^{(t)}$：第 $t$ 轮语义增益联想结果。

传统做法会固定：

- 轮数
- path budget
- semantic edge budget
- semantic node budget

本文后面将说明，正是这一点需要改为自适应控制。

---

## 九、自适应联想强度控制

### 9.1 为什么必须自适应

从 `mix` 与多数据集实验可以看出：

- 对 `legal / art / news`，较强联想通常有效；
- 对 `agriculture`，相同联想强度会导致过度扩展和回答泛化。

因此不能按数据集写死 preset，也不能按所有 query 统一预算。

我们需要一个统一原则：

> **在 query-time，根据 query form、检索广度和当前 root graph 结构，动态决定联想应该扩到什么程度。**

### 9.2 Query-Time Graph Features

早期版本曾尝试用：

- root support dispersion
- top-k root chunk dispersion

来直接估计 query 是否更“综述化”。但实验表明，这两类指标在当前图和检索流程上都容易饱和：一旦 top-k root chunks 都被 rerank 到较高相关区间，它们的分布往往天然较平，难以真正区分“聚焦型 query”和“概括型 query”。因此，本文最终将 style 判别改写为三个层次：

1. **Query Form Classifier**
2. **Retrieval Breadth Estimator**
3. **Graph Structure Corrector**

也就是说，style 不再由内容词决定，而是由：

- 问句形式，
- 检索是否呈现“断崖式聚焦”或“多候选并行”，
- root graph 是稠密还是破碎

共同决定。

#### （1）Query Form Classifier

这一层借鉴 question classification 的思路，但不做开放域语义类别预测，而只预测与联想强度相关的**问句功能类型**。与经典 question classification 工作类似，这里重点使用问句头词、句法模式和问题功能，而不是内容主题词。对应地，它在思想上接近 Li and Roth 一类的问句分类工作：不是预测“问的是哪个领域实体”，而是先识别“这是解释型问题、比较型问题、还是定位型问题”。

可将 query form 粗分为三类：

- **Synthesis-oriented**
  - 典型形式：`why`, `how did`, `in what ways`, `what role`, `how do ... compare`, `what patterns`
  - 特征：要求原因、机制、比较、趋势、主题抽象
- **Concrete-oriented**
  - 典型形式：`which`, `when`, `where`, `who`, `what are the steps`, `what specific`, `list`
  - 特征：要求列举、定位、步骤、明确对象
- **Balanced**
  - 既非强综述，也非强定位
  - 往往是“围绕一个主题做有限归纳”的问题

这一层的输出不是最终联想强度，而是给出一个初始 style prior：

$$
z_q^{form}\in\{\text{synthesis}, \text{balanced}, \text{concrete}\}
$$

这里的关键约束是：

- 只使用通用问句功能词和句法模式；
- 不使用 `agriculture`、`art`、`model`、`dataset` 之类内容词；
- 假设系统第一次见到 query，也能完成 style 估计。

#### （2）Retrieval Breadth Estimator

这一层用于估计 query 在当前语料中的“展开宽度”。它比单看 query 文本更可靠，因为同一个句法形式在不同语料中可能呈现不同检索形态。

这里建议重点使用三类 retrieval 特征。

**a. Candidate-Stage Retrieval Cliff**

设初始 candidate pool 的检索分数为：

$$
s_1 \ge s_2 \ge \dots \ge s_n
$$

这里的关键是：

> **cliff 必须在 root rerank 之前统计。**

原因是 rerank 之后保留下来的本来就是“最相关的少数 chunk”，它们的分数天然会被压缩到相近区间，再去估计 cliff 往往看不出 query 是聚焦还是概括。相反，真正有判别力的是 pre-rerank candidate pool 的前段分数形态。

因此，本文建议至少在 top-15，必要时在 top-20 或 top-30 的 candidate 范围内统计 cliff，而不是只看 top-2 或最终 top-5 root chunks。

定义 candidate-stage cliff 为：

$$
\kappa_q^{cand} = \frac{s_1-\frac{1}{m-1}\sum_{i=2}^{m}s_i}{\max(|s_1|,\epsilon)}, \quad m\in\{15,20,30\}
$$

直觉：

- $\kappa_q^{cand}$ 高：top-1 或极少数 chunks 显著领先，query 更聚焦；
- $\kappa_q^{cand}$ 低：多个候选同时接近，query 更可能需要综述式覆盖。

这一路线与 query clarity / query performance prediction 的思想一致：检索分布越集中，query 越可能具有更明确的局部意图；分布越平，query 越可能涉及多个方面。也就是说，我们不是直接复用传统 IR 的 query performance prediction 目标，而是借用它的核心判断逻辑，把“query 是否清晰、是否宽泛”转译成“本次是否值得做更强联想”。

**b. Candidate Retrieval Dispersion**

不要只看最终 top-5 root chunks，而要看更前面的 candidate set，例如 top-10 或 top-12 候选 chunks。设其 softmax 归一化权重为：

$$
p_i = \frac{\exp(\tau s_i)}{\sum_j \exp(\tau s_j)}
$$

定义分布熵：

$$
H_q^{cand} = -\sum_i p_i \log p_i
$$

再归一化为：

$$
\sigma_q^{cand} = \frac{H_q^{cand}}{\log n}
$$

并给出有效候选数：

$$
N_q^{eff} = \frac{1}{\sum_i p_i^2}
$$

直觉：

- $\sigma_q^{cand}$ 高、$N_q^{eff}$ 大：query 需要更宽覆盖；
- $\sigma_q^{cand}$ 低、$N_q^{eff}$ 小：query 更像集中命中。

不过实验也表明，这类分布指标如果放在过晚的 rerank 阶段容易再次饱和。因此它更适合作为**辅助信号**，而不是唯一主指标；真正的一阶控制信号仍然应当是 pre-rerank candidate-stage retrieval cliff。

**c. Optional Clarity-Like Signal**

进一步可以引入简化版 clarity score，用 top retrieved chunks 的词分布与语料背景词分布之间的 KL 散度衡量 query clarity。若 clarity 高，说明 query 与一个较稳定的局部主题对齐；若 clarity 低，则更像 broad / faceted query。

在工程上，若不希望增加额外代价，这一项可以先作为后续扩展，而非第一版必选项。

综合这一层，得到 retrieval breadth 判别：

$$
z_q^{breadth}\in\{\text{focused}, \text{middle}, \text{broad}\}
$$

#### （3）Graph Structure Corrector

前两层只描述 query 与检索形态，还需要第三层判断：

> **当前 root graph 是否真的值得继续在图上发散。**

本文建议至少使用以下两类结构特征。

**a. Root Density**

$$
\rho_q=\frac{|E_q^{root}|}{\max(|V_q^{root}|,1)}
$$

直觉：

- $\rho_q$ 高：root graph 已较稠密，往往已有较完整局部结构；
- $\rho_q$ 低：root graph 更碎，需要一定桥接。

**b. Root Fragmentation**

设 root components 数为 $M_q$，则定义：

$$
\phi_q=\frac{M_q}{\max(|V_q^{root}|,1)}
$$

或直接使用 component count / largest component ratio。

直觉：

- $\phi_q$ 高：root graph 更碎，更需要结构桥接；
- 但若同时 $\rho_q$ 很低，说明图本身缺乏可靠结构支撑，这时应优先做有限桥接，而不是无约束语义扩张。

### 9.3 从单一联想强度到 Expansion Profile Selection

基于上述分析，本文不再主张用单个连续 $\alpha_q$ 去统管所有预算。原因有二：

1. `root_density`、`root_fragmentation`、`retrieval_cliff` 都是 **query 在当前图与检索系统中的反应**，而不是 query 语义意图本身；
2. 实验表明，不同 query 真正偏好的并不是“更强”或“更弱”的单轴扩展，而是**不同类型的扩展策略**。

因此，本文将自适应控制重写为：

> **Query-form prior + evidence-shape prior -> expansion profile selection**

也就是说，系统不再先估计一个“联想强度分数”，而是先判断：

- 用户更像希望得到哪种回答组织方式；
- 当前 root evidence 在图上呈现出怎样的形状；
- 然后在几种可解释的扩展 profile 中选择一种。

#### （1）为什么需要两类先验

本文将先验分为两类。

**a. Query-form prior**

query wording 中的问法 cue 并不依赖领域词本身，而主要体现用户希望得到的回答组织方式。例如：

- `what are the main / in what ways / how did ... influence / what patterns`
  - 倾向于综合、比较、机制解释；
- `what are the initial / what steps / which resources / what specific examples`
  - 倾向于枚举、操作、具体归纳。

这一层提供的是：

> **answer-form prior**

即用户更像要：

- `precision-oriented`
- `bridge-oriented`
- `coverage-oriented`

中的哪一类回答形态。

**b. Evidence-shape prior**

图侧先验并不直接编码 query intent，它们编码的是：

> **当前 query 在自动图中能触达的证据形状。**

其中：

- `root_density`
  - 表示 root evidence 的局部凝聚程度；
- `root_fragmentation`
  - 表示 root evidence 是否分散在多个相互松散的局部块上；
- `retrieval_cliff`
  - 表示初始文本检索是否已形成明显强锚点。

因此，evidence-shape prior 回答的是：

> 当前证据更像是一个已经成形的局部证据块，还是一组需要进一步桥接或补面的分散片段。

#### （2）为什么不能只靠图特征

这一点需要明确承认：

> `root_density`、`root_fragmentation`、`retrieval_cliff` 不能单独决定 query 是“概括型”还是“具体型”。

因为它们描述的是 query 在当前图上的反应，而不是 query 语言意图本身。  
同一主题可以用更技术细节的问法提出，也可以用更综述性的问法提出；这两种问法即使命中了相近的实体集合，也不意味着最终应该采用同样的扩展策略。

因此，本文的控制逻辑不是：

> graph priors -> directly infer query intent

而是：

> query-form prior 给出用户期望的回答组织方式，  
> evidence-shape prior 给出当前证据距离这种组织方式还有多远。

这也正是为什么本文最终采用的是 **两类先验联合控制**。

### 9.4 三类 Expansion Profiles

基于大规模预实验，本文最终将扩展策略划分为三类 profile，而不是一个连续的强弱轴。

#### （1）Precision Profile

适用于：

- root evidence 已较凝聚；
- query 更偏具体归纳、操作建议、成本/步骤/失败原因等；
- 系统应避免过多外围联想破坏收束性。

其目标是：

> **保留高精度核心证据，减少噪声扩散。**

一个可用的参数实例为：

- `top_root_nodes = 10`
- `top_root_edges = 12`
- `path_budget = 8`
- `semantic_edge_budget = 14`
- `semantic_node_budget = 8`
- `semantic_edge_min_score = 0.05`
- `semantic_node_min_score = 0.05`

#### （2）Bridge Profile

适用于：

- root evidence 分散在多个局部块；
- query 需要系统把多个子主题、多个机制或多个局部证据组织起来；
- 当前问题更缺连接，而不一定缺更多外围内容。

其目标是：

> **优先补足跨块桥接，增强主题组织能力，而不是无约束扩厚语义内容。**

一个可用的参数实例为：

- `top_root_nodes = 14`
- `top_root_edges = 18`
- `path_budget = 20`
- `semantic_edge_budget = 16`
- `semantic_node_budget = 10`
- `semantic_edge_min_score = 0.03`
- `semantic_node_min_score = 0.035`

#### （3）Coverage Profile

适用于：

- root evidence 本身较碎；
- query 期待更宽的主题面、趋势总结、跨方面比较；
- root 阶段尚不足以支撑高质量综述，需要更厚的 evidence horizon。

其目标是：

> **扩大支持总结的证据覆盖面与视角多样性。**

一个可用的参数实例为：

- `top_root_nodes = 14`
- `top_root_edges = 20`
- `path_budget = 12`
- `semantic_edge_budget = 28`
- `semantic_node_budget = 18`
- `semantic_edge_min_score = 0.02`
- `semantic_node_min_score = 0.02`

### 9.5 Profile 选择逻辑

本文的核心控制器不再输出 `association_strength`，而是输出：

$$
\pi_q \in \{\text{precision},\text{bridge},\text{coverage}\}
$$

其中：

$$
\pi_q = g\left(z_q^{form}, \rho_q, \phi_q, \kappa_q\right)
$$

这里：

- $z_q^{form}$：query-form prior；
- $\rho_q$：root density；
- $\phi_q$：root fragmentation；
- $\kappa_q$：candidate-stage retrieval cliff。

#### （1）逻辑解释

这套划分的主观逻辑是：

- **如果 query wording 更像枚举、操作、具体归纳，而 root evidence 已较凝聚**
  - 说明系统更应该“守住核心证据”
  - 选择 `precision`

- **如果 query wording 更像综述或机制解释，而 root evidence 分散在多个局部块**
  - 说明系统更需要把这些块联起来
  - 选择 `bridge`

- **如果 query wording 明显需要宽覆盖，且 root evidence 同时稀疏且破碎**
  - 说明当前证据面还不足以支撑高质量总结
  - 选择 `coverage`

换句话说：

- `precision` 解决的是“缺收束”
- `bridge` 解决的是“缺连接”
- `coverage` 解决的是“缺内容面”

#### （2）一版可执行规则

基于当前实验，本文给出一版可实现的 profile selector：

- 若 `root_density >= 0.55` 且 `root_fragmentation <= 0.72`
  - 优先 `precision`

- 若 `root_density <= 0.43` 且 `root_fragmentation >= 0.84`
  - 优先 `coverage`

- 若 `0.48 <= root_density <= 0.55` 且 `root_fragmentation >= 0.74`
  - 优先 `bridge`

- 其余情况
  - 先由 `query-form prior` 在 `precision / bridge / coverage` 之间给出初选
  - 再由 `retrieval_cliff` 做收缩修正：
    - `retrieval_cliff` 高，向 `precision` 收缩；
    - `retrieval_cliff` 低，不额外压制 `bridge/coverage`。

这一规则不是声称“精确恢复 query 真实语义类别”，而是声称：

> **基于 query-form prior 与 evidence-shape prior，可以较稳定地选择更合适的扩展策略。**

### 9.6 为什么 profile selection 比单一 alpha 更合理

实验显示，单一 `alpha` 有三个问题：

1. 它会把“是否扩展内容面”和“是否增加桥接连接”混成一件事；
2. 它默认所有预算可沿同一轴同步增减，但实际最优控制往往不是这样；
3. 它很难解释“为什么某些 query 需要更多连接而不是更多内容”。

而 profile selection 的优势在于：

- **解释性更强**
  - 直接回答“为什么选这个扩展策略”；
- **控制对象更清楚**
  - `precision`、`bridge`、`coverage` 各自对应不同的系统缺口；
- **更符合实验事实**
  - `agriculture` 明显怕 `coverage`
  - `art` 明显吃 `coverage`
  - `bridge` 在跨块组织型问题上又是单独有价值的一类

因此，本文将自适应控制正式改写为：

> **prior-guided expansion profile selection**

而不是：

> **single-axis association strength tuning**

### 9.7 在线早停

无论选择哪类 profile，都保留统一的在线 early stopping：

若第 $t$ 轮满足以下任一条件，则停止：

1. 新增结构路径数很少；
2. 新增语义边平均分数低于阈值；
3. 新增边的信息增益均值低于阈值；
4. 新增 source chunks 很少；
5. 当前知识空间已接近 source word budget 上限。

因此，最终联想行为由两层共同决定：

- **offline-selected profile**
  - 决定本轮允许怎样的扩展倾向；
- **online marginal gain**
  - 决定扩展是否还值得继续。

共同决定。

---

## 十、Knowledge Group 构造

### 10.1 基本思想

完成联想扩展后，我们不直接把所有 nodes/edges 平铺给 LLM，而是先将扩展后的知识空间切分为若干 **query-focused knowledge groups**。

每个 group 由以下元素组成：

- 核心节点
- 核心关系
- supporting chunks
- 必要桥接路径
- relation themes
- group score

### 10.2 Group Scoring

对 group $g$，定义：

$$
S_{group}(g|q)=
\omega_1 QueryRel(g,q)+
\omega_2 SupportSpan(g)+
\omega_3 RelationCoverage(g)+
\omega_4 RootDensity(g)+
\omega_5 StructureDensity(g)-
\omega_6 NoisePenalty(g)
$$

其中：

- $QueryRel(g,q)$：group 与 query 的整体相关性；
- $SupportSpan(g)$：group 覆盖的不同 source chunks；
- $RelationCoverage(g)$：group 中关系主题的覆盖度；
- $RootDensity(g)$：group 内 root evidence 占比；
- $StructureDensity(g)$：group 内结构桥接占比；
- $NoisePenalty(g)$：技术元信息、低支持 peripheral items、与 query 主体偏离的扩展支路等惩罚。

这里的 $NoisePenalty(g)$ 正是从 `mix` 经验中抽象出的统一控制项。它不是为某个数据集写规则，而是用于压制那些“图上能联到、但 QFS 不值得写”的知识组。

### 10.3 为什么 knowledge groups 是必要的

knowledge group 是本文区别于“图扩展后直接拼上下文”的关键中间层。

它的作用是把：

- root evidence
- bridge relations
- semantic expansion

重新组织成局部可总结单元，使 LLM 不必面对未经整理的大图，而是面对若干局部主题组。

这比传统 node/path 输出更贴近 QFS。

---

## 十一、Context Assembly 与生成策略

### 11.1 给 LLM 的不是大图，而是知识包

最终上下文由五部分组成：

1. Root Chunks
2. Focused Entities
3. Focused Relations
4. Knowledge Groups
5. Selected Sources

这里的原则是：

> **不要把分析态的大图塞给 LLM，而要给它压缩后的、可阅读的 knowledge package。**

### 11.2 从 `mix` 学到的生成侧修正

`mix` 预实验表明，若不控制生成提示，模型容易把：

- dataset 元信息
- NLP 方法描述
- benchmark 叙述

当成 query 本身的回答内容。

因此最终生成提示需要显式强调：

- 默认优先回答 passage / event / theme / character 内容；
- 除非 query 明确问方法，否则不要围绕 dataset / model / benchmark 展开；
- 对历史、比较、趋势类问题，要突出结构化比较而非散点罗列。

这一步不是数据集调参，而是 QFS 任务的必要约束。

---

## 十二、方法与实验结果之间的统一解释

### 12.1 为什么 `legal / art / news` 表现强

这些数据集都更接近真正的 synthesis-heavy QFS：

- `legal`：需要跨 section 聚合；
- `art`：需要抽象模式与因果链；
- `news`：需要多角度组织趋势与舆论变化。

在这些场景中：

- 结构联想能桥接分散 section / article / case；
- 语义增益联想能补足关系维度；
- knowledge groups 能把多源证据组织成可总结单元。

因此本文方法天然占优。

### 12.2 为什么 `agriculture` 表现弱

`agriculture` 并不是纯粹的综述型 QFS，很多 query 更像：

- 具体原因归纳；
- 初学者建议；
- 实操型总结。

在这种场景中，联想太强会出现：

- 泛化背景过多；
- 具体操作细节被稀释；
- 综述能力有余，具体性不足。

这一现象恰好说明：

> 固定联想预算是不合理的，自适应联想强度控制是必要的。

### 12.3 因此，本文的方法升级不是 dataset-specific tuning

不是为 `agriculture` 单独做 preset，也不是为 `legal` 单独加规则。

而是提出一个统一原则：

> **由 query-time graph state 决定联想强度。**

如果这一点成立，那么方法就可以在统一框架下同时兼顾：

- synthesis-heavy QFS
- concrete/practical QFS

这比“按数据集写多套 preset”更有方法性，也更适合写论文。

---

## 十三、论文中的方法主张

本文最终可以归纳为三个核心主张：

### Claim 1

QFS 中图结构的价值不仅在于检索更多实体和路径，更在于将分散知识组织成可总结的中间层。

### Claim 2

固定强度的图联想并不适合所有 QFS query。真正有效的做法不是用单个强度系数统一调节所有预算，而是结合 query-form prior 与 query-time evidence shape，在 `precision / bridge / coverage` 三类扩展 profile 之间自适应选择。

### Claim 3

以 chunk anchoring 为 grounding、以 relation-aware association 为组织手段、以 profile-based adaptive control 为约束，可以在统一框架下兼顾跨文档综合能力、联想噪声控制以及最终 evidence horizon 的可解释调节。

---

## 十四、可直接写进论文的方法概括

如果用一段话概括本文方法，可以写成：

> We propose an adaptive associative graph retrieval framework for query-focused summarization.  
> Instead of directly relying on query-to-graph entry matching, our method first anchors the query on top-ranked chunks, projects them onto the graph as a root evidence layer, and then performs relation-aware structural and semantic-gain association to organize query-relevant knowledge into knowledge groups.  
> Crucially, the system does not rely on a single association-strength scalar. Instead, it combines query-form priors with query-time evidence-shape features such as root density, fragmentation, and candidate-stage retrieval cliff to select among precision, bridge, and coverage expansion profiles, thereby adapting the evidence horizon to the needs of each query.

对应中文就是：

> 本文提出一种面向 QFS 的自适应联想式图检索框架。该方法不再主要依赖 query 对图节点的直接命中，而是先将 query 锚定到高相关 chunks，再投影到图上构造 root evidence layer，随后通过关系感知的结构联想与语义增益联想，将多源知识组织为 query-focused knowledge groups。最关键的是，系统不再用单个联想强度去控制所有预算，而是联合 query-form prior 与 query-time evidence shape（如 root density、fragmentation 与 candidate-stage retrieval cliff），在 `precision / bridge / coverage` 三类扩展 profile 中自适应选择，从而为不同类型的 QFS query 提供更合适的 evidence horizon。

---

## 十五、实现落点

当前原型已经对应到以下模块：

- retrieval
  - dense + lexical hybrid chunk retrieval
  - root chunk rerank
- projection
  - source_id based root graph projection
- relation-aware scoring
  - root node / edge scoring
  - relation information gain
- association
  - structural association
  - semantic-gain association
- organization
  - knowledge groups
  - compact prompt context
- generation
  - QFS-oriented answer prompt
- evaluation
  - pairwise win-rate vs FG-RAG

下一步实现重点不再是补更多 heuristic，而是补上：

1. query-time adaptive association controller
2. section / clause / heading anchors for legal-style corpora
3. 更严格的 early stopping 与 noise gating

这三点基本就是论文里的下一阶段方法增量。
