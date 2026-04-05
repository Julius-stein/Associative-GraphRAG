# 知识组织研究备忘录

本文档不是最终论文写法，也不是当前实现说明，而是基于近期实验结果，对“知识组织阶段下一步该怎么研究”做的内部备忘录。

对应背景：

- 联想阶段已经被显著增强，尤其加入了：
  - 图路径桥接
  - chunk 邻接桥接
  - 图覆盖扩张
  - chunk 邻接覆盖扩张
- 但在 `runs_expand` 这一轮中，整体效果明显退步，说明：
  - 问题已不主要在 recall
  - 问题已经转移到 organization

---

## 1. 当前结论：联想增强后，组织成为新瓶颈

`runs_expand` 相比此前较稳的 `runs_derector3`，有一个非常清楚的现象：

- `final subgraph` 规模暴涨
- 但最终送给 LLM 的 source chunk 数量和词数几乎没变
- facet group 数也几乎没变

这意味着：

> 系统已经拿到了更多信息，但 organization 没有把这些新增信息有效地压缩、分配、转化为更好的 answer evidence units。

换句话说，当前问题不是：

- “没有召回到”

而是：

- “召回到了，但没有组织好”

---

## 2. 当前 organization 的根本局限

当前组织层已经做了几件正确的事：

- 引入了 `section / mechanism / comparison / theme` 四类 contract
- 不再把 connected component 直接当 group
- 允许 root/theme/bridge 多类 region 进入 organizer

但它仍有几个根本局限。

### 2.1 Group 选择仍然更像“挑几组”，而不是“分配信息”

现在的流程更像：

1. 从 final subgraph 里收集一些 root / bridge / theme regions
2. 按 contract 组织成若干 group
3. 再从中选前几个

这本质上仍然是：

> **region packing + group selection**

而不是：

> **global evidence allocation**

也就是说，现在系统更像是在“挑哪些组保留”，而不是“整个大图中的信息应该如何分配到少量 answer units 上”。

### 2.2 新增 evidence 没有被显式分配给 facet

联想增强以后，尤其 chunk-side association 会引入很多新的 nodes / edges / chunks。

但当前 organizer 没有一个显式机制来回答：

- 这些新增证据应该归到哪个 facet？
- 它们是补充已有 facet，还是形成新 facet？
- 如果多个 facet 都能吃到它们，应该优先给谁？

所以现在很多新增 evidence 最终：

- 要么根本没进入最终 source package
- 要么被粗暴地揉进已有 group
- 要么在 group 间造成主题重叠

此外，当前 organizer 还没有充分利用一个非常关键的信息源：

> **结构联想的生长过程本身。**

也就是说，不只是“最后有哪些 nodes / edges / chunks 被纳入”，还包括：

- 它们是沿哪些 graph bridge path 生长出来的
- 它们是通过哪些 chunk bridge 接上当前 frontier 的
- 哪些 evidence 是 root 周围直接生长出来的
- 哪些 evidence 是通过结构桥接才出现的

这些 growth traces 本身就很适合成为 organization 的重要信号：

- 对 `mechanism-grounded` 而言，它们可以提示因果或过程链是如何展开的；
- 对 `comparison-grounded` 而言，它们可以提示两侧 evidence 是如何被桥接到同一问题上的；
- 对 `theme-grounded` 而言，它们可以帮助区分“核心 root 周边主题”和“后续桥接引入的补充主题”；
- 对 `section-grounded` 而言，它们可以帮助判断哪些 section band 是局部连续生长出来的，而不是仅靠最终 overlap 拼出来的。

### 2.3 contract 已经存在，但 contract-specific allocation 还不存在

当前我们已经做了 contract 路由：

- `section-grounded`
- `mechanism-grounded`
- `comparison-grounded`
- `theme-grounded`

这一步已经证明是有价值的。

但现在 contract 更多是在决定：

- 用哪个 organizer

而不是决定：

- evidence 在该 contract 下应当如何被组织

也就是说，当前缺的是：

> **contract-specific evidence allocation policy**

### 2.4 source budget 压缩仍是最后一步的被动截断

现在最终 source chunk 选择仍然偏后处理。

这导致：

- 前面 final subgraph 再大
- 最后还是在固定 budget 下截成差不多大小的一包文本

如果 organization 没有在此之前把证据变成少量高质量 answer units，那么最终压缩一定会丢信息。

所以：

> **organization 真正要做的是先压出结构，再交给 source selection。**

而不是先有大图，再在最后临时截文本。

---

## 3. 当前我们真正要解决的两个问题

这两个问题应继续作为组织层设计的目标。

### 3.1 Focus Mismatch

query 要求的组织方式，和最终 answer 实际采用的组织方式不一致。

典型表现：

- 要 section-grounded，结果给 thematic overview
- 要 mechanism-grounded，结果给 generic advice
- 要 comparison-grounded，结果给 broad summary

这不是生成质量问题，而是 retrieval-to-organization pipeline 的结构性偏差。

### 3.2 Evidence Anchoring Weakness

答案表面合理，但没有稳定回到 corpus 中能支撑该组织方式的证据块。

典型表现：

- 回答流畅
- 信息也不完全错
- 但证据支撑不稳定
- 或证据和组织方式之间并没有被明确绑定

这同样不是纯生成问题，而是 organization 没有把 evidence 组织成真正可回答的单元。

---

## 4. 下一阶段 organization 应该长什么样

我目前认为，下一版 organization 不该继续只做“group 生成”，而应该升级为：

> **从 final subgraph 中，把信息分配给少量 contract-aware answer units。**

这里的关键词不是：

- group

而是：

- allocation
- assignment
- compression

也就是说，要从“大图里挑几组”转成“把大图压成几个可回答的单元”。

---

## 5. 可以考虑的几条方案

下面是几条比较成体系的研究路线。

### 方案 A：Facet-first Evidence Allocation

这是我目前最推荐的一条。

核心思想：

1. 先根据 contract 确定“本题理论上允许的 answer unit 类型”
2. 再把 final subgraph 中的 evidence 分配给这些 unit
3. 最后每个 unit 只保留最能支撑它的 evidence

例如：

- `theme-grounded`
  - 若干并列 aspect units
- `mechanism-grounded`
  - 若干 cause / process / effect units
- `comparison-grounded`
  - 若干 side / contrast-dimension units
- `section-grounded`
  - 若干 section band units

优点：

- organization 从“挑组”变成“分配”
- 更容易解释为什么某条证据属于某个 unit
- 更适合大图压缩

难点：

- 需要先定义 unit 的类型
- 需要明确 evidence assignment 规则

---

### 方案 B：Section / Mechanism / Comparison / Theme 四种专用压缩器

这条路线和当前代码最连续。

做法是：

- 保留四类 contract
- 但不再只让它们“组织 region”
- 而让它们各自实现一套专用 compression policy

例如：

#### section-grounded

- 目标：压成少量 `section band`
- 每个 band 必须：
  - 同 `full_doc_id`
  - chunk 连续
  - 对 query 的某个部分有明确支撑

#### mechanism-grounded

- 目标：压成少量 `原因 -> 过程 -> 结果` 单元
- 每个单元中边比点更重要
- bridge region 不只是补充，而是骨架

#### comparison-grounded

- 目标：压成少量 `side A / side B / contrast dimension` 单元
- 重叠允许存在，但每个单元必须自圆其说

#### theme-grounded

- 目标：压成少量并列 aspect
- 每个 aspect 尽量覆盖不同维度，而不是简单主题重复

优点：

- 和当前代码兼容性高
- 好落地

难点：

- 四个 organizer 内部要重新设计
- 不再只是“从 region 构 group”

---

### 方案 C：Global Budgeted Assignment

这条路线更偏“优化问题”。

核心思想：

1. final subgraph 全部 evidence 先进入候选池
2. 定义少量 target groups
3. 对每条 node / edge / chunk，决定：
   - 分给哪个 group
   - 是否被保留
4. 在总 source budget 下最大化：
   - coverage
   - distinctness
   - anchoring
   - contract fit

这条线更像：

> `subset selection + assignment`

优点：

- 非常贴当前真实问题
- 直接面向“有限 budget 如何压缩大图”

难点：

- 实现复杂
- 如果引入太多加权项，容易又回到“默认假设过多”

所以如果走这条，必须保持规则简单。

---

### 方案 D：Hierarchical Organization

这条路线适合处理“大图太大但 final answer 预算固定”的矛盾。

思路：

1. 先做粗粒度 organizer
   - 得到 2~4 个 macro regions
2. 再在每个 macro region 内做细粒度 facet
3. 最终从每个宏区里各取少量 evidence

这样做的好处是：

- 保证 breadth
- 同时保留局部 coherence

坏处是：

- 结构更复杂
- 第一版实现成本偏高

---

## 6. 我目前最推荐的方向

如果只选一条最值得先做的，我会推荐：

> **方案 B：四类专用压缩器**
> 再吸收方案 A 的 evidence allocation 思想。

原因是：

- 我们已经有 contract detector
- 四类 query contract 已经证明有效
- 用户也明确不想继续无限细分组织类型

因此下一版最自然的研究问题不是：

- “要不要再加第五种组织方式”

而是：

- “在这四种组织方式下，怎样把大图压成真正有用的 answer units”

---

## 7. 一个更具体的下一步研究顺序

如果按风险和收益排序，我建议这样推进。

### Step 1：明确每种 contract 的 answer unit 长什么样

先不写代码，先定义：

- `section unit`
- `mechanism unit`
- `comparison unit`
- `theme unit`

各自最小包含什么：

- anchor chunks
- support chunks
- node / edge skeleton
- label / summary

### Step 2：为每种 contract 定义 assignment 规则

重点不是打分，而是：

- 哪些 evidence 可以归入该 unit
- 哪些 evidence 不该归入
- unit 之间允许怎样的重叠

### Step 3：为每种 contract 定义 compression 规则

即：

- 在预算有限时先保留什么
- 去掉什么
- 哪些 evidence 必须保

### Step 4：再考虑 prompt 如何利用这些 units

注意：

这一步应该放在最后。

当前实验已经说明：

> 如果 organization 没做好，prompt 很难硬救。

---

## 8. 现在不建议再做的事

在 knowledge organization 研究清楚之前，我不建议继续做这些：

- 再继续增强联想
- 再继续细调 detector
- 再增加 prompt 范式约束
- 再引入新的 organizer 类别

因为这些都会让问题被掩盖，而不是被解决。

---

## 9. 一句话总结

当前系统已经说明：

- recall 不再是唯一瓶颈
- organization 才是下一阶段的主问题

更准确地说：

> 我们已经能从 query 出发联想到很大的 final subgraph；
> 下一步的真正挑战，是如何把这个大图压缩成少量 query-appropriate、evidence-anchored 的 answer units。

这应当成为接下来 organization 研究的中心。
