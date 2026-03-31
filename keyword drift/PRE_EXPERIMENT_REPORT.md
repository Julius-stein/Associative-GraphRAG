# LightRAG Keyword 入口稳定性预实验报告（阶段一）

## 1. 问题聚焦

当前 LightRAG 在 query 的第一步会由 LLM 生成 `high_level_keywords` 和 `low_level_keywords`，再用关键词去做图入口召回（`local_entities` / `global_edges`）。  
核心担忧是：

- 同一语义的不同问法会触发明显不同的关键词集合；
- 关键词差异会传导到图入口召回差异；
- 最终导致子图范围飘移，回答稳定性下降。

本预实验目标是做“存在性证明”：证明这种不稳定性在真实数据集上显著存在。

---

## 2. 实验设计（已完成）

- 数据集：`agriculture`、`legal`、`mix`
- 每个数据集：10 组 query，每组 5 个语义等价变体（共 50 条）
- 模式：`hybrid`
- 记录内容：`keywords`、`v0.local_entities`、`v0.global_edges`、`subgraph`
- 分析方式：组内两两比较（pairwise）
  - keyword 一致性：`high/low keyword jaccard`, `overlap@10`
  - 入口一致性：
    - low keyword -> local 实体入口一致性（`local_entities`）
    - high keyword -> global 边入口一致性（`global_edges`）

---

## 3. 总体统计结果

### 3.1 Agriculture

- `high_keyword_jaccard_avg = 0.2369`
- `low_keyword_jaccard_avg = 0.1883`
- `low->local entry jaccard = 0.4958`
- `high->global entry jaccard = 0.5199`
- `subgraph_zero_ratio = 0.0`

结论：keyword 有明显漂移；入口一致性中等偏低。

### 3.2 Legal

- `high_keyword_jaccard_avg = 0.1926`
- `low_keyword_jaccard_avg = 0.0615`
- `low->local entry jaccard = 0.1969`
- `high->global entry jaccard = 0.3388`
- `subgraph_zero_ratio = 0.0`

结论：Legal 数据集漂移最严重，尤其 low keyword 与 local 入口一致性极差。

### 3.3 Mix

- `high_keyword_jaccard_avg = 0.2641`
- `low_keyword_jaccard_avg = 0.1546`
- `low->local entry jaccard = 0.3682`
- `high->global entry jaccard = 0.4942`
- `subgraph_zero_ratio = 0.0`

结论：Mix 也存在显著漂移，介于 agriculture 和 legal 之间。

---

## 4. 关键例子（用于“讲清楚问题”）

## 4.1 Agriculture（q002）

原始语义：新手养蜂人如何保持投入  

- Query A: `How does the book suggest new beekeepers to maintain their commitment to beekeeping?`
- Query B: `How does the book advise novice beekeepers to stay dedicated to beekeeping?`

组内对比：

- `high keyword jaccard = 0.1667`
- `low keyword jaccard = 0.0`
- `local entry jaccard = 0.5789`
- `global entry jaccard = 0.4458`
- 子图规模：
  - A: `(127, 189, 211)`
  - B: `(110, 194, 184)`

说明：语义近似但 low keyword 完全不重合，入口与子图范围均发生明显变化。

## 4.2 Legal（q003）

原始语义：哪些实体合规要求最复杂

- Query A: `Which entities have the most complex compliance requirements based on the dataset?`
- Query B: `Which organizations face the most intricate compliance rules according to the dataset?`

组内对比：

- `high keyword jaccard = 0.0`
- `low keyword jaccard = 0.0`
- `local entry jaccard = 0.3333`
- `global entry jaccard = 0.2121`
- 子图规模：
  - A: `(127, 203, 764)`
  - B: `(137, 97, 804)`

说明：同义改写导致关键词入口几乎“重置”，全局边入口重合度极低。

## 4.3 Mix（q001）

原始语义：叙事如何反映社会政治语境

- Query A: `How do the narratives in different passages reflect the socio-political contexts of their times?`
- Query B: `In what ways do the stories in various texts mirror the social and political environments of their periods?`

组内对比：

- `high keyword jaccard = 0.0`
- `low keyword jaccard = 0.0`
- `local entry jaccard = 0.3187`
- `global entry jaccard = 0.25`
- 子图规模：
  - A: `(137, 111, 300)`
  - B: `(124, 93, 261)`

说明：词面改写后，入口集合与子图规模均显著变化，直接支撑“不稳定入口”假设。

---

## 5. 预实验结论（可直接汇报）

- 在 3 个数据集上，keyword 稳定性均偏低，尤其 `low keywords`。
- keyword 不稳定会传导到入口召回（`local_entities/global_edges`）不稳定。
- 子图范围随改写明显波动，证明“同义 query -> 图检索路径飘移”客观存在。
- 该问题在 `legal` 数据集最突出，说明在抽象/规范类语义上风险更高。

---

## 6. 初步解决方案（阶段二方向）

建议从“入口稳态化”入手，分三层推进：

1. **关键词生成约束化（Prompt + Schema）**
   - high/low keyword 固定槽位与数量上限；
   - 约束输出使用图谱词表/别名表中的 canonical term；
   - 对抽象词（如“complex”, “important”, “issues”）进行降权或替换。

2. **入口检索做多路融合而非单路 keyword**
   - `keyword retrieval` + `query embedding retrieval` 并联；
   - local/global 入口按加权融合，降低单次 keyword 抖动影响；
   - 对入口做 top-k 稳定化（例如取多次生成/多路召回交并集）。

3. **增加重排与稳定性反馈**
   - 对候选入口实体/边引入 cross-encoder 或 LLM re-ranker；
   - 将“组内一致性指标”（本实验指标）纳入离线优化目标；
   - 以“答案质量 + 入口稳定性”双目标调参。

---

## 7. 下一步执行计划

1. 在 `legal` 先做小范围 ablation（最不稳定集优先），验证稳态化收益。  
2. 接入 RAGAS，对改造前后回答进行质量评估（faithfulness, answer relevancy, context precision/recall）。  
3. 形成阶段二报告：`稳定性提升幅度 + 质量变化 + 代价（时延/成本）`。

