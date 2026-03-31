这次预实验我关注一件事：验证 LightRAG 的第一步（LLM 生成 keyword 作为图入口）是否稳定。

预实验为每个原始问题做 4 个同义改写，形成 1 组 5 个 query。改写方式是同义替换、语序变化、轻度约束、显式化表达。目标是“人看起来没有歧义、语义等价”。然后跑同一套图检索流程，记录每个 query 的 high/low keywords、local 实体入口、global 边入口和子图规模。

对组内 query 做两两比较，指标是 Jaccard：A 交 B 除以 A 并 B。  
具体算四类集合：
1. high keywords
2. low keywords
3. low keyword 对应 local 入口（local_entities）
4. high keyword 对应 global 入口（global_edges）

先给两个最能说明问题的例子。

例子 1（agriculture，q002）
原始问题：How does the book suggest new beekeepers to maintain their commitment to beekeeping?  
同义改写：How does the book advise novice beekeepers to stay dedicated to beekeeping?

这两个问题人看起来基本等价，但系统行为差异明显：
- high keyword Jaccard = 0.1667
- low keyword Jaccard = 0.0（完全不重合）
- local 入口 Jaccard = 0.5789
- global 入口 Jaccard = 0.4458

更关键的是，global 入口里出现了不少和“如何保持投入”关系不强的内容，比如 "SWARM SEASON"、"PACKAGE BEES"、"MILLER"、"THE HIVE" 等，检索焦点明显漂移到养蜂技术/人物语境，而不是“维持承诺的方法”本身。

例子 2（mix，q001）
原始问题：How do the narratives in different passages reflect the socio-political contexts of their times?  
同义改写：In what ways do the stories in various texts mirror the social and political environments of their periods?

同样是语义等价，但：
- high keyword Jaccard = 0.0
- low keyword Jaccard = 0.0
- local 入口 Jaccard = 0.3187
- global 入口 Jaccard = 0.25

并且入口内容偏题明显。比如 global top 命中里出现 "RESEARCH STUDIES"-"SOCIAL MEDIA"、"LE MONDE"-"THE KING'S SPEECH"、"BLACK PLAGUE"-"MONICA H. GREEN" 等，说明入口已经被拉向跨主题噪声，而不是稳定围绕“叙事与社会政治语境”。

整体统计（3 个数据集）也支持这个问题是普遍存在，不是个例：

agriculture：
- high keyword Jaccard 平均 0.2369
- low keyword Jaccard 平均 0.1883
- low->local 入口一致性 0.4958
- high->global 入口一致性 0.5199

legal：
- high keyword Jaccard 平均 0.1926
- low keyword Jaccard 平均 0.0615
- low->local 入口一致性 0.1969
- high->global 入口一致性 0.3388

mix：
- high keyword Jaccard 平均 0.2641
- low keyword Jaccard 平均 0.1546
- low->local 入口一致性 0.3682
- high->global 入口一致性 0.4942

结论很清楚：同义 query 会触发明显不同的 keyword，keyword 差异继续放大为图入口差异。这件事在 agriculture、legal、mix 都存在，尤其 legal 最严重（low keyword 和 local 入口稳定性最低）。  
所以“keyword 入口不稳定导致子图漂移”这个问题是成立的，而且是系统性问题。

