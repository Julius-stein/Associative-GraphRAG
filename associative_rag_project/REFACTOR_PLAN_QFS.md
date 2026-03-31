# QFS Refactor Plan

This document records the next code refactor direction. It is not a paper write-up. It is a concrete implementation plan for simplifying the current system so that the method is easier to explain and the experiments are easier to interpret.

## 1. Why Refactor

The current pipeline works as an engineering system, but it has become difficult to explain as a research method.

The main issue is that many stages currently mix too many objectives into one score:

- relevance to the query
- support from root chunks
- graph richness
- relation diversity
- bridge value
- technical-noise suppression

Most of these are combined by fixed weighted sums. This is convenient for engineering iteration, but from a research perspective it introduces too many default assumptions.

As a result:

- the system is hard to interpret
- ablations are hard to trust
- it is unclear which module is responsible for which behavior
- early local query relevance can dominate the whole pipeline
- mid-level structures that are useful for summarization can be suppressed too early

## 2. New Core Principle

The new framework should explicitly split QFS graph retrieval into two stages:

1. **Divergence**
   Goal: expand from grounded evidence to improve coverage and recall.

2. **Organization**
   Goal: take the expanded evidence and organize it into structures that an LLM can summarize well.

This means we should stop letting the association step perform both divergence and organization at the same time.

## 3. High-Level New Pipeline

The target pipeline becomes:

1. `query -> dense chunk retrieval`
2. `candidate roots -> diversified root selection`
3. `root graph -> divergence by information gain`
4. `expanded graph -> query-aware organization`
5. `knowledge groups -> answer generation`

In other words:

- grounding happens first
- divergence happens second
- query-aware summarization logic happens later

## 4. Design Rule for Scores

New rule:

- avoid large weighted sums
- prefer one dominant factor
- allow at most three factors per decision
- use hard constraints or lexicographic tie-breaking where possible

This means:

- fewer composite scores
- more stage-specific criteria
- clearer explanations of what each module is doing

## 5. Planned Refactor by Stage

## 5.1 Stage A: Root Grounding

### Problem in current code

Current root selection is in:

- [rerank_root_chunks()](/Users/Admin/projects/Association/associative_rag_project/retrieval.py)

It currently mixes:

- retrieval score
- lexical query overlap
- graph yield
- dense term
- BM25 term
- technical penalty

This makes root selection neither a pure retrieval step nor a pure diversity step.

### New target behavior

Root selection should serve one purpose:

> choose a set of starting chunks that cover different evidence directions

### New rule

Use **dense retrieval only** as the main starting signal for the next version.

Reason:

- it removes one major source of mixed assumptions
- it matches the intended chunk-first grounding story
- it makes later comparison cleaner

BM25 can remain as an ablation, but not as the default research story.

### New root selection structure

Step 1:

- retrieve top-N dense chunks

Step 2:

- select final root chunks by **diversified root selection**, not by a weighted rerank score

### Diversified root selection criteria

Each chosen root should ideally contribute a different evidence direction.

The new root selector should use:

- primary signal: dense retrieval rank
- hard constraint: avoid over-selecting adjacent chunks from the same document band
- hard constraint: avoid high provenance overlap
- tie-break: prefer chunks that introduce new root nodes or new root edges

### Key change

This stage should not ask:

> which chunk is most query-like?

It should ask:

> which chunk adds a new grounded starting point?

## 5.2 Stage B: Divergence

### Problem in current code

The current association code is in:

- [structural_association()](/Users/Admin/projects/Association/associative_rag_project/association.py)
- [semantic_association()](/Users/Admin/projects/Association/associative_rag_project/association.py)

The current semantic association is still dominated by `query_rel`.

That means divergence is not really divergence. It is still early query-focused filtering.

### New target behavior

Divergence should answer:

> what is worth adding in order to increase evidence coverage?

This stage should be much less concerned with fine-grained query similarity.

### New divergence objects

The divergence stage should expand two kinds of things:

1. **Bridge structures**
   - paths or nodes that connect disconnected root regions

2. **Coverage structures**
   - new relations, chunks, bands, or neighborhoods that increase evidence breadth

### New divergence features

The main signals should be information-gain style signals:

- `component_gain`
  - does this connect previously separate root regions?

- `relation_gain`
  - does this introduce a new relation category?

- `source_gain`
  - does this introduce evidence from a new source band or provenance region?

- `band_gain`
  - does this expand into a new neighboring chunk region within a document?

### New score policy

Prefer one dominant factor plus simple tie-breaks.

Examples:

- **bridge path ranking**
  - primary: `component_gain`
  - tie-break 1: `source_gain`
  - tie-break 2: shorter path length

- **coverage edge ranking**
  - primary: `relation_gain`
  - tie-break 1: `source_gain`
  - tie-break 2: root-support overlap

- **coverage node ranking**
  - primary: `band_gain` or `bridge_gain`
  - tie-break 1: root-support overlap

Crucially:

- `query_rel` should not be the main divergence driver
- at most it can be used as a weak filter or late tie-break

## 5.3 Stage C: Query-Aware Organization

### Problem in current code

The current system builds groups mainly from connected components:

- [build_knowledge_groups()](/Users/Admin/projects/Association/associative_rag_project/context.py)

This does not guarantee:

- one group corresponds to one coherent topic
- different groups are distinct
- the groups jointly cover the query’s main aspects

### New target behavior

Organization should answer:

> from the already expanded evidence, what is worth keeping and how should it be organized for summarization?

This is where query relevance should return as a main driver.

### New organization tasks

1. **group formation**
   - split or merge expanded structures into coherent topic units

2. **group deduplication**
   - remove or merge highly overlapping groups

3. **anchor assignment**
   - ensure each group has real grounding evidence

4. **query-facing ranking**
   - rank groups by usefulness for answering the query

### New organization features

This stage can use:

- `group_query_rel`
- `group_cohesion`
- `group_anchor_support`
- `group_redundancy`

### New score policy

Again, avoid large weighted mixtures.

Example:

- primary: `group_query_rel`
- tie-break 1: `group_cohesion`
- tie-break 2: `group_anchor_support`
- hard penalty: high `group_redundancy`

This keeps the organization phase clearly different from the divergence phase.

## 6. Root Diversity Is a First-Class Problem

One of the most important issues is that current root chunks may collapse onto one section or one narrow theme.

If the top `5` chunks are effectively five consecutive chunks from the same section, then:

- root graph diversity is low
- later graph expansion starts from one topic band
- divergence has little real room to work

Therefore root diversity must be enforced directly.

### Planned root diversity constraints

For any pair of candidate root chunks, compute:

- same `full_doc_id`
- distance in `chunk_order_index`
- provenance overlap on nodes
- provenance overlap on edges

Then reject or down-prioritize a candidate if:

- it is in the same local chunk band as an already selected root
- it has very high provenance overlap with an already selected root

This turns root selection into a real multi-start procedure.

## 7. Chunk Adjacency as a New Evidence Signal

The project currently does not explicitly know section boundaries.

However, chunks already contain:

- `full_doc_id`
- `chunk_order_index`

This makes it possible to add **chunk adjacency** without re-chunking the corpus.

### Why this matters

A section often spans multiple contiguous chunks. If only one chunk is selected, the system may lose:

- continuation
- surrounding conditions
- clause boundaries
- local thematic completeness

### Planned use of chunk adjacency

Chunk adjacency should not replace graph association. It should act as an additional grounded signal.

Potential uses:

1. **divergence-side band gain**
   - selecting a node/edge/chunk can expose nearby chunk bands

2. **organization-side anchor completion**
   - a group’s anchor evidence can include adjacent chunks from the same document band

3. **section-like evidence unit construction**
   - multiple contiguous chunks can be merged into one local evidence band

This is preferable to semantic re-chunking because it preserves current provenance and experiment comparability.

## 8. What Will Be Removed or Weakened

The refactor should explicitly remove the following patterns:

- large linear combinations with 4 to 6 terms
- using `query_rel` as the dominant signal in every stage
- root selection that collapses onto one section band
- connected component = final group by default
- letting the same score simultaneously represent recall, organization, and denoising

## 9. Proposed Code Changes

## 9.1 Retrieval layer

Files to modify:

- [retrieval.py](/Users/Admin/projects/Association/associative_rag_project/retrieval.py)
- [pipeline.py](/Users/Admin/projects/Association/associative_rag_project/pipeline.py)

Planned changes:

- add a dense-only default path for root retrieval
- replace `rerank_root_chunks()` with:
  - `score_root_candidates_dense()`
  - `select_diverse_roots()`

Expected result:

- root selection becomes easier to explain
- root diversity becomes explicit

## 9.2 Divergence layer

Files to modify:

- [association.py](/Users/Admin/projects/Association/associative_rag_project/association.py)

Planned changes:

- split current association logic into:
  - `bridge_divergence()`
  - `coverage_divergence()`

- reduce score complexity
- use information-gain style signals
- move query-focused filtering out of this early stage

Expected result:

- divergence becomes a genuine recall-expansion step

## 9.3 Organization layer

Files to modify:

- [context.py](/Users/Admin/projects/Association/associative_rag_project/context.py)

Planned changes:

- replace direct component-to-group mapping with:
  - group refinement
  - group merge/split
  - group dedup
  - anchor evidence assignment

Expected result:

- groups become more coherent
- repeated groups are reduced
- LLM sees cleaner evidence structure

## 9.4 Optional chunk-band support

Files to modify later:

- [data.py](/Users/Admin/projects/Association/associative_rag_project/data.py)
- [context.py](/Users/Admin/projects/Association/associative_rag_project/context.py)
- possibly [pipeline.py](/Users/Admin/projects/Association/associative_rag_project/pipeline.py)

Planned changes:

- build same-document chunk neighborhood maps
- expose adjacent chunk bands as optional contextual evidence

## 10. Refactor Order

To keep experiments interpretable, the refactor should proceed in this order:

1. freeze current version as backup
2. simplify root retrieval and root selection
3. simplify divergence scores
4. rebuild knowledge organization
5. add chunk adjacency / section-band support

This order is important because:

- root diversity affects everything downstream
- organization quality cannot be judged fairly if divergence is still malformed

## 11. Immediate First Step

The first concrete coding step should be:

> **replace current root reranking with dense-first candidate retrieval plus diversified root selection**

This is the cleanest place to start because:

- it directly addresses the “all roots come from one section” problem
- it removes one of the most over-composite scores
- it creates a clearer starting point for the new divergence stage

## 12. Summary

The refactor direction is:

- less score mixing
- less early query dominance
- more explicit divergence
- more explicit organization
- more grounded root diversity

The target is not just a better engineering system, but a system whose intermediate decisions can be explained as a research method.
