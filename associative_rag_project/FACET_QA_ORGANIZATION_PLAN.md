# Facet Group Organization Redesign

This note records the next organization-stage redesign. It is not the final
paper wording. It is a concrete implementation note for replacing the current
knowledge-group packing logic with a more query-grounded facet decomposition
pipeline.

## 1. Why the Current Organization Stage Is Not Enough

The current pipeline already separates:

- grounded chunk retrieval
- graph association / divergence
- LLM answer generation

However, the current organization stage still has a core weakness:

> it turns the expanded graph into local groups, but it does not explicitly
> turn the query into multiple evidence-grounded answer facets.

As a result:

- a group is still just a local evidence packet
- group boundaries are only partially query-grounded
- final answer quality depends too much on the LLM inferring the missing
  aspect decomposition by itself

This is too close to dense-RAG-plus-packing and does not fully realize the
intended QFS story.

## 2. New Principle

The organization stage should be defined as:

> Given a high-recall final subgraph, construct a small set of
> evidence-grounded answer facets, where each facet represents one major
> answer aspect of the original query.

This means each final unit is no longer just a generic "group". Instead it becomes:

- one local answer facet
- one root-grounded evidence region
- one evidence package for a distinct fact function

QFS can then be interpreted as:

> synthesize several evidence-grounded answer facets into the final summary.

## 3. Stage Split

The full method becomes:

1. **Grounding**
   - `query -> dense candidate chunks -> diversified root chunks`

2. **Association**
   - `root graph -> high-recall final subgraph`
   - goal: maximize recall / evidence breadth

3. **Organization**
   - `final subgraph + query -> answer facets`
   - goal: decide what answer aspects exist in the evidence

4. **Generation**
   - `facet groups -> final QFS answer`

## 4. Inputs to Organization

The organization stage should have access to:

- `query`
- `root_chunk_hits`
- `top_root_nodes`
- `top_root_edges`
- `final_nodes`
- `final_edges`
- `node_roles`
- `edge_roles`
- `node_to_chunks`
- `edge_to_chunks`
- `chunk_store`

The key difference from the current implementation is:

> organization should explicitly consume the whole final subgraph and root
> grounding together, instead of only consuming local component fragments.

## 5. Desired Output Structure

The new output unit should be an **answer facet**, not just a generic knowledge
group.

Each answer facet should contain:

- `facet_id`
- `facet_prompt`
  - a short answer target
- `anchor_chunk_ids`
  - grounded entry evidence
- `support_chunk_ids`
  - additional evidence chunks for this facet
- `node_ids`
- `edge_ids`
- `relation_themes`
- `focus_entities`
- `focus_relations`
- `facet_summary`
  - short orienting description

## 6. Proposed Algorithm

### Step A. Build root-anchored evidence regions

For each root chunk:

1. collect the final nodes and final edges directly supported by that root
2. attach a small amount of nearby support evidence
3. form one **root-anchored region**

This keeps the organization stage grounded in the original evidence entry.

### Step B. Expand region descriptors

For each root-anchored region, compute:

- main relation themes
- focus entities
- support chunk previews
- root-to-support bridge patterns

These descriptors are used to understand what kind of answer facet this region
can answer.

### Step C. Induce candidate answer facets

Facet induction should be done from:

- the original query form
- the region descriptors
- the relation themes observed in the final subgraph

The facet is not a free LLM rewrite of the query. It is a constrained
answer-aspect proposal grounded in actual evidence regions.

The first implementation should use deterministic rules, not full LLM
generation.

Examples:

- if several regions emphasize causes / risks / failures
  -> induce a "cause/failure" facet
- if several regions emphasize institutions / support / organizations
  -> induce a "support mechanism" facet
- if regions emphasize procedures / requirements / steps
  -> induce a "procedure" facet

### Step D. Assign evidence regions to facets

Each root-anchored region is assigned to the facet that best matches:

1. query-side sub-intent
2. relation-theme compatibility
3. anchor evidence compatibility

This assignment should not use large weighted sums.

Initial decision rule:

1. hard filter by facet-compatible cue family
2. tie-break by lexical overlap between query and region descriptor
3. tie-break by anchor support count

### Step E. Build one facet group per facet

For each facet:

- merge the assigned root-anchored regions
- deduplicate support chunks
- keep the strongest anchors
- derive a facet summary

The resulting facet group is then a real answer unit:

- it has an answer target
- it has evidence
- it has a local answer space

## 7. Why This Is Better Than the Current Group Logic

The current local-group logic improves grounding, but it still leaves the
query decomposition implicit.

The new facet logic makes the organization stage explicit:

- association handles recall
- organization handles decomposition

This makes the system easier to explain:

- no need to pretend that local groups themselves are the final answer units
- no need to force the LLM to infer all query aspects from packed evidence

## 8. First Implementation Scope

The first implementation should stay small and deterministic.

It should:

- keep the current association stage
- add a new `organization.py` module
- build root-anchored regions from the final subgraph
- induce a small number of candidate answer facets from query cues plus relation
  themes
- merge regions into facet groups

It should **not** yet:

- call another LLM in the organization stage
- add heavy adaptive control
- add many composite ranking scores

## 9. Practical Coding Plan

### Phase 1: Scaffold

Add a new module:

- `associative_rag_project/organization.py`

It should define:

- region data structures
- facet data structures
- region collection helpers
- facet induction helpers
- facet-group construction helpers

### Phase 2: Dual path in pipeline

Temporarily keep the current `build_knowledge_groups()` path.

Add an alternate path:

- `build_qa_facet_groups()`

This allows comparison without breaking the whole pipeline immediately.

### Phase 3: Prompt transition

Once QA facets are stable, replace the prompt wording from:

- "knowledge groups"

to:

- "facet groups" / "question facets"

The answer prompt should ask the LLM to:

1. answer each facet briefly
2. merge overlapping facets
3. produce the final QFS answer

## 10. Minimal Success Criteria

The redesign is moving in the right direction if:

- groups are no longer obviously off-topic despite a large final subgraph
- each group can be read as one answerable sub-question
- agriculture-like queries recover breadth without falling back into generic
  off-topic expansion
- art-like queries preserve broad thematic synthesis
