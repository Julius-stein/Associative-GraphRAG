# Associative RAG Technical Report

This document describes how the current `associative_rag_project` codebase works as implemented today. It is a code-grounded technical note rather than a method proposal. All details below are based on the current Python implementation in this repository.

## 1. Scope

The current system is a query-focused summarization pipeline with five major stages:

1. chunk retrieval
2. root reranking and root graph construction
3. graph association
4. knowledge-group packaging
5. answer generation and LLM judging

The main CLI entrypoint is [main.py](/Users/Admin/projects/Association/associative_rag_project/main.py). The end-to-end retrieval pipeline is implemented in [pipeline.py](/Users/Admin/projects/Association/associative_rag_project/pipeline.py).

## 2. Runtime Inputs

For one corpus directory, the current pipeline expects these files:

- `graph_chunk_entity_relation.graphml`
- `kv_store_text_chunks.json`
- `vdb_chunks.json` for dense or hybrid retrieval
- `kv_store_full_docs.json` is present in corpora but is not yet used in the main retrieval path

The corpus loader is [load_graph_corpus()](/Users/Admin/projects/Association/associative_rag_project/data.py), which currently loads:

- the GraphML graph
- the chunk KV store

Chunk records in `kv_store_text_chunks.json` contain:

- `tokens`
- `content`
- `chunk_order_index`
- `full_doc_id`

This schema is also declared in [base.py](/Users/Admin/projects/Association/lightrag/base.py).

## 3. CLI and Current Default Configuration

The system exposes five subcommands in [main.py](/Users/Admin/projects/Association/associative_rag_project/main.py):

- `retrieve`
- `answer`
- `judge`
- `run`
- `run-all`

### 3.1 Current default retrieval/generation hyperparameters

These are the current CLI defaults from [build_parser()](/Users/Admin/projects/Association/associative_rag_project/main.py):

| Parameter | Default | Function |
|---|---:|---|
| `top_chunks` | `5` | Number of reranked root chunks kept after the candidate retrieval stage. These chunks become the initial evidence anchors. |
| `chunk_candidate_multiplier` | `3` | Expands the candidate retrieval pool before root reranking. With `top_chunks=5`, this contributes `15` candidates unless a larger pool is required elsewhere. |
| `adaptive_candidate_pool_size` | `30` | Minimum candidate retrieval pool size used for adaptive-feature estimation and root reranking. This is why the system usually retrieves `30` chunk candidates even though only `5` roots are kept. |
| `retrieval_mode` | `hybrid` | Chooses whether initial chunk retrieval uses BM25 only, dense only, or weighted fusion of both. |
| `dense_weight` | `0.75` | Weight assigned to normalized dense retrieval scores in hybrid retrieval. |
| `bm25_weight` | `0.25` | Weight assigned to normalized BM25 scores in hybrid retrieval. |
| `top_root_nodes` | `12` | Maximum number of scored root nodes retained as high-priority anchors for graph expansion. |
| `top_root_edges` | `16` | Maximum number of scored root edges retained as relation-first anchors for graph expansion. |
| `max_hop` | `4` | Shortest-path cutoff used in structural association when searching for bridge paths between root components. |
| `path_budget` | `12` | Upper bound on how many structural bridge paths can be selected in one query. |
| `semantic_edge_budget` | `20` | Upper bound on how many semantically useful edges can be added during semantic association. |
| `semantic_node_budget` | `12` | Upper bound on how many semantically useful nodes can be added during semantic association. |
| `semantic_edge_min_score` | `0.03` | Minimum score threshold for a candidate semantic edge before it is eligible for selection. |
| `semantic_node_min_score` | `0.03` | Minimum score threshold for a candidate semantic node before it is eligible for selection. |
| `association_rounds` | `2` | Number of alternating `structural association -> semantic association` rounds. |
| `group_limit` | `8` | Maximum number of knowledge groups kept after group scoring and sorting. |
| `max_source_chunks` | `14` | Maximum number of source chunks that may appear in the final answer evidence package. |
| `max_source_word_budget` | `4500` | Approximate word budget for the final source bundle given to the answer LLM. |
| `adaptive_control` | enabled unless `--disable-adaptive-control` | Enables the weak adaptive controller that nudges graph expansion budgets around the baseline. |

### 3.2 Current model configuration

Configuration is loaded from [config.py](/Users/Admin/projects/Association/associative_rag_project/config.py), which reuses [llm_config.py](/Users/Admin/projects/Association/lightrag/llm_config.py).

Current defaults:

- generation model: `gpt-4o-mini`
- generation base URL: inherited from `SCHEME4_LLM_CONFIG`
- embedding model: `text-embedding-3-small` unless overridden by env
- judge model: `gpt-5.4-mini`

The generation client is [OpenAICompatibleClient](/Users/Admin/projects/Association/associative_rag_project/llm_client.py). The embedding client is [OpenAICompatibleEmbeddingClient](/Users/Admin/projects/Association/associative_rag_project/embedding_client.py).

## 4. End-to-End Retrieval Pipeline

The main retrieval routine is [retrieve_corpus_queries()](/Users/Admin/projects/Association/associative_rag_project/pipeline.py), which performs:

1. load graph and chunk store
2. build BM25 index
3. optionally load dense chunk matrix
4. build chunk-to-node and chunk-to-edge provenance mappings
5. load query rows
6. run `run_query()` for every query

## 5. Query Representation and Question Loading

[load_query_rows()](/Users/Admin/projects/Association/associative_rag_project/data.py) supports two formats:

- plain question lists
- grouped rewrite files

Each query row is normalized into:

- `group_id`
- `variant_id`
- `query`
- `base_query`

For standard benchmark runs, `group_id` is formatted as `q001`, `q002`, etc.

## 6. Chunk Retrieval

Chunk retrieval is implemented in [retrieval.py](/Users/Admin/projects/Association/associative_rag_project/retrieval.py).

### 6.1 BM25 retrieval

[BM25Index.build()](/Users/Admin/projects/Association/associative_rag_project/retrieval.py) builds an in-memory inverted index over `chunk["content"]`.

Current BM25 constants:

- `k1 = 1.5`
- `b = 0.75`

Search is performed by [BM25Index.search()](/Users/Admin/projects/Association/associative_rag_project/retrieval.py), which returns:

- `chunk_id`
- raw BM25 `score`
- `score_norm`

where `score_norm = score / top_score`.

### 6.2 Dense retrieval

[DenseChunkIndex.load()](/Users/Admin/projects/Association/associative_rag_project/retrieval.py) loads:

- `payload["data"]` for chunk ids
- `payload["matrix"]` as a base64-encoded dense embedding matrix

The code uses cosine similarity over row-normalized chunk embeddings. Query embeddings are obtained through [OpenAICompatibleEmbeddingClient.embed_text()](/Users/Admin/projects/Association/associative_rag_project/embedding_client.py).

Dense search returns:

- `chunk_id`
- raw cosine `dense_score`
- normalized `dense_score_norm`

### 6.3 Hybrid retrieval

[HybridChunkRetriever.search()](/Users/Admin/projects/Association/associative_rag_project/retrieval.py) merges BM25 and dense hits by chunk id.

Current fusion rule:

- if `bm25`: use BM25 only
- if `dense`: use dense only
- if `hybrid`: use weighted sum

Current hybrid score:

`retrieval_score = 0.75 * dense_score_norm + 0.25 * bm25_score_norm`

Merged hits keep:

- `bm25_score`, `bm25_score_norm`
- `dense_score`, `dense_score_norm`
- `retrieval_score`

Then results are sorted by:

1. `retrieval_score` descending
2. `dense_score_norm` descending
3. `bm25_score_norm` descending
4. `chunk_id`

## 7. Candidate Pool and Root Chunk Reranking

Inside [run_query()](/Users/Admin/projects/Association/associative_rag_project/pipeline.py), the initial candidate pool size is:

`candidate_top_k = max(top_chunks, top_chunks * chunk_candidate_multiplier, adaptive_candidate_pool_size)`

With default settings this becomes:

- `max(5, 5*3, 30) = 30`

So even though the final root chunk count is `5`, the current system first retrieves `30` candidate chunks.

Root reranking is performed by [rerank_root_chunks()](/Users/Admin/projects/Association/associative_rag_project/retrieval.py).

For each candidate chunk, the reranker computes:

- `item["score_norm"]`: retrieval score after BM25/dense fusion
- `query_rel`: lexical overlap between query and chunk text
- `graph_yield`: normalized graph richness of the chunk
  - `len(chunk_to_nodes[chunk]) + 2 * len(chunk_to_edges[chunk])`
- `dense_term`
- `bm25_term`
- `technical_penalty`

Current rerank formula:

`rerank_score = 0.35 * retrieval + 0.25 * query_rel + 0.15 * graph_yield + 0.15 * dense_term + 0.10 * bm25_term - technical_penalty`

Technical penalty is:

- `0.0` if the query itself appears technical
- otherwise `0.20 * technical_density(chunk_text)`

Only the top `top_chunks` reranked chunks become root chunks.

## 8. Graph Provenance Mapping

[build_chunk_mappings()](/Users/Admin/projects/Association/associative_rag_project/data.py) constructs four maps:

- `chunk_to_nodes`
- `chunk_to_edges`
- `node_to_chunks`
- `edge_to_chunks`

These are built from LightRAG provenance fields:

- node `source_id`
- edge `source_id`

The `source_id` field is split by the LightRAG separator `<SEP>` via [parse_source_ids()](/Users/Admin/projects/Association/associative_rag_project/common.py).

This many-to-many mapping is the basis for:

- root graph construction
- support estimation
- chunk ranking
- semantic association
- knowledge-group packaging

## 9. Adaptive Control

Adaptive control is implemented in [adaptive_control.py](/Users/Admin/projects/Association/associative_rag_project/adaptive_control.py).

Important note:

- the project still computes adaptive signals even when adaptive control is disabled
- when `--disable-adaptive-control` is used, these signals are logged and stored, but the actual budgets are reset to CLI defaults

### 9.1 Query-form priors

Current query-style detection uses generic lexical cues only.

Overview cues include items such as:

- `overall`
- `compare`
- `patterns`
- `themes`
- `across`
- `impact`
- `influence`

Focused cues include items such as:

- `which`
- `when`
- `specific`
- `section`
- `list`
- `steps`
- `cost`

The resulting `query_style` is one of:

- `synthesis`
- `balanced`
- `concrete`

### 9.2 Retrieval-shape features

Two retrieval-shape signals are computed:

1. `candidate_chunk_dispersion`
2. `retrieval_cliff`

`candidate_chunk_dispersion` is based on a softmax over the top candidate retrieval scores. It also produces:

- `candidate_chunk_concentration`
- `effective_candidate_chunk_count`

`retrieval_cliff` is computed from the pre-rerank candidate pool and measures how dominant the top retrieved chunk is relative to the remaining top-N candidates.

### 9.3 Root-graph features

[compute_root_graph_features()](/Users/Admin/projects/Association/associative_rag_project/adaptive_control.py) computes:

- `root_density = len(root_edges) / len(root_nodes)`
- `fragmentation = 1 - largest_component_ratio`
- `largest_component_ratio`
- `component_count`

### 9.4 Current adaptive formula

Current adaptive control is intentionally weak. It starts from the non-adaptive baseline and only nudges retrieval-time expansion budgets.

The current association strength is:

`association_strength = 0.50 + 0.26*(density_signal - 0.50) + 0.12*(fragmentation_signal - 0.50) + style_prior - cliff_brake`

Where:

- `density_signal` is inverse-normalized from `root_density` over `[0.30, 0.75]`
- `fragmentation_signal` is normalized from `fragmentation` over `[0.50, 0.95]`
- `style_prior = +0.08` for `synthesis`, `-0.08` for `concrete`, else `0`
- `cliff_brake = 0.18 * retrieval_cliff`

Then `association_strength` is clamped to `[0.15, 0.85]`.

### 9.5 What adaptive control is currently allowed to change

Adaptive control does not change:

- `association_rounds`
- `group_limit`
- `max_source_chunks`
- `max_source_word_budget`

It only changes:

- `path_budget`
- `semantic_edge_budget`
- `semantic_node_budget`
- `semantic_edge_min_score`
- `semantic_node_min_score`

This is done by scaling around the no-adaptive baseline.

## 10. Root Node and Root Edge Scoring

Root node scoring is implemented by [score_root_nodes()](/Users/Admin/projects/Association/associative_rag_project/retrieval.py).

For each root node:

- `query_rel`: lexical overlap with node id, entity type, description
- `support`: overlap between node provenance chunks and root chunks
- `chunk_alignment`: mean normalized root-chunk score over overlapping chunks

Current node score:

`node_score = 0.55 * query_rel + 0.25 * support + 0.20 * chunk_alignment`

Root edge scoring is implemented by [score_root_edges()](/Users/Admin/projects/Association/associative_rag_project/retrieval.py).

For each root edge:

- `query_rel`
- `support`
- `chunk_alignment`
- `weight_term = log1p(edge_weight) / 5`

Current edge score:

`edge_score = 0.50 * query_rel + 0.20 * support + 0.15 * chunk_alignment + 0.15 * weight_term`

Only the top `top_root_nodes` and top `top_root_edges` are used as root anchors for graph association.

## 11. Structural Association

Structural association is implemented by [structural_association()](/Users/Admin/projects/Association/associative_rag_project/association.py).

Its purpose is to bridge disconnected root components using short query-relevant paths.

### 11.1 Seed nodes

Seed nodes are built from:

- ranked root nodes
- endpoints of ranked root edges

### 11.2 Path enumeration

For each seed node, the code runs:

- `nx.single_source_shortest_path(graph, source_id, cutoff=max_hop)`

Only paths that:

- connect different root components
- contain at least one new edge not already in the current edge set

are kept as structural candidates.

### 11.3 Path score

[build_path_score()](/Users/Admin/projects/Association/associative_rag_project/association.py) computes:

- `root_reach = 1.0` if the path bridges two different root components, else `0.4`
- `support_span`: number of provenance chunks touched by the path, normalized by root chunk count and capped at `2.0`
- `rel_path`: mean lexical relevance of edges along the path
- `length_penalty = number_of_edges`

Current path score:

`path_score = 0.45 * root_reach + 0.30 * support_span + 0.35 * rel_path - 0.10 * length_penalty`

Top `path_budget` paths are kept.

## 12. Semantic Association

Semantic association is implemented by [semantic_association()](/Users/Admin/projects/Association/associative_rag_project/association.py).

Its purpose is to add nearby graph content that improves coverage or relation diversity without expanding blindly.

### 12.1 Candidate generation

Candidates come from:

- immediate graph neighbors of current nodes
- chunks already supporting current nodes or edges
- nodes and edges connected to those chunks

### 12.2 Edge scoring

For each candidate edge:

- `query_rel`
- `support`
- `chunk_alignment`
- `info_gain`

`info_gain` is the increase in relation-category entropy if this edge category were added to the current edge set.

Current semantic edge score:

`semantic_edge_score = 0.55 * query_rel + 0.20 * support + 0.15 * chunk_alignment + 0.10 * info_gain`

Edges are retained only if:

- `score >= semantic_edge_min_score`
- and at least one of:
  - `query_rel >= semantic_edge_min_score / 2`
  - `support > 0`
  - `chunk_alignment >= 0.6`
  - `info_gain >= 0.15`

Top `semantic_edge_budget` edges are kept.

### 12.3 Node scoring

For each candidate node:

- `query_rel`
- `support`
- `chunk_alignment`
- `bridge_strength`

`bridge_strength` is based on how many neighbors connect the node back to the current graph or newly selected semantic nodes, capped at `1.0`.

Current semantic node score:

`semantic_node_score = 0.50 * query_rel + 0.20 * support + 0.15 * chunk_alignment + 0.15 * bridge_strength`

Nodes are skipped entirely if they have:

- no graph-bridge signal
- and no overlap with root-supporting chunks

Top `semantic_node_budget` nodes above `semantic_node_min_score` are kept.

## 13. Multi-round Graph Expansion

[expand_associative_graph()](/Users/Admin/projects/Association/associative_rag_project/association.py) alternates:

1. structural association
2. semantic association

for `association_rounds` rounds.

The function records round-level diagnostics:

- structural path count
- structural nodes/edges added
- semantic nodes/edges added
- current graph size after the round
- previews of selected paths, semantic edges, and semantic nodes

These diagnostics are stored in each retrieval JSON under `rounds`.

## 14. Node/Edge Role Tagging

After expansion, the pipeline tags:

- nodes as `root`, `structural`, or `semantic`
- edges as `root`, `structural`, or `semantic`

This is done by:

- [build_node_role_sets()](/Users/Admin/projects/Association/associative_rag_project/association.py)
- [build_edge_role_sets()](/Users/Admin/projects/Association/associative_rag_project/association.py)

These roles are later used in:

- evidence presentation
- ranking entities and relations for the prompt
- statistics

## 15. Knowledge Group Construction

Knowledge groups are built by [build_knowledge_groups()](/Users/Admin/projects/Association/associative_rag_project/context.py).

### 15.1 Group base unit

The final selected subgraph is converted into an undirected graph, and connected components become initial group candidates.

Each candidate group contains:

- component nodes
- component edges
- supporting chunk ids
- relation themes
- source previews

### 15.2 Group score

For each group, the code computes:

- `query_rel`: lexical overlap between the query and a group text built from nodes, relation themes, and source previews
- `root_density`: fraction of group nodes/edges that came from root roles
- `structure_density`: fraction of group nodes/edges that came from structural roles
- `support_span = min(num_supporting_chunks / 8, 1.0)`
- `size_term = min((num_nodes + num_edges) / 14, 1.0)`
- `relation_term = min(num_relation_categories / 4, 1.0)`
- `technical_penalty`

Current group score:

`group_score = 0.42 * query_rel + 0.18 * support_span + 0.12 * size_term + 0.10 * relation_term + 0.10 * root_density + 0.08 * structure_density - technical_penalty`

Additional penalties:

- `-0.06` if supporting chunks < 2 and nodes < 4
- `-0.04` if there are no relation themes and `query_rel < 0.05`

Groups are sorted by:

1. `group_score`
2. `node_count`
3. first node id

Then truncated to `group_limit`.

### 15.3 Group summary

Each group receives a lightweight textual summary from [_build_group_summary()](/Users/Admin/projects/Association/associative_rag_project/context.py), which currently uses:

- lead entities
- top relation themes
- chunk count
- first source preview

This is a rule-based summary, not an extra LLM call.

## 16. Final Prompt Context Construction

Prompt-context construction is implemented by [build_prompt_context()](/Users/Admin/projects/Association/associative_rag_project/context.py).

### 16.1 Supporting-chunk ranking

[rank_supporting_chunks()](/Users/Admin/projects/Association/associative_rag_project/context.py) scores chunks by how much final graph evidence they support:

- `+1` for each final node supported
- `+2` for each final edge supported
- `+1000` if the chunk is a root chunk

This extremely strong root boost is deliberate: it keeps final answers grounded in the initial dense retrieval.

### 16.2 Diverse source selection

[choose_diverse_source_chunks()](/Users/Admin/projects/Association/associative_rag_project/context.py) currently selects sources in two stages:

1. per-group stage
   - try to add one root chunk for the group
   - then one non-root support chunk for the group
2. fallback stage
   - fill remaining budget from the global chunk ranking

Selection is constrained by:

- `max_source_chunks`
- `max_source_word_budget`

### 16.3 Prompt sections

The current final prompt context contains:

- `Root Chunks`
- `Focused Entities`
- `Focused Relations`
- `Knowledge Groups`
- `Knowledge Group Dossiers`
- `Sources`

The `Knowledge Group Dossiers` section is the most query-facing organization layer. Each group dossier includes:

- score
- node/edge counts
- summary
- themes
- key entities
- key relations
- linked sources as a CSV block

## 17. Answer Generation

Answer generation is implemented by:

- [build_generation_prompt()](/Users/Admin/projects/Association/associative_rag_project/llm_client.py)
- [generate_answers()](/Users/Admin/projects/Association/associative_rag_project/llm_client.py)

### 17.1 Current answering behavior

The generation prompt explicitly tells the model to:

- do query-focused summarization
- organize around major themes or aspects
- synthesize across sources
- prefer themes supported by multiple groups or sources
- read the evidence package group-first rather than table-first
- select the best `2-4` groups
- use `Sources` through linked source ids
- avoid centering on metadata unless the query explicitly asks for it

Additional prompt constraints are injected for:

- passage/character/narrative queries
- historical/socio-political queries
- `synthesis` query style
- `concrete` query style

### 17.2 Generation parallelism

[generate_answers()](/Users/Admin/projects/Association/associative_rag_project/llm_client.py) runs requests in a `ThreadPoolExecutor`.

By default:

- `max_workers = llm_client.max_concurrency`
- this comes from the loaded generation config unless overridden by CLI

## 18. Judge and Evaluation

LLM judging is implemented by [judge.py](/Users/Admin/projects/Association/associative_rag_project/judge.py).

### 18.1 Judge prompt

The code uses an FG-RAG-compatible evaluation prompt with four outputs:

- `Comprehensiveness`
- `Diversity`
- `Empowerment`
- `Overall Winner`

### 18.2 Double-order judging

Each answer pair is judged twice:

1. `candidate` as Answer 1 vs `baseline` as Answer 2
2. `baseline` as Answer 1 vs `candidate` as Answer 2

The second result is remapped to reduce position bias.

### 18.3 Robust parsing

The judge parser:

- strips prose/code-fence wrappers
- retries up to `3` times on malformed JSON
- falls back to a tie verdict if parsing repeatedly fails

### 18.4 Output statistics

The judge outputs:

- overall win/loss/tie summary
- per-criterion vote counts and probabilities
- full verdict objects for each query

## 19. Output JSON Structure

A retrieval JSON currently contains, per query:

- root and candidate chunks
- query style
- adaptive profile
- stats
- top root nodes and edges
- round-by-round expansion diagnostics
- structural association output
- semantic association output
- knowledge groups
- final `prompt_context`

An answer JSON contains:

- `group_id`
- `query`
- `query_style`
- `model_answer`
- `stats`

A judge JSON contains:

- `summary`
- `criteria_summary`
- `verdicts`

## 20. Important Current Design Properties

Several implementation properties are especially important for interpreting results:

1. The system is chunk-first, not graph-entry-first.
2. Hybrid retrieval is the current default.
3. Root chunks are very strongly favored in final source selection because of the `+1000` grounding boost.
4. Adaptive control is currently weak steering, not a full policy controller.
5. Knowledge groups are connected-component packages over the final selected graph, not externally supervised topic clusters.
6. Group summaries are currently rule-based, not LLM-generated.
7. The final generation prompt is group-first, but root chunks still heavily influence the final evidence package.

## 21. Current Default End-to-End Command

The current default experimental pattern is effectively:

```bash
python -m associative_rag_project.main run-all \
  --corpus-dir <corpus> \
  --retrieval-mode hybrid \
  --top-chunks 5 \
  --top-root-nodes 12 \
  --top-root-edges 16 \
  --max-hop 4 \
  --path-budget 12 \
  --semantic-edge-budget 20 \
  --semantic-node-budget 12 \
  --semantic-edge-min-score 0.03 \
  --semantic-node-min-score 0.03 \
  --association-rounds 2 \
  --group-limit 8 \
  --max-source-chunks 14 \
  --max-source-word-budget 4500
```

with adaptive control enabled unless `--disable-adaptive-control` is passed.

## 22. What This Report Does Not Cover

This report intentionally does not describe:

- idealized method narratives
- proposed future changes
- paper-level claims about why the method should work

It only describes the current executable implementation and its default parameterization.
