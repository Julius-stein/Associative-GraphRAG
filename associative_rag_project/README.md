# Associative RAG Project

Research prototype for query-focused summarization over chunk-entity-relation graphs.

## What This Module Does

The current implementation answers a query with the following pipeline:

1. detect the query contract
2. retrieve chunk candidates from lexical/dense retrieval plus graph-side recall
3. select diverse root chunks
4. run multi-round chunk-level association on the graph
5. organize the final subgraph into facet groups
6. pack evidence into a prompt context
7. generate the final answer with an OpenAI-compatible LLM
8. judge the answer against a baseline with a contract-aware LLM judge

The important architectural choice in the current code is:

- all contracts now share one theme-style retrieval and association backbone
- the contracts mainly diverge at the organization layer
- `section-grounded` keeps a strict single-`full_doc_id` constraint
- `mechanism-grounded` uses pathway-style grouping
- `comparison-grounded` uses side/axis-style grouping
- `theme-grounded` uses slot-based multi-aspect grouping

## Code Map

- [main.py](/Users/Admin/projects/Association/associative_rag_project/main.py): CLI entrypoint
- [pipeline.py](/Users/Admin/projects/Association/associative_rag_project/pipeline.py): end-to-end retrieval pipeline
- [data.py](/Users/Admin/projects/Association/associative_rag_project/data.py): corpus / query loading and graph-chunk mappings
- [retrieval.py](/Users/Admin/projects/Association/associative_rag_project/retrieval.py): BM25/dense retrieval, graph-side recall, root selection, root node/edge scoring
- [association.py](/Users/Admin/projects/Association/associative_rag_project/association.py): multi-round chunk association and root reseeding
- [organization.py](/Users/Admin/projects/Association/associative_rag_project/organization.py): contract detection, region construction, facet grouping
- [context.py](/Users/Admin/projects/Association/associative_rag_project/context.py): prompt context assembly and source packing
- [llm_client.py](/Users/Admin/projects/Association/associative_rag_project/llm_client.py): answer prompt construction and generation
- [judge.py](/Users/Admin/projects/Association/associative_rag_project/judge.py): pairwise FG-RAG-style evaluation

## Data Assumptions

Each indexed corpus under `Datasets/<name>/index` should provide:

- `graph_chunk_entity_relation.graphml`
- `kv_store_text_chunks.json`
- `vdb_chunks.json` if dense or hybrid retrieval is used

Typical query file:

- `Datasets/<name>/query/<name>.json`

Typical baseline answer file:

- `Datasets/<name>/output/FG-RAG-4o-mini.json`

## Main Commands

### Retrieve

```bash
python -m associative_rag_project.main retrieve \
  --corpus-dir Datasets/art/index \
  --output-dir associative_rag_project/runs_demo \
  --retrieval-mode dense
```

### Answer

```bash
python -m associative_rag_project.main answer \
  --retrieval-file associative_rag_project/runs_demo/art_top5_hop4_assoc_project_retrieval.json \
  --output-file associative_rag_project/runs_demo/art_top5_hop4_assoc_project_answers.json
```

### Judge

```bash
python -m associative_rag_project.main judge \
  --questions-file Datasets/art/query/art.json \
  --candidate-file associative_rag_project/runs_demo/art_top5_hop4_assoc_project_answers.json \
  --baseline-file Datasets/art/output/FG-RAG-4o-mini.json \
  --output-file associative_rag_project/runs_demo/art_top5_hop4_assoc_project_vs_FG-RAG-4o-mini_winrate.json
```

### End-to-End

```bash
python -m associative_rag_project.main run-all \
  --corpus-dir Datasets/art/index \
  --output-dir associative_rag_project/runs_demo \
  --max-workers 4
```

## Useful CLI Arguments

Retrieval:

- `--retrieval-mode {bm25,dense,hybrid}`
- `--top-chunks`
- `--chunk-candidate-multiplier`
- `--candidate-pool-size`
- `--dense-weight`
- `--bm25-weight`
- `--top-root-nodes`
- `--top-root-edges`
- `--max-hop`
- `--path-budget`
- `--semantic-edge-budget`
- `--semantic-node-budget`
- `--semantic-edge-min-score`
- `--semantic-node-min-score`
- `--association-rounds`
- `--group-limit`
- `--max-source-chunks`
- `--max-source-word-budget`

Embedding:

- `--embedding-provider {openai_compatible,bge_m3_local}`
- `--embedding-model`
- `--embedding-base-url`
- `--embedding-api-key`
- `--local-embedding-model`
- `--local-embedding-device`
- `--local-embedding-batch-size`

Execution:

- `--limit-groups`
- `--output-dir`
- `--max-workers`

## Current Retrieval and Organization Logic

### 1. Unified candidate retrieval

`run_query()` in [pipeline.py](/Users/Admin/projects/Association/associative_rag_project/pipeline.py) first retrieves chunk candidates from:

- the primary retriever: `bm25`, `dense`, or `hybrid`
- `search_graph_focus_chunks()`
- `search_graph_keyword_chunks()`

The merged candidate pool is therefore both query-relevant and graph-aware.

### 2. Diverse root selection

[select_diverse_root_chunks()](/Users/Admin/projects/Association/associative_rag_project/retrieval.py) selects a small set of high-value roots while discouraging:

- same-doc local-band duplication
- provenance overlap
- staying inside one query-term basin

### 3. Multi-round association

[expand_associative_graph()](/Users/Admin/projects/Association/associative_rag_project/association.py) now routes all contracts into the theme-style chunk association loop:

- bridge chunks connect current roots to new graph areas
- support chunks bring new information mass
- peripheral chunks add local graph context
- root seeds are refreshed after each round

### 4. Contract-specific grouping

[build_answer_facet_groups()](/Users/Admin/projects/Association/associative_rag_project/organization.py) is where contracts diverge:

- `section-grounded`: section bands inside one `full_doc_id`
- `mechanism-grounded`: `pathway: ...` facets
- `comparison-grounded`: `comparison side ...` and `contrast axis ...` facets
- `theme-grounded`: slot-based aspect groups such as examples, drivers, contexts, outcomes

## Output Artifacts

Each retrieval record includes:

- `candidate_root_chunks`
- `root_chunks`
- `promoted_root_chunks`
- `theme_selected_chunks`
- `facet_groups`
- `candidate_points`
- `prompt_context`
- per-query `stats`

This makes the system easy to inspect when you want to debug whether a problem comes from retrieval, association, organization, or prompting.

## Recommended Reading Order

1. [pipeline.py](/Users/Admin/projects/Association/associative_rag_project/pipeline.py)
2. [retrieval.py](/Users/Admin/projects/Association/associative_rag_project/retrieval.py)
3. [association.py](/Users/Admin/projects/Association/associative_rag_project/association.py)
4. [organization.py](/Users/Admin/projects/Association/associative_rag_project/organization.py)
5. [context.py](/Users/Admin/projects/Association/associative_rag_project/context.py)
6. [llm_client.py](/Users/Admin/projects/Association/associative_rag_project/llm_client.py)
7. [judge.py](/Users/Admin/projects/Association/associative_rag_project/judge.py)

## Detailed Report

For the algorithm-level Chinese report, see:

- [TECHNICAL_REPORT_CN.md](/Users/Admin/projects/Association/associative_rag_project/TECHNICAL_REPORT_CN.md)
