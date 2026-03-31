# Associative RAG Project

Independent research prototype for query-focused summarization built around:

- `chunk-first` anchoring
- relation-aware structural association
- semantic-gain association
- adaptive control of association strength
- knowledge-group context construction
- OpenAI-compatible answer generation
- FG-RAG-style pairwise evaluation

## Overview

The main retrieval flow is:

1. retrieve root chunks with `bm25`, `dense`, or `hybrid`
2. rerank root chunks with query-aware and graph-aware signals
3. project root chunks into the entity-relation graph
4. expand with alternating `structural association` and `semantic association`
5. package the final graph into `knowledge groups`
6. build a final evidence package for answer generation

The adaptive controller can shrink or expand association budgets per query based on:

- generic query-form cues
- root graph density
- root graph fragmentation
- support dispersion
- retrieval score cliff

## Code Map

- [main.py](/Users/Admin/projects/Association/associative_rag_project/main.py): CLI entrypoint
- [pipeline.py](/Users/Admin/projects/Association/associative_rag_project/pipeline.py): end-to-end retrieval pipeline
- [retrieval.py](/Users/Admin/projects/Association/associative_rag_project/retrieval.py): chunk retrieval and root scoring
- [association.py](/Users/Admin/projects/Association/associative_rag_project/association.py): structural and semantic association
- [adaptive_control.py](/Users/Admin/projects/Association/associative_rag_project/adaptive_control.py): per-query association-strength controller
- [context.py](/Users/Admin/projects/Association/associative_rag_project/context.py): knowledge groups and prompt context assembly
- [llm_client.py](/Users/Admin/projects/Association/associative_rag_project/llm_client.py): answer generation
- [judge.py](/Users/Admin/projects/Association/associative_rag_project/judge.py): FG-RAG-style evaluation
- [analyze_query_shape.py](/Users/Admin/projects/Association/associative_rag_project/analyze_query_shape.py): inspect retrieval-side signals against manual broad/focused judgments
- [data.py](/Users/Admin/projects/Association/associative_rag_project/data.py): corpus / question / baseline loading
- [config.py](/Users/Admin/projects/Association/associative_rag_project/config.py): shared config loading from `lightrag/llm_config.py`

## Data Assumptions

Each corpus directory is expected to contain at least:

- `graph_chunk_entity_relation.graphml`
- `kv_store_text_chunks.json`

Dense retrieval additionally expects:

- `vdb_chunks.json`

Question files are resolved from:

- `datasets/questions/<corpus>_questions.txt`

Baseline files default to:

- `FG-RAG/<corpus>/output/FG-RAG-4o-mini.json`

You can override both with explicit CLI arguments.

## Main Commands

### 1. End-to-End Run

Run retrieval, answer generation, and pairwise judging in one command:

```bash
python -m associative_rag_project.main run-all \
  --corpus-dir agriculture \
  --max-workers 4
```

This writes files like:

- `associative_rag_project/runs/agriculture_top5_hop4_assoc_project_retrieval.json`
- `associative_rag_project/runs/agriculture_top5_hop4_assoc_project_answers.json`
- `associative_rag_project/runs/agriculture_top5_hop4_assoc_project_vs_FG-RAG-4o-mini_winrate.json`

### 2. Retrieval Only

Useful when you want to inspect graph growth, root chunks, and evidence packages before generating answers.

```bash
python -m associative_rag_project.main retrieve \
  --corpus-dir mix \
  --max-workers 4
```

### 3. Answer Only

Generate answers from an existing retrieval file:

```bash
python -m associative_rag_project.main answer \
  --retrieval-file associative_rag_project/runs/mix_top5_hop4_assoc_project_retrieval.json \
  --max-workers 4
```

### 4. Judge Only

Compare a candidate answer file against a baseline:

```bash
python -m associative_rag_project.main judge \
  --questions-file datasets/questions/mix_questions.txt \
  --candidate-file associative_rag_project/runs/mix_top5_hop4_assoc_project_answers.json \
  --baseline-file FG-RAG/mix/output/FG-RAG-4o-mini.json \
  --output-file associative_rag_project/runs/mix_top5_hop4_assoc_project_vs_FG-RAG-4o-mini_winrate.json \
  --max-workers 4
```

## Common Examples

### Small Sanity Run

Run only the first 5 questions:

```bash
python -m associative_rag_project.main run-all \
  --corpus-dir legal \
  --limit-groups 5 \
  --max-workers 4
```

### Hybrid Retrieval

This is the default mode:

```bash
python -m associative_rag_project.main retrieve \
  --corpus-dir art \
  --retrieval-mode hybrid
```

### Dense-Only Retrieval

```bash
python -m associative_rag_project.main retrieve \
  --corpus-dir art \
  --retrieval-mode dense
```

### BM25-Only Retrieval

```bash
python -m associative_rag_project.main retrieve \
  --corpus-dir art \
  --retrieval-mode bm25
```

### No-Adaptive Ablation

Run the same pipeline with fixed association strength:

```bash
python -m associative_rag_project.main run-all \
  --corpus-dir agriculture \
  --max-workers 4 \
  --disable-adaptive-control \
  --output-dir associative_rag_project/runs_noadaptive \
  --answer-output-file associative_rag_project/runs_noadaptive/agriculture_top5_hop4_assoc_project_answers_noadaptive.json \
  --judge-output-file associative_rag_project/runs_noadaptive/agriculture_top5_hop4_assoc_project_vs_FG-RAG-4o-mini_winrate_noadaptive.json
```

### Query-Shape Inspection

Inspect which queries sit at the low / high end of each retrieval-side metric:

```bash
python associative_rag_project/analyze_query_shape.py \
  associative_rag_project/runs_linear2_60/art_top5_hop4_assoc_project_retrieval.json \
  associative_rag_project/runs_linear2_60/agriculture_top5_hop4_assoc_project_retrieval.json
```

Export a CSV template for manual `broad / focused / mixed` annotation:

```bash
python associative_rag_project/analyze_query_shape.py \
  associative_rag_project/runs_linear2_60/art_top5_hop4_assoc_project_retrieval.json \
  associative_rag_project/runs_linear2_60/agriculture_top5_hop4_assoc_project_retrieval.json \
  --top-n 10 \
  --export-csv associative_rag_project/query_shape_annotation_template.csv
```

### Custom Retrieval Budget

```bash
python -m associative_rag_project.main retrieve \
  --corpus-dir mix \
  --top-chunks 5 \
  --max-hop 4 \
  --path-budget 12 \
  --semantic-edge-budget 20 \
  --semantic-node-budget 12
```

### Custom Output Directory

```bash
python -m associative_rag_project.main run-all \
  --corpus-dir legal \
  --output-dir associative_rag_project/exp_legal_v2 \
  --max-workers 4
```

## Important CLI Arguments

Retrieval-related:

- `--retrieval-mode {bm25,dense,hybrid}`
- `--top-chunks`
- `--chunk-candidate-multiplier`
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
- `--disable-adaptive-control`

Run-control:

- `--limit-groups`
- `--max-workers`
- `--output-dir`
- `--questions-file`
- `--baseline-file`

## Output Files

### Retrieval JSON

Contains:

- selected root chunks
- adaptive profile
- round-by-round structural / semantic association traces
- final graph size statistics
- prompt context

Typical file:

- [mix_top5_hop4_assoc_project_retrieval.json](/Users/Admin/projects/Association/associative_rag_project/runs/mix_top5_hop4_assoc_project_retrieval.json)

### Answer JSON

Contains:

- `group_id`
- `query`
- `query_style`
- `model_answer`
- retrieval statistics

Typical file:

- [mix_top5_hop4_assoc_project_answers.json](/Users/Admin/projects/Association/associative_rag_project/runs/mix_top5_hop4_assoc_project_answers.json)

### Winrate JSON

Contains:

- `summary`
- `criteria_summary`
- per-query `verdicts`

`criteria_summary` follows the FG-RAG evaluation dimensions:

- `Comprehensiveness`
- `Diversity`
- `Empowerment`
- `Overall Winner`

Typical file:

- [mix_top5_hop4_assoc_project_vs_FG-RAG-4o-mini_winrate.json](/Users/Admin/projects/Association/associative_rag_project/runs/mix_top5_hop4_assoc_project_vs_FG-RAG-4o-mini_winrate.json)

## Evaluation Notes

- The judge uses the FG-RAG evaluation prompt template from [EvaluatorPrompt.py](/Users/Admin/projects/Association/FG-RAG/prompt/EvaluatorPrompt.py).
- Judging is done in both answer orders to reduce position bias.
- The output keeps both overall win rate and per-dimension probabilities.

## Config Notes

- Generation and embedding endpoints are loaded from [llm_config.py](/Users/Admin/projects/Association/lightrag/llm_config.py).
- Judge model defaults to `gpt-5.4-mini`.
- You can override generation / embedding / judge settings with environment variables consumed by [config.py](/Users/Admin/projects/Association/associative_rag_project/config.py).

## Practical Workflow

For normal experiments:

1. run `retrieve` first when debugging graph behavior
2. inspect retrieval JSON or sample context
3. run `answer`
4. run `judge`
5. only use `run-all` once the setup is stable

For ablations:

1. keep the same corpus and output naming pattern
2. change only one switch at a time
3. place ablation outputs in a different directory such as `runs_noadaptive`
