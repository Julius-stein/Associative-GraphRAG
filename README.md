# Association

<div align="center">

## Associative RAG for Query-Focused Summarization

Query-focused summarization over chunk-entity-relation graphs with chunk-centric multi-round association, trace-grounded knowledge grouping, and LLM-as-a-judge evaluation.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

## Overview

This repository is a research workspace centered on [associative_rag_project](/Users/Admin/projects/Association/associative_rag_project), a query-focused summarization system built on top of document chunks and a chunk-entity-relation graph.

The current system follows one unified QFS retrieval-and-organization backbone:

1. retrieve chunk candidates with `bm25`, `dense`, or `hybrid`
2. inject graph-side recall through `graph_focus` and `graph_keyword`
3. select diverse root chunks
4. expand the evidence graph with multi-round chunk-centric association and root reseeding
5. build root traces and compress the final graph into edge-skeleton plus evidence-dossier knowledge groups
6. pack grounded evidence for the answer LLM
7. compare candidate answers against one or many baselines with QFS-oriented pairwise and groundedness judges

The repository also includes:

- baseline or reference code such as [lightrag](/Users/Admin/projects/Association/lightrag) and [FG-RAG](/Users/Admin/projects/Association/FG-RAG)
- local corpora under [Datasets](/Users/Admin/projects/Association/Datasets)
- experiment logs and outputs under the project directories

## Highlights

- Unified associative retrieval for QFS
- Chunk-centric multi-round expansion with per-round root reseeding
- Explicit `root / bridge / support / context` chunk roles
- Root-trace-preserving organization into edge skeletons and evidence dossiers
- Degree-normalized graph features to reduce high-connectivity basin traps
- QFS-oriented LLM judge with `Comprehensiveness`, `Diversity`, `Empowerment`, `Overall`, and claim groundedness support

## Repository Layout

- [associative_rag_project](/Users/Admin/projects/Association/associative_rag_project): main research system
- [Datasets](/Users/Admin/projects/Association/Datasets): indexed corpora and query files
- [lightrag](/Users/Admin/projects/Association/lightrag): upstream/reference code kept in the workspace
- [FG-RAG](/Users/Admin/projects/Association/FG-RAG): baseline outputs and utilities
- [examples](/Users/Admin/projects/Association/examples): auxiliary examples

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Configure models

Create or edit the local, git-ignored [llm_config.py](/Users/Admin/projects/Association/llm_config.py) so that:

- answer generation points to your preferred chat model
- judge generation points to your preferred judge model
- dense retrieval uses a compatible embedding backend if `--retrieval-mode dense` or `hybrid`

### 3. Check corpus layout

Each corpus should live under `Datasets/<name>/index` and provide:

- `graph_chunk_entity_relation.graphml`
- `kv_store_text_chunks.json`
- `vdb_chunks.json` for dense or hybrid retrieval

Queries are typically under `Datasets/<name>/query/<name>.json`.

### 4. Run the pipeline

End-to-end `retrieve -> answer -> judge`:

```bash
python -m associative_rag_project.main run-all \
  --corpus-dir Datasets/art/index \
  --output-dir associative_rag_project/runs_v3_ppr \
  --max-workers 4
```

Retrieval only:

```bash
python -m associative_rag_project.main retrieve \
  --corpus-dir Datasets/art/index \
  --output-dir associative_rag_project/runs_v3_ppr \
  --retrieval-mode dense
```

Retrieve + answer without judge:

```bash
python -m associative_rag_project.main run \
  --corpus-dir Datasets/art/index \
  --output-dir associative_rag_project/runs_v3_ppr \
  --max-workers 4
```

Answer only:

```bash
python -m associative_rag_project.main answer \
  --retrieval-file associative_rag_project/runs_v3_ppr/art_top5_hop4_assoc_project_retrieval.json \
  --max-workers 4
```

Judge against a single baseline:

```bash
python -m associative_rag_project.main judge \
  --questions-file Datasets/art/query/art.json \
  --candidate-file associative_rag_project/runs_v3_ppr/art_top5_hop4_assoc_project_answers.json \
  --baseline-file Datasets/art/output/FG-RAG-4o-mini.json \
  --output-file associative_rag_project/runs_v3_ppr/art_top5_hop4_assoc_project_vs_FG-RAG-4o-mini_winrate.json
```

Judge against all baselines in one directory and write a combined Markdown summary:

```bash
python -m associative_rag_project.main judge \
  --questions-file Datasets/agriculture/query/agriculture.json \
  --candidate-file associative_rag_project/runs_v3_ppr/agriculture_top5_hop4_assoc_project_answers.json \
  --corpus-dir Datasets/agriculture/index/ \
  --baseline-dir Datasets/agriculture/output \
  --candidate-label assoc-project \
  --summary-file associative_rag_project/result/agriculture_baseline_winrate_tables.md \
  --max-workers 12
```

Useful options:

- `--retrieval-mode bm25|dense|hybrid`
- `--limit-groups` or `--limit` on `retrieve/run/run-all`
- `--baseline-dir` on `judge` for bulk baseline comparison
- `--summary-file` on `judge` to render one Markdown file containing all win-rate tables

## Output Files

For a corpus named `art`, the pipeline typically writes:

- `art_top5_hop4_assoc_project_retrieval.json`
- `art_top5_hop4_assoc_project_answers.json`
- `art_top5_hop4_assoc_project_vs_FG-RAG-4o-mini_winrate.json`
- `art_top5_hop4_assoc_project_sample_context.txt`

When `judge --baseline-dir` is used, the project also writes:

- one `*_vs_<baseline>_winrate.json` per baseline file
- one combined Markdown table file such as `agriculture_baseline_winrate_tables.md`
- one JSON manifest beside that Markdown summary

## Documentation

- Detailed Chinese technical report: [associative_rag_project/TECHNICAL_REPORT_CN.md](/Users/Admin/projects/Association/associative_rag_project/TECHNICAL_REPORT_CN.md)
- Chinese method draft: [associative_rag_project/METHOD_DRAFT_CN.md](/Users/Admin/projects/Association/associative_rag_project/METHOD_DRAFT_CN.md)
- Chinese figure guide: [associative_rag_project/METHOD_FIGURE_GUIDE_CN.md](/Users/Admin/projects/Association/associative_rag_project/METHOD_FIGURE_GUIDE_CN.md)

## Current Positioning

This codebase is a research prototype rather than a polished library package. The emphasis is:

- inspecting retrieval traces
- iterating on chunk-centric association logic
- studying graph-based knowledge grouping for QFS
- comparing against multiple external baselines in a unified judge format

If you want the shortest path to the current implementation, start with:

- [associative_rag_project/main.py](/Users/Admin/projects/Association/associative_rag_project/main.py)
- [associative_rag_project/pipeline.py](/Users/Admin/projects/Association/associative_rag_project/pipeline.py)
- [associative_rag_project/association.py](/Users/Admin/projects/Association/associative_rag_project/association.py)
- [associative_rag_project/organization.py](/Users/Admin/projects/Association/associative_rag_project/organization.py)
- [associative_rag_project/TECHNICAL_REPORT_CN.md](/Users/Admin/projects/Association/associative_rag_project/TECHNICAL_REPORT_CN.md)
