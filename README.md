# Association

<div align="center">

## Associative RAG for Query-Focused Summarization

Query-focused summarization over chunk-entity-relation graphs with multi-round associative retrieval, contract-aware organization, and LLM-as-a-judge evaluation.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

## Overview

This repository is a research workspace centered on [associative_rag_project](/Users/Admin/projects/Association/associative_rag_project), a query-focused summarization system built on top of document chunks and a chunk-entity-relation graph.

The current system follows one unified retrieval-and-association backbone:

1. retrieve chunk candidates with `bm25`, `dense`, or `hybrid`
2. inject graph-side recall through `graph_focus` and `graph_keyword`
3. select diverse root chunks
4. expand the evidence graph with multi-round chunk-level association
5. organize the final subgraph into answer facets
6. pack grounded evidence for the answer LLM
7. compare candidate answers against FG-RAG-style baselines with a contract-aware judge

The repository also includes:

- baseline or reference code such as [lightrag](/Users/Admin/projects/Association/lightrag) and [FG-RAG](/Users/Admin/projects/Association/FG-RAG)
- local corpora under [Datasets](/Users/Admin/projects/Association/Datasets)
- experiment logs and outputs under the project directories

## Highlights

- Unified associative retrieval for all query contracts
- Multi-round root reseeding instead of one-shot graph crawling
- Chunk-level bridge/support/peripheral scheduling
- Theme-style broad evidence expansion with section/mechanism/comparison-specific organization
- Contract-aware LLM judge with `Comprehensiveness`, `Diversity`, `Focus Match`, `Evidence Anchoring`, `Scope Discipline`, and `Scenario Fidelity`

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

Edit [llm_config.py](/Users/Admin/projects/Association/llm_config.py) so that:

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

End-to-end run:

```bash
python -m associative_rag_project.main run-all \
  --corpus-dir Datasets/art/index \
  --output-dir associative_rag_project/runs_demo \
  --max-workers 4
```

Retrieval only:

```bash
python -m associative_rag_project.main retrieve \
  --corpus-dir Datasets/art/index \
  --output-dir associative_rag_project/runs_demo \
  --retrieval-mode dense
```

Judge only:

```bash
python -m associative_rag_project.main judge \
  --questions-file Datasets/art/query/art.json \
  --candidate-file associative_rag_project/runs_demo/art_top5_hop4_assoc_project_answers.json \
  --baseline-file Datasets/art/output/FG-RAG-4o-mini.json \
  --output-file associative_rag_project/runs_demo/art_top5_hop4_assoc_project_vs_FG-RAG-4o-mini_winrate.json
```

## Output Files

For a corpus named `art`, the pipeline typically writes:

- `art_top5_hop4_assoc_project_retrieval.json`
- `art_top5_hop4_assoc_project_answers.json`
- `art_top5_hop4_assoc_project_vs_FG-RAG-4o-mini_winrate.json`
- `art_top5_hop4_assoc_project_sample_context.txt`

## Documentation

- Project guide: [associative_rag_project/README.md](/Users/Admin/projects/Association/associative_rag_project/README.md)
- Detailed Chinese technical report: [associative_rag_project/TECHNICAL_REPORT_CN.md](/Users/Admin/projects/Association/associative_rag_project/TECHNICAL_REPORT_CN.md)
- Chinese method draft: [associative_rag_project/METHOD_DRAFT_CN.md](/Users/Admin/projects/Association/associative_rag_project/METHOD_DRAFT_CN.md)

## Current Positioning

This codebase is a research prototype rather than a polished library package. The emphasis is:

- inspecting retrieval traces
- iterating on graph association logic
- studying contract-aware answer organization
- evaluating against FG-RAG-style baselines

If you want the shortest path to the current implementation, start with:

- [associative_rag_project/main.py](/Users/Admin/projects/Association/associative_rag_project/main.py)
- [associative_rag_project/pipeline.py](/Users/Admin/projects/Association/associative_rag_project/pipeline.py)
- [associative_rag_project/TECHNICAL_REPORT_CN.md](/Users/Admin/projects/Association/associative_rag_project/TECHNICAL_REPORT_CN.md)
