# Pre-Experiment Transfer Notes

This repo now keeps pre-experiment scripts under `lightrag/` (same as your production layout).

## Files To Sync To Production

- `lightrag/pre_experiment_main.py`
- `lightrag/generate_query_rewrites.py`
- `lightrag/query_trace.py`
- `lightrag/operate.py`
- `lightrag/utils.py`
- `lightrag/llm.py`

## Environment

Set these before running:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_API_BASE="https://gptgod.cloud/v1"
```

PowerShell:

```powershell
$env:OPENAI_API_KEY="your-key"
$env:OPENAI_API_BASE="https://gptgod.cloud/v1"
```

## Run Order (from `lightrag/` directory)

Generate rewrites first:

```bash
python generate_query_rewrites.py \
  --questions-file ../datasets/questions/agriculture_questions.txt \
  --output-file agriculture_rewrites.json \
  --limit 10 \
  --num-rewrites 4
```

Then run pre-experiment:

```bash
python pre_experiment_main.py \
  --working-dir ../agriculture \
  --rewrites-file agriculture_rewrites.json \
  --mode hybrid \
  --output-file agriculture_pre_experiment.json
```

## Outputs

- `pre_experiment_main.py` writes query-level outputs to `--output-file`.
- LightRAG writes trace logs to `<working-dir>/query_trace.log`.

Each trace line is JSON with core fields:

- `query`
- `mode`
- `keywords` (LLM-generated keyword entry)
- `v0` (first-stage local/global hits)
- `subgraph` (`num_entities`, `num_relations`, `num_sources`)
