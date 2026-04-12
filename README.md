# GraphJailbreakBench

Research codebase for **Graph-Jailbreaking**, an AIES-oriented project studying
whether semantic graph structure provides complementary signal for jailbreak
detection, robustness analysis, motif discovery, and fairness-oriented
overblocking evaluation.

The project is driven by the experiment design in `Graph Jailbreaking.xlsx` and
currently builds from real public benchmark sources only.

## Scope

This repository includes:

- dataset source manifests and reproducibility checks;
- graph annotation taxonomy and first-pass graph feature extraction;
- detection baselines, source-balanced robustness experiments, and ablations;
- motif frequency/enrichment analysis and graph visualizations;
- fairness/overblocking baseline analysis across six safe-prompt style groups;
- proof bundles with command logs, file hashes, environment metadata, and result
  summaries.

The repository does **not** commit raw or processed prompt datasets. Those files
are reproduced locally from the source manifest and scripts.

## Data Sources

The local benchmark is built from these real sources:

- AdvBench (`llm-attacks/llm-attacks`)
- JailbreakBench JBB-Behaviors (`JailbreakBench/JBB-Behaviors`)
- HarmBench (`centerforaisafety/HarmBench`)
- JailbreakV-28K (`JailbreakV-28K/JailBreakV-28k`)
- JailbreakDB (`youbin2014/JailbreakDB`)

Source URLs, row counts, and checksums are recorded in
`configs/data_sources.json`.

## Current Status

- Real-source processed dataset: 3,599 rows locally.
- Jailbreak/harmful rows: 2,999.
- Safe rows: 600.
- Synthetic source rows: 0.
- E1 baseline and robustness experiments: completed.
- RQ3 motif frequency/enrichment: completed without success-rate labels.
- RQ4 six-group fairness baseline: completed with heuristic subgroup assignment.

Blocked items are intentionally not fabricated:

- RQ0 final annotation agreement requires filled human annotator templates.
- RQ2 transferability requires real cross-model success labels.
- RQ3 motif success rates require real model success labels.
- E5 multi-turn ASR requires real multi-turn variants and success labels.

## Project Layout

- `configs/`: experiment specs, taxonomy, and data-source manifests.
- `data/`: annotation templates and data recreation notes. Raw/processed prompt
  files are ignored by Git.
- `docs/`: project notes and next-step roadmap.
- `logs/proof/`: reproducibility proof summaries and command logs.
- `results/`: committed result summaries, tables, figures, and Google Sheets TSV
  exports.
- `scripts/`: reproducible data, experiment, plotting, and proof-export entry
  points.
- `src/gjb/`: reusable GraphJailbreakBench package code.
- `tests/`: unit tests for core graph, metrics, and annotation utilities.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
PIP_CACHE_DIR=.pip-cache python -m pip install -r requirements-dev.txt
```

The local `.venv`, `.pip-cache`, and `.hf-cache` directories are intentionally
ignored by Git.

## Reproduce Local Data

```bash
python scripts/verify_sources.py
python scripts/build_dataset.py --output data/processed/gjb_real_v1.jsonl
python scripts/assign_rq4_safe_subgroups.py \
  --dataset data/processed/gjb_real_v1.jsonl \
  --output data/processed/gjb_real_v1_rq4_balanced.jsonl
```

## Run Experiments

```bash
python scripts/run_baseline_experiments.py \
  --dataset data/processed/gjb_real_v1.jsonl \
  --output-dir results/tables

python scripts/run_e1_ml_baselines.py \
  --dataset data/processed/gjb_real_v1.jsonl \
  --output-dir results/rq/rq1

python scripts/run_e1_embedding_baselines.py \
  --dataset data/processed/gjb_real_v1.jsonl \
  --output-dir results/rq/rq1/embedding_baseline

python scripts/run_unblocked_experiments.py \
  --dataset data/processed/gjb_real_v1.jsonl \
  --balanced-dataset data/processed/gjb_real_v1_rq4_balanced.jsonl \
  --output-dir results/unblocked
```

## Proof and Validation

```bash
python -m unittest discover -s tests
python scripts/export_proof_bundle.py
```

The proof bundle records checksums, row counts, command outputs, and environment
metadata while avoiding duplicate raw prompt text in logs.

## Safety and Data Boundary

Raw benchmark prompts are kept in local ignored data files. Committed artifacts
use row IDs, prompt hashes, source names, aggregate metrics, and plots instead of
duplicating raw prompt text.
