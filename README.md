# GraphJailbreakBench

Research codebase for the graph-based jailbreak detection experiments described in
`Graph Jailbreaking.xlsx`.

This repository now builds from real source datasets only:

- AdvBench (`llm-attacks/llm-attacks`)
- JailbreakBench JBB-Behaviors (`JailbreakBench/JBB-Behaviors`)
- HarmBench (`centerforaisafety/HarmBench`)

Synthetic dataset generation has been removed. Model success labels and human
annotation labels are still separate requirements because the source CSVs do not
include GPT-4/LLaMA success outcomes or independent annotator graphs.

## Workbook Coverage

- RQ0 / E0: annotation validation helpers for node agreement, edge agreement,
  and normalized graph edit distance.
- RQ1 / E1: graph-vs-text detection scaffolding and reusable metrics.
- RQ2 / E2: transferability prediction scaffolding with graph/text features.
- RQ3 / E3: motif enumeration for directed graph patterns of size 2-4.
- RQ4 / E4: fairness / overblocking metrics for the six safe-prompt subgroups.
- E5: single-turn vs multi-turn attack-success-rate comparison.

## Project Layout

- `configs/`: workbook-derived experiment, dataset, and taxonomy specs.
- `configs/data_sources.json`: official source URLs, row counts, and checksums.
- `data/raw/`: downloaded AdvBench, JailbreakBench, and HarmBench source CSVs.
- `data/processed/`: cleaned/merged benchmark outputs.
- `docs/`: extracted workbook understanding and methodology notes.
- `src/gjb/`: reusable package code.
- `scripts/`: repeatable command-line entry points.
- `results/`: generated result tables and figures.
- `tests/`: lightweight correctness checks.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
PIP_CACHE_DIR=.pip-cache python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python3 scripts/build_dataset.py --output data/processed/gjb_real_v1.jsonl
python3 scripts/verify_sources.py
python3 scripts/run_baseline_experiments.py \
  --dataset data/processed/gjb_real_v1.jsonl \
  --output-dir results/tables
python3 scripts/export_proof_bundle.py
python3 -m unittest discover -s tests
```

## Data Boundary

The processed JSONL preserves real source prompts for research use. Scripts and
result tables avoid printing prompt text by default; motif tables reference row
IDs and sources instead of example prompts.
