# Next Steps To Complete GraphJailbreakBench

This project now has a real-source dataset foundation from AdvBench,
JailbreakBench, and HarmBench. The remaining work is about completing the labels,
models, and paper-grade validation promised in the Excel workbook.

## Current Readiness

- Expanded processed dataset: `data/processed/gjb_real_v1.jsonl`.
- Total rows: 3,599.
- Jailbreak/harmful rows: 2,999, within the workbook target of 2,000-3,000.
- Safe rows: 600, matching the workbook target.
- Synthetic source rows: 0.
- Baseline experiments can run now.
- RQ2 and E5 runner code is now implemented and will automatically produce
  final metrics when real label files are populated.
- Final paper-grade RQ0/RQ2/E5 still require human annotation, cross-model
  success labels, and real multi-turn rows.

## 1. Lock the data provenance

- Keep `configs/data_sources.json` as the source manifest.
- Run `python scripts/verify_sources.py` before every major experiment.
- Run `python scripts/export_proof_bundle.py` after every major experiment batch.
- Do not manually edit `data/raw/*`; regenerate processed files through
  `scripts/build_dataset.py`.

## 2. Complete RQ0 annotation validation

- Select 200 real harmful prompts from `data/processed/gjb_real_v1.jsonl`.
- Create independent annotation files for 2-3 annotators.
- Each annotator should assign graph nodes and edges from the 12-label taxonomy
  in `configs/taxonomy.json`.
- Run Cohen's kappa for node labels, edge kappa, and normalized graph edit
  distance.
- If node kappa is below 0.70 or edge kappa is below 0.60, revise the annotation
  guide before running final experiments.

## 3. Expand safe controls for RQ4

- The current real sources include 100 JailbreakBench benign rows.
- The workbook target is 600 safe prompts across six user-population-linked
  style groups.
- Add or source real benign prompts for formal/academic, informal/colloquial,
  non-native English, security/technical, medical/health, and creative writing.
- Keep subgroup labels in `safe_subgroup`.

## 4. Collect model response and success labels

- Run the real prompts through the approved target models.
- Store response text and success labels in the schema fields:
  `model_gpt4_response`, `model_gpt4_success`, `model_llama_response`,
  `model_llama_success`.
- Record model name, model version/date, decoding settings, and classifier or
  judge protocol in `metadata`.
- Import validated labels with `python scripts/import_model_labels.py`.
- Run transferability with `python scripts/run_rq2_transferability_experiments.py`.
- Do not fabricate success labels; E2 and E5 remain blocked until these are real.

## 5. Build the final E1 detection models

- Text-only baseline: BERT/RoBERTa or sentence-transformer embeddings with a
  classifier.
- Graph-only baseline: graph features first, then a GNN if needed.
- LR + graph features: depth, node counts, edge density, motif counts, node-label
  indicators.
- Hybrid: text embeddings concatenated with graph features.
- Report accuracy, precision, recall, and F1 overall and by difficulty.

## 6. Build E2 transferability prediction

- Use prompts with both Model A and Model B success labels.
- Compare text-feature and graph-feature classifiers.
- Report accuracy and ROC-AUC.
- Keep the split grouped by prompt/source to avoid leakage.
- The executable runner is `scripts/run_rq2_transferability_experiments.py`.

## 7. Complete E3 motif discovery

- Replace first-pass heuristic graph labels with validated human annotations
  where available.
- Enumerate motifs of size 2-4.
- Rank motifs by frequency, then by success rate once success labels exist.
- Export top-10 motif tables and small graph visualizations.

## 8. Complete E4 fairness / overblocking

- Use the expanded 600-row safe set.
- Compare false positive rates per subgroup for text-only and graph-based
  defenses.
- Report max-min FPR gap and FPR standard deviation.

## 9. Complete E5 multi-turn analysis

- Generate or collect real multi-turn variants with provenance.
- Import validated variants with `python scripts/import_multiturn_variants.py`.
- Run model evaluations on single-turn and multi-turn prompts.
- Report attack success rate only after real success labels exist.
- The executable runner is `scripts/run_e5_asr_analysis.py`.

## 10. Paper-ready packaging

- Keep all result CSVs in `results/tables`.
- Put figures in `results/figures`.
- Export proof bundles to `logs/proof`.
- Keep a frozen dependency file in `requirements-lock.txt`.
- Before submission, rerun source verification, tests, all experiments, and proof
  export from the local `.venv`.
