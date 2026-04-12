#!/usr/bin/env python3
"""Export one saved result/status artifact per workbook research question."""

from __future__ import annotations

import csv
import json
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gjb.io import read_jsonl

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "gjb_real_v1.jsonl"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
OUTPUT_DIR = PROJECT_ROOT / "results" / "rq"


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DATASET_PATH.relative_to(PROJECT_ROOT)))
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = resolve_project_path(args.dataset)
    rows = read_jsonl(dataset_path)
    now = datetime.now(timezone.utc).isoformat()

    dataset_path_text = str(dataset_path.relative_to(PROJECT_ROOT)) if dataset_path.is_relative_to(PROJECT_ROOT) else str(dataset_path)
    dataset_summary = {
        "dataset_path": dataset_path_text,
        "total_rows": len(rows),
        "by_type": dict(sorted(Counter(row["type"] for row in rows).items())),
        "by_source_dataset": dict(sorted(Counter(row["source_dataset"] for row in rows).items())),
        "by_difficulty": dict(sorted(Counter(row["difficulty"] for row in rows).items())),
        "by_attack_type": dict(sorted(Counter(row["attack_type"] for row in rows).items())),
        "safe_subgroups": dict(sorted(Counter(row.get("safe_subgroup") for row in rows if row["type"] == "safe").items())),
        "model_gpt4_success_labels": sum(row.get("model_gpt4_success") is not None for row in rows),
        "model_llama_success_labels": sum(row.get("model_llama_success") is not None for row in rows),
        "multiturn_rows": sum(row["type"] == "multiturn" for row in rows),
        "prompt_text_logged": False,
    }

    rq0 = {
        "created_at_utc": now,
        "rq": "RQ0",
        "question": "Can jailbreak prompts be reliably decomposed into semantic graphs using a structured annotation protocol?",
        "status": "blocked_pending_human_annotation",
        "current_artifacts": {
            "taxonomy": "configs/taxonomy.json",
            "annotation_metrics_code": "src/gjb/metrics.py",
        },
        "needed_to_complete": [
            "Select 200 real harmful prompts.",
            "Create 2-3 independent human annotation files with graph nodes and edges.",
            "Run node kappa, edge kappa, and normalized graph edit distance.",
        ],
        "dataset_summary": dataset_summary,
    }
    rq1 = {
        "created_at_utc": now,
        "rq": "RQ1",
        "question": "Do structural graph features provide complementary signal to text features for jailbreak detection?",
        "status": "baseline_results_available_not_final",
        "result_tables": [
            "results/tables/table_2_detection_real.csv",
            "results/rq/rq1/e1_ml_metrics.csv",
            "results/rq/rq1/embedding_baseline/e1_embedding_metrics.csv",
        ],
        "quick_rule_results": read_csv_dicts(TABLE_DIR / "table_2_detection_real.csv"),
        "ml_baseline_results": read_csv_dicts(PROJECT_ROOT / "results" / "rq" / "rq1" / "e1_ml_metrics.csv"),
        "ml_config": json.loads((PROJECT_ROOT / "results" / "rq" / "rq1" / "e1_ml_config.json").read_text(encoding="utf-8")),
        "embedding_baseline_results": read_csv_dicts(
            PROJECT_ROOT / "results" / "rq" / "rq1" / "embedding_baseline" / "e1_embedding_metrics.csv"
        ),
        "embedding_config": json.loads(
            (PROJECT_ROOT / "results" / "rq" / "rq1" / "embedding_baseline" / "e1_embedding_config.json").read_text(
                encoding="utf-8"
            )
        ),
        "caveat": "ML baselines use source-derived labels and first-pass heuristic graph annotations. Full-taxonomy graph variants include benign_query/harmful_intent as sanity checks; NoIntent and embedding variants are stricter comparisons.",
        "dataset_summary": dataset_summary,
    }
    rq2_runner_status = read_json_if_exists(PROJECT_ROOT / "results" / "rq" / "rq2" / "rq2_transferability_status.json")
    rq2_real_table = PROJECT_ROOT / "results" / "tables" / "table_3_transferability_real.csv"
    rq2_status = rq2_runner_status.get("status") if rq2_runner_status else "blocked_pending_cross_model_success_labels"
    rq2 = {
        "created_at_utc": now,
        "rq": "RQ2",
        "question": "Do graph-based features generalize better across models for predicting jailbreak success?",
        "status": rq2_status,
        "result_table": "results/tables/table_3_transferability_status.csv",
        "results": read_csv_dicts(TABLE_DIR / "table_3_transferability_status.csv"),
        "label_dependent_runner": "scripts/run_rq2_transferability_experiments.py",
        "needed_to_complete": [
            "Collect real Model A success labels.",
            "Collect real Model B success labels.",
            "Train text-feature and graph-feature transfer classifiers.",
        ],
    }
    if rq2_runner_status:
        rq2["runner_status"] = rq2_runner_status
    if rq2_status == "completed" and rq2_real_table.exists():
        rq2["result_table"] = "results/tables/table_3_transferability_real.csv"
        rq2["results"] = read_csv_dicts(rq2_real_table)
        rq2["needed_to_complete"] = []
    rq3 = {
        "created_at_utc": now,
        "rq": "RQ3",
        "question": "What structural motifs are most associated with successful jailbreak attacks?",
        "status": "frequency_results_available_success_rate_blocked",
        "result_table": "results/tables/table_4_top_motifs_real.csv",
        "results": read_csv_dicts(TABLE_DIR / "table_4_top_motifs_real.csv"),
        "caveat": "Motif frequency is available. Motif success rate remains NA until real model success labels exist.",
    }
    rq4 = {
        "created_at_utc": now,
        "rq": "RQ4",
        "question": "Do graph-based defenses reduce disparate overblocking across prompt styles of different user populations?",
        "status": "baseline_results_available_safe_grouping_not_final",
        "result_tables": [
            "results/tables/table_5_fairness_real.csv",
            "results/tables/table_6_fairness_disparity_real.csv",
        ],
        "subgroup_results": read_csv_dicts(TABLE_DIR / "table_5_fairness_real.csv"),
        "disparity_results": read_csv_dicts(TABLE_DIR / "table_6_fairness_disparity_real.csv"),
        "caveat": "Safe count is 600, but 500 rows are currently grouped as jailbreakdb_regular. The workbook's six demographic-adjacent style groups still need targeted real safe controls.",
    }
    balanced_fairness_dir = PROJECT_ROOT / "results" / "tables" / "rq4_balanced"
    if balanced_fairness_dir.exists():
        rq4["balanced_six_group_result_tables"] = [
            "results/tables/rq4_balanced/table_5_fairness_real.csv",
            "results/tables/rq4_balanced/table_6_fairness_disparity_real.csv",
        ]
        rq4["balanced_six_group_subgroup_results"] = read_csv_dicts(
            balanced_fairness_dir / "table_5_fairness_real.csv"
        )
        rq4["balanced_six_group_disparity_results"] = read_csv_dicts(
            balanced_fairness_dir / "table_6_fairness_disparity_real.csv"
        )
        assignment_summary = PROJECT_ROOT / "results" / "rq" / "rq4" / "safe_subgroup_assignment_summary.json"
        if assignment_summary.exists():
            rq4["balanced_assignment_summary"] = json.loads(assignment_summary.read_text(encoding="utf-8"))
        rq4["caveat"] = (
            "Baseline six-group fairness results are available using deterministic heuristic safe-subgroup assignment. "
            "Human review is still recommended before final demographic-adjacent fairness claims."
        )
    e5_runner_status = read_json_if_exists(PROJECT_ROOT / "results" / "rq" / "e5" / "e5_asr_status.json")
    e5_real_table = PROJECT_ROOT / "results" / "tables" / "table_10_asr_real.csv"
    e5_status = e5_runner_status.get("status") if e5_runner_status else "blocked_pending_real_multiturn_variants_and_success_labels"
    e5 = {
        "created_at_utc": now,
        "experiment": "E5",
        "name": "Multi-Turn vs Single-Turn",
        "status": e5_status,
        "result_table": "results/tables/table_10_asr_status.csv",
        "results": read_csv_dicts(TABLE_DIR / "table_10_asr_status.csv"),
        "label_dependent_runner": "scripts/run_e5_asr_analysis.py",
        "needed_to_complete": [
            "Collect or construct real multi-turn variants with provenance.",
            "Run target model evaluations.",
            "Store real attack success labels.",
        ],
    }
    if e5_runner_status:
        e5["runner_status"] = e5_runner_status
    if e5_status == "completed" and e5_real_table.exists():
        e5["result_table"] = "results/tables/table_10_asr_real.csv"
        e5["results"] = read_csv_dicts(e5_real_table)
        e5["needed_to_complete"] = []
    readiness = {
        "created_at_utc": now,
        "dataset_enough_for_baseline_runs": True,
        "dataset_enough_for_final_all_rqs": all(
            status in {"completed", "passed"}
            for status in [rq0["status"], rq1["status"], rq2["status"], rq3["status"], rq4["status"], e5["status"]]
        ),
        "why": [
            "The expanded dataset has 3,599 rows, within the workbook's overall target range.",
            "It has 2,999 harmful/jailbreak rows, within the workbook's 2,000-3,000 target range.",
            "It has 600 safe rows, matching the workbook safe-count target.",
            "It still lacks human annotator labels, model success labels, and real multi-turn rows.",
        ],
        "dataset_summary": dataset_summary,
        "rq_status": {
            "RQ0": rq0["status"],
            "RQ1": rq1["status"],
            "RQ2": rq2["status"],
            "RQ3": rq3["status"],
            "RQ4": rq4["status"],
            "E5": e5["status"],
        },
    }

    artifacts = {
        "rq0_annotation_validation_status.json": rq0,
        "rq1_detection_results.json": rq1,
        "rq2_transferability_status.json": rq2,
        "rq3_motif_results.json": rq3,
        "rq4_fairness_results.json": rq4,
        "e5_multiturn_status.json": e5,
        "readiness_summary.json": readiness,
    }
    for filename, payload in artifacts.items():
        write_json(OUTPUT_DIR / filename, payload)

    summary = [
        "# RQ Results Summary",
        "",
        f"Created UTC: {now}",
        f"Dataset: {dataset_summary['dataset_path']}",
        f"Rows: {dataset_summary['total_rows']}",
        f"By type: {dataset_summary['by_type']}",
        "",
        "Status:",
        f"- RQ0: {rq0['status']}",
        f"- RQ1: {rq1['status']}",
        f"- RQ2: {rq2['status']}",
        f"- RQ3: {rq3['status']}",
        f"- RQ4: {rq4['status']}",
        f"- E5: {e5['status']}",
        "",
        "Raw prompt text is not duplicated in these RQ result JSON files.",
    ]
    (OUTPUT_DIR / "SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(json.dumps(readiness, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
