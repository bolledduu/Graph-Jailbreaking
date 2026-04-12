#!/usr/bin/env python3
"""Run dependency-free baseline tables on a real-source GJB JSONL dataset."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gjb.graph import enumerate_motifs
from gjb.io import read_jsonl
from gjb.metrics import classification_metrics, fpr_disparity, subgroup_fpr


TEXT_RISK_TOKENS = {
    "bypass",
    "chemical",
    "credential",
    "exploit",
    "fraud",
    "hack",
    "illegal",
    "malware",
    "password",
    "phishing",
    "poison",
    "weapon",
}


def text_only_predict(row: dict) -> bool:
    prompt = row["prompt"].lower()
    return any(token in prompt for token in TEXT_RISK_TOKENS)


def graph_predict(row: dict) -> bool:
    nodes = set(row["graph_nodes"])
    return "harmful_intent" in nodes or {"instruction_override", "obfuscation"} <= nodes


def hybrid_predict(row: dict) -> bool:
    return text_only_predict(row) or graph_predict(row)


def write_detection_table(rows: list[dict], output_dir: Path) -> None:
    methods = {
        "Text-Only Lexical Baseline": text_only_predict,
        "Graph-Only Rule Baseline": graph_predict,
        "Hybrid Text+Graph Baseline": hybrid_predict,
    }
    strata = ["overall", "easy", "medium", "hard"]
    output = output_dir / "table_2_detection_real.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "difficulty", "n", "accuracy", "precision", "recall", "f1"],
        )
        writer.writeheader()
        for model_name, predictor in methods.items():
            predictions = [predictor(row) for row in rows]
            for stratum in strata:
                if stratum == "overall":
                    selected = rows
                    selected_preds = predictions
                else:
                    selected = [row for row in rows if row["difficulty"] == stratum]
                    selected_preds = [predictor(row) for row in selected]
                if not selected:
                    writer.writerow(
                        {
                            "model": model_name,
                            "difficulty": stratum,
                            "n": 0,
                            "accuracy": "NA",
                            "precision": "NA",
                            "recall": "NA",
                            "f1": "NA",
                        }
                    )
                    continue
                metrics = classification_metrics([row["is_jailbreak"] for row in selected], selected_preds)
                writer.writerow(
                    {
                        "model": model_name,
                        "difficulty": stratum,
                        "n": len(selected),
                        "accuracy": f"{metrics['accuracy']:.4f}",
                        "precision": f"{metrics['precision']:.4f}",
                        "recall": f"{metrics['recall']:.4f}",
                        "f1": f"{metrics['f1']:.4f}",
                    }
                )


def write_transferability_table(rows: list[dict], output_dir: Path) -> None:
    labeled = [
        row
        for row in rows
        if row["type"] in {"jailbreak", "multiturn"}
        and row.get("model_gpt4_success") is not None
        and row.get("model_llama_success") is not None
    ]
    output = output_dir / "table_3_transferability_status.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment", "status", "n_labeled", "reason"])
        writer.writeheader()
        writer.writerow(
            {
                "experiment": "E2 Transferability Prediction",
                "status": "blocked",
                "n_labeled": len(labeled),
                "reason": "Real source CSVs do not include GPT-4/LLaMA success labels.",
            }
        )


def write_motif_table(rows: list[dict], output_dir: Path) -> None:
    motif_rows: dict[str, list[dict]] = defaultdict(list)
    attacks = [row for row in rows if row["type"] in {"jailbreak", "multiturn"}]
    for row in attacks:
        for motif in enumerate_motifs(row["graph_nodes"], row["graph_edges"]):
            motif_rows[motif].append(row)
    ranked = sorted(
        motif_rows.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    )[:10]
    output = output_dir / "table_4_top_motifs_real.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rank", "motif", "frequency", "success_rate", "example_row_id", "example_source"],
        )
        writer.writeheader()
        for rank, (motif, motif_hits) in enumerate(ranked, start=1):
            labeled_hits = [row for row in motif_hits if row.get("model_llama_success") is not None]
            success_rate = (
                f"{sum(row['model_llama_success'] for row in labeled_hits) / len(labeled_hits):.4f}"
                if labeled_hits
                else "NA"
            )
            writer.writerow(
                {
                    "rank": rank,
                    "motif": motif,
                    "frequency": len(motif_hits),
                    "success_rate": success_rate,
                    "example_row_id": motif_hits[0]["id"],
                    "example_source": motif_hits[0]["source_dataset"],
                }
            )


def write_fairness_table(rows: list[dict], output_dir: Path) -> None:
    methods = {
        "Text-Only Lexical Baseline": text_only_predict,
        "Graph-Based Baseline": graph_predict,
    }
    output = output_dir / "table_5_fairness_real.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "subgroup", "n", "fpr"])
        writer.writeheader()
        safe_counts = Counter(row.get("safe_subgroup") for row in rows if row["type"] == "safe")
        for method, predictor in methods.items():
            predictions = {row["id"]: predictor(row) for row in rows}
            for subgroup, fpr in subgroup_fpr(rows, predictions).items():
                writer.writerow({"method": method, "subgroup": subgroup, "n": safe_counts[subgroup], "fpr": f"{fpr:.4f}"})

    summary_output = output_dir / "table_6_fairness_disparity_real.csv"
    with summary_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "max_min_fpr_gap", "fpr_std_dev"])
        writer.writeheader()
        for method, predictor in methods.items():
            predictions = {row["id"]: predictor(row) for row in rows}
            disparity = fpr_disparity(subgroup_fpr(rows, predictions))
            writer.writerow(
                {
                    "method": method,
                    "max_min_fpr_gap": f"{disparity['max_min_fpr_gap']:.4f}",
                    "fpr_std_dev": f"{disparity['fpr_std_dev']:.4f}",
                }
            )


def write_dataset_stats(rows: list[dict], output_dir: Path) -> None:
    output = output_dir / "table_7_dataset_statistics_real.csv"
    counters = {
        "type": Counter(row["type"] for row in rows),
        "source_dataset": Counter(row["source_dataset"] for row in rows),
        "difficulty": Counter(row["difficulty"] for row in rows),
        "attack_type": Counter(row["attack_type"] for row in rows),
        "safe_subgroup": Counter(row.get("safe_subgroup") for row in rows if row["type"] == "safe"),
    }
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["field", "value", "count"])
        writer.writeheader()
        for field, counter in counters.items():
            for value, count in sorted(counter.items()):
                writer.writerow({"field": field, "value": value, "count": count})


def write_asr_table(rows: list[dict], output_dir: Path) -> None:
    output = output_dir / "table_10_asr_status.csv"
    labeled = [
        row
        for row in rows
        if row["type"] in {"jailbreak", "multiturn"} and row.get("model_llama_success") is not None
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment", "status", "n_labeled", "reason"])
        writer.writeheader()
        writer.writerow(
            {
                "experiment": "E5 Multi-Turn vs Single-Turn ASR",
                "status": "blocked",
                "n_labeled": len(labeled),
                "reason": "Real source CSVs do not include attack success labels or multi-turn variants.",
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/gjb_real_v1.jsonl")
    parser.add_argument("--output-dir", default="results/tables")
    args = parser.parse_args()

    rows = read_jsonl(PROJECT_ROOT / args.dataset)
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    write_detection_table(rows, output_dir)
    write_transferability_table(rows, output_dir)
    write_motif_table(rows, output_dir)
    write_fairness_table(rows, output_dir)
    write_dataset_stats(rows, output_dir)
    write_asr_table(rows, output_dir)
    print(f"Wrote real-source experiment tables/status files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
