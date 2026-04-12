#!/usr/bin/env python3
"""Run RQ2 transferability prediction when real cross-model labels exist."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gjb.graph import graph_features
from gjb.io import read_jsonl
from gjb.taxonomy import taxonomy_labels


NOINTENT_LABELS = {"benign_query", "harmful_intent"}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fmt(value: float) -> str:
    return "NA" if math.isnan(value) else f"{value:.6f}"


def metric_row(model: str, y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    roc_auc = roc_auc_score(y_true, y_score) if len(set(y_true.tolist())) == 2 else float("nan")
    return {
        "model": model,
        "n": len(y_true),
        "accuracy": fmt(accuracy_score(y_true, y_pred)),
        "precision": fmt(precision_score(y_true, y_pred, zero_division=0)),
        "recall": fmt(recall_score(y_true, y_pred, zero_division=0)),
        "f1": fmt(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": fmt(roc_auc),
    }


def graph_matrix(rows: list[dict[str, Any]]) -> csr_matrix:
    numeric_names = ["num_nodes", "num_edges", "graph_depth", "edge_density", "motif_count_2_4"]
    numeric = np.asarray(
        [[graph_features(row["graph_nodes"], row["graph_edges"])[name] for name in numeric_names] for row in rows],
        dtype=float,
    )
    labels = sorted(taxonomy_labels() - NOINTENT_LABELS)
    label_index = {label: index for index, label in enumerate(labels)}
    label_values = np.zeros((len(rows), len(labels)), dtype=float)
    for row_index, row in enumerate(rows):
        for node in row["graph_nodes"]:
            if node in label_index:
                label_values[row_index, label_index[node]] = 1.0
    return hstack([csr_matrix(numeric), csr_matrix(label_values)], format="csr")


def fit_predict(x_train: csr_matrix, y_train: np.ndarray, x_test: csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    model = LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced", random_state=52)
    model.fit(x_train, y_train)
    score = model.predict_proba(x_test)[:, 1]
    return (score >= 0.5).astype(int), score


def blocked_payload(args: argparse.Namespace, reason: str, candidate_rows: int, target_counts: Counter) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rq": "RQ2",
        "status": "blocked_pending_cross_model_success_labels",
        "reason": reason,
        "dataset": args.dataset,
        "candidate_rows": candidate_rows,
        "target_success_counts": dict(sorted(target_counts.items())),
        "source_success_field": args.source_success_field,
        "target_success_field": args.target_success_field,
        "source_success_only": not args.include_source_failures,
        "prompt_text_logged": False,
    }


def write_status_table(path: Path, status: str, n_labeled: int, reason: str) -> None:
    write_csv(
        path,
        [
            {
                "experiment": "E2 Transferability Prediction",
                "status": status,
                "n_labeled": n_labeled,
                "reason": reason,
            }
        ],
        ["experiment", "status", "n_labeled", "reason"],
    )


def run_completed(rows: list[dict[str, Any]], labeled_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    y = np.asarray([1 if row[args.target_success_field] else 0 for row in labeled_rows], dtype=int)
    indices = np.arange(len(labeled_rows))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_features=30000, strip_accents="unicode")
    text_train = vectorizer.fit_transform([labeled_rows[index]["prompt"] for index in train_idx])
    text_test = vectorizer.transform([labeled_rows[index]["prompt"] for index in test_idx])

    graph_all = graph_matrix(labeled_rows)
    scaler = StandardScaler(with_mean=False)
    graph_train = scaler.fit_transform(graph_all[train_idx])
    graph_test = scaler.transform(graph_all[test_idx])

    model_inputs = {
        "Text TFIDF LR": (text_train, text_test),
        "Graph NoIntent LR": (graph_train, graph_test),
        "Hybrid TFIDF+Graph NoIntent LR": (
            hstack([text_train, graph_train], format="csr"),
            hstack([text_test, graph_test], format="csr"),
        ),
    }

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for model_name, (x_train, x_test) in model_inputs.items():
        pred, score = fit_predict(x_train, y[train_idx], x_test)
        metric_rows.append(metric_row(model_name, y[test_idx], pred, score))
        for local_index, dataset_index in enumerate(test_idx):
            row = labeled_rows[dataset_index]
            prediction_rows.append(
                {
                    "id": row["id"],
                    "model": model_name,
                    "y_true": int(y[dataset_index]),
                    "y_pred": int(pred[local_index]),
                    "score": f"{score[local_index]:.8f}",
                    "source_dataset": row["source_dataset"],
                    "difficulty": row["difficulty"],
                    "prompt_sha256": row.get("metadata", {}).get("prompt_sha256", ""),
                }
            )

    write_csv(output_dir / "rq2_transferability_metrics.csv", metric_rows, ["model", "n", "accuracy", "precision", "recall", "f1", "roc_auc"])
    write_csv(
        output_dir / "rq2_transferability_predictions.csv",
        prediction_rows,
        ["id", "model", "y_true", "y_pred", "score", "source_dataset", "difficulty", "prompt_sha256"],
    )
    write_csv(
        PROJECT_ROOT / "results" / "tables" / "table_3_transferability_real.csv",
        metric_rows,
        ["model", "n", "accuracy", "precision", "recall", "f1", "roc_auc"],
    )
    status = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rq": "RQ2",
        "status": "completed",
        "dataset": args.dataset,
        "output_dir": args.output_dir,
        "source_success_field": args.source_success_field,
        "target_success_field": args.target_success_field,
        "source_success_only": not args.include_source_failures,
        "candidate_rows": len(labeled_rows),
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "models": [row["model"] for row in metric_rows],
        "prompt_text_logged": False,
        "result_table": "results/tables/table_3_transferability_real.csv",
    }
    write_json(output_dir / "rq2_transferability_status.json", status)
    write_status_table(PROJECT_ROOT / "results" / "tables" / "table_3_transferability_status.csv", "completed", len(labeled_rows), "Real cross-model labels available; metrics written to table_3_transferability_real.csv.")
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/gjb_real_v1.jsonl")
    parser.add_argument("--output-dir", default="results/rq/rq2")
    parser.add_argument("--source-success-field", default="model_gpt4_success")
    parser.add_argument("--target-success-field", default="model_llama_success")
    parser.add_argument("--include-source-failures", action="store_true")
    parser.add_argument("--min-labeled", type=int, default=50)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=52)
    args = parser.parse_args()

    rows = read_jsonl(PROJECT_ROOT / args.dataset)
    attack_rows = [row for row in rows if row["type"] in {"jailbreak", "multiturn"}]
    candidates = [
        row
        for row in attack_rows
        if row.get(args.source_success_field) is not None and row.get(args.target_success_field) is not None
    ]
    if not args.include_source_failures:
        candidates = [row for row in candidates if row.get(args.source_success_field) is True]
    target_counts = Counter(str(row.get(args.target_success_field)) for row in candidates)

    reason = ""
    if len(candidates) < args.min_labeled:
        reason = f"Need at least {args.min_labeled} real paired labels; found {len(candidates)}."
    elif len(target_counts) < 2:
        reason = "Need both successful and unsuccessful target-model outcomes for supervised transferability."
    elif min(target_counts.values()) < 2:
        reason = "Need at least two rows in each target outcome class for stratified train/test split."

    if reason:
        payload = blocked_payload(args, reason, len(candidates), target_counts)
        output_dir = PROJECT_ROOT / args.output_dir
        write_json(output_dir / "rq2_transferability_status.json", payload)
        write_status_table(PROJECT_ROOT / "results" / "tables" / "table_3_transferability_status.csv", "blocked", len(candidates), reason)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    status = run_completed(rows, candidates, args)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
