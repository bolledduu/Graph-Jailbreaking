#!/usr/bin/env python3
"""Run E1 real ML baselines for graph-vs-text jailbreak detection."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

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


@dataclass(frozen=True)
class DatasetSplit:
    train_indices: np.ndarray
    test_indices: np.ndarray


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, str]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if len(set(y_true.tolist())) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    else:
        metrics["roc_auc"] = float("nan")
    return {key: f"{value:.6f}" for key, value in metrics.items()}


def graph_numeric_matrix(rows: list[dict]) -> tuple[csr_matrix, list[str]]:
    feature_names = ["num_nodes", "num_edges", "graph_depth", "edge_density", "motif_count_2_4"]
    values = []
    for row in rows:
        features = graph_features(row["graph_nodes"], row["graph_edges"])
        values.append([features[name] for name in feature_names])
    return csr_matrix(np.asarray(values, dtype=float)), feature_names


def graph_taxonomy_matrix(rows: list[dict], exclude_labels: set[str] | None = None) -> tuple[csr_matrix, list[str]]:
    numeric, numeric_names = graph_numeric_matrix(rows)
    excluded = exclude_labels or set()
    labels = sorted(taxonomy_labels() - excluded)
    label_values = np.zeros((len(rows), len(labels)), dtype=float)
    label_index = {label: index for index, label in enumerate(labels)}
    for row_index, row in enumerate(rows):
        for node in row["graph_nodes"]:
            if node in label_index:
                label_values[row_index, label_index[node]] = 1.0
    feature_names = [*numeric_names, *[f"node:{label}" for label in labels]]
    return hstack([numeric, csr_matrix(label_values)], format="csr"), feature_names


def scale_sparse_train_test(x_train: csr_matrix, x_test: csr_matrix) -> tuple[csr_matrix, csr_matrix]:
    scaler = StandardScaler(with_mean=False)
    return scaler.fit_transform(x_train), scaler.transform(x_test)


def fit_predict_lr(x_train: csr_matrix, y_train: np.ndarray, x_test: csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=30)
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_test)[:, 1]
    predictions = (scores >= 0.5).astype(int)
    return predictions, scores


def run_models(rows: list[dict], split: DatasetSplit) -> tuple[list[dict], list[dict]]:
    y = np.asarray([1 if row["is_jailbreak"] else 0 for row in rows], dtype=int)
    difficulties = np.asarray([row["difficulty"] for row in rows], dtype=object)
    row_ids = np.asarray([row["id"] for row in rows], dtype=object)
    train_idx = split.train_indices
    test_idx = split.test_indices

    model_outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    text_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000,
        strip_accents="unicode",
    )
    text_train = text_vectorizer.fit_transform([rows[index]["prompt"] for index in train_idx])
    text_test = text_vectorizer.transform([rows[index]["prompt"] for index in test_idx])
    model_outputs["Text-Only TFIDF LR"] = fit_predict_lr(text_train, y[train_idx], text_test)

    graph_struct, _ = graph_numeric_matrix(rows)
    graph_struct_train, graph_struct_test = scale_sparse_train_test(graph_struct[train_idx], graph_struct[test_idx])
    model_outputs["Graph-Structural LR"] = fit_predict_lr(graph_struct_train, y[train_idx], graph_struct_test)

    graph_taxonomy, _ = graph_taxonomy_matrix(rows)
    graph_tax_train, graph_tax_test = scale_sparse_train_test(graph_taxonomy[train_idx], graph_taxonomy[test_idx])
    model_outputs["Graph-Taxonomy LR"] = fit_predict_lr(graph_tax_train, y[train_idx], graph_tax_test)

    graph_nointent, _ = graph_taxonomy_matrix(rows, exclude_labels={"benign_query", "harmful_intent"})
    graph_nointent_train, graph_nointent_test = scale_sparse_train_test(graph_nointent[train_idx], graph_nointent[test_idx])
    model_outputs["Graph-Taxonomy NoIntent LR"] = fit_predict_lr(graph_nointent_train, y[train_idx], graph_nointent_test)

    hybrid_train = hstack([text_train, graph_tax_train], format="csr")
    hybrid_test = hstack([text_test, graph_tax_test], format="csr")
    model_outputs["Hybrid Text+Graph LR"] = fit_predict_lr(hybrid_train, y[train_idx], hybrid_test)

    hybrid_nointent_train = hstack([text_train, graph_nointent_train], format="csr")
    hybrid_nointent_test = hstack([text_test, graph_nointent_test], format="csr")
    model_outputs["Hybrid Text+Graph NoIntent LR"] = fit_predict_lr(
        hybrid_nointent_train, y[train_idx], hybrid_nointent_test
    )

    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    for model_name, (predictions, scores) in model_outputs.items():
        y_test = y[test_idx]
        for stratum in ["overall", "easy", "medium", "hard"]:
            if stratum == "overall":
                mask = np.ones(len(test_idx), dtype=bool)
            else:
                mask = difficulties[test_idx] == stratum
            if not np.any(mask):
                metric_rows.append(
                    {
                        "model": model_name,
                        "difficulty": stratum,
                        "n": 0,
                        "accuracy": "NA",
                        "precision": "NA",
                        "recall": "NA",
                        "f1": "NA",
                        "roc_auc": "NA",
                    }
                )
                continue
            metrics = classification_metrics(y_test[mask], predictions[mask], scores[mask])
            metric_rows.append({"model": model_name, "difficulty": stratum, "n": int(mask.sum()), **metrics})

        for local_index, dataset_index in enumerate(test_idx):
            prediction_rows.append(
                {
                    "id": row_ids[dataset_index],
                    "model": model_name,
                    "split": "test",
                    "y_true": int(y[dataset_index]),
                    "y_pred": int(predictions[local_index]),
                    "score": f"{scores[local_index]:.8f}",
                    "difficulty": difficulties[dataset_index],
                    "source_dataset": rows[dataset_index]["source_dataset"],
                }
            )
    return metric_rows, prediction_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/gjb_real_v1.jsonl")
    parser.add_argument("--output-dir", default="results/rq/rq1")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=30)
    args = parser.parse_args()

    rows = read_jsonl(PROJECT_ROOT / args.dataset)
    labels = np.asarray([1 if row["is_jailbreak"] else 0 for row in rows], dtype=int)
    indices = np.arange(len(rows))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )
    split = DatasetSplit(train_indices=np.asarray(train_idx), test_indices=np.asarray(test_idx))
    metric_rows, prediction_rows = run_models(rows, split)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "e1_ml_metrics.csv",
        metric_rows,
        ["model", "difficulty", "n", "accuracy", "precision", "recall", "f1", "roc_auc"],
    )
    write_csv(
        output_dir / "e1_ml_predictions.csv",
        prediction_rows,
        ["id", "model", "split", "y_true", "y_pred", "score", "difficulty", "source_dataset"],
    )
    split_payload = {
        "train_ids": [rows[index]["id"] for index in train_idx.tolist()],
        "test_ids": [rows[index]["id"] for index in test_idx.tolist()],
    }
    (output_dir / "e1_split_ids.json").write_text(json.dumps(split_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "test_size": args.test_size,
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "models": sorted({row["model"] for row in metric_rows}),
        "prompt_text_logged": False,
        "caveat": "Uses source-derived labels and first-pass heuristic graph annotations. Full taxonomy variants include benign_query/harmful_intent and are sanity checks; NoIntent variants exclude those direct intent labels. Suitable for baseline pipeline evidence, not final paper claims.",
    }
    (output_dir / "e1_ml_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": str(output_dir / "e1_ml_metrics.csv"), "predictions": str(output_dir / "e1_ml_predictions.csv"), **config}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
