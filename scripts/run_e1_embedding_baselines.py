#!/usr/bin/env python3
"""Run E1 sentence-transformer embedding baselines with local cached embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gjb.graph import graph_features
from gjb.io import read_jsonl


def graph_numeric(rows: list[dict]) -> np.ndarray:
    names = ["num_nodes", "num_edges", "graph_depth", "edge_density", "motif_count_2_4"]
    return np.asarray([[graph_features(row["graph_nodes"], row["graph_edges"])[name] for name in names] for row in rows], dtype=float)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, str]:
    values = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score) if len(set(y_true.tolist())) == 2 else float("nan"),
    }
    return {key: f"{value:.6f}" for key, value in values.items()}


def fit_lr(x_train: np.ndarray | csr_matrix, y_train: np.ndarray, x_test: np.ndarray | csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=31)
    model.fit(x_train, y_train)
    score = model.predict_proba(x_test)[:, 1]
    pred = (score >= 0.5).astype(int)
    return pred, score


def load_or_create_embeddings(rows: list[dict], model_name: str, cache_path: Path, batch_size: int) -> np.ndarray:
    if cache_path.exists():
        return np.load(cache_path)["embeddings"]
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, cache_folder=str(PROJECT_ROOT / ".hf-cache" / "sentence-transformers"))
    embeddings = model.encode(
        [row["prompt"] for row in rows],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, embeddings=embeddings.astype(np.float32))
    return embeddings


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/gjb_real_v1.jsonl")
    parser.add_argument("--output-dir", default="results/rq/rq1/embedding_baseline")
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf-cache"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".hf-cache" / "transformers"))

    rows = read_jsonl(PROJECT_ROOT / args.dataset)
    output_dir = PROJECT_ROOT / args.output_dir
    safe_model_name = args.model_name.replace("/", "__")
    embedding_path = output_dir / f"{safe_model_name}_embeddings.npz"
    embeddings = load_or_create_embeddings(rows, args.model_name, embedding_path, args.batch_size)

    y = np.asarray([1 if row["is_jailbreak"] else 0 for row in rows], dtype=int)
    difficulties = np.asarray([row["difficulty"] for row in rows], dtype=object)
    indices = np.arange(len(rows))
    train_idx, test_idx = train_test_split(indices, test_size=args.test_size, random_state=args.seed, stratify=y)

    graph_values = graph_numeric(rows)
    scaler = StandardScaler()
    graph_train = scaler.fit_transform(graph_values[train_idx])
    graph_test = scaler.transform(graph_values[test_idx])

    model_inputs = {
        "SentenceEmbedding LR": (embeddings[train_idx], embeddings[test_idx]),
        "SentenceEmbedding+GraphStruct LR": (
            np.hstack([embeddings[train_idx], graph_train]),
            np.hstack([embeddings[test_idx], graph_test]),
        ),
    }

    metric_rows = []
    prediction_rows = []
    for model_label, (x_train, x_test) in model_inputs.items():
        pred, score = fit_lr(x_train, y[train_idx], x_test)
        for stratum in ["overall", "easy", "medium", "hard"]:
            mask = np.ones(len(test_idx), dtype=bool) if stratum == "overall" else difficulties[test_idx] == stratum
            metric_rows.append(
                {
                    "model": model_label,
                    "difficulty": stratum,
                    "n": int(mask.sum()),
                    **metrics(y[test_idx][mask], pred[mask], score[mask]),
                }
            )
        for local_index, dataset_index in enumerate(test_idx):
            prediction_rows.append(
                {
                    "id": rows[dataset_index]["id"],
                    "model": model_label,
                    "y_true": int(y[dataset_index]),
                    "y_pred": int(pred[local_index]),
                    "score": f"{score[local_index]:.8f}",
                    "difficulty": rows[dataset_index]["difficulty"],
                    "source_dataset": rows[dataset_index]["source_dataset"],
                }
            )

    write_csv(output_dir / "e1_embedding_metrics.csv", metric_rows, ["model", "difficulty", "n", "accuracy", "precision", "recall", "f1", "roc_auc"])
    write_csv(output_dir / "e1_embedding_predictions.csv", prediction_rows, ["id", "model", "y_true", "y_pred", "score", "difficulty", "source_dataset"])
    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "model_name": args.model_name,
        "embedding_cache": str(embedding_path.relative_to(PROJECT_ROOT)),
        "hf_home": os.environ["HF_HOME"],
        "test_size": args.test_size,
        "seed": args.seed,
        "prompt_text_logged": False,
    }
    (output_dir / "e1_embedding_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": str((output_dir / "e1_embedding_metrics.csv").relative_to(PROJECT_ROOT)), **config}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
