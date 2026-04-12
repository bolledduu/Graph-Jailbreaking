#!/usr/bin/env python3
"""Run all currently unblocked robustness, fairness, and motif experiments.

Outputs intentionally avoid raw prompt text. Row IDs, source names, hashes,
metrics, and figure paths are saved for auditability.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import calibration_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gjb.graph import enumerate_motifs, graph_features
from gjb.io import read_jsonl
from gjb.taxonomy import taxonomy_labels


NOINTENT_LABELS = {"benign_query", "harmful_intent"}
RISK_TOKENS = {
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


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def metric_values(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": float("nan"),
        "brier": float("nan"),
    }
    if len(set(y_true.tolist())) == 2:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
        out["brier"] = brier_score_loss(y_true, y_score)
    return out


def format_metric(value: float) -> str:
    return "NA" if math.isnan(value) else f"{value:.6f}"


def format_metric_row(model: str, split: str, stratum: str, y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    metrics = metric_values(y_true, y_pred, y_score)
    return {
        "model": model,
        "split": split,
        "stratum": stratum,
        "n": len(y_true),
        "accuracy": format_metric(metrics["accuracy"]),
        "precision": format_metric(metrics["precision"]),
        "recall": format_metric(metrics["recall"]),
        "f1": format_metric(metrics["f1"]),
        "roc_auc": format_metric(metrics["roc_auc"]),
        "brier": format_metric(metrics["brier"]),
    }


def graph_struct_matrix(rows: list[dict]) -> tuple[csr_matrix, list[str]]:
    names = ["num_nodes", "num_edges", "graph_depth", "edge_density", "motif_count_2_4"]
    values = [[graph_features(row["graph_nodes"], row["graph_edges"])[name] for name in names] for row in rows]
    return csr_matrix(np.asarray(values, dtype=float)), names


def graph_nointent_matrix(rows: list[dict]) -> tuple[csr_matrix, list[str]]:
    numeric, numeric_names = graph_struct_matrix(rows)
    labels = sorted(taxonomy_labels() - NOINTENT_LABELS)
    label_index = {label: index for index, label in enumerate(labels)}
    label_values = np.zeros((len(rows), len(labels)), dtype=float)
    for row_index, row in enumerate(rows):
        for node in row["graph_nodes"]:
            if node in label_index:
                label_values[row_index, label_index[node]] = 1.0
    return hstack([numeric, csr_matrix(label_values)], format="csr"), [*numeric_names, *[f"node:{label}" for label in labels]]


def scale_sparse(train: csr_matrix, test: csr_matrix) -> tuple[csr_matrix, csr_matrix]:
    scaler = StandardScaler(with_mean=False)
    return scaler.fit_transform(train), scaler.transform(test)


def fit_lr(x_train: csr_matrix, y_train: np.ndarray, x_test: csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    model = LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced", random_state=42)
    model.fit(x_train, y_train)
    score = model.predict_proba(x_test)[:, 1]
    pred = (score >= 0.5).astype(int)
    return pred, score


def source_balanced_split(rows: list[dict], test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row["source_dataset"], row["type"])].append(index)
    train: list[int] = []
    test: list[int] = []
    for key in sorted(groups):
        group = groups[key][:]
        rng.shuffle(group)
        if len(group) == 1:
            train.extend(group)
            continue
        n_test = max(1, int(round(len(group) * test_size)))
        n_test = min(n_test, len(group) - 1)
        test.extend(group[:n_test])
        train.extend(group[n_test:])
    return np.asarray(sorted(train)), np.asarray(sorted(test))


def load_cached_embeddings(rows: list[dict], path: Path) -> csr_matrix | None:
    if not path.exists():
        return None
    embeddings = np.load(path)["embeddings"]
    if embeddings.shape[0] != len(rows):
        return None
    return csr_matrix(embeddings)


def run_ablation(rows: list[dict], output_dir: Path, seed: int, test_size: float) -> dict:
    y = np.asarray([1 if row["is_jailbreak"] else 0 for row in rows], dtype=int)
    ids = np.asarray([row["id"] for row in rows], dtype=object)
    sources = np.asarray([row["source_dataset"] for row in rows], dtype=object)
    difficulties = np.asarray([row["difficulty"] for row in rows], dtype=object)
    train_idx, test_idx = source_balanced_split(rows, test_size, seed)

    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=40000, strip_accents="unicode")
    x_text_train = vectorizer.fit_transform([rows[index]["prompt"] for index in train_idx])
    x_text_test = vectorizer.transform([rows[index]["prompt"] for index in test_idx])

    graph_struct, _ = graph_struct_matrix(rows)
    gs_train, gs_test = scale_sparse(graph_struct[train_idx], graph_struct[test_idx])
    graph_no, _ = graph_nointent_matrix(rows)
    gn_train, gn_test = scale_sparse(graph_no[train_idx], graph_no[test_idx])

    models: dict[str, tuple[csr_matrix, csr_matrix]] = {
        "Text TFIDF LR": (x_text_train, x_text_test),
        "Graph Structural LR": (gs_train, gs_test),
        "Graph NoIntent LR": (gn_train, gn_test),
        "Hybrid TFIDF+Graph NoIntent LR": (hstack([x_text_train, gn_train], format="csr"), hstack([x_text_test, gn_test], format="csr")),
    }
    embedding_path = PROJECT_ROOT / "results" / "rq" / "rq1" / "embedding_baseline" / "sentence-transformers__all-MiniLM-L6-v2_embeddings.npz"
    embeddings = load_cached_embeddings(rows, embedding_path)
    if embeddings is not None:
        models["SentenceEmbedding LR"] = (embeddings[train_idx], embeddings[test_idx])
        models["SentenceEmbedding+Graph Structural LR"] = (
            hstack([embeddings[train_idx], gs_train], format="csr"),
            hstack([embeddings[test_idx], gs_test], format="csr"),
        )

    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    error_rows: list[dict] = []
    calibration_rows: list[dict] = []
    bootstrap_input: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for model_name, (x_train, x_test) in models.items():
        pred, score = fit_lr(x_train, y[train_idx], x_test)
        y_test = y[test_idx]
        bootstrap_input[model_name] = (y_test, pred, score)
        metric_rows.append(format_metric_row(model_name, "source_balanced_test", "overall", y_test, pred, score))
        for difficulty in ["easy", "medium", "hard"]:
            mask = difficulties[test_idx] == difficulty
            if np.any(mask):
                metric_rows.append(format_metric_row(model_name, "source_balanced_test", difficulty, y_test[mask], pred[mask], score[mask]))
        for source in sorted(set(sources[test_idx].tolist())):
            mask = sources[test_idx] == source
            metric_rows.append(format_metric_row(model_name, "source_balanced_test", f"source:{source}", y_test[mask], pred[mask], score[mask]))

        if len(set(y_test.tolist())) == 2:
            frac_pos, mean_pred = calibration_curve(y_test, score, n_bins=10, strategy="uniform")
            for bin_index, (prob, frac) in enumerate(zip(mean_pred, frac_pos), start=1):
                calibration_rows.append(
                    {
                        "model": model_name,
                        "bin": bin_index,
                        "mean_predicted_probability": f"{prob:.6f}",
                        "fraction_positive": f"{frac:.6f}",
                    }
                )

        for local_index, dataset_index in enumerate(test_idx):
            record = {
                "id": ids[dataset_index],
                "model": model_name,
                "y_true": int(y[dataset_index]),
                "y_pred": int(pred[local_index]),
                "score": f"{score[local_index]:.8f}",
                "source_dataset": sources[dataset_index],
                "difficulty": difficulties[dataset_index],
                "prompt_sha256": rows[dataset_index].get("metadata", {}).get("prompt_sha256", ""),
            }
            prediction_rows.append(record)
            if int(y[dataset_index]) != int(pred[local_index]):
                error_rows.append(
                    {
                        **record,
                        "error_type": "false_positive" if pred[local_index] == 1 else "false_negative",
                    }
                )

    write_csv(output_dir / "ablation_metrics.csv", metric_rows, ["model", "split", "stratum", "n", "accuracy", "precision", "recall", "f1", "roc_auc", "brier"])
    write_csv(output_dir / "ablation_predictions.csv", prediction_rows, ["id", "model", "y_true", "y_pred", "score", "source_dataset", "difficulty", "prompt_sha256"])
    write_csv(output_dir / "error_analysis.csv", error_rows, ["id", "model", "y_true", "y_pred", "score", "source_dataset", "difficulty", "prompt_sha256", "error_type"])
    write_csv(output_dir / "calibration_bins.csv", calibration_rows, ["model", "bin", "mean_predicted_probability", "fraction_positive"])
    write_json(output_dir / "source_balanced_split.json", {"train_ids": ids[train_idx].tolist(), "test_ids": ids[test_idx].tolist()})

    audit = {
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "train_type_counts": dict(sorted(Counter(rows[index]["type"] for index in train_idx).items())),
        "test_type_counts": dict(sorted(Counter(rows[index]["type"] for index in test_idx).items())),
        "train_source_counts": dict(sorted(Counter(rows[index]["source_dataset"] for index in train_idx).items())),
        "test_source_counts": dict(sorted(Counter(rows[index]["source_dataset"] for index in test_idx).items())),
        "duplicate_prompt_hashes_between_train_test": len(
            {rows[index].get("metadata", {}).get("prompt_sha256", "") for index in train_idx}
            & {rows[index].get("metadata", {}).get("prompt_sha256", "") for index in test_idx}
        ),
        "prompt_text_logged": False,
    }
    write_json(output_dir / "split_leakage_audit.json", audit)
    return {"bootstrap_input": bootstrap_input, "test_idx": test_idx, "summary": audit}


def bootstrap_cis(bootstrap_input: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], output_dir: Path, seed: int, n_bootstrap: int) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for model_name, (y_true, y_pred, y_score) in bootstrap_input.items():
        n = len(y_true)
        samples: dict[str, list[float]] = defaultdict(list)
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            values = metric_values(y_true[idx], y_pred[idx], y_score[idx])
            for key, value in values.items():
                if not math.isnan(value):
                    samples[key].append(value)
        for metric, values in sorted(samples.items()):
            if not values:
                continue
            arr = np.asarray(values)
            rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "n_bootstrap": n_bootstrap,
                    "mean": f"{float(np.mean(arr)):.6f}",
                    "ci95_low": f"{float(np.percentile(arr, 2.5)):.6f}",
                    "ci95_high": f"{float(np.percentile(arr, 97.5)):.6f}",
                }
            )
    write_csv(output_dir / "bootstrap_95ci.csv", rows, ["model", "metric", "n_bootstrap", "mean", "ci95_low", "ci95_high"])


def run_leave_one_source(rows: list[dict], output_dir: Path) -> None:
    y_all = np.asarray([1 if row["is_jailbreak"] else 0 for row in rows], dtype=int)
    sources = sorted(set(row["source_dataset"] for row in rows))
    metric_rows = []
    prediction_rows = []
    rng = random.Random(202)
    for source in sources:
        heldout = [index for index, row in enumerate(rows) if row["source_dataset"] == source]
        train = [index for index, row in enumerate(rows) if row["source_dataset"] != source]
        heldout_labels = {int(y_all[index]) for index in heldout}
        contrast_used = False
        test = heldout[:]
        if len(heldout_labels) == 1:
            needed_label = 1 - next(iter(heldout_labels))
            candidates = [index for index in train if int(y_all[index]) == needed_label]
            rng.shuffle(candidates)
            test.extend(candidates[: min(len(candidates), len(heldout))])
            contrast_used = True

        y_train = y_all[train]
        if len(set(y_train.tolist())) < 2 or len(set(y_all[test].tolist())) < 2:
            continue
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=40000, strip_accents="unicode")
        x_train = vectorizer.fit_transform([rows[index]["prompt"] for index in train])
        x_test = vectorizer.transform([rows[index]["prompt"] for index in test])
        pred, score = fit_lr(x_train, y_train, x_test)
        metric_rows.append(
            {
                **format_metric_row("Text TFIDF LR", "leave_one_source_out", f"heldout:{source}", y_all[test], pred, score),
                "heldout_source": source,
                "heldout_rows": len(heldout),
                "contrast_rows_added": len(test) - len(heldout),
                "contrast_set_used": contrast_used,
            }
        )
        for local_index, dataset_index in enumerate(test):
            prediction_rows.append(
                {
                    "heldout_source": source,
                    "id": rows[dataset_index]["id"],
                    "source_dataset": rows[dataset_index]["source_dataset"],
                    "is_contrast_row": rows[dataset_index]["source_dataset"] != source,
                    "y_true": int(y_all[dataset_index]),
                    "y_pred": int(pred[local_index]),
                    "score": f"{score[local_index]:.8f}",
                    "prompt_sha256": rows[dataset_index].get("metadata", {}).get("prompt_sha256", ""),
                }
            )
    write_csv(
        output_dir / "leave_one_source_out_metrics.csv",
        metric_rows,
        ["model", "split", "stratum", "n", "accuracy", "precision", "recall", "f1", "roc_auc", "brier", "heldout_source", "heldout_rows", "contrast_rows_added", "contrast_set_used"],
    )
    write_csv(
        output_dir / "leave_one_source_out_predictions.csv",
        prediction_rows,
        ["heldout_source", "id", "source_dataset", "is_contrast_row", "y_true", "y_pred", "score", "prompt_sha256"],
    )


def text_lexical_predict(row: dict) -> bool:
    prompt = row["prompt"].lower()
    return any(token in prompt for token in RISK_TOKENS)


def graph_rule_predict(row: dict) -> bool:
    nodes = set(row["graph_nodes"])
    return "harmful_intent" in nodes or {"instruction_override", "obfuscation"} <= nodes


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return (centre - margin) / denom, (centre + margin) / denom


def run_fairness_ci(balanced_rows: list[dict], output_dir: Path) -> None:
    methods: dict[str, Callable[[dict], bool]] = {
        "Text-Only Lexical Baseline": text_lexical_predict,
        "Graph-Based Baseline": graph_rule_predict,
    }
    rows = []
    for method, predictor in methods.items():
        for subgroup in sorted({row["safe_subgroup"] for row in balanced_rows if row["type"] == "safe"}):
            group = [row for row in balanced_rows if row["type"] == "safe" and row["safe_subgroup"] == subgroup]
            fp = sum(1 for row in group if predictor(row))
            low, high = wilson_interval(fp, len(group))
            rows.append(
                {
                    "method": method,
                    "subgroup": subgroup,
                    "n": len(group),
                    "false_positives": fp,
                    "fpr": f"{fp / len(group):.6f}" if group else "NA",
                    "fpr_ci95_low": format_metric(low),
                    "fpr_ci95_high": format_metric(high),
                }
            )
    write_csv(output_dir / "rq4_fpr_wilson_ci.csv", rows, ["method", "subgroup", "n", "false_positives", "fpr", "fpr_ci95_low", "fpr_ci95_high"])


def run_motif_enrichment(rows: list[dict], output_dir: Path) -> None:
    motif_counts = defaultdict(lambda: Counter())
    totals = Counter()
    for row in rows:
        label = row["type"]
        totals[label] += 1
        for motif in enumerate_motifs(row["graph_nodes"], row["graph_edges"]):
            motif_counts[motif][label] += 1
    out = []
    total_jb = totals["jailbreak"]
    total_safe = totals["safe"]
    for motif, counts in motif_counts.items():
        jb = counts["jailbreak"]
        safe = counts["safe"]
        jb_rate = jb / total_jb if total_jb else 0.0
        safe_rate = safe / total_safe if total_safe else 0.0
        odds_ratio = ((jb + 0.5) / (total_jb - jb + 0.5)) / ((safe + 0.5) / (total_safe - safe + 0.5))
        out.append(
            {
                "motif": motif,
                "jailbreak_count": jb,
                "safe_count": safe,
                "jailbreak_rate": f"{jb_rate:.6f}",
                "safe_rate": f"{safe_rate:.6f}",
                "smoothed_odds_ratio": f"{odds_ratio:.6f}",
            }
        )
    out.sort(key=lambda row: (float(row["smoothed_odds_ratio"]), int(row["jailbreak_count"])), reverse=True)
    write_csv(output_dir / "rq3_motif_enrichment.csv", out[:100], ["motif", "jailbreak_count", "safe_count", "jailbreak_rate", "safe_rate", "smoothed_odds_ratio"])


def plot_distributions(rows: list[dict], output_dir: Path) -> None:
    graph_depths = [row["graph_depth"] for row in rows]
    node_counts = [row["num_nodes"] for row in rows]
    types = [row["type"] for row in rows]
    output_dir.mkdir(parents=True, exist_ok=True)
    for values, label, filename in [
        (graph_depths, "Graph Depth", "graph_depth_by_type.png"),
        (node_counts, "Node Count", "node_count_by_type.png"),
    ]:
        plt.figure(figsize=(6, 4))
        for row_type in sorted(set(types)):
            subset = [value for value, typ in zip(values, types) if typ == row_type]
            plt.hist(subset, bins=range(min(values), max(values) + 2), alpha=0.55, label=row_type)
        plt.xlabel(label)
        plt.ylabel("Rows")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=180)
        plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/gjb_real_v1.jsonl")
    parser.add_argument("--balanced-dataset", default="data/processed/gjb_real_v1_rq4_balanced.jsonl")
    parser.add_argument("--output-dir", default="results/unblocked")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    rows = read_jsonl(PROJECT_ROOT / args.dataset)
    balanced_rows = read_jsonl(PROJECT_ROOT / args.balanced_dataset) if (PROJECT_ROOT / args.balanced_dataset).exists() else rows
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ablation = run_ablation(rows, output_dir / "e1", args.seed, args.test_size)
    bootstrap_cis(ablation["bootstrap_input"], output_dir / "e1", args.seed + 1, args.bootstrap)
    run_leave_one_source(rows, output_dir / "e1")
    run_motif_enrichment(rows, output_dir / "rq3")
    run_fairness_ci(balanced_rows, output_dir / "rq4")
    plot_distributions(rows, output_dir / "figures")

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "balanced_dataset": args.balanced_dataset,
        "output_dir": args.output_dir,
        "experiments_completed": [
            "source_balanced_detection_ablation",
            "bootstrap_confidence_intervals",
            "leave_one_source_out_detection",
            "deduplication_leakage_audit",
            "error_analysis",
            "calibration_bins",
            "rq3_motif_enrichment",
            "rq4_fpr_wilson_confidence_intervals",
            "graph_feature_distribution_plots",
        ],
        "source_balanced_split_summary": ablation["summary"],
        "prompt_text_logged": False,
    }
    write_json(output_dir / "unblocked_experiment_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
