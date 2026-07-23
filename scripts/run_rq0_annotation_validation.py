#!/usr/bin/env python3
"""Run RQ0 annotation validation once annotator templates are filled."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gjb.metrics import cohen_kappa, krippendorff_alpha_nominal, normalized_graph_edit_distance
from gjb.taxonomy import taxonomy_labels


def load_annotator_models(results_dir: Path) -> dict[str, dict]:
    """Map each annotator_id -> {provider, model} from LLM run metadata.

    Reads every results/rq/rq0/llm_annotation_run*.json and keeps the most
    recent provider/model per annotator (later created_at_utc wins), so the
    final model that produced each template is recorded alongside the metrics.
    """
    models: dict[str, dict] = {}
    for meta_path in sorted(results_dir.glob("llm_annotation_run*.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        created = meta.get("created_at_utc", "")
        for run in meta.get("runs", []):
            annotator = run.get("annotator")
            if not annotator:
                continue
            if annotator not in models or created >= models[annotator]["_created"]:
                models[annotator] = {
                    "provider": run.get("provider"),
                    "model": run.get("model"),
                    "run_metadata": meta_path.name,
                    "_created": created,
                }
    for info in models.values():
        info.pop("_created", None)
    return models


def read_annotations(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            node_text = record.get("node_labels_json", "").strip()
            edge_text = record.get("edge_list_json", "").strip()
            rows[record["id"]] = {
                "nodes_raw": node_text,
                "edges_raw": edge_text,
                "nodes": json.loads(node_text) if node_text else None,
                "edges": json.loads(edge_text) if edge_text else None,
                "notes": record.get("notes", ""),
            }
    return rows


def blank_count(annotations: dict[str, dict]) -> int:
    return sum(1 for row in annotations.values() if row["nodes"] is None or row["edges"] is None)


def _micro_f1(tp: float, size_left: int, size_right: int) -> float | None:
    # For two annotators FP+FN = (|L|-TP)+(|R|-TP), so micro-F1 = 2*TP/(|L|+|R|).
    denom = size_left + size_right
    if denom == 0:
        return None  # neither annotator drew any edges -> undefined, not 0
    return 2 * tp / denom


def pairwise_scores(left: dict[str, dict], right: dict[str, dict], labels: list[str]) -> dict:
    # Only score rows BOTH annotators actually completed (skip blanks). Some
    # rows are unrecoverable (model refusals), so we report agreement over the
    # completed intersection and surface N rather than blocking on full coverage.
    shared_ids = sorted(
        row_id
        for row_id in (set(left) & set(right))
        if left[row_id]["nodes"] is not None and right[row_id]["nodes"] is not None
    )

    # ---- Node agreement (A1) ----------------------------------------------
    # Cohen's kappa per label, but ONLY for labels at least one annotator used
    # on the shared rows. A label nobody applied carries no information; scoring
    # it as 1.0 (the old behaviour) spuriously inflated the mean.
    per_label_kappa: dict[str, float] = {}
    label_support: dict[str, int] = {}
    excluded_no_support: list[str] = []
    for label in labels:
        left_binary = ["1" if label in set(left[row_id]["nodes"]) else "0" for row_id in shared_ids]
        right_binary = ["1" if label in set(right[row_id]["nodes"]) else "0" for row_id in shared_ids]
        support = left_binary.count("1") + right_binary.count("1")
        label_support[label] = support
        if support == 0:
            excluded_no_support.append(label)
            continue
        per_label_kappa[label] = cohen_kappa(left_binary, right_binary)
    mean_node_kappa = (
        sum(per_label_kappa.values()) / len(per_label_kappa) if per_label_kappa else None
    )

    # ---- Edge agreement (A5) ----------------------------------------------
    # Replace kappa-over-132-possible-edges (sparsity paradox) with set-overlap
    # F1, split into PRESENCE (is there a connection, ignoring direction) and
    # DIRECTION (given a shared connection, do they orient it the same way).
    dir_tp = dir_l = dir_r = 0          # directed edge F1 components
    pres_tp = pres_l = pres_r = 0       # undirected (presence) F1 components
    jaccard_values: list[float] = []
    dir_match = dir_shared = 0          # direction agreement among shared pairs
    ged_values: list[float] = []
    for row_id in shared_ids:
        l_dir = {tuple(edge) for edge in left[row_id]["edges"]}
        r_dir = {tuple(edge) for edge in right[row_id]["edges"]}
        l_und = {frozenset(edge) for edge in l_dir}
        r_und = {frozenset(edge) for edge in r_dir}

        dir_tp += len(l_dir & r_dir); dir_l += len(l_dir); dir_r += len(r_dir)
        pres_tp += len(l_und & r_und); pres_l += len(l_und); pres_r += len(r_und)

        union = l_dir | r_dir
        jaccard_values.append(len(l_dir & r_dir) / len(union) if union else 1.0)

        for pair in (l_und & r_und):  # both connected this pair -> compare orientation
            dir_shared += 1
            if {e for e in l_dir if frozenset(e) == pair} == {e for e in r_dir if frozenset(e) == pair}:
                dir_match += 1

        ged_values.append(
            normalized_graph_edit_distance(
                {"nodes": left[row_id]["nodes"], "edges": left[row_id]["edges"]},
                {"nodes": right[row_id]["nodes"], "edges": right[row_id]["edges"]},
            )
        )

    return {
        "shared_rows": len(shared_ids),
        "mean_node_kappa": mean_node_kappa,
        "node_labels_scored": len(per_label_kappa),
        "node_labels_excluded_no_support": excluded_no_support,
        "per_label_node_kappa": {k: round(v, 4) for k, v in sorted(per_label_kappa.items())},
        "edge_f1_directed": _micro_f1(dir_tp, dir_l, dir_r),
        "edge_f1_presence": _micro_f1(pres_tp, pres_l, pres_r),
        "edge_direction_agreement": (dir_match / dir_shared) if dir_shared else None,
        "mean_edge_jaccard": sum(jaccard_values) / len(jaccard_values) if jaccard_values else None,
        "mean_normalized_ged": sum(ged_values) / len(ged_values) if ged_values else None,
    }


def joint_node_alpha(annotations: dict[str, dict], labels: list[str]) -> dict:
    """Krippendorff's alpha across ALL annotators at once (not pairwise).

    For each (row, label) a binary present/absent decision is collected from
    every annotator that completed the row (>=2 needed). Reports an overall
    pooled alpha plus per-label alpha, excluding labels with no support (A1).
    """
    all_ids = sorted({row_id for ann in annotations.values() for row_id in ann})
    per_label_alpha: dict[str, float] = {}
    label_support: dict[str, int] = {}
    pooled_units: list[list[str]] = []
    for label in labels:
        units: list[list[str]] = []
        for row_id in all_ids:
            vals = [
                "1" if label in set(ann[row_id]["nodes"]) else "0"
                for ann in annotations.values()
                if row_id in ann and ann[row_id]["nodes"] is not None
            ]
            if len(vals) >= 2:
                units.append(vals)
        support = sum(v.count("1") for v in units)
        label_support[label] = support
        if support == 0:
            continue
        alpha = krippendorff_alpha_nominal(units)
        if alpha is not None:
            per_label_alpha[label] = round(alpha, 4)
        pooled_units.extend(units)
    overall = krippendorff_alpha_nominal(pooled_units)
    return {
        "annotators": sorted(annotations),
        "node_krippendorff_alpha_overall": overall,
        "node_krippendorff_alpha_per_label": per_label_alpha,
        "node_labels_excluded_no_support": [l for l in labels if label_support.get(l, 0) == 0],
    }


def _mean(values: list) -> float | None:
    nums = [v for v in values if v is not None]
    return sum(nums) / len(nums) if nums else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", default="data/annotations/rq0")
    parser.add_argument("--output", default="results/rq/rq0/rq0_annotation_validation.json")
    parser.add_argument("--run-metadata-dir", default="results/rq/rq0")
    args = parser.parse_args()

    annotation_dir = PROJECT_ROOT / args.annotation_dir
    files = sorted(annotation_dir.glob("annotator_*_template.csv"))
    annotations = {path.stem: read_annotations(path) for path in files}
    blank_by_file = {path.name: blank_count(annotations[path.stem]) for path in files}
    completed_ids = [
        {row_id for row_id, row in annotations[path.stem].items() if row["nodes"] is not None}
        for path in files
    ]
    common_completed = sorted(set.intersection(*completed_ids)) if completed_ids else []

    # Record which LLM produced each annotator's labels (provider + model).
    models_by_id = load_annotator_models(PROJECT_ROOT / args.run_metadata_dir)
    annotator_models = {
        path.stem: models_by_id.get(path.stem.replace("_template", ""), {"provider": None, "model": None})
        for path in files
    }

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "blocked_pending_human_annotation",
        "annotation_files": [str(path.relative_to(PROJECT_ROOT)) for path in files],
        "annotator_models": annotator_models,
        "blank_rows_by_file": blank_by_file,
        "rows_completed_by_all": len(common_completed),
        "prompt_text_logged": False,
        "targets": {
            "node_alpha_min": 0.667,
            "edge_f1_directed_min": 0.60,
            "mean_normalized_ged_max": 0.30,
        },
        "joint_agreement": None,
        "pairwise_results": [],
    }

    # Score agreement over the rows every annotator completed. Rows that any
    # annotator could not label (e.g. model refusals) are excluded, not blocking.
    if common_completed:
        labels = sorted(taxonomy_labels())
        for left_name, right_name in itertools.combinations(sorted(annotations), 2):
            scores = pairwise_scores(annotations[left_name], annotations[right_name], labels)
            payload["pairwise_results"].append({"annotator_pair": [left_name, right_name], **scores})

        # Joint agreement across ALL annotators at once: Krippendorff's alpha on
        # nodes, and edge metrics aggregated as the mean of the pairwise values.
        joint = joint_node_alpha(annotations, labels)
        joint["edge_f1_directed_mean"] = _mean([r["edge_f1_directed"] for r in payload["pairwise_results"]])
        joint["edge_f1_presence_mean"] = _mean([r["edge_f1_presence"] for r in payload["pairwise_results"]])
        joint["edge_direction_agreement_mean"] = _mean([r["edge_direction_agreement"] for r in payload["pairwise_results"]])
        joint["mean_edge_jaccard_mean"] = _mean([r["mean_edge_jaccard"] for r in payload["pairwise_results"]])
        joint["mean_normalized_ged_mean"] = _mean([r["mean_normalized_ged"] for r in payload["pairwise_results"]])
        payload["joint_agreement"] = joint

        t = payload["targets"]
        node_ok = joint["node_krippendorff_alpha_overall"] is not None and joint["node_krippendorff_alpha_overall"] >= t["node_alpha_min"]
        edge_ok = joint["edge_f1_directed_mean"] is not None and joint["edge_f1_directed_mean"] >= t["edge_f1_directed_min"]
        ged_ok = joint["mean_normalized_ged_mean"] is not None and joint["mean_normalized_ged_mean"] <= t["mean_normalized_ged_max"]
        payload["status"] = "passed" if (node_ok and edge_ok and ged_ok) else "needs_taxonomy_revision"

    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
