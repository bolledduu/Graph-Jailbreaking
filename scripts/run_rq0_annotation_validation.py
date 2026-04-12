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

from gjb.metrics import cohen_kappa, normalized_graph_edit_distance
from gjb.taxonomy import taxonomy_labels


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


def pairwise_scores(left: dict[str, dict], right: dict[str, dict], labels: list[str]) -> dict:
    shared_ids = sorted(set(left) & set(right))
    all_edges = [(source, target) for source in labels for target in labels if source != target]
    node_kappas = []
    for label in labels:
        left_binary = ["1" if label in set(left[row_id]["nodes"]) else "0" for row_id in shared_ids]
        right_binary = ["1" if label in set(right[row_id]["nodes"]) else "0" for row_id in shared_ids]
        node_kappas.append(cohen_kappa(left_binary, right_binary))

    left_edge_labels = []
    right_edge_labels = []
    ged_values = []
    for row_id in shared_ids:
        left_edges = {tuple(edge) for edge in left[row_id]["edges"]}
        right_edges = {tuple(edge) for edge in right[row_id]["edges"]}
        for edge in all_edges:
            left_edge_labels.append("1" if edge in left_edges else "0")
            right_edge_labels.append("1" if edge in right_edges else "0")
        ged_values.append(
            normalized_graph_edit_distance(
                {"nodes": left[row_id]["nodes"], "edges": left[row_id]["edges"]},
                {"nodes": right[row_id]["nodes"], "edges": right[row_id]["edges"]},
            )
        )

    return {
        "shared_rows": len(shared_ids),
        "mean_node_kappa": sum(node_kappas) / len(node_kappas) if node_kappas else None,
        "edge_kappa": cohen_kappa(left_edge_labels, right_edge_labels) if left_edge_labels else None,
        "mean_normalized_ged": sum(ged_values) / len(ged_values) if ged_values else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", default="data/annotations/rq0")
    parser.add_argument("--output", default="results/rq/rq0/rq0_annotation_validation.json")
    args = parser.parse_args()

    annotation_dir = PROJECT_ROOT / args.annotation_dir
    files = sorted(annotation_dir.glob("annotator_*_template.csv"))
    annotations = {path.stem: read_annotations(path) for path in files}
    blank_by_file = {path.name: blank_count(annotations[path.stem]) for path in files}
    is_complete = bool(files) and all(count == 0 for count in blank_by_file.values())

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ready" if is_complete else "blocked_pending_human_annotation",
        "annotation_files": [str(path.relative_to(PROJECT_ROOT)) for path in files],
        "blank_rows_by_file": blank_by_file,
        "prompt_text_logged": False,
        "targets": {
            "node_kappa_min": 0.70,
            "edge_kappa_min": 0.60,
            "mean_normalized_ged_max": 0.30,
        },
        "pairwise_results": [],
    }

    if is_complete:
        labels = sorted(taxonomy_labels())
        for left_name, right_name in itertools.combinations(sorted(annotations), 2):
            scores = pairwise_scores(annotations[left_name], annotations[right_name], labels)
            payload["pairwise_results"].append({"annotator_pair": [left_name, right_name], **scores})
        payload["status"] = "passed" if all(
            result["mean_node_kappa"] >= payload["targets"]["node_kappa_min"]
            and result["edge_kappa"] >= payload["targets"]["edge_kappa_min"]
            and result["mean_normalized_ged"] <= payload["targets"]["mean_normalized_ged_max"]
            for result in payload["pairwise_results"]
        ) else "needs_taxonomy_revision"

    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
