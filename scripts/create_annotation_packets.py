#!/usr/bin/env python3
"""Create RQ0 annotation packets and model-label templates.

By default this script does not duplicate raw prompt text. It stores row IDs,
source metadata, prompt hashes, and blank annotation/label columns so annotators
and model-evaluation scripts can join back to the canonical dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gjb.io import read_jsonl


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stable_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    if len(rows) <= n:
        return sorted(rows, key=lambda row: row["id"])
    return sorted(rng.sample(rows, n), key=lambda row: row["id"])


def metadata_row(row: dict) -> dict:
    metadata = row.get("metadata", {})
    return {
        "id": row["id"],
        "source_dataset": row["source_dataset"],
        "source_record_id": row.get("source_record_id") or "",
        "source_category": row.get("source_category") or "",
        "prompt_sha256": metadata.get("prompt_sha256", ""),
        "prompt_char_length": len(row["prompt"]),
        "attack_type_first_pass": row["attack_type"],
        "difficulty_first_pass": row["difficulty"],
        "graph_nodes_first_pass_json": json.dumps(row["graph_nodes"], sort_keys=True),
        "graph_edges_first_pass_json": json.dumps(row["graph_edges"], sort_keys=True),
    }


def create_rq0_packets(rows: list[dict], sample_size: int, seed: int) -> dict:
    harmful = [row for row in rows if row["type"] == "jailbreak"]
    sample = stable_sample(harmful, sample_size, seed)
    base_fields = [
        "id",
        "source_dataset",
        "source_record_id",
        "source_category",
        "prompt_sha256",
        "prompt_char_length",
        "attack_type_first_pass",
        "difficulty_first_pass",
        "graph_nodes_first_pass_json",
        "graph_edges_first_pass_json",
    ]
    sample_rows = [metadata_row(row) for row in sample]
    write_csv(PROJECT_ROOT / "data" / "annotations" / "rq0" / "rq0_sample_ids.csv", sample_rows, base_fields)

    annotation_fields = [
        "id",
        "annotator_id",
        "node_labels_json",
        "edge_list_json",
        "notes",
    ]
    for annotator_id in ["annotator_a", "annotator_b", "annotator_c"]:
        template_rows = [
            {
                "id": row["id"],
                "annotator_id": annotator_id,
                "node_labels_json": "",
                "edge_list_json": "",
                "notes": "",
            }
            for row in sample
        ]
        write_csv(
            PROJECT_ROOT / "data" / "annotations" / "rq0" / f"{annotator_id}_template.csv",
            template_rows,
            annotation_fields,
        )

    return {
        "sample_size": len(sample),
        "sample_seed": seed,
        "sample_path": "data/annotations/rq0/rq0_sample_ids.csv",
        "annotator_templates": [
            "data/annotations/rq0/annotator_a_template.csv",
            "data/annotations/rq0/annotator_b_template.csv",
            "data/annotations/rq0/annotator_c_template.csv",
        ],
        "by_source_dataset": dict(sorted(Counter(row["source_dataset"] for row in sample).items())),
        "prompt_text_duplicated": False,
    }


def create_model_success_template(rows: list[dict]) -> dict:
    harmful = [row for row in rows if row["type"] == "jailbreak"]
    fields = [
        "id",
        "source_dataset",
        "source_record_id",
        "source_category",
        "prompt_sha256",
        "prompt_char_length",
        "model_a_name",
        "model_a_version",
        "model_a_success",
        "model_a_response_path",
        "model_b_name",
        "model_b_version",
        "model_b_success",
        "model_b_response_path",
        "judge_protocol",
        "notes",
    ]
    template_rows = []
    for row in harmful:
        meta = metadata_row(row)
        template_rows.append(
            {
                "id": meta["id"],
                "source_dataset": meta["source_dataset"],
                "source_record_id": meta["source_record_id"],
                "source_category": meta["source_category"],
                "prompt_sha256": meta["prompt_sha256"],
                "prompt_char_length": meta["prompt_char_length"],
                "model_a_name": "",
                "model_a_version": "",
                "model_a_success": "",
                "model_a_response_path": "",
                "model_b_name": "",
                "model_b_version": "",
                "model_b_success": "",
                "model_b_response_path": "",
                "judge_protocol": "",
                "notes": "",
            }
        )
    path = PROJECT_ROOT / "data" / "annotations" / "rq2" / "model_success_template.csv"
    write_csv(path, template_rows, fields)
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "rows": len(template_rows),
        "prompt_text_duplicated": False,
    }


def create_multiturn_template(rows: list[dict], sample_size: int, seed: int) -> dict:
    harmful = [row for row in rows if row["type"] == "jailbreak"]
    sample = stable_sample(harmful, sample_size, seed)
    fields = [
        "base_id",
        "variant_id",
        "source_dataset",
        "prompt_sha256",
        "turns_json",
        "model_name",
        "model_version",
        "attack_success",
        "judge_protocol",
        "notes",
    ]
    template_rows = []
    for row in sample:
        metadata = row.get("metadata", {})
        for variant_index in range(1, 3):
            template_rows.append(
                {
                    "base_id": row["id"],
                    "variant_id": f"{row['id']}_mt_{variant_index}",
                    "source_dataset": row["source_dataset"],
                    "prompt_sha256": metadata.get("prompt_sha256", ""),
                    "turns_json": "",
                    "model_name": "",
                    "model_version": "",
                    "attack_success": "",
                    "judge_protocol": "",
                    "notes": "",
                }
            )
    path = PROJECT_ROOT / "data" / "annotations" / "e5" / "multiturn_variants_template.csv"
    write_csv(path, template_rows, fields)
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "base_prompt_sample_size": len(sample),
        "variant_rows": len(template_rows),
        "sample_seed": seed,
        "prompt_text_duplicated": False,
    }


def write_guidelines() -> None:
    taxonomy_path = PROJECT_ROOT / "configs" / "taxonomy.json"
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    labels = "\n".join(f"- `{node['label']}`: {node['definition']}" for node in taxonomy["nodes"])
    text = f"""# RQ0 Annotation Packet Guide

Use the 12-label taxonomy from `configs/taxonomy.json`.

{labels}

## Rules

- Annotate each row independently.
- Use `node_labels_json` as a JSON list of node labels, for example:
  `["roleplay", "fiction", "harmful_intent"]`.
- Use `edge_list_json` as a JSON list of directed pairs, for example:
  `[["roleplay", "fiction"], ["fiction", "harmful_intent"]]`.
- Do not discuss labels across annotators until all templates are complete.
- Use `notes` only for ambiguity, not for final labels.
- Prompt text is not duplicated in these templates. Join by `id` against the
  canonical processed dataset when an annotator interface needs to display the
  prompt.
"""
    (PROJECT_ROOT / "data" / "annotations" / "rq0" / "ANNOTATION_GUIDE.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/gjb_real_v1.jsonl")
    parser.add_argument("--rq0-sample-size", type=int, default=200)
    parser.add_argument("--e5-base-sample-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=30)
    args = parser.parse_args()

    rows = read_jsonl(PROJECT_ROOT / args.dataset)
    write_guidelines()
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "rq0": create_rq0_packets(rows, args.rq0_sample_size, args.seed),
        "rq2": create_model_success_template(rows),
        "e5": create_multiturn_template(rows, args.e5_base_sample_size, args.seed),
    }
    summary_path = PROJECT_ROOT / "data" / "annotations" / "annotation_packet_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
