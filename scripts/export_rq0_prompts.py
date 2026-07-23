#!/usr/bin/env python3
"""Export RQ0 sample prompts by joining rq0_sample_ids.csv to the processed dataset.

Output is local-only (contains raw prompt text). Default path is gitignored.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gjb.io import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Export RQ0 prompts for annotators or LLM runs.")
    parser.add_argument("--dataset", default="data/processed/gjb_real_v1.jsonl")
    parser.add_argument("--sample-ids", default="data/annotations/rq0/rq0_sample_ids.csv")
    parser.add_argument(
        "--output",
        default="data/annotations/rq0/rq0_sample_prompts_local.csv",
        help="Local CSV with id + prompt_text (not for public commit).",
    )
    args = parser.parse_args()

    sample_path = PROJECT_ROOT / args.sample_ids
    with sample_path.open(newline="", encoding="utf-8") as handle:
        sample_rows = list(csv.DictReader(handle))
    sample_ids = [row["id"] for row in sample_rows]
    id_set = set(sample_ids)

    dataset_path = PROJECT_ROOT / args.dataset
    by_id = {row["id"]: row for row in read_jsonl(dataset_path) if row["id"] in id_set}
    missing = [row_id for row_id in sample_ids if row_id not in by_id]
    if missing:
        print(f"ERROR: {len(missing)} RQ0 ids missing from {dataset_path}", file=sys.stderr)
        print("First missing:", missing[:5], file=sys.stderr)
        return 1

    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "source_dataset",
        "attack_type",
        "difficulty",
        "prompt_text",
        "graph_nodes_first_pass_json",
        "graph_edges_first_pass_json",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for sample in sample_rows:
            row = by_id[sample["id"]]
            writer.writerow(
                {
                    "id": row["id"],
                    "source_dataset": row["source_dataset"],
                    "attack_type": row["attack_type"],
                    "difficulty": row["difficulty"],
                    "prompt_text": row["prompt"],
                    "graph_nodes_first_pass_json": json.dumps(row["graph_nodes"], sort_keys=True),
                    "graph_edges_first_pass_json": json.dumps(row["graph_edges"], sort_keys=True),
                }
            )

    print(f"Wrote {len(sample_ids)} prompts to {output.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
