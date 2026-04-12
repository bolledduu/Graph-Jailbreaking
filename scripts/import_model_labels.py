#!/usr/bin/env python3
"""Import real cross-model success labels into a local processed dataset.

This script validates row IDs and prompt hashes before writing labels. It does
not print prompt text or model responses.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gjb.io import read_jsonl, write_jsonl
from gjb.labeling import parse_optional_bool, project_relative, require_unique, row_prompt_hash


SUCCESS_FIELDS = {"model_gpt4_success", "model_llama_success"}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_success_field(field_name: str) -> None:
    if field_name not in SUCCESS_FIELDS:
        raise ValueError(f"Unsupported success field {field_name!r}. Use one of {sorted(SUCCESS_FIELDS)}")


def clone_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cloned = []
    for row in rows:
        copied = dict(row)
        copied["metadata"] = dict(row.get("metadata") or {})
        cloned.append(copied)
    return cloned


def import_labels(
    dataset_rows: list[dict[str, Any]],
    label_rows: list[dict[str, str]],
    label_path: Path,
    output_path: Path,
    model_a_success_field: str,
    model_b_success_field: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_success_field(model_a_success_field)
    validate_success_field(model_b_success_field)
    require_unique([row["id"] for row in dataset_rows], "dataset id")

    nonempty_label_ids = [row.get("id", "").strip() for row in label_rows if row.get("id", "").strip()]
    require_unique(nonempty_label_ids, "label id")

    updated_rows = clone_rows(dataset_rows)
    by_id = {row["id"]: row for row in updated_rows}
    errors: list[str] = []
    imported_ids: set[str] = set()
    partial_ids: set[str] = set()
    blank_rows = 0

    now = datetime.now(timezone.utc).isoformat()
    for csv_index, record in enumerate(label_rows, start=2):
        row_id = record.get("id", "").strip()
        if not row_id:
            blank_rows += 1
            continue
        if row_id not in by_id:
            errors.append(f"CSV row {csv_index}: unknown dataset id {row_id!r}")
            continue

        row = by_id[row_id]
        supplied_hash = record.get("prompt_sha256", "").strip()
        expected_hash = row_prompt_hash(row)
        if supplied_hash and supplied_hash != expected_hash:
            errors.append(f"CSV row {csv_index}: prompt hash mismatch for id {row_id!r}")
            continue

        try:
            model_a_success = parse_optional_bool(record.get("model_a_success"), "model_a_success")
            model_b_success = parse_optional_bool(record.get("model_b_success"), "model_b_success")
        except ValueError as exc:
            errors.append(f"CSV row {csv_index}: {exc}")
            continue

        if model_a_success is None and model_b_success is None:
            blank_rows += 1
            continue

        if model_a_success is not None:
            row[model_a_success_field] = model_a_success
        if model_b_success is not None:
            row[model_b_success_field] = model_b_success

        metadata = dict(row.get("metadata") or {})
        imports = list(metadata.get("model_label_imports") or [])
        imports.append(
            {
                "imported_at_utc": now,
                "label_file": project_relative(label_path, PROJECT_ROOT),
                "model_a_name": record.get("model_a_name", "").strip(),
                "model_a_version": record.get("model_a_version", "").strip(),
                "model_a_success_field": model_a_success_field,
                "model_a_response_path": record.get("model_a_response_path", "").strip(),
                "model_b_name": record.get("model_b_name", "").strip(),
                "model_b_version": record.get("model_b_version", "").strip(),
                "model_b_success_field": model_b_success_field,
                "model_b_response_path": record.get("model_b_response_path", "").strip(),
                "judge_protocol": record.get("judge_protocol", "").strip(),
            }
        )
        metadata["model_label_imports"] = imports
        row["metadata"] = metadata
        imported_ids.add(row_id)
        if model_a_success is None or model_b_success is None:
            partial_ids.add(row_id)

    if errors:
        raise ValueError("Model label import failed validation:\n" + "\n".join(errors[:25]))

    paired_labeled_rows = [
        row
        for row in updated_rows
        if row["type"] in {"jailbreak", "multiturn"}
        and row.get(model_a_success_field) is not None
        and row.get(model_b_success_field) is not None
    ]
    target_counts = Counter(str(row.get(model_b_success_field)) for row in paired_labeled_rows)
    status = (
        "ready_for_rq2_transferability"
        if len(paired_labeled_rows) >= 50 and len(target_counts) >= 2
        else "blocked_pending_cross_model_success_labels"
    )
    summary = {
        "created_at_utc": now,
        "status": status,
        "dataset": project_relative(output_path, PROJECT_ROOT),
        "label_file": project_relative(label_path, PROJECT_ROOT),
        "rows_in_dataset": len(updated_rows),
        "rows_in_label_file": len(label_rows),
        "rows_with_any_imported_label": len(imported_ids),
        "rows_with_partial_imported_label": len(partial_ids),
        "blank_label_rows": blank_rows,
        "paired_labeled_attack_rows": len(paired_labeled_rows),
        "target_success_counts": dict(sorted(target_counts.items())),
        "model_a_success_field": model_a_success_field,
        "model_b_success_field": model_b_success_field,
        "prompt_text_logged": False,
        "response_text_logged": False,
    }
    return updated_rows, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/gjb_real_v1.jsonl")
    parser.add_argument("--labels", default="data/annotations/rq2/model_success_template.csv")
    parser.add_argument("--output", default="data/processed/gjb_real_v1_with_model_labels.jsonl")
    parser.add_argument("--summary", default="results/rq/rq2/model_label_import_summary.json")
    parser.add_argument("--model-a-success-field", default="model_gpt4_success")
    parser.add_argument("--model-b-success-field", default="model_llama_success")
    args = parser.parse_args()

    dataset_path = PROJECT_ROOT / args.dataset
    label_path = PROJECT_ROOT / args.labels
    output_path = PROJECT_ROOT / args.output
    summary_path = PROJECT_ROOT / args.summary

    dataset_rows = read_jsonl(dataset_path)
    label_rows = read_csv_rows(label_path)
    updated_rows, summary = import_labels(
        dataset_rows,
        label_rows,
        label_path,
        output_path,
        args.model_a_success_field,
        args.model_b_success_field,
    )
    write_jsonl(updated_rows, output_path)
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
