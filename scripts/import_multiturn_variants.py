#!/usr/bin/env python3
"""Import real multi-turn variants into a local processed dataset."""

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

from gjb.annotation import annotate_prompt
from gjb.io import read_jsonl, write_jsonl
from gjb.labeling import (
    parse_optional_bool,
    parse_turns_json,
    project_relative,
    require_unique,
    row_prompt_hash,
    sha256_text,
)
from gjb.schema import validate_row
from gjb.taxonomy import taxonomy_labels


SUCCESS_FIELDS = {"model_gpt4_success", "model_llama_success"}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_variant_row(
    base_row: dict[str, Any],
    record: dict[str, str],
    turns: list[str],
    success: bool,
    success_field: str,
    imported_at_utc: str,
    template_path: Path,
) -> dict[str, Any]:
    full_prompt = "\n".join(turns)
    graph = annotate_prompt(full_prompt, is_harmful=True)
    variant_id = record["variant_id"].strip()
    metadata = {
        "base_id": base_row["id"],
        "base_prompt_sha256": row_prompt_hash(base_row),
        "prompt_sha256": sha256_text(full_prompt),
        "turn_count": len(turns),
        "multiturn_import": {
            "imported_at_utc": imported_at_utc,
            "template_file": project_relative(template_path, PROJECT_ROOT),
            "model_name": record.get("model_name", "").strip(),
            "model_version": record.get("model_version", "").strip(),
            "success_field": success_field,
            "judge_protocol": record.get("judge_protocol", "").strip(),
        },
    }
    row = {
        "id": variant_id,
        "type": "multiturn",
        "source_dataset": base_row["source_dataset"],
        "prompt": full_prompt,
        "conversation": turns,
        "attack_type": graph["attack_type"],
        "difficulty": graph["difficulty"],
        "graph_nodes": graph["graph_nodes"],
        "graph_edges": graph["graph_edges"],
        "graph_depth": graph["graph_depth"],
        "num_nodes": graph["num_nodes"],
        "model_gpt4_response": None,
        "model_gpt4_success": None,
        "model_llama_response": None,
        "model_llama_success": None,
        "is_jailbreak": True,
        "is_harmful": True,
        "safe_subgroup": None,
        "source_record_id": base_row["id"],
        "source_category": base_row.get("source_category"),
        "metadata": metadata,
    }
    row[success_field] = success
    validate_row(row, taxonomy_labels())
    return row


def import_variants(
    dataset_rows: list[dict[str, Any]],
    template_rows: list[dict[str, str]],
    template_path: Path,
    output_path: Path,
    success_field: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if success_field not in SUCCESS_FIELDS:
        raise ValueError(f"Unsupported success field {success_field!r}. Use one of {sorted(SUCCESS_FIELDS)}")
    require_unique([row["id"] for row in dataset_rows], "dataset id")

    by_id = {row["id"]: row for row in dataset_rows}
    existing_ids = set(by_id)
    variant_ids = [row.get("variant_id", "").strip() for row in template_rows if row.get("variant_id", "").strip()]
    require_unique(variant_ids, "variant id")

    imported: list[dict[str, Any]] = []
    errors: list[str] = []
    blank_rows = 0
    partial_rows = 0
    now = datetime.now(timezone.utc).isoformat()

    for csv_index, record in enumerate(template_rows, start=2):
        base_id = record.get("base_id", "").strip()
        variant_id = record.get("variant_id", "").strip()
        if not base_id and not variant_id:
            blank_rows += 1
            continue
        if not base_id or not variant_id:
            partial_rows += 1
            errors.append(f"CSV row {csv_index}: base_id and variant_id are both required")
            continue
        if base_id not in by_id:
            errors.append(f"CSV row {csv_index}: unknown base_id {base_id!r}")
            continue
        if variant_id in existing_ids:
            errors.append(f"CSV row {csv_index}: variant_id already exists in dataset: {variant_id!r}")
            continue

        base_row = by_id[base_id]
        supplied_hash = record.get("prompt_sha256", "").strip()
        expected_hash = row_prompt_hash(base_row)
        if supplied_hash and supplied_hash != expected_hash:
            errors.append(f"CSV row {csv_index}: prompt hash mismatch for base_id {base_id!r}")
            continue

        try:
            turns = parse_turns_json(record.get("turns_json"), "turns_json")
            success = parse_optional_bool(record.get("attack_success"), "attack_success")
        except ValueError as exc:
            errors.append(f"CSV row {csv_index}: {exc}")
            continue

        if turns is None and success is None:
            blank_rows += 1
            continue
        if turns is None or success is None:
            partial_rows += 1
            errors.append(f"CSV row {csv_index}: turns_json and attack_success must be filled together")
            continue

        imported.append(build_variant_row(base_row, record, turns, success, success_field, now, template_path))
        existing_ids.add(variant_id)

    if errors:
        raise ValueError("Multi-turn import failed validation:\n" + "\n".join(errors[:25]))

    updated_rows = [*dataset_rows, *imported]
    by_type = Counter(row["type"] for row in updated_rows)
    status = "ready_for_e5_asr" if by_type["multiturn"] > 0 else "blocked_pending_real_multiturn_variants_and_success_labels"
    summary = {
        "created_at_utc": now,
        "status": status,
        "dataset": project_relative(output_path, PROJECT_ROOT),
        "template_file": project_relative(template_path, PROJECT_ROOT),
        "success_field": success_field,
        "rows_in_dataset_before_import": len(dataset_rows),
        "rows_in_dataset_after_import": len(updated_rows),
        "template_rows": len(template_rows),
        "imported_multiturn_rows": len(imported),
        "blank_template_rows": blank_rows,
        "partial_template_rows": partial_rows,
        "by_type_after_import": dict(sorted(by_type.items())),
        "prompt_text_logged": False,
        "response_text_logged": False,
    }
    return updated_rows, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/gjb_real_v1.jsonl")
    parser.add_argument("--template", default="data/annotations/e5/multiturn_variants_template.csv")
    parser.add_argument("--output", default="data/processed/gjb_real_v1_with_multiturn.jsonl")
    parser.add_argument("--summary", default="results/rq/e5/multiturn_import_summary.json")
    parser.add_argument("--success-field", default="model_llama_success")
    args = parser.parse_args()

    dataset_path = PROJECT_ROOT / args.dataset
    template_path = PROJECT_ROOT / args.template
    output_path = PROJECT_ROOT / args.output
    summary_path = PROJECT_ROOT / args.summary

    dataset_rows = read_jsonl(dataset_path)
    template_rows = read_csv_rows(template_path)
    updated_rows, summary = import_variants(dataset_rows, template_rows, template_path, output_path, args.success_field)
    write_jsonl(updated_rows, output_path)
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
