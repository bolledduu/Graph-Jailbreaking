#!/usr/bin/env python3
"""Run E5 single-turn vs multi-turn ASR analysis when real labels exist."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gjb.io import read_jsonl


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return (centre - margin) / denom, (centre + margin) / denom


def fmt(value: float) -> str:
    return "NA" if math.isnan(value) else f"{value:.6f}"


def write_status_table(path: Path, status: str, n_labeled: int, reason: str) -> None:
    write_csv(
        path,
        [
            {
                "experiment": "E5 Multi-Turn vs Single-Turn ASR",
                "status": status,
                "n_labeled": n_labeled,
                "reason": reason,
            }
        ],
        ["experiment", "status", "n_labeled", "reason"],
    )


def blocked_payload(args: argparse.Namespace, reason: str, labeled_counts: Counter) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "E5",
        "status": "blocked_pending_real_multiturn_variants_and_success_labels",
        "reason": reason,
        "dataset": args.dataset,
        "success_field": args.success_field,
        "labeled_counts_by_type": dict(sorted(labeled_counts.items())),
        "prompt_text_logged": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/gjb_real_v1.jsonl")
    parser.add_argument("--output-dir", default="results/rq/e5")
    parser.add_argument("--success-field", default="model_llama_success")
    args = parser.parse_args()

    rows = read_jsonl(PROJECT_ROOT / args.dataset)
    labeled = [
        row
        for row in rows
        if row["type"] in {"jailbreak", "multiturn"} and row.get(args.success_field) is not None
    ]
    counts = Counter(row["type"] for row in labeled)
    reason = ""
    if counts["jailbreak"] == 0:
        reason = f"No labeled single-turn jailbreak rows found for {args.success_field}."
    elif counts["multiturn"] == 0:
        reason = f"No labeled multi-turn rows found for {args.success_field}."

    output_dir = PROJECT_ROOT / args.output_dir
    if reason:
        payload = blocked_payload(args, reason, counts)
        write_json(output_dir / "e5_asr_status.json", payload)
        write_status_table(PROJECT_ROOT / "results" / "tables" / "table_10_asr_status.csv", "blocked", len(labeled), reason)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    asr_rows: list[dict[str, Any]] = []
    for group_name, group_rows in [
        ("single_turn", [row for row in labeled if row["type"] == "jailbreak"]),
        ("multi_turn", [row for row in labeled if row["type"] == "multiturn"]),
    ]:
        successes = sum(1 for row in group_rows if row[args.success_field] is True)
        low, high = wilson(successes, len(group_rows))
        asr_rows.append(
            {
                "success_field": args.success_field,
                "group": group_name,
                "n": len(group_rows),
                "successes": successes,
                "asr": fmt(successes / len(group_rows)),
                "asr_ci95_low": fmt(low),
                "asr_ci95_high": fmt(high),
            }
        )

    single = next(row for row in asr_rows if row["group"] == "single_turn")
    multi = next(row for row in asr_rows if row["group"] == "multi_turn")
    single_asr = float(single["asr"])
    multi_asr = float(multi["asr"])
    comparison_rows = [
        {
            "success_field": args.success_field,
            "single_turn_asr": single["asr"],
            "multi_turn_asr": multi["asr"],
            "risk_difference_multi_minus_single": fmt(multi_asr - single_asr),
            "risk_ratio_multi_over_single": fmt(multi_asr / single_asr) if single_asr > 0 else "NA",
        }
    ]

    write_csv(output_dir / "e5_asr_by_group.csv", asr_rows, ["success_field", "group", "n", "successes", "asr", "asr_ci95_low", "asr_ci95_high"])
    write_csv(
        output_dir / "e5_asr_comparison.csv",
        comparison_rows,
        ["success_field", "single_turn_asr", "multi_turn_asr", "risk_difference_multi_minus_single", "risk_ratio_multi_over_single"],
    )
    write_csv(
        PROJECT_ROOT / "results" / "tables" / "table_10_asr_real.csv",
        asr_rows,
        ["success_field", "group", "n", "successes", "asr", "asr_ci95_low", "asr_ci95_high"],
    )
    write_status_table(PROJECT_ROOT / "results" / "tables" / "table_10_asr_status.csv", "completed", len(labeled), "Real single-turn and multi-turn success labels available; ASR metrics written to table_10_asr_real.csv.")

    status = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "E5",
        "status": "completed",
        "dataset": args.dataset,
        "output_dir": args.output_dir,
        "success_field": args.success_field,
        "labeled_counts_by_type": dict(sorted(counts.items())),
        "result_table": "results/tables/table_10_asr_real.csv",
        "prompt_text_logged": False,
    }
    write_json(output_dir / "e5_asr_status.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
