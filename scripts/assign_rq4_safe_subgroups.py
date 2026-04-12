#!/usr/bin/env python3
"""Assign real safe prompts into the six workbook RQ4 safe subgroups.

This is a deterministic heuristic assignment for baseline experiments. It does
not change prompt text and does not fabricate rows. It updates only safe rows'
`safe_subgroup` field and records an audit CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gjb.io import read_jsonl, write_jsonl


SUBGROUPS = [
    "formal_academic",
    "informal_colloquial",
    "non_native_english",
    "security_technical",
    "medical_health",
    "creative_writing",
]

KEYWORDS = {
    "formal_academic": [
        "research",
        "analyze",
        "analysis",
        "explain",
        "policy",
        "study",
        "evidence",
        "academic",
        "theory",
        "summarize",
    ],
    "informal_colloquial": [
        "u",
        "yo",
        "pls",
        "bro",
        "lol",
        "help me",
        "what's",
        "kinda",
        "stuff",
        "easy way",
    ],
    "non_native_english": [
        "please explain",
        "i want learn",
        "how to know",
        "is working",
        "in simple english",
        "basic",
        "meaning",
        "can you describe",
    ],
    "security_technical": [
        "security",
        "password",
        "malware",
        "hacking",
        "vulnerability",
        "penetration",
        "encryption",
        "network",
        "threat",
        "software",
        "computer",
        "privacy",
    ],
    "medical_health": [
        "medical",
        "health",
        "doctor",
        "patient",
        "medicine",
        "dosage",
        "symptom",
        "treatment",
        "drug",
        "hospital",
        "mental",
    ],
    "creative_writing": [
        "story",
        "write",
        "scene",
        "character",
        "fiction",
        "novel",
        "dialogue",
        "poem",
        "screenplay",
        "creative",
    ],
}


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def score_prompt(prompt: str) -> dict[str, int]:
    lowered = prompt.lower()
    token_set = tokens(prompt)
    scores: dict[str, int] = {}
    for subgroup, keywords in KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if " " in keyword:
                score += 2 if keyword in lowered else 0
            elif keyword in token_set:
                score += 1
        scores[subgroup] = score
    return scores


def assign_balanced(safe_rows: list[dict], per_group: int) -> dict[str, str]:
    capacities = {subgroup: per_group for subgroup in SUBGROUPS}
    assignments: dict[str, str] = {}
    scored = []
    for row in safe_rows:
        scores = score_prompt(row["prompt"])
        best_score = max(scores.values())
        specificity = sum(1 for value in scores.values() if value > 0)
        scored.append((best_score, specificity, row["id"], row, scores))

    for _, _, _, row, scores in sorted(scored, key=lambda item: (-item[0], item[1], item[2])):
        ranked_groups = sorted(SUBGROUPS, key=lambda subgroup: (-scores[subgroup], -capacities[subgroup], subgroup))
        chosen = next((group for group in ranked_groups if capacities[group] > 0), None)
        if chosen is None:
            chosen = min(SUBGROUPS)
        assignments[row["id"]] = chosen
        capacities[chosen] -= 1
    return assignments


def write_audit(path: Path, safe_rows: list[dict], assignments: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "id",
            "source_dataset",
            "old_safe_subgroup",
            "new_safe_subgroup",
            "prompt_sha256",
            "prompt_char_length",
            *[f"score_{subgroup}" for subgroup in SUBGROUPS],
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in safe_rows:
            scores = score_prompt(row["prompt"])
            writer.writerow(
                {
                    "id": row["id"],
                    "source_dataset": row["source_dataset"],
                    "old_safe_subgroup": row.get("safe_subgroup") or "",
                    "new_safe_subgroup": assignments[row["id"]],
                    "prompt_sha256": row.get("metadata", {}).get("prompt_sha256", ""),
                    "prompt_char_length": len(row["prompt"]),
                    **{f"score_{subgroup}": scores[subgroup] for subgroup in SUBGROUPS},
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/gjb_real_v1.jsonl")
    parser.add_argument("--output", default="data/processed/gjb_real_v1_rq4_balanced.jsonl")
    parser.add_argument("--audit", default="results/rq/rq4/safe_subgroup_assignment_audit.csv")
    parser.add_argument("--per-group", type=int, default=100)
    args = parser.parse_args()

    rows = read_jsonl(PROJECT_ROOT / args.dataset)
    safe_rows = [row for row in rows if row["type"] == "safe"]
    expected_safe = args.per_group * len(SUBGROUPS)
    if len(safe_rows) != expected_safe:
        raise ValueError(f"Expected {expected_safe} safe rows for balanced assignment; found {len(safe_rows)}")

    assignments = assign_balanced(safe_rows, args.per_group)
    for row in rows:
        if row["type"] == "safe":
            old_subgroup = row.get("safe_subgroup")
            row["safe_subgroup"] = assignments[row["id"]]
            row.setdefault("metadata", {})["safe_subgroup_assignment"] = {
                "method": "deterministic_keyword_balanced_v1",
                "old_safe_subgroup": old_subgroup,
                "assigned_at_utc": datetime.now(timezone.utc).isoformat(),
                "caveat": "Heuristic baseline subgroup assignment, not human-validated demographic labeling.",
            }

    output_path = PROJECT_ROOT / args.output
    write_jsonl(rows, output_path)
    write_audit(PROJECT_ROOT / args.audit, safe_rows, assignments)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": args.dataset,
        "output": args.output,
        "audit": args.audit,
        "safe_counts": dict(sorted(Counter(row["safe_subgroup"] for row in rows if row["type"] == "safe").items())),
        "method": "deterministic_keyword_balanced_v1",
        "prompt_text_logged": False,
        "caveat": "Baseline subgroup assignment only. Human review recommended before final fairness claims.",
    }
    summary_path = PROJECT_ROOT / "results" / "rq" / "rq4" / "safe_subgroup_assignment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
