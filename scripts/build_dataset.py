#!/usr/bin/env python3
"""Build GraphJailbreakBench from real source datasets only."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gjb.io import write_jsonl
from gjb.real_sources import load_real_sources


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output", default="data/processed/gjb_real_v1.jsonl")
    parser.add_argument("--dedupe-prompts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-jailbreak", type=int, default=3000)
    parser.add_argument("--target-safe", type=int, default=600)
    parser.add_argument("--sample-seed", type=int, default=30)
    args = parser.parse_args()

    rows = load_real_sources(
        PROJECT_ROOT / args.raw_dir,
        dedupe_prompts=args.dedupe_prompts,
        target_jailbreak=args.target_jailbreak,
        target_safe=args.target_safe,
        seed=args.sample_seed,
    )
    output = PROJECT_ROOT / args.output
    write_jsonl(rows, output)

    by_type = Counter(row["type"] for row in rows)
    by_source = Counter(row["source_dataset"] for row in rows)
    by_safe_group = Counter(row.get("safe_subgroup") for row in rows if row["type"] == "safe")
    print(f"Wrote {len(rows)} real-source rows to {output}")
    print("By source:", dict(sorted(by_source.items())))
    print("By type:", dict(sorted(by_type.items())))
    print("Safe groups:", dict(sorted(by_safe_group.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
