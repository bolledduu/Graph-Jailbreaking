#!/usr/bin/env python3
"""Run all label-dependent experiments that are ready to run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_command(command: list[str]) -> dict:
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/gjb_real_v1.jsonl")
    parser.add_argument("--output", default="results/rq/label_dependent_run_summary.json")
    args = parser.parse_args()

    commands = [
        [sys.executable, "scripts/run_rq0_annotation_validation.py"],
        [sys.executable, "scripts/run_rq2_transferability_experiments.py", "--dataset", args.dataset],
        [sys.executable, "scripts/run_e5_asr_analysis.py", "--dataset", args.dataset],
        [sys.executable, "scripts/export_rq_results.py", "--dataset", args.dataset],
    ]
    results = [run_command(command) for command in commands]
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "commands": results,
        "status": "completed" if all(result["returncode"] == 0 for result in results) else "failed",
        "prompt_text_logged": False,
    }
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
