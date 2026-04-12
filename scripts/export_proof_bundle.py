#!/usr/bin/env python3
"""Export reproducibility proof artifacts for the project.

The proof bundle intentionally avoids copying raw prompt text into logs. It
records row counts, file hashes, source checksums, result table summaries, and
environment details so the project can be audited without duplicating harmful
prompt content in extra places.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "logs" / "proof"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def jsonl_row_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def project_files() -> list[Path]:
    ignored_parts = {".venv", ".pip-cache", "__pycache__", ".git"}
    files: list[Path] = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in ignored_parts for part in path.relative_to(PROJECT_ROOT).parts):
            continue
        files.append(path)
    return sorted(files)


def build_file_manifest(files: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": relative(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]


def build_source_verification() -> dict[str, Any]:
    manifest_path = PROJECT_ROOT / "configs" / "data_sources.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = []
    for source in manifest["sources"]:
        path = PROJECT_ROOT / source["local_path"]
        exists = path.exists()
        row_count = csv_row_count(path) if exists else None
        checksum = sha256(path) if exists else None
        sources.append(
            {
                "name": source["name"],
                "url": source["url"],
                "local_path": source["local_path"],
                "exists": exists,
                "expected_rows": source["rows"],
                "actual_rows": row_count,
                "expected_sha256": source["sha256"],
                "actual_sha256": checksum,
                "rows_match": row_count == source["rows"],
                "sha256_match": checksum == source["sha256"],
            }
        )
    return {
        "manifest_path": relative(manifest_path),
        "all_sources_verified": all(item["exists"] and item["rows_match"] and item["sha256_match"] for item in sources),
        "sources": sources,
    }


def build_dataset_summary() -> dict[str, Any]:
    path = PROJECT_ROOT / "data" / "processed" / "gjb_real_v1.jsonl"
    rows = jsonl_rows(path)
    return {
        "path": relative(path),
        "sha256": sha256(path),
        "rows": len(rows),
        "by_source_dataset": dict(sorted(Counter(row["source_dataset"] for row in rows).items())),
        "by_type": dict(sorted(Counter(row["type"] for row in rows).items())),
        "by_attack_type": dict(sorted(Counter(row["attack_type"] for row in rows).items())),
        "by_difficulty": dict(sorted(Counter(row["difficulty"] for row in rows).items())),
        "safe_subgroups": dict(sorted(Counter(row.get("safe_subgroup") for row in rows if row["type"] == "safe").items())),
        "gpt4_success_labels_present": sum(1 for row in rows if row.get("model_gpt4_success") is not None),
        "llama_success_labels_present": sum(1 for row in rows if row.get("model_llama_success") is not None),
        "synthetic_source_rows": sum(1 for row in rows if "synthetic" in row["source_dataset"].lower()),
        "prompt_text_logged": False,
    }


def build_results_summary() -> dict[str, Any]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    tables = []
    for path in sorted(tables_dir.glob("*.csv")):
        tables.append(
            {
                "path": relative(path),
                "rows_excluding_header": csv_row_count(path),
                "sha256": sha256(path),
            }
        )
    return {"tables": tables}


def build_environment_summary() -> dict[str, Any]:
    pip_freeze = run_command([str(PROJECT_ROOT / ".venv" / "bin" / "python"), "-m", "pip", "freeze", "--all"])
    pip_check = run_command([str(PROJECT_ROOT / ".venv" / "bin" / "python"), "-m", "pip", "check"])
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "venv_python": str(PROJECT_ROOT / ".venv" / "bin" / "python"),
        "pip_check": pip_check,
        "pip_freeze": pip_freeze,
    }


def build_command_log() -> list[dict[str, Any]]:
    commands = [
        [str(PROJECT_ROOT / ".venv" / "bin" / "python"), "scripts/verify_sources.py"],
        [str(PROJECT_ROOT / ".venv" / "bin" / "python"), "scripts/build_dataset.py", "--output", "data/processed/gjb_real_v1.jsonl"],
        [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            "scripts/create_annotation_packets.py",
            "--dataset",
            "data/processed/gjb_real_v1.jsonl",
            "--rq0-sample-size",
            "200",
            "--e5-base-sample-size",
            "250",
            "--seed",
            "30",
        ],
        [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            "scripts/run_rq0_annotation_validation.py",
            "--annotation-dir",
            "data/annotations/rq0",
            "--output",
            "results/rq/rq0/rq0_annotation_validation.json",
        ],
        [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            "scripts/run_e1_ml_baselines.py",
            "--dataset",
            "data/processed/gjb_real_v1.jsonl",
            "--output-dir",
            "results/rq/rq1",
            "--test-size",
            "0.20",
            "--seed",
            "30",
        ],
        [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            "scripts/run_e1_embedding_baselines.py",
            "--dataset",
            "data/processed/gjb_real_v1.jsonl",
            "--output-dir",
            "results/rq/rq1/embedding_baseline",
            "--model-name",
            "sentence-transformers/all-MiniLM-L6-v2",
            "--batch-size",
            "64",
            "--test-size",
            "0.20",
            "--seed",
            "31",
        ],
        [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            "scripts/run_baseline_experiments.py",
            "--dataset",
            "data/processed/gjb_real_v1.jsonl",
            "--output-dir",
            "results/tables",
        ],
        [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            "scripts/assign_rq4_safe_subgroups.py",
            "--dataset",
            "data/processed/gjb_real_v1.jsonl",
            "--output",
            "data/processed/gjb_real_v1_rq4_balanced.jsonl",
            "--audit",
            "results/rq/rq4/safe_subgroup_assignment_audit.csv",
            "--per-group",
            "100",
        ],
        [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            "scripts/run_baseline_experiments.py",
            "--dataset",
            "data/processed/gjb_real_v1_rq4_balanced.jsonl",
            "--output-dir",
            "results/tables/rq4_balanced",
        ],
        [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            "scripts/plot_rq3_motifs.py",
            "--motifs",
            "results/tables/table_4_top_motifs_real.csv",
            "--output-dir",
            "results/figures/rq3",
            "--top-k",
            "5",
        ],
        [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            "scripts/run_unblocked_experiments.py",
            "--dataset",
            "data/processed/gjb_real_v1.jsonl",
            "--balanced-dataset",
            "data/processed/gjb_real_v1_rq4_balanced.jsonl",
            "--output-dir",
            "results/unblocked",
            "--seed",
            "44",
            "--test-size",
            "0.20",
            "--bootstrap",
            "1000",
        ],
        [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            "scripts/import_model_labels.py",
            "--dataset",
            "data/processed/gjb_real_v1.jsonl",
            "--labels",
            "data/annotations/rq2/model_success_template.csv",
            "--output",
            "data/processed/gjb_real_v1_with_model_labels.jsonl",
        ],
        [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            "scripts/import_multiturn_variants.py",
            "--dataset",
            "data/processed/gjb_real_v1.jsonl",
            "--template",
            "data/annotations/e5/multiturn_variants_template.csv",
            "--output",
            "data/processed/gjb_real_v1_with_multiturn.jsonl",
        ],
        [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            "scripts/run_label_dependent_experiments.py",
            "--dataset",
            "data/processed/gjb_real_v1.jsonl",
        ],
        [str(PROJECT_ROOT / ".venv" / "bin" / "python"), "scripts/export_rq_results.py", "--dataset", "data/processed/gjb_real_v1.jsonl"],
        [str(PROJECT_ROOT / ".venv" / "bin" / "python"), "-m", "unittest", "discover", "-s", "tests"],
    ]
    return [run_command(command) for command in commands]


def main() -> int:
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    files = project_files()
    file_manifest = build_file_manifest(files)
    source_verification = build_source_verification()
    dataset_summary = build_dataset_summary()
    results_summary = build_results_summary()
    environment_summary = build_environment_summary()
    command_log = build_command_log()

    bundle_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "proof_dir": relative(output_dir),
        "prompt_text_logged": False,
        "artifacts": {
            "file_manifest": "logs/proof/file_manifest.json",
            "source_verification": "logs/proof/source_verification.json",
            "dataset_summary": "logs/proof/dataset_summary.json",
            "results_summary": "logs/proof/results_summary.json",
            "environment_summary": "logs/proof/environment_summary.json",
            "command_log": "logs/proof/command_log.json",
        },
        "summary": {
            "project_file_count": len(file_manifest),
            "real_dataset_rows": dataset_summary["rows"],
            "all_sources_verified": source_verification["all_sources_verified"],
            "synthetic_source_rows": dataset_summary["synthetic_source_rows"],
            "result_table_count": len(results_summary["tables"]),
            "commands_all_passed": all(item["returncode"] == 0 for item in command_log),
        },
    }

    write_json(output_dir / "file_manifest.json", file_manifest)
    write_json(output_dir / "source_verification.json", source_verification)
    write_json(output_dir / "dataset_summary.json", dataset_summary)
    write_json(output_dir / "results_summary.json", results_summary)
    write_json(output_dir / "environment_summary.json", environment_summary)
    write_json(output_dir / "command_log.json", command_log)
    write_json(output_dir / "bundle_manifest.json", bundle_manifest)

    summary_lines = [
        "# Proof Bundle Summary",
        "",
        f"Created UTC: {bundle_manifest['created_at_utc']}",
        f"Project root: {PROJECT_ROOT}",
        f"Real dataset rows: {dataset_summary['rows']}",
        f"All sources verified: {source_verification['all_sources_verified']}",
        f"Synthetic source rows: {dataset_summary['synthetic_source_rows']}",
        f"Result tables: {len(results_summary['tables'])}",
        f"Commands all passed: {bundle_manifest['summary']['commands_all_passed']}",
        "",
        "Raw prompt text is intentionally not duplicated in the proof logs.",
    ]
    (output_dir / "SUMMARY.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(json.dumps(bundle_manifest["summary"], indent=2, sort_keys=True))
    return 0 if bundle_manifest["summary"]["commands_all_passed"] and source_verification["all_sources_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
