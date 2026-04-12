#!/usr/bin/env python3
"""Verify downloaded real-source CSVs against the source manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/data_sources.json")
    args = parser.parse_args()

    manifest_path = PROJECT_ROOT / args.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = 0
    for source in manifest["sources"]:
        path = PROJECT_ROOT / source["local_path"]
        exists = path.exists()
        row_count = csv_rows(path) if exists else None
        checksum = sha256(path) if exists else None
        ok = exists and row_count == source["rows"] and checksum == source["sha256"]
        failures += 0 if ok else 1
        print(
            f"{source['name']}: "
            f"exists={exists} rows={row_count} checksum_match={checksum == source['sha256'] if exists else False}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
