#!/usr/bin/env python3
"""Build a leak-free, style-matched detection set from allenai/wildjailbreak.

Both classes come from the SAME sub-source (allenai/wildjailbreak) so the
dataset-of-origin cannot predict the label -- the corpus-style leak that
inflated the earlier text baseline is removed by construction. Benign rows are
drawn from JailbreakDB's regular pool (jailbreak=0), harmful rows from its
jailbreak pool (jailbreak=1). Prompts are deduped and length-filtered.

Graphs are left as single-node placeholders here; the safe AND jailbreak rows
are annotated afterwards by the standard gpt-oss protocol.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path

csv.field_size_limit(10**7)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE = "allenai/wildjailbreak"
REGULAR = PROJECT_ROOT / "data/raw/jailbreakdb/text_regular_unique.csv"
JAILBREAK = PROJECT_ROOT / "data/raw/jailbreakdb/text_jailbreak_unique.csv"
OUT = PROJECT_ROOT / "data/processed/wildjailbreak_500x500.jsonl"

N_PER_CLASS = 500
MIN_LEN, MAX_LEN = 60, 4000
SEED = 42


def collect(path: Path, want_jb: str) -> list[dict]:
    """Return deduped rows from `path` where source==WildJailbreak and jailbreak==want_jb."""
    seen: set[str] = set()
    out: list[dict] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("source") != SOURCE:
                continue
            if (row.get("jailbreak") or "").strip() != want_jb:
                continue
            prompt = (row.get("user_prompt") or "").strip()
            if not (MIN_LEN <= len(prompt) <= MAX_LEN):
                continue
            key = " ".join(prompt.lower().split())
            if key in seen:
                continue
            seen.add(key)
            out.append({"prompt": prompt, "tactic": row.get("tactic"), "jailbreak": want_jb})
    return out


def make_row(index: int, prompt: str, is_jb: bool, tactic: str | None) -> dict:
    prefix = "wj_harmful" if is_jb else "wj_benign"
    node = "harmful_intent" if is_jb else "benign_query"
    return {
        "id": f"{prefix}_{index:04d}",
        "prompt": prompt,
        "type": "jailbreak" if is_jb else "safe",
        "is_jailbreak": is_jb,
        "is_harmful": is_jb,
        "difficulty": "medium",  # unknown; strata not used for the honest metric
        "attack_type": "wrapped" if is_jb else "wrapped_benign",
        "conversation": [],
        "source_dataset": "WildJailbreak",  # SAME source for both classes (by design)
        "source_category": None,
        "source_record_id": None,
        "safe_subgroup": None if is_jb else "wildjailbreak_adversarial_benign",
        "graph_nodes": [node],  # placeholder; real graph filled by annotation
        "graph_edges": [],
        "num_nodes": 1,
        "graph_depth": 1,
        "model_gpt4_response": None,
        "model_gpt4_success": None,
        "model_llama_response": None,
        "model_llama_success": None,
        "metadata": {
            "graph_source": None,  # not yet annotated -> annotator will pick it up
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "source": SOURCE,
            "source_path": str(REGULAR if not is_jb else JAILBREAK),
            "raw_fields": {"tactic": tactic, "jailbreak": "1" if is_jb else "0"},
        },
    }


def main() -> int:
    rng = random.Random(SEED)
    print("Scanning benign pool (jailbreak=0)...")
    benign = collect(REGULAR, "0")
    print(f"  wildjailbreak benign available (deduped, length-ok): {len(benign)}")
    print("Scanning jailbreak pool (jailbreak=1)...")
    harmful = collect(JAILBREAK, "1")
    print(f"  wildjailbreak harmful available (deduped, length-ok): {len(harmful)}")

    if len(benign) < N_PER_CLASS or len(harmful) < N_PER_CLASS:
        print("ERROR: not enough rows to sample", file=sys.stderr)
        return 1

    benign_s = rng.sample(benign, N_PER_CLASS)
    harmful_s = rng.sample(harmful, N_PER_CLASS)

    rows = []
    for i, r in enumerate(benign_s):
        rows.append(make_row(i, r["prompt"], False, r["tactic"]))
    for i, r in enumerate(harmful_s):
        rows.append(make_row(i, r["prompt"], True, r["tactic"]))
    rng.shuffle(rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")

    lens = [len(r["prompt"]) for r in rows]
    print(f"\nWrote {len(rows)} rows -> {OUT}")
    print(f"  benign={sum(1 for r in rows if not r['is_jailbreak'])}, "
          f"harmful={sum(1 for r in rows if r['is_jailbreak'])}")
    print(f"  prompt length: min={min(lens)} median={sorted(lens)[len(lens)//2]} max={max(lens)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
