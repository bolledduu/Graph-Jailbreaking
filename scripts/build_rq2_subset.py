#!/usr/bin/env python3
"""Build a diversified 500-prompt multi-node harmful subset for RQ2.

Pools multi-node (>=2 node) harmful prompts across all annotated sources, then
selects 500 that are diverse along three axes:
  1. source        - balance so no single corpus dominates
  2. structure     - round-robin across distinct tactic-combinations (motifs)
  3. wording       - embedding dedup drops near-identical templated prompts
"""
from __future__ import annotations
import json, random, sys
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SINK = {"harmful_intent", "benign_query"}
FILES = ["data/processed/wildjailbreak_500x500.jsonl", "data/processed/gjb_real_v1_graphs.jsonl"]
OUT = PROJECT_ROOT / "data/processed/rq2_multinode_500.jsonl"
TARGET = 500
SIM_THRESHOLD = 0.90       # drop a candidate if cosine >= this to any selected prompt
SEED = 42


def load_pool():
    seen, pool = set(), []
    for f in FILES:
        p = PROJECT_ROOT / f
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            if not r.get("is_jailbreak"):
                continue
            if not (r.get("metadata") or {}).get("graph_source"):
                continue
            if len(set(r.get("graph_nodes") or [])) < 2:   # multi-node only
                continue
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            pool.append(r)
    return pool


def combo(r):
    return frozenset(set(r["graph_nodes"]) - SINK)


def main() -> int:
    rng = random.Random(SEED)
    pool = load_pool()
    print(f"multi-node harmful pool: {len(pool)}")

    enc = SentenceTransformer("all-MiniLM-L6-v2")
    emb = enc.encode([r["prompt"] for r in pool], batch_size=64,
                     show_progress_bar=False, normalize_embeddings=True)
    emb_by_id = {r["id"]: emb[i] for i, r in enumerate(pool)}

    selected, sel_emb = [], []

    def try_add(r) -> bool:
        e = emb_by_id[r["id"]]
        if sel_emb:
            if float(np.max(np.asarray(sel_emb) @ e)) >= SIM_THRESHOLD:
                return False                                  # too similar in wording
        selected.append(r); sel_emb.append(e)
        return True

    # per-source quotas: take all of the small sources, balance the two big ones
    by_src = {}
    for r in pool:
        by_src.setdefault(r["source_dataset"], []).append(r)
    small = [s for s in by_src if len(by_src[s]) <= 40]
    big = [s for s in by_src if len(by_src[s]) > 40]           # WildJailbreak, JailbreakV-28K

    # 1) small sources: take all (deduped)
    for s in small:
        for r in by_src[s]:
            try_add(r)
    print(f"after small sources: {len(selected)}")

    # 2) big sources: round-robin across structure-combos, split remaining quota evenly
    def fill_source(rows, quota):
        buckets = {}
        for r in rows:
            buckets.setdefault(combo(r), []).append(r)
        for b in buckets.values():
            rng.shuffle(b)
        order = list(buckets.values()); rng.shuffle(order)
        added, active = 0, True
        while added < quota and active:
            active = False
            for b in order:
                if not b:
                    continue
                r = b.pop()
                active = True
                if try_add(r):
                    added += 1
                    if added >= quota:
                        break
        return added

    remaining = TARGET - len(selected)
    per = remaining // len(big)
    for i, s in enumerate(sorted(big, key=lambda x: len(by_src[x]))):   # smaller big source first
        quota = per if i < len(big) - 1 else TARGET - len(selected)     # last takes the rest
        got = fill_source(by_src[s], quota)
        print(f"  {s}: +{got}")

    # 3) if still short (dedup removed too many), top up from remaining pool ignoring source balance
    if len(selected) < TARGET:
        leftover = [r for r in pool if r not in selected]
        rng.shuffle(leftover)
        for r in leftover:
            if len(selected) >= TARGET:
                break
            try_add(r)

    rng.shuffle(selected)
    with OUT.open("w", encoding="utf-8") as h:
        for r in selected:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")

    # report diversity
    import collections
    print(f"\nSELECTED {len(selected)} -> {OUT}")
    print("by source:", dict(collections.Counter(r["source_dataset"] for r in selected)))
    combos = collections.Counter(combo(r) for r in selected)
    print(f"distinct structure types: {len(combos)}")
    nc = collections.Counter(len(set(r["graph_nodes"]) - SINK) for r in selected)
    print("tactic-count distribution:", dict(sorted(nc.items())))
    # mean pairwise similarity (lower = more diverse)
    S = np.asarray(sel_emb)
    sims = S @ S.T; n = len(S)
    mean_sim = (sims.sum() - n) / (n * (n - 1))
    print(f"mean pairwise cosine similarity: {mean_sim:.3f} (lower = more diverse)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
