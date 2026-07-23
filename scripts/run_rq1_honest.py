#!/usr/bin/env python3
"""Honest RQ1 on the leak-free WildJailbreak set.

Adds (1) a sentence-embedding text baseline, (2) an interpretable request-form
baseline, (3) correctly-scaled hybrids, and (4) a counterfactual probe that
isolates request-form shortcut from genuine harm signal. Uses 5-fold
out-of-fold predictions so every row is scored on a held-out fold.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from gjb.graph import graph_features
from gjb.taxonomy import taxonomy_labels

DATA = PROJECT_ROOT / "data/processed/wildjailbreak_500x500.jsonl"
SEED = 30
GEN = re.compile(r"\b(write|create|generate|compose|draft|develop|design|produce|make a|give me)\b", re.I)
Q = re.compile(r"\b(what|how|why|which|can you|could you|is there|are there|do you|explain|tell me)\b", re.I)


def lr():
    return LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=SEED)


def graph_nointent_matrix(rows):
    labs = sorted(taxonomy_labels() - {"benign_query", "harmful_intent"})
    li = {l: i for i, l in enumerate(labs)}
    M = np.zeros((len(rows), len(labs) + 5))
    for k, r in enumerate(rows):
        f = graph_features(r["graph_nodes"], r["graph_edges"])
        M[k, :5] = [f["num_nodes"], f["num_edges"], f["graph_depth"], f["edge_density"], f["motif_count_2_4"]]
        for n in set(r["graph_nodes"]):
            if n in li:
                M[k, 5 + li[n]] = 1
    return csr_matrix(M)


def form_matrix(rows):
    M = np.zeros((len(rows), 4))
    for k, r in enumerate(rows):
        p = r["prompt"]
        M[k] = [len(GEN.findall(p)) > 0, len(Q.findall(p)) > 0, len(GEN.findall(p)), len(p.split())]
    return csr_matrix(M)


def oof(estimator, X, y):
    """Out-of-fold predictions + probabilities (5-fold stratified)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    proba = cross_val_predict(estimator, X, y, cv=skf, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    return pred, proba


def report(name, y, pred, proba, results):
    results[name] = {
        "f1": f1_score(y, pred), "auc": roc_auc_score(y, proba), "acc": accuracy_score(y, pred),
        "pred": pred,
    }
    print(f"   {name:<32} F1={results[name]['f1']:.3f}  AUC={results[name]['auc']:.3f}  Acc={results[name]['acc']:.3f}")


def main():
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    rows = [r for r in rows if (r.get("metadata") or {}).get("graph_source")]  # annotated only
    y = np.array([1 if r["is_jailbreak"] else 0 for r in rows])
    texts = [r["prompt"] for r in rows]
    print(f"rows={len(rows)} (jb={y.sum()}, safe={(y==0).sum()})\n")

    results = {}
    print("=== MODELS (5-fold out-of-fold) ===")

    # 1. TF-IDF
    tfidf_pipe = make_pipeline(
        TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=30000, strip_accents="unicode"),
        lr(),
    )
    report("Text TF-IDF", y, *oof(tfidf_pipe, texts, y), results)

    # 2. Sentence embeddings
    from sentence_transformers import SentenceTransformer
    print("   (encoding embeddings...)", flush=True)
    emb = SentenceTransformer("all-MiniLM-L6-v2").encode(texts, batch_size=64, show_progress_bar=False,
                                                         normalize_embeddings=True)
    report("Text Embeddings (MiniLM)", y, *oof(make_pipeline(StandardScaler(), lr()), emb, y), results)

    # 3. Request-form only (interpretable shortcut baseline)
    report("Request-form only", y, *oof(make_pipeline(MaxAbsScaler(), lr()), form_matrix(rows), y), results)

    # 4. Graph NoIntent
    G = graph_nointent_matrix(rows)
    report("Graph NoIntent", y, *oof(make_pipeline(MaxAbsScaler(), lr()), G, y), results)

    # 5. Hybrids (correctly scaled together)
    tfidf_only = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=30000, strip_accents="unicode").fit_transform(texts)
    report("Hybrid TFIDF+Graph", y, *oof(make_pipeline(MaxAbsScaler(), lr()), hstack([tfidf_only, G]).tocsr(), y), results)
    report("Hybrid Embed+Graph", y, *oof(make_pipeline(StandardScaler(), lr()), np.hstack([emb, G.toarray()]), y), results)

    # ---- Counterfactual probe: does the model resist the request-form shortcut? ----
    print("\n=== COUNTERFACTUAL PROBE (against-the-shortcut subsets) ===")
    has_gen = np.array([bool(GEN.search(t)) for t in texts])
    # benign-but-looks-harmful (has generation verb) ; harmful-but-looks-benign (no generation verb)
    cf_benign = (y == 0) & has_gen
    cf_harm = (y == 1) & ~has_gen
    print(f"   benign w/ generation-verb (shortcut says HARMFUL): n={cf_benign.sum()}")
    print(f"   harmful w/o generation-verb (shortcut says BENIGN): n={cf_harm.sum()}")
    print(f"   {'model':<32}{'acc: cf-benign':>16}{'acc: cf-harm':>14}")
    for name, r in results.items():
        pred = r["pred"]
        ab = accuracy_score(y[cf_benign], pred[cf_benign]) if cf_benign.sum() else float("nan")
        ah = accuracy_score(y[cf_harm], pred[cf_harm]) if cf_harm.sum() else float("nan")
        print(f"   {name:<32}{ab:>16.3f}{ah:>14.3f}")
    print("\n(High counterfactual accuracy = resists the form shortcut = detects real harm.)")


if __name__ == "__main__":
    main()
