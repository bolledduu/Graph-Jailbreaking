# GraphJailbreakBench — Experiment Tracker (RQ0–RQ4 + E5)

Status legend: ✅ done · 🔶 partial · ⛔ blocked (needs upstream data) · ⬜ not started

Each RQ is broken into sub-parts with **what's done (findings/metrics)** and **what's next (solution)**.
Detailed write-ups: `RQ1_update_wildjailbreak.md`, `RQ1_reverification_update(_brief).md`, `RQ0_SHARE_SUMMARY.md`.

---

## RQ0 — Can jailbreak structure be reliably annotated? ✅ PASSED (foundation)

| Sub-part | Status | Finding / Next step |
|---|---|---|
| Node agreement | ✅ | Krippendorff's α = **0.85** (target ≥0.667) |
| Edge agreement | ✅ | Edge F1 (directed) = **0.64** (target ≥0.60); direction agreement 0.99 |
| Shape agreement | ✅ | Mean normalized GED = **0.21** (≤0.30); edge Jaccard 0.77 |
| Panel | ✅ | 3 LLM annotators (Gemini, gpt-oss-120b, GLM-4.7), n=182; Llama dropped as outlier |

**Open items (RQ0):**
- 🔶 **Paper inconsistency:** Methods text still says "Cohen's κ, κ>0.70/0.60"; Table 1 correctly uses α + edge F1. Reconcile prose to table.
- ⬜ Paper says "human annotators" but panel is LLMs → either reword to "LLM annotators" or add a small human-adjudicated subset.
- ⬜ GLM-4.7 is a Chinese model → state it was used only for the RQ0 agreement study; full annotation used gpt-oss (non-Chinese).

---

## RQ1 — Do graph features complement text for detection? ✅ ANSWERED (No, except one narrow regime)

| Sub-part | Status | Finding / Next step |
|---|---|---|
| 1a. Is the original benchmark leak-free? | ✅ | **No.** 6/7 sources single-label → source ≡ label. Text 0.956 → **0.558** once shortcut removed; leave-one-source-out flagged 87/100 safe as jailbreak. |
| 1b. Build leak-free set | ✅ | **WildJailbreak 500+500** (same source both classes), both annotated (goal-sink protocol). Benign now **66% multi-node** (was 0%). Label agreement 93%. |
| 1c. Honest text-only detection | ✅ | TF-IDF **F1 0.82 / AUC 0.90**; MiniLM embeddings 0.75/0.83; **request-form-only shortcut alone = 0.75** → most of text is surface speech-act ("write" vs "what"), not harm. |
| 1d. LLM zero-shot (correct basis?) | ✅ | **Form-invariant** (benign cf 0.963 ≈ ctrl 0.960; harmful cf 0.687 ≈ ctrl 0.700) → predicts on harm, not form. But misses ~30% of adversarial-harmful prompts. |
| 1e. Graph-only detection | ✅ | NoIntent LR **F1 0.54**; sink-removed 0.56; **GCN no-sink AUC 0.60 (chance)**; GCN all-12 = 0.93 but **circular** (sink = label). |
| 1f. Complementarity (hybrid) | ✅ | Hybrid ≈ text (**0.86**, after scaling fix). Graph adds nothing. |
| 1g. Can better graph features help? | ✅ | **No** — information ceiling. GCN doesn't beat counts; content-decomposition (sentence pooling) ≤ flat text < TF-IDF. Harm is lexical → sparse text is the ceiling. |
| 1h. Obfuscation "hard attack" regime | ✅ | Graph wins **only** when text is destroyed (char-spaced/base64 → text 0.50, graph 0.63). Reversible obfuscation (leet/rot13) doesn't hurt text. Weak win + annotator-recoverability caveat. |

**Verdict:** Detection is a **text/LLM job**; graph structure is harm-agnostic and does not complement text on recoverable prompts. Confirmed the paper's "hard attack" hypothesis narrowly (obfuscation only, weak).

**Open items (RQ1):**
- 🔶 **Code fix:** `run_e1_ml_baselines.py` scales graph features with `StandardScaler` then hstacks unscaled TF-IDF → hybrid artificially hurt. Fix = `MaxAbsScaler` over combined matrix (verified: hybrid recovers to ≈ text).
- 🔶 **Broken difficulty strata:** old easy/medium/hard were label-confounded (medium/hard 100% jailbreak → fake 1.0). Drop or rebuild balanced.
- ⬜ Fill paper **Table 2** with honest numbers incl. GCN row.
- ⬜ (optional) Human audit of the ~7% label-noise in WildJailbreak benign split.
- ⬜ (optional) Fine-tuned transformer / LLM as the strong text baseline in the paper.

---

## RQ2 — Does graph structure predict cross-model transferability? ⛔ BLOCKED

| Sub-part | Status | Finding / Next step |
|---|---|---|
| 2a. Cross-model success labels | ⛔ | **Not collected.** THE blocker for RQ2/RQ3-success/E5. |
| 2b. Graph vs text predicting target-model success | ⬜ | Depends on 2a. |

**Solution / next step (highest-value data task):**
1. Pick **2 non-Chinese target models** (e.g., gpt-oss-120b + Gemini, or a paid GPT-4o-mini + Llama).
2. Run the harmful prompts against each; record response + **success label** (LLM-judge or refusal-classifier).
3. Train: input = graph features + source-model success → predict target-model success; baseline = text features. Compare accuracy / ROC-AUC.
> Rationale: here the outcome variable is **structural/behavioral**, not harm-content, so the "harm-agnostic" limitation that sinks RQ1 detection is **not** a problem — this is where the graph could genuinely win.

---

## RQ3 — Which structural motifs characterize attacks? 🔶 PARTIAL

| Sub-part | Status | Finding / Next step |
|---|---|---|
| 3a. Enumerate + rank motifs by frequency | ✅ | Top motifs (old jailbreak set): `instruction_override→harmful_intent` (1065), `roleplay→instruction_override` (924), **DAN chain** `roleplay→instruction_override→harmful_intent` (877), `context_shift→harmful_intent` (318), `fiction→harmful_intent` (213). ~10 distinct motifs. |
| 3b. Rank by empirical success rate | ⛔ | Blocked — needs success labels (RQ2 2a). |
| 3c. Top-k coverage | 🔶 | Frequency-coverage computable now; success-coverage blocked. |

**Open items (RQ3):**
- 🔶 Recompute motifs on the **new goal-sink annotated data** (current numbers are from the older jailbreak-only annotation).
- ⬜ Add success-rate ranking once RQ2 labels exist.
- ℹ️ Note limitation: sink/star edge protocol trades structural richness for reliability → only ~10 motifs (expected, disclose in paper).

---

## RQ4 — Do graph defences reduce disparate overblocking? ⬜ NOT STARTED

| Sub-part | Status | Finding / Next step |
|---|---|---|
| 4a. Build 6 safe subgroups | ⬜ | Paper defines: formal-academic, informal-colloquial, non-native-English, security-technical, medical-health, creative-writing. Current safe data isn't organized this way. |
| 4b. Per-subgroup FPR (text vs graph) | ⬜ | Depends on 4a + trained detectors. |
| 4c. Disparity (max-min gap, std dev) | ⬜ | Depends on 4b. |

**Solution / next step:**
1. Assemble ~50–100 benign prompts per subgroup (style-controlled, must **not** be trivially separable — apply the same leak-free lesson).
2. Run the trained text and graph detectors; compute per-subgroup FPR + max-min gap + std dev.
3. Hypothesis: text overblocks security-technical / medical-health (shared vocab with attacks); a structural classifier should be flatter.
> ⚠️ Caveat: since graph is a near-chance detector, its "fairness" comes partly from not detecting much — frame carefully.

---

## E5 — Multi-turn vs single-turn ASR ⛔ BLOCKED

| Sub-part | Status | Finding / Next step |
|---|---|---|
| 5a. Multi-turn expansions | ⬜ | Expand single-turn jailbreaks into 3–5 turn variants (same graph). |
| 5b. ASR single vs multi-turn | ⛔ | Needs success labels (RQ2 2a). |

---

## Cross-cutting priorities

1. **Collect model-success labels** (RQ2 2a) — single highest-value task; unblocks RQ2, RQ3-success, E5.
2. **RQ1 write-up + code fixes** (scaling, difficulty strata, Table 2) — closest to done; finish it.
3. **RQ4 safe-subgroup set** — independent, buildable now; apply the leak-free discipline.
4. **Reconcile paper** — metrics prose (RQ0), source-data list (add WildJailbreak/JailbreakV/JailbreakDB), annotator description.

**The through-line:** detection (RQ1) is settled as a text/LLM job; the graph's genuine payoff lives in **RQ2/RQ3-success** (structure → transfer/success) and **RQ4** (fairness), all of which hinge on **model-success labels**.
