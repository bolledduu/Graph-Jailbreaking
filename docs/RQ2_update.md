# RQ2 Update — Cross-Model Transferability

**Question.** Given a jailbreak's success on one model, does its **graph structure** predict
success on a *different* model better than its **text** does? (The hypothesis: attack
*structure* is model-agnostic and should transfer more predictably than surface wording.)

---

1. **Built a diversified attack set.** Sampled **500 multi-node harmful prompts** (≥2 tactic
   nodes — i.e., prompts that actually *have* structure) pooled across WildJailbreak,
   JailbreakV, HarmBench, JailbreakBench, AdvBench, JailbreakDB. Balanced across sources,
   round-robin across 104 distinct tactic-combinations, and **embedding-deduped** (mean
   pairwise cosine = 0.18) so it isn't 500 copies of the same DAN template.
   File: `data/processed/rq2_multinode_500.jsonl`.

2. **Ran the attacks against 3 target models** (different families, for a real
   transfer test), sending each prompt as-is with **default safety settings** to observe
   real refusal behaviour:
   - `gemma-4-31b` (Google, via Cerebras)
   - `llama-3.3-70b` (Meta, via Groq)
   - `gpt-oss-20b` (OpenAI, via Groq)
   Responses: `results/rq/rq2/rq2_responses.jsonl` (500 × 3, zero technical failures).

3. **Labeled success** (complied with the harmful task) vs **fail** (refused) from each
   response. Caught and fixed a labeling bug: gpt-oss uses curly apostrophes (`I can’t`),
   which the refusal heuristic missed — inflating gpt-oss to a false 99% success. After
   apostrophe-normalization + spot-check validation, the labels are reliable.

4. **Built the transfer predictor.** For each ordered model pair, input = prompt features
   + source-model success bit; target = target-model success. Compared **graph features**
   (structural + tactic indicators) vs **text features** (TF-IDF), 5-fold CV, ROC-AUC.

---

## 2. Per-model success & non-nesting

| Target model | Success rate | Behaviour |
|---|---:|---|
| gpt-oss-20b | **10%** | strictest |
| gemma-4-31b | **20%** | strict |
| llama-3.3-70b | **75%** | permissive |

The three models are **genuinely non-nested** (not a simple strict→lax ordering): e.g. 19
prompts that **gemma refuses, gpt-oss accepts**, and 65 that gpt-oss uniquely refuses. So
"which attacks transfer" depends on the *type* of attack, not just overall strictness —
exactly the setting needed to give structure a fair test.

**Structure does carry a transfer signal.** Among attacks that beat the permissive model
(llama), the ones that *also* beat the strict model differ structurally:

| Tactic | Transferred | Blocked | Δ |
|---|---:|---:|---:|
| fiction | 51% | 28% | **+22** |
| hypothetical | 32% | 20% | +13 |
| roleplay | 63% | 51% | +12 |
| instruction_override | 32% | 49% | **−17** |
| obfuscation | 2% | 9% | −7 |

→ narrative wrappers (fiction/roleplay/hypothetical) transfer; blatant override/obfuscation
gets blocked by stricter models.

---

## 3. Transfer prediction on clean text — text ≥ graph

Predicting target-model success (graph features vs text features), all 6 ordered directions:

| source → target | GRAPH AUC | TEXT AUC | winner |
|---|---:|---:|---|
| gemma → llama | 0.728 | 0.833 | text |
| gemma → gpt-oss | 0.811 | 0.827 | text (close) |
| llama → gemma | 0.741 | 0.756 | text (close) |
| llama → gpt-oss | 0.753 | 0.778 | text |
| gpt-oss → gemma | 0.729 | 0.746 | text (close) |
| gpt-oss → llama | 0.697 | 0.808 | text |

**Graph 0 / Text 6.** Both predict transfer well above chance, but on clean text **text
matches or beats the graph everywhere.** Why: the tactic signal that drives transfer
(fiction/roleplay) is *also lexically visible* ("story", "imagine", "hypothetical"), so text
re-derives the structure from words **and** adds content severity. The graph's structural
information is real but **not orthogonal** to text on clean prompts.

*(The §3 table uses hand-crafted graph features + Logistic Regression; the text column is TF-IDF.)*

---

## 3b. With the paper's architecture (RoBERTa + 3-layer GCN)

To check that the §3 verdict isn't an artifact of the simple LR/TF-IDF models, we re-ran the
same clean-text transfer task with the paper's actual **RoBERTa + 3-layer GCN hybrid**
(frozen `roberta-base` text branch on CPU; trainable GCN over the tactic graph; concat → MLP;
5-fold out-of-fold ROC-AUC):

| source → target | TEXT (RoBERTa) | GRAPH (GCN) | HYBRID |
|---|---:|---:|---:|
| gemma → llama | 0.808 | 0.723 | 0.810 |
| gemma → gpt-oss | 0.794 | 0.785 | 0.804 |
| llama → gemma | 0.739 | 0.735 | 0.754 |
| llama → gpt-oss | 0.790 | 0.739 | 0.801 |
| gpt-oss → gemma | 0.740 | 0.738 | 0.753 |
| gpt-oss → llama | 0.805 | 0.694 | 0.806 |

**Same verdict as §3.** The GCN is **≤ RoBERTa in all 6 directions**, and the **hybrid only
edges text by ~+0.009 AUC on average** (range +0.002 to +0.015) — a real but *marginal*
complementary bump, not the large gain the paper claims for detection. The GCN also matches
the LR-on-features graph model (~0.69–0.79), so the neural graph encoder adds nothing over
hand-crafted structural features. Structure is **largely redundant with the text embedding**
on clean transfer prediction.

*Caveats:* frozen `roberta-base`, **not** fine-tuned `roberta-large` (CPU limit) — a
fine-tuned large text model would raise the text number and only *shrink* the hybrid's edge;
small data with 10–20% positives for the strict models → AUCs are noisy (±0.02–0.03).
Script: `scripts/run_rq2_hybrid.py`.

---

## 4. The obfuscation case — where the graph wins

This is the key result. We re-ran the transfer prediction under **surface obfuscation**: the
same transform is applied to every prompt's *text* (so obfuscation itself can't leak the
label), while the **graph is held fixed** (structure is assumed recoverable). This strips
text of its lexical proxy for structure and leaves the graph as the only structural signal.

**Predicting transfer to the strict model (conditional on llama-success):**

| Predictor | AUC | |
|---|---:|---|
| GRAPH (structure) | 0.679 | fixed |
| text — clean | 0.714 | text wins |
| text — leetspeak (reversible) | 0.712 | text still fine |
| **text — char-spaced** | **0.500** | ← **graph wins** |
| **text — base64** | **0.493** | ← **graph wins** |

(All-500 version: graph 0.699; clean text 0.721; destroyed text → 0.50.)

**Interpretation.**
- **Reversible obfuscation (leetspeak/rot13) doesn't help the graph** — a retrained text
  model just relearns the transformed tokens (0.712 ≈ 0.714).
- **Destructive obfuscation (char-spacing, base64) collapses text to chance (0.50)** while
  the graph holds at ~0.68 → **the graph becomes the better transfer predictor.**

So the graph's distinctive value is **robustness**: it retains transfer-predictive signal
when the surface text is destroyed, which text cannot. This is the same story as RQ1 —
text handles clean attacks; the graph is the obfuscation-robust structural fallback.

---


