# RQ0 — Annotation Reliability: Progress & Results

**Question:** Can jailbreak prompts be reliably decomposed into semantic graphs
(12-label taxonomy: nodes = attack tactics, edges = how they connect) using a
structured annotation protocol?

## Progress
- Sampled **200 harmful prompts** (AdvBench / HarmBench / JailbreakBench /
  JailbreakV / JailbreakDB).
- Annotated independently by LLMs from different model families:

  | Annotator | Provider | Model | Rows completed | In analysis |
  |-----------|----------|-------|----------------|-------------|
  | a | Gemini   | gemini-3.1-flash-lite   | 198/200 | yes |
  | c | Cerebras | gpt-oss-120b            | 200/200 | yes |
  | d | Cerebras | zai-glm-4.7 (GLM-4.7)   | 200/200 | yes |
  | b | Groq     | llama-3.3-70b-versatile | 161/200 | **excluded (outlier)** |

- Annotator **b (Llama) excluded** from the analysis: it refused 39 of the
  harshest prompts and over-labeled the rest (worst agreement in every pairing).
  Its labels are retained, not deleted.
- Agreement computed over the 198 rows the trio (a/c/d) completed, using:
  Krippendorff's α (node labels, all three jointly), edge F1 (split into
  presence vs direction), and normalized graph edit distance (GED).

## Results

| Metric | Value | Target | Met? |
|--------|------:|:------:|:----:|
| **Node agreement — Krippendorff's α (a/c/d jointly)** | **0.82** | ≥ 0.667 | ✅ |
| Edge F1 — directed | 0.42 | ≥ 0.60 | ❌ |
| Edge F1 — presence (undirected) | 0.49 | — | — |
| Edge direction agreement (when both connect a pair) | 0.84 | — | ✅ |
| Mean GED | 0.31 | ≤ 0.30 | ❌ (marginal) |

**Overall status: `needs_taxonomy_revision`** — node labels are reliable; the gap
is **edge structure**, specifically *which* tactics to connect (edge presence),
not their direction.

### Per-label node agreement (Krippendorff's α)
- Strong: instruction_override 0.94, roleplay 0.89, harmful_intent 0.80,
  hypothetical 0.74.
- Moderate: emotional_manipulation 0.63, fiction 0.62, obfuscation 0.57,
  benign_query 0.50.
- Weak (overlapping/fuzzy): authority_claim 0.26, context_shift 0.31.

### Pairwise (diagnostic)
| Pair | Models | Node κ | Edge F1 (dir) | Edge dir. agree | GED |
|------|--------|-------:|--------------:|----------------:|----:|
| a–c | Gemini / gpt-oss | 0.65 | 0.38 | 0.82 | 0.33 |
| a–d | Gemini / GLM     | 0.59 | 0.52 | 0.89 | 0.29 |
| c–d | gpt-oss / GLM    | 0.55 | 0.35 | 0.82 | 0.29 |

## Artifacts
- Metrics: `results/rq/rq0/rq0_annotation_validation.json`
- Per-annotator run metadata: `results/rq/rq0/llm_annotation_run_*.json`
- Annotations: `data/annotations/rq0/annotator_{a,c,d}_template.csv` (IDs + labels only; no raw prompt text)
