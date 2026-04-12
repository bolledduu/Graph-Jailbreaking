# Workbook Understanding

Source workbook: `/Users/pradeep/Downloads/Graph Jailbreaking.xlsx`

## Research Questions

- RQ0: Can jailbreak prompts be reliably decomposed into semantic graphs using
  a structured annotation protocol?
- RQ1: Do structural graph features provide complementary signal to text
  features for jailbreak detection?
- RQ2: Do graph-based features generalize better across models for predicting
  jailbreak success?
- RQ3: What structural motifs are most associated with successful jailbreak
  attacks?
- RQ4: Do graph-based defenses reduce disparate overblocking across prompt
  styles of different user populations?

## Dataset Requirements

The workbook specifies GraphJailbreakBench (GJB), with these components:

- 2,000-3,000 single-turn jailbreak prompts.
- 500-1,000 multi-turn variants.
- 600 safe prompts, exactly 100 in each of six subgroups.
- Total target: approximately 3,100-4,600 rows.

The dataset schema is represented in `configs/dataset_plan.json`. The added
extension columns are `safe_subgroup`, `source_record_id`, and
`source_category` so the merged JSONL preserves real-source provenance and can
support fairness-style grouping when safe source rows are available.

## Annotation Taxonomy

The 12 node labels from the workbook are represented in `configs/taxonomy.json`:

1. benign_query
2. roleplay
3. fiction
4. authority_claim
5. hypothetical
6. obfuscation
7. context_shift
8. gradual_escalation
9. indirect_request
10. harmful_intent
11. instruction_override
12. emotional_manipulation

## Current Implementation Boundary

The repository now downloads and builds from AdvBench, JailbreakBench, and
HarmBench only. Synthetic generation was removed.

The real source CSVs provide harmful and benign behavior prompts, but they do not
provide the workbook's 2-3 independent graph annotator labels, GPT-4/LLaMA
response text, GPT-4/LLaMA success labels, or generated multi-turn variants.
Those pieces remain separate data collection steps before RQ0, RQ2, and E5 can
be reported as final empirical results.
