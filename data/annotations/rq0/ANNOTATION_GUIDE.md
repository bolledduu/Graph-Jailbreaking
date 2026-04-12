# RQ0 Annotation Packet Guide

Use the 12-label taxonomy from `configs/taxonomy.json`.

- `benign_query`: A genuine, harmless question or request
- `roleplay`: Asking model to assume a character or persona
- `fiction`: Framing request as story, movie, or game
- `authority_claim`: Claiming special permissions or overrides
- `hypothetical`: Framing harmful content as thought experiment
- `obfuscation`: Encoding, translating, or disguising the request
- `context_shift`: Changing frame to normalize harmful content
- `gradual_escalation`: Incrementally increasing severity across turns
- `indirect_request`: Asking through implication rather than directly
- `harmful_intent`: The actual harmful request or goal
- `instruction_override`: Telling model to ignore training/guidelines
- `emotional_manipulation`: Using guilt, flattery, or urgency

## Rules

- Annotate each row independently.
- Use `node_labels_json` as a JSON list of node labels, for example:
  `["roleplay", "fiction", "harmful_intent"]`.
- Use `edge_list_json` as a JSON list of directed pairs, for example:
  `[["roleplay", "fiction"], ["fiction", "harmful_intent"]]`.
- Do not discuss labels across annotators until all templates are complete.
- Use `notes` only for ambiguity, not for final labels.
- Prompt text is not duplicated in these templates. Join by `id` against the
  canonical processed dataset when an annotator interface needs to display the
  prompt.
