# Data Directory

This repo intentionally does not commit raw or processed benchmark prompt data.

Use the source manifest and scripts to recreate local data:

```bash
python scripts/verify_sources.py
python scripts/build_dataset.py --output data/processed/gjb_real_v1.jsonl
python scripts/assign_rq4_safe_subgroups.py \
  --dataset data/processed/gjb_real_v1.jsonl \
  --output data/processed/gjb_real_v1_rq4_balanced.jsonl
```

Committed files under `data/annotations/` are templates and metadata-only
annotation/label packets. They do not duplicate raw prompt text.
