"""Load real benchmark sources into the GraphJailbreakBench schema."""

from __future__ import annotations

import csv
import hashlib
import random
import re
from pathlib import Path
from typing import Any, Iterable

from gjb.annotation import annotate_prompt
from gjb.schema import GJBRow, validate_row
from gjb.taxonomy import taxonomy_labels


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"


def _slug(value: str | None) -> str | None:
    if not value:
        return None
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _row_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.strip().lower().encode("utf-8")).hexdigest()


def _read_csv(path: Path) -> Iterable[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        yield from csv.DictReader(handle)


def _reservoir_sample(items: Iterable[dict[str, str]], sample_size: int, seed: int) -> list[tuple[int, dict[str, str]]]:
    """Deterministic streaming sample that avoids loading huge CSVs fully."""
    if sample_size <= 0:
        return []
    rng = random.Random(seed)
    reservoir: list[tuple[int, dict[str, str]]] = []
    for index, item in enumerate(items, start=1):
        item_copy = dict(item)
        if len(reservoir) < sample_size:
            reservoir.append((index, item_copy))
            continue
        replacement = rng.randint(1, index)
        if replacement <= sample_size:
            reservoir[replacement - 1] = (index, item_copy)
    return sorted(reservoir, key=lambda pair: pair[0])


def _make_row(
    *,
    row_id: str,
    row_type: str,
    source_dataset: str,
    prompt: str,
    is_harmful: bool,
    safe_subgroup: str | None,
    metadata: dict[str, Any],
) -> GJBRow:
    graph = annotate_prompt(prompt, is_harmful=is_harmful)
    return GJBRow(
        id=row_id,
        type=row_type,
        source_dataset=source_dataset,
        prompt=prompt,
        conversation=[],
        attack_type=graph["attack_type"],
        difficulty=graph["difficulty"],
        graph_nodes=graph["graph_nodes"],
        graph_edges=graph["graph_edges"],
        graph_depth=graph["graph_depth"],
        num_nodes=graph["num_nodes"],
        model_gpt4_response=None,
        model_gpt4_success=None,
        model_llama_response=None,
        model_llama_success=None,
        is_jailbreak=is_harmful,
        is_harmful=is_harmful,
        safe_subgroup=safe_subgroup,
        source_record_id=metadata.get("source_record_id"),
        source_category=metadata.get("source_category"),
        metadata=metadata,
    )


def load_advbench(raw_dir: Path = DEFAULT_RAW_DIR) -> list[GJBRow]:
    path = raw_dir / "advbench" / "harmful_behaviors.csv"
    rows: list[GJBRow] = []
    for index, record in enumerate(_read_csv(path), start=1):
        prompt = record["goal"].strip()
        rows.append(
            _make_row(
                row_id=f"advbench_{index:04d}",
                row_type="jailbreak",
                source_dataset="AdvBench",
                prompt=prompt,
                is_harmful=True,
                safe_subgroup=None,
                metadata={
                    "source_record_id": str(index),
                    "source_category": None,
                    "raw_fields": {"target": record.get("target", "")},
                    "source_path": str(path),
                    "prompt_sha256": _row_hash(prompt),
                },
            )
        )
    return rows


def load_jailbreakbench(raw_dir: Path = DEFAULT_RAW_DIR) -> list[GJBRow]:
    rows: list[GJBRow] = []
    sources = [
        ("harmful-behaviors.csv", True, "jbb_harmful", "jailbreak"),
        ("benign-behaviors.csv", False, "jbb_benign", "safe"),
    ]
    for filename, is_harmful, id_prefix, row_type in sources:
        path = raw_dir / "jailbreakbench" / filename
        for index, record in enumerate(_read_csv(path), start=1):
            source_id = record.get("Index") or str(index)
            category = record.get("Category")
            prompt = record["Goal"].strip()
            rows.append(
                _make_row(
                    row_id=f"{id_prefix}_{int(source_id):04d}" if str(source_id).isdigit() else f"{id_prefix}_{index:04d}",
                    row_type=row_type,
                    source_dataset="JailbreakBench",
                    prompt=prompt,
                    is_harmful=is_harmful,
                    safe_subgroup=_slug(category) if not is_harmful else None,
                    metadata={
                        "source_record_id": source_id,
                        "source_category": category,
                        "raw_fields": {
                            "target": record.get("Target", ""),
                            "behavior": record.get("Behavior", ""),
                            "source": record.get("Source", ""),
                        },
                        "source_path": str(path),
                        "prompt_sha256": _row_hash(prompt),
                    },
                )
            )
    return rows


def load_harmbench(raw_dir: Path = DEFAULT_RAW_DIR) -> list[GJBRow]:
    path = raw_dir / "harmbench" / "harmbench_behaviors_text_all.csv"
    rows: list[GJBRow] = []
    for index, record in enumerate(_read_csv(path), start=1):
        behavior = record["Behavior"].strip()
        context = record.get("ContextString", "").strip()
        prompt = f"{context}\n\n{behavior}" if context else behavior
        source_id = record.get("BehaviorID") or str(index)
        rows.append(
            _make_row(
                row_id=f"harmbench_{index:04d}",
                row_type="jailbreak",
                source_dataset="HarmBench",
                prompt=prompt,
                is_harmful=True,
                safe_subgroup=None,
                metadata={
                    "source_record_id": source_id,
                    "source_category": record.get("SemanticCategory", ""),
                    "raw_fields": {
                        "functional_category": record.get("FunctionalCategory", ""),
                        "semantic_category": record.get("SemanticCategory", ""),
                        "tags": record.get("Tags", ""),
                        "has_context": bool(context),
                    },
                    "source_path": str(path),
                    "prompt_sha256": _row_hash(prompt),
                },
            )
        )
    return rows


def load_jailbreakv_28k(raw_dir: Path = DEFAULT_RAW_DIR, limit: int | None = None, seed: int = 28) -> list[GJBRow]:
    path = raw_dir / "jailbreakv_28k" / "JailBreakV_28K.csv"
    source_records = (
        _reservoir_sample(_read_csv(path), limit, seed) if limit is not None else list(enumerate(_read_csv(path), start=1))
    )
    rows: list[GJBRow] = []
    for source_index, record in source_records:
        prompt = record["jailbreak_query"].strip()
        category = record.get("policy", "")
        source_id = record.get("id") or str(source_index)
        rows.append(
            _make_row(
                row_id=f"jailbreakv28k_{source_index:05d}",
                row_type="jailbreak",
                source_dataset="JailbreakV-28K",
                prompt=prompt,
                is_harmful=True,
                safe_subgroup=None,
                metadata={
                    "source_record_id": source_id,
                    "source_category": category,
                    "raw_fields": {
                        "redteam_query_sha256": _row_hash(record.get("redteam_query", "")),
                        "format": record.get("format", ""),
                        "from": record.get("from", ""),
                        "selected_mini": record.get("selected_mini", ""),
                        "transfer_from_llm": record.get("transfer_from_llm", ""),
                        "image_path": record.get("image_path", ""),
                    },
                    "source_path": str(path),
                    "prompt_sha256": _row_hash(prompt),
                    "selection_seed": seed,
                    "selection_limit": limit,
                },
            )
        )
    return rows


def load_jailbreakv_redteam(raw_dir: Path = DEFAULT_RAW_DIR, limit: int | None = None, seed: int = 29) -> list[GJBRow]:
    path = raw_dir / "jailbreakv_28k" / "RedTeam_2K.csv"
    source_records = (
        _reservoir_sample(_read_csv(path), limit, seed) if limit is not None else list(enumerate(_read_csv(path), start=1))
    )
    rows: list[GJBRow] = []
    for source_index, record in source_records:
        prompt = record["question"].strip()
        category = record.get("policy", "")
        source_id = record.get("id") or str(source_index)
        rows.append(
            _make_row(
                row_id=f"jailbreakv_redteam_{source_index:05d}",
                row_type="jailbreak",
                source_dataset="JailbreakV-28K-RedTeam",
                prompt=prompt,
                is_harmful=True,
                safe_subgroup=None,
                metadata={
                    "source_record_id": source_id,
                    "source_category": category,
                    "raw_fields": {"from": record.get("from", "")},
                    "source_path": str(path),
                    "prompt_sha256": _row_hash(prompt),
                    "selection_seed": seed,
                    "selection_limit": limit,
                },
            )
        )
    return rows


def _jailbreakdb_prompt(record: dict[str, str]) -> str:
    system_prompt = record.get("system_prompt", "").strip()
    user_prompt = record.get("user_prompt", "").strip()
    if system_prompt and user_prompt:
        return f"{system_prompt}\n\n{user_prompt}"
    return user_prompt or system_prompt


def load_jailbreakdb(
    raw_dir: Path = DEFAULT_RAW_DIR,
    *,
    filename: str,
    row_type: str,
    source_dataset: str,
    is_harmful: bool,
    safe_subgroup: str | None,
    limit: int | None = None,
    seed: int = 30,
) -> list[GJBRow]:
    path = raw_dir / "jailbreakdb" / filename
    source_records = (
        _reservoir_sample(_read_csv(path), limit, seed) if limit is not None else list(enumerate(_read_csv(path), start=1))
    )
    rows: list[GJBRow] = []
    for source_index, record in source_records:
        prompt = _jailbreakdb_prompt(record)
        if not prompt:
            continue
        rows.append(
            _make_row(
                row_id=f"{_slug(source_dataset) or 'jailbreakdb'}_{source_index:07d}",
                row_type=row_type,
                source_dataset=source_dataset,
                prompt=prompt,
                is_harmful=is_harmful,
                safe_subgroup=safe_subgroup,
                metadata={
                    "source_record_id": str(source_index),
                    "source_category": record.get("tactic", ""),
                    "raw_fields": {
                        "source": record.get("source", ""),
                        "tactic": record.get("tactic", ""),
                        "jailbreak": record.get("jailbreak", ""),
                    },
                    "source_path": str(path),
                    "prompt_sha256": _row_hash(prompt),
                    "selection_seed": seed,
                    "selection_limit": limit,
                },
            )
        )
    return rows


def _dedupe(rows: list[GJBRow]) -> list[GJBRow]:
    deduped: list[GJBRow] = []
    seen: set[str] = set()
    for row in rows:
        prompt_hash = row.metadata["prompt_sha256"]
        if prompt_hash in seen:
            continue
        seen.add(prompt_hash)
        deduped.append(row)
    return deduped


def load_real_sources(
    raw_dir: Path = DEFAULT_RAW_DIR,
    dedupe_prompts: bool = False,
    target_jailbreak: int | None = None,
    target_safe: int | None = None,
    seed: int = 30,
) -> list[dict]:
    rows = [*load_advbench(raw_dir), *load_jailbreakbench(raw_dir), *load_harmbench(raw_dir)]
    if dedupe_prompts:
        rows = _dedupe(rows)

    if target_jailbreak is not None:
        current_jailbreak = sum(row.type == "jailbreak" for row in rows)
        needed_jailbreak = max(target_jailbreak - current_jailbreak, 0)
        if needed_jailbreak:
            additions = load_jailbreakv_28k(raw_dir, limit=needed_jailbreak, seed=seed)
            rows.extend(additions)
            if dedupe_prompts:
                rows = _dedupe(rows)
                current_jailbreak = sum(row.type == "jailbreak" for row in rows)
                still_needed = max(target_jailbreak - current_jailbreak, 0)
                if still_needed:
                    rows.extend(load_jailbreakv_redteam(raw_dir, limit=still_needed, seed=seed + 1))
                    rows = _dedupe(rows)
                current_jailbreak = sum(row.type == "jailbreak" for row in rows)
                still_needed = max(target_jailbreak - current_jailbreak, 0)
                if still_needed:
                    rows.extend(
                        load_jailbreakdb(
                            raw_dir,
                            filename="text_jailbreak_unique.csv",
                            row_type="jailbreak",
                            source_dataset="JailbreakDB-Jailbreak",
                            is_harmful=True,
                            safe_subgroup=None,
                            limit=still_needed,
                            seed=seed + 3,
                        )
                    )
                    rows = _dedupe(rows)

    if target_safe is not None:
        current_safe = sum(row.type == "safe" for row in rows)
        needed_safe = max(target_safe - current_safe, 0)
        if needed_safe:
            rows.extend(
                load_jailbreakdb(
                    raw_dir,
                    filename="text_regular_unique.csv",
                    row_type="safe",
                    source_dataset="JailbreakDB-Regular",
                    is_harmful=False,
                    safe_subgroup="jailbreakdb_regular",
                    limit=needed_safe,
                    seed=seed + 2,
                )
            )
            if dedupe_prompts:
                rows = _dedupe(rows)

    valid_labels = taxonomy_labels()
    dict_rows = [row.to_dict() for row in rows]
    for row in dict_rows:
        validate_row(row, valid_labels)
    return dict_rows
