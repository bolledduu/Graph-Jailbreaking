"""Dataset row schema and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from gjb.graph import graph_depth, normalize_edges


REQUIRED_COLUMNS = {
    "id",
    "type",
    "source_dataset",
    "prompt",
    "conversation",
    "attack_type",
    "difficulty",
    "graph_nodes",
    "graph_edges",
    "graph_depth",
    "num_nodes",
    "model_gpt4_response",
    "model_gpt4_success",
    "model_llama_response",
    "model_llama_success",
    "is_jailbreak",
    "is_harmful",
}

VALID_TYPES = {"jailbreak", "multiturn", "safe"}
VALID_ATTACK_TYPES = {"roleplay", "obfuscation", "reasoning", "encoding", "none"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


@dataclass(frozen=True)
class GJBRow:
    id: str
    type: str
    source_dataset: str
    prompt: str
    conversation: list[str]
    attack_type: str
    difficulty: str
    graph_nodes: list[str]
    graph_edges: list[list[str]]
    graph_depth: int
    num_nodes: int
    model_gpt4_response: str | None
    model_gpt4_success: bool | None
    model_llama_response: str | None
    model_llama_success: bool | None
    is_jailbreak: bool
    is_harmful: bool
    safe_subgroup: str | None = None
    source_record_id: str | None = None
    source_category: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_row(row: dict[str, Any], valid_node_labels: set[str]) -> None:
    missing = REQUIRED_COLUMNS - set(row)
    if missing:
        raise ValueError(f"Dataset row {row.get('id', '<missing id>')} missing columns: {sorted(missing)}")
    if row["type"] not in VALID_TYPES:
        raise ValueError(f"Invalid row type for {row['id']}: {row['type']}")
    if row["attack_type"] not in VALID_ATTACK_TYPES:
        raise ValueError(f"Invalid attack_type for {row['id']}: {row['attack_type']}")
    if row["difficulty"] not in VALID_DIFFICULTIES:
        raise ValueError(f"Invalid difficulty for {row['id']}: {row['difficulty']}")
    unknown_nodes = set(row["graph_nodes"]) - valid_node_labels
    if unknown_nodes:
        raise ValueError(f"Unknown graph nodes for {row['id']}: {sorted(unknown_nodes)}")
    normalized_edges = normalize_edges(row["graph_edges"])
    node_set = set(row["graph_nodes"])
    bad_edges = [edge for edge in normalized_edges if edge[0] not in node_set or edge[1] not in node_set]
    if bad_edges:
        raise ValueError(f"Edges reference absent nodes for {row['id']}: {bad_edges}")
    expected_depth = graph_depth(row["graph_nodes"], normalized_edges)
    if int(row["graph_depth"]) != expected_depth:
        raise ValueError(
            f"graph_depth mismatch for {row['id']}: got {row['graph_depth']}, expected {expected_depth}"
        )
    if int(row["num_nodes"]) != len(set(row["graph_nodes"])):
        raise ValueError(f"num_nodes mismatch for {row['id']}")
