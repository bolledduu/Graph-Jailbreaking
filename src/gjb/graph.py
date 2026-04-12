"""Small graph utilities used by the benchmark.

The functions here deliberately use the Python standard library so the initial
codebase can run before optional research dependencies such as NetworkX or
PyTorch Geometric are installed.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations


Edge = tuple[str, str]


def normalize_edges(edges: list[list[str]] | list[tuple[str, str]]) -> list[Edge]:
    normalized: list[Edge] = []
    for edge in edges:
        if len(edge) != 2:
            raise ValueError(f"Graph edge must contain exactly two labels: {edge!r}")
        source, target = str(edge[0]), str(edge[1])
        if source == target:
            continue
        normalized.append((source, target))
    return sorted(set(normalized))


def graph_depth(nodes: list[str], edges: list[list[str]] | list[tuple[str, str]]) -> int:
    """Return longest directed path length, counting nodes in the path."""
    if not nodes:
        return 0
    adjacency: dict[str, list[str]] = defaultdict(list)
    for source, target in normalize_edges(edges):
        adjacency[source].append(target)

    visiting: set[str] = set()
    memo: dict[str, int] = {}

    def depth_from(node: str) -> int:
        if node in memo:
            return memo[node]
        if node in visiting:
            return 1
        visiting.add(node)
        child_depth = 0
        for child in adjacency.get(node, []):
            child_depth = max(child_depth, depth_from(child))
        visiting.remove(node)
        memo[node] = 1 + child_depth
        return memo[node]

    return max(depth_from(node) for node in nodes)


def motif_key(nodes: list[str], edges: list[Edge]) -> str:
    """Create a stable key for a small directed labeled graph."""
    node_part = ",".join(sorted(nodes))
    edge_part = ",".join(f"{source}>{target}" for source, target in sorted(edges))
    return f"nodes:{node_part}|edges:{edge_part}"


def enumerate_motifs(
    nodes: list[str],
    edges: list[list[str]] | list[tuple[str, str]],
    min_size: int = 2,
    max_size: int = 4,
) -> set[str]:
    """Enumerate induced labeled directed motifs for node subset sizes 2-4."""
    if min_size < 1 or max_size < min_size:
        raise ValueError("Invalid motif size bounds")
    unique_nodes = sorted(set(nodes))
    edge_set = set(normalize_edges(edges))
    motifs: set[str] = set()
    for size in range(min_size, min(max_size, len(unique_nodes)) + 1):
        for subset in combinations(unique_nodes, size):
            subset_set = set(subset)
            subset_edges = [
                edge for edge in edge_set if edge[0] in subset_set and edge[1] in subset_set
            ]
            if subset_edges:
                motifs.add(motif_key(list(subset), subset_edges))
    return motifs


def graph_features(nodes: list[str], edges: list[list[str]] | list[tuple[str, str]]) -> dict[str, float]:
    """Return model-ready graph features without relying on a GNN dependency."""
    normalized = normalize_edges(edges)
    node_count = len(set(nodes))
    edge_count = len(normalized)
    return {
        "num_nodes": float(node_count),
        "num_edges": float(edge_count),
        "graph_depth": float(graph_depth(nodes, normalized)),
        "edge_density": float(edge_count / (node_count * (node_count - 1))) if node_count > 1 else 0.0,
        "motif_count_2_4": float(len(enumerate_motifs(nodes, normalized))),
    }
