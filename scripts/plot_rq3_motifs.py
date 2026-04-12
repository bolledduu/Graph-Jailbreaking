#!/usr/bin/env python3
"""Plot top RQ3 directed graph motifs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_motif(motif: str) -> tuple[list[str], list[tuple[str, str]]]:
    node_match = re.search(r"nodes:([^|]+)", motif)
    edge_match = re.search(r"edges:(.+)$", motif)
    nodes = node_match.group(1).split(",") if node_match else []
    edges: list[tuple[str, str]] = []
    if edge_match:
        for edge in edge_match.group(1).split(","):
            if ">" in edge:
                source, target = edge.split(">", 1)
                edges.append((source, target))
    return nodes, edges


def plot_motif(row: dict, output_path: Path) -> None:
    nodes, edges = parse_motif(row["motif"])
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    pos = nx.spring_layout(graph, seed=7) if len(nodes) > 1 else {nodes[0]: (0.0, 0.0)}

    plt.figure(figsize=(5.6, 3.8))
    nx.draw_networkx_nodes(graph, pos, node_color="#d9f0ee", edgecolors="#1c4945", node_size=1800, linewidths=1.2)
    nx.draw_networkx_edges(graph, pos, edge_color="#525252", arrows=True, arrowsize=18, width=1.5, connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_labels(graph, pos, font_size=8, font_family="DejaVu Sans")
    plt.title(f"Rank {row['rank']} | Frequency {row['frequency']} | Success {row['success_rate']}", fontsize=10)
    plt.axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motifs", default="results/tables/table_4_top_motifs_real.csv")
    parser.add_argument("--output-dir", default="results/figures/rq3")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    motif_path = PROJECT_ROOT / args.motifs
    output_dir = PROJECT_ROOT / args.output_dir
    with motif_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[: args.top_k]

    figures = []
    for row in rows:
        output = output_dir / f"motif_rank_{int(row['rank']):02d}.png"
        plot_motif(row, output)
        figures.append({"rank": row["rank"], "motif": row["motif"], "frequency": row["frequency"], "path": str(output.relative_to(PROJECT_ROOT))})

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "motif_table": args.motifs,
        "figures": figures,
        "prompt_text_logged": False,
    }
    summary_path = output_dir / "motif_figure_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
