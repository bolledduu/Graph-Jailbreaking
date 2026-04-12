from __future__ import annotations

import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gjb.graph import enumerate_motifs, graph_depth
from gjb.metrics import classification_metrics, cohen_kappa, normalized_graph_edit_distance, roc_auc_score
from gjb.annotation import annotate_prompt


class CoreTest(unittest.TestCase):
    def test_graph_depth(self) -> None:
        nodes = ["roleplay", "fiction", "harmful_intent"]
        edges = [["roleplay", "fiction"], ["fiction", "harmful_intent"]]
        self.assertEqual(graph_depth(nodes, edges), 3)

    def test_motif_enumeration(self) -> None:
        nodes = ["roleplay", "fiction", "harmful_intent"]
        edges = [["roleplay", "fiction"], ["fiction", "harmful_intent"]]
        motifs = enumerate_motifs(nodes, edges)
        self.assertTrue(any("roleplay>fiction" in motif for motif in motifs))

    def test_classification_metrics(self) -> None:
        metrics = classification_metrics([True, False, True, False], [True, False, False, True])
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["tp"], 1.0)
        self.assertEqual(roc_auc_score([False, True], [0.1, 0.9]), 1.0)

    def test_annotation_metrics(self) -> None:
        self.assertGreater(cohen_kappa(["a", "b", "b"], ["a", "b", "a"]), 0.0)
        ged = normalized_graph_edit_distance(
            {"nodes": ["a", "b"], "edges": [["a", "b"]]},
            {"nodes": ["a", "c"], "edges": [["a", "c"]]},
        )
        self.assertGreater(ged, 0.0)

    def test_annotation_uses_workbook_taxonomy(self) -> None:
        graph = annotate_prompt("Hypothetical educational request about an exploit", is_harmful=True)
        self.assertIn("hypothetical", graph["graph_nodes"])
        self.assertIn("harmful_intent", graph["graph_nodes"])


if __name__ == "__main__":
    unittest.main()
