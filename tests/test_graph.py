"""Tests for graph representation, edge-list loading, and DOT export."""

from __future__ import annotations

import os
import tempfile
import unittest

from graph import Graph, GraphError, load_edgelist, to_dot, visualize


class GraphRepresentationTests(unittest.TestCase):
    def test_undirected_edge_adds_both_directions(self):
        g = Graph(directed=False)
        g.add_edge("a", "b", 2.5)
        self.assertEqual(g.neighbors("a"), [("b", 2.5)])
        self.assertEqual(g.neighbors("b"), [("a", 2.5)])
        self.assertEqual(g.edge_count(), 1)
        self.assertEqual(g.nodes(), ["a", "b"])

    def test_directed_edge_adds_one_direction(self):
        g = Graph(directed=True)
        g.add_edge(1, 2, 3.0)
        self.assertEqual(g.neighbors(1), [(2, 3.0)])
        self.assertEqual(g.neighbors(2), [])
        self.assertEqual(g.weight(1, 2), 3.0)
        self.assertIsNone(g.weight(2, 1))

    def test_matrix_roundtrip_preserves_edges(self):
        g = Graph(directed=True)
        g.add_edge("s", "a", 4.0)
        g.add_edge("s", "b", 1.5)
        g.add_edge("a", "b", 2.0)
        nodes, matrix = g.to_matrix()
        self.assertEqual(nodes, ["s", "a", "b"])
        self.assertEqual(matrix[0][1], 4.0)
        self.assertIsNone(matrix[1][0])
        rebuilt = Graph.from_matrix(matrix, directed=True, nodes=list(nodes))
        self.assertEqual(rebuilt.edges(), g.edges())

    def test_from_matrix_rejects_nonsquare(self):
        with self.assertRaises(GraphError):
            Graph.from_matrix([[None, 1.0]])

    def test_from_matrix_rejects_undirected_self_loop(self):
        with self.assertRaises(GraphError):
            Graph.from_matrix([[5.0]], directed=False)

    def test_duplicate_edge_last_write_wins_in_matrix(self):
        g = Graph(directed=True)
        g.add_edge("a", "b", 1.0)
        g.add_edge("a", "b", 9.0)
        _, matrix = g.to_matrix()
        self.assertEqual(matrix[0][1], 9.0)


class EdgeListLoaderTests(unittest.TestCase):
    def _write(self, text: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".txt")
        self.addCleanup(os.remove, path)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def test_loads_weighted_unweighted_and_comments(self):
        path = self._write(
            "# a comment\n"
            "a b\n"
            "b c 2.5\n"
            "\n"
            "a c -1\n"
        )
        g = load_edgelist(path, directed=True)
        self.assertEqual(g.node_count(), 3)
        self.assertEqual(g.edge_count(), 3)
        self.assertEqual(g.weight("a", "b"), 1.0)
        self.assertEqual(g.weight("b", "c"), 2.5)
        self.assertEqual(g.weight("a", "c"), -1.0)

    def test_malformed_line_reports_line_number(self):
        path = self._write("a b\nx y z 4\n")
        with self.assertRaises(GraphError) as ctx:
            load_edgelist(path)
        self.assertIn(":2:", str(ctx.exception))

    def test_bad_weight_reports_line_number(self):
        path = self._write("a b notanumber\n")
        with self.assertRaises(GraphError) as ctx:
            load_edgelist(path)
        self.assertIn(":1:", str(ctx.exception))
        self.assertIn("notanumber", str(ctx.exception))

    def test_directed_flag_honored(self):
        path = self._write("u v 3\n")
        g = load_edgelist(path, directed=True)
        self.assertEqual(g.neighbors("v"), [])
        self.assertEqual(g.neighbors("u"), [("v", 3.0)])


class DotExportTests(unittest.TestCase):
    def test_directed_dot_syntax(self):
        g = Graph(directed=True)
        g.add_edge("a", "b", 2.0)
        text = to_dot(g, name="demo")
        self.assertIn('digraph "demo"', text)
        self.assertIn('"a" -> "b" [label="2"]', text)
        self.assertIn('"a";', text)
        self.assertTrue(text.rstrip().endswith("}"))

    def test_undirected_dot_uses_dashes(self):
        g = Graph(directed=False)
        g.add_edge("x", "y")  # unit weight: no label attribute
        text = to_dot(g)
        self.assertIn('graph "G"', text)
        self.assertIn('"x" -- "y";', text)

    def test_visualize_writes_file(self):
        g = Graph(directed=False)
        g.add_edge(1, 2, 7.5)
        fd, path = tempfile.mkstemp(suffix=".dot")
        self.addCleanup(os.remove, path)
        os.close(fd)
        text = visualize(g, path)
        with open(path, "r", encoding="utf-8") as fh:
            self.assertEqual(fh.read(), text)
        self.assertIn('[label="7.5"]', text)

    def test_dot_escapes_quotes_in_node_names(self):
        g = Graph(directed=True)
        g.add_edge('we"ird', "n2")
        text = to_dot(g)
        self.assertIn('"we\\"ird"', text)


if __name__ == "__main__":
    unittest.main()
