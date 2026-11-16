"""Tests for the graph-algos CLI."""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest

import cli


class CliTests(unittest.TestCase):
    def _edge_file(self, text: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".txt")
        self.addCleanup(os.remove, path)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def _run(self, argv: list[str]) -> tuple[int, str]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            code = cli.main(argv)
        return code, buf.getvalue()

    def test_info_counts_nodes_and_edges(self):
        path = self._edge_file("a b 2\nb c\n")
        code, out = self._run(["info", path])
        self.assertEqual(code, 0)
        self.assertIn("undirected graph: 3 nodes, 2 edges", out)
        self.assertIn("nodes: a b c", out)

    def test_dijkstra_prints_distances(self):
        path = self._edge_file("A B 4\nA C 2\nC B 1\nB D 5\nC D 8\n")
        code, out = self._run(["dijkstra", path, "A", "--directed"])
        self.assertEqual(code, 0)
        self.assertIn("B: A -> C -> B  (distance 3)", out)
        self.assertIn("C: A -> C  (distance 2)", out)
        self.assertIn("D: A -> C -> B -> D  (distance 8)", out)

    def test_dijkstra_reports_unreachable(self):
        path = self._edge_file("A B 1\nZ Y\n")
        code, out = self._run(["dijkstra", path, "A", "--directed"])
        self.assertEqual(code, 0)
        self.assertIn("Z: unreachable", out)
        self.assertIn("Y: unreachable", out)

    def test_astar_returns_two_when_unreachable(self):
        # Q exists in the graph but has no path from A
        path = self._edge_file("A B 1\nQ Z\n")
        code, out = self._run(["astar", path, "A", "Q", "--directed"])
        self.assertEqual(code, 2)
        self.assertIn("unreachable", out)

    def test_bellman_flags_negative_cycle(self):
        path = self._edge_file("s x 1\nx y -2\ny x -1\n")
        code, out = self._run(["bellman", path, "s", "--directed"])
        self.assertEqual(code, 3)
        self.assertIn("negative cycle detected", out)

    def test_mst_prints_total(self):
        path = self._edge_file("a b 1\nb c 2\na c 3\nc d 4\n")
        code, out = self._run(["mst", path])
        self.assertEqual(code, 0)
        self.assertIn("total: 7", out)
        code, out_prim = self._run(["mst", path, "--prim"])
        self.assertEqual(code, 0)
        self.assertIn("total: 7", out_prim)

    def test_topo_prints_valid_order(self):
        path = self._edge_file("shirt belt\nshirt pants\npants belt\n")
        code, out = self._run(["topo", path])
        self.assertEqual(code, 0)
        order = out.strip().split(" -> ")
        self.assertEqual(len(order), 3)
        self.assertLess(order.index("pants"), order.index("belt"))
        self.assertLess(order.index("shirt"), order.index("pants"))

    def test_topo_cycle_exits_nonzero(self):
        path = self._edge_file("a b\nb a\n")
        code, _ = self._run(["topo", path])
        self.assertEqual(code, 1)

    def test_scc_prints_components(self):
        path = self._edge_file("a b\nb c\nc a\nd e\ne d\n")
        code, out = self._run(["scc", path])
        self.assertEqual(code, 0)
        self.assertIn("scc 1:", out)
        self.assertEqual(out.count("scc "), 2)

    def test_flow_reports_value_and_cut(self):
        path = self._edge_file("s a 10\ns b 10\na c 8\nb c 4\nb t 10\nc t 6\n")
        code, out = self._run(["flow", path, "s", "t"])
        self.assertEqual(code, 0)
        self.assertIn("max flow: 16", out)
        self.assertIn("min cut:", out)
        self.assertGreaterEqual(out.count("cap "), 1)

    def test_dot_writes_stdout_or_file(self):
        path = self._edge_file("a b 3\n")
        code, out = self._run(["dot", path, "--directed"])
        self.assertEqual(code, 0)
        self.assertIn('digraph "G"', out)
        fd, out_path = tempfile.mkstemp(suffix=".dot")
        self.addCleanup(os.remove, out_path)
        os.close(fd)
        code, out = self._run(["dot", path, "--directed", "-o", out_path])
        self.assertEqual(code, 0)
        self.assertIn("wrote", out)
        with open(out_path, "r", encoding="utf-8") as fh:
            self.assertIn('-> "b"', fh.read())

    def test_missing_file_exits_one(self):
        code, _ = self._run(["info", "definitely-not-here-xyz.txt"])
        self.assertEqual(code, 1)

    def test_mst_on_disconnected_file_exits_one(self):
        path = self._edge_file("a b 1\nc d 1\n")
        code, _ = self._run(["mst", path])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
