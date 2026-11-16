"""Tests for Kruskal and Prim (hand-computed CLRS graph + randomized)."""

from __future__ import annotations

import random
import unittest

from graph import Graph, GraphError
from mst import kruskal, prim


def _clrs_mst_graph() -> Graph:
    """The CLRS Figure 23.4 undirected graph; its MST weight is exactly 37."""
    g = Graph(directed=False)
    for u, v, w in [
        ("a", "b", 4), ("a", "h", 8), ("b", "c", 8), ("b", "h", 11),
        ("c", "d", 7), ("c", "i", 2), ("c", "f", 4), ("d", "e", 9),
        ("d", "f", 14), ("e", "f", 10), ("f", "g", 2), ("g", "h", 1),
        ("g", "i", 6), ("h", "i", 7),
    ]:
        g.add_edge(u, v, w)
    return g


def _random_connected_undirected(rng: random.Random, n: int) -> Graph:
    """Random connected graph: random spanning chain plus extra edges."""
    g = Graph(directed=False)
    for i in range(n):
        g.add_node(i)
    order = list(range(n))
    rng.shuffle(order)
    for a, b in zip(order, order[1:]):
        g.add_edge(a, b, rng.randint(1, 30))
    for _ in range(n * 2):
        u, v = rng.sample(range(n), 2)
        if g.weight(u, v) is None:
            g.add_edge(u, v, rng.randint(1, 30))
    return g


def _is_spanning_tree(g: Graph, edges: list[tuple[object, object, float]]) -> bool:
    if len(edges) != g.node_count() - 1:
        return False
    parent = {u: u for u in g.nodes()}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for u, v, _ in edges:
        ru, rv = find(u), find(v)
        if ru == rv:
            return False  # cycle
        parent[ru] = rv
    return len({find(u) for u in g.nodes()}) == 1


class KruskalTests(unittest.TestCase):
    def test_clrs_graph_total_weight_37(self):
        tree = kruskal(_clrs_mst_graph())
        self.assertEqual(tree.total, 37.0)
        self.assertTrue(_is_spanning_tree(_clrs_mst_graph(), tree.edges))

    def test_known_small_graph_exact_edges(self):
        g = Graph(directed=False)
        for u, v, w in [("a", "b", 1), ("b", "c", 2), ("a", "c", 3),
                        ("c", "d", 4), ("b", "d", 5)]:
            g.add_edge(u, v, w)
        tree = kruskal(g)
        self.assertEqual(tree.total, 7.0)
        self.assertEqual(
            {frozenset((u, v)) for u, v, _ in tree.edges},
            {frozenset(("a", "b")), frozenset(("b", "c")), frozenset(("c", "d"))},
        )

    def test_directed_graph_rejected(self):
        g = Graph(directed=True)
        g.add_edge("a", "b", 1.0)
        with self.assertRaises(GraphError):
            kruskal(g)

    def test_disconnected_graph_rejected(self):
        g = Graph(directed=False)
        g.add_edge("a", "b", 1.0)
        g.add_edge("c", "d", 2.0)
        with self.assertRaises(GraphError) as ctx:
            kruskal(g)
        self.assertIn("not connected", str(ctx.exception))

    def test_empty_graph_returns_empty_tree(self):
        tree = kruskal(Graph(directed=False))
        self.assertEqual(tree.total, 0.0)
        self.assertEqual(tree.edges, [])

    def test_single_node_returns_empty_tree(self):
        g = Graph(directed=False)
        g.add_node("only")
        tree = kruskal(g)
        self.assertEqual(tree.total, 0.0)
        self.assertEqual(tree.edge_pairs(), [])


class PrimTests(unittest.TestCase):
    def test_clrs_graph_total_weight_37(self):
        tree = prim(_clrs_mst_graph(), start="a")
        self.assertEqual(tree.total, 37.0)
        self.assertTrue(_is_spanning_tree(_clrs_mst_graph(), tree.edges))

    def test_kruskal_and_prim_agree_on_random_graphs(self):
        rng = random.Random(9001)
        for trial in range(10):
            g = _random_connected_undirected(rng, rng.randint(3, 10))
            k = kruskal(g)
            p = prim(g, start=g.nodes()[0])
            self.assertEqual(k.total, p.total, f"trial {trial} diverges")
            self.assertTrue(_is_spanning_tree(g, p.edges))

    def test_prim_rejects_directed_and_disconnected(self):
        d = Graph(directed=True)
        d.add_edge(1, 2, 1.0)
        with self.assertRaises(GraphError):
            prim(d)
        u = Graph(directed=False)
        u.add_edge(1, 2, 1.0)
        u.add_edge(3, 4, 1.0)
        with self.assertRaises(GraphError):
            prim(u)

    def test_prim_rejects_unknown_start(self):
        g = _clrs_mst_graph()
        with self.assertRaises(GraphError):
            prim(g, start="zz")

    def test_tie_breaking_is_deterministic(self):
        # four unit edges: two valid MSTs exist; repeated runs must agree
        g = Graph(directed=False)
        g.add_edge("a", "b", 1.0)
        g.add_edge("b", "c", 1.0)
        g.add_edge("c", "d", 1.0)
        g.add_edge("d", "a", 1.0)
        first = kruskal(g).edge_pairs()
        second = kruskal(g).edge_pairs()
        self.assertEqual(first, second)
        self.assertEqual(prim(g).edge_pairs(), prim(g).edge_pairs())


if __name__ == "__main__":
    unittest.main()
