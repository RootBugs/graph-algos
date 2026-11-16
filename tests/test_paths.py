"""Tests for Dijkstra, A*, and Bellman-Ford (hand-computed + randomized)."""

from __future__ import annotations

import math
import random
import unittest

from graph import Graph, GraphError
from shortest import (
    BellmanFordResult,
    astar,
    bellman_ford,
    dijkstra,
    reconstruct_path,
)


def _hand_graph() -> Graph:
    """Directed graph whose shortest distances from A are hand-computed.

    A->B=4, A->C=2, C->B=1, B->D=5, C->D=8, C->E=10, D->E=2, plus an
    unreachable F.  Shortest from A: A=0 B=3 (via C) C=2 D=8 (via C,B) E=10.
    """
    g = Graph(directed=True)
    for u, v, w in [
        ("A", "B", 4), ("A", "C", 2), ("C", "B", 1), ("B", "D", 5),
        ("C", "D", 8), ("C", "E", 10), ("D", "E", 2),
    ]:
        g.add_edge(u, v, w)
    g.add_node("F")
    return g


class DijkstraTests(unittest.TestCase):
    def test_hand_computed_distances_and_predecessors(self):
        dist, pred = dijkstra(_hand_graph(), "A")
        self.assertEqual(dist, {"A": 0.0, "B": 3.0, "C": 2.0, "D": 8.0, "E": 10.0})
        self.assertEqual(pred["B"], "C")  # indirect route beats direct edge
        self.assertEqual(pred["C"], "A")
        self.assertEqual(pred["D"], "B")
        self.assertEqual(pred["E"], "D")
        self.assertIsNone(pred["A"])

    def test_unreachable_node_absent_and_path_none(self):
        dist, pred = dijkstra(_hand_graph(), "A")
        self.assertNotIn("F", dist)
        self.assertIsNone(reconstruct_path(pred, "A", "F"))

    def test_path_reconstruction_follows_predecessors(self):
        dist, pred = dijkstra(_hand_graph(), "A")
        path = reconstruct_path(pred, "A", "E")
        self.assertEqual(path, ["A", "C", "B", "D", "E"])
        self.assertEqual(dist["E"], 10.0)

    def test_unknown_source_raises(self):
        with self.assertRaises(GraphError):
            dijkstra(_hand_graph(), "ZZ")

    def test_negative_weight_rejected(self):
        g = Graph(directed=True)
        g.add_edge("a", "b", -1.0)
        with self.assertRaises(GraphError):
            dijkstra(g, "a")


class AStarTests(unittest.TestCase):
    def test_zero_heuristic_matches_dijkstra(self):
        g = _hand_graph()
        dist_d, _ = dijkstra(g, "A")
        cost, path = astar(g, "A", "E", lambda _u: 0.0)
        self.assertEqual(cost, dist_d["E"])
        self.assertEqual(path, ["A", "C", "B", "D", "E"])

    def test_admissible_heuristic_stays_optimal(self):
        """2-D grid with Euclidean weights; straight-line h is admissible."""
        g = Graph(directed=True)
        coords = {
            (0, 0): "s", (1, 0): "a", (2, 0): "b", (2, 1): "c", (2, 2): "t",
            (1, 2): "d", (0, 1): "e", (0, 2): "f", (1, 1): "m",
        }
        edges = [
            ("s", "a"), ("a", "b"), ("b", "c"), ("c", "t"),
            ("s", "e"), ("e", "f"), ("f", "d"), ("d", "t"),
            ("a", "m"), ("m", "d"), ("b", "m"), ("m", "c"),
            ("e", "m"),
        ]
        pos = {v: k for k, v in coords.items()}

        def he(p, q):
            return math.dist(p, q)

        for u, v in edges:
            g.add_edge(u, v, he(pos[u], pos[v]))
            g.add_edge(v, u, he(pos[v], pos[u]))

        h = lambda node: he(pos[node], pos["t"])
        dist_d, _ = dijkstra(g, "s")
        cost, path = astar(g, "s", "t", h)
        self.assertEqual(cost, dist_d["t"])
        self.assertEqual(cost, 4.0)  # Manhattan lower bound, hit exactly
        self.assertEqual(path[0], "s")
        self.assertEqual(path[-1], "t")
        self.assertEqual(len(path), 5)  # 4 unit edges — no detour possible

    def test_unreachable_target_returns_infinite(self):
        g = _hand_graph()
        cost, path = astar(g, "A", "F", lambda _u: 0.0)
        self.assertEqual(cost, math.inf)
        self.assertIsNone(path)

    def test_inadmissible_zero_shortcut_still_terminates(self):
        # even a bad heuristic must not loop forever or crash
        g = _hand_graph()
        cost, path = astar(g, "A", "E", lambda _u: -1000.0)
        self.assertEqual(cost, 10.0)
        self.assertIsNotNone(path)


class BellmanFordTests(unittest.TestCase):
    def test_negative_edges_without_negative_cycle(self):
        g = Graph(directed=True)
        for u, v, w in [
            ("s", "a", 5), ("s", "b", 2), ("a", "b", -4), ("b", "c", 1),
        ]:
            g.add_edge(u, v, w)
        result = bellman_ford(g, "s")
        self.assertFalse(result.has_negative_cycle)
        self.assertEqual(result.dist["s"], 0.0)
        self.assertEqual(result.dist["a"], 5.0)
        self.assertEqual(result.dist["b"], 1.0)
        self.assertEqual(result.dist["c"], 2.0)

    def test_negative_cycle_detected_and_extracted(self):
        g = Graph(directed=True)
        for u, v, w in [
            ("s", "x", 1), ("x", "y", -2), ("y", "x", -1), ("y", "z", 3),
        ]:
            g.add_edge(u, v, w)
        result = bellman_ford(g, "s")
        self.assertIsInstance(result, BellmanFordResult)
        self.assertTrue(result.has_negative_cycle)
        cycle = result.negative_cycle
        self.assertEqual(cycle[0], cycle[-1])
        self.assertEqual(set(cycle), {"x", "y"})
        # every consecutive pair of the reported cycle is a real edge
        for a, b in zip(cycle, cycle[1:]):
            self.assertIsNotNone(g.weight(a, b), f"missing cycle edge {a}->{b}")

    def test_negative_self_loop_detected(self):
        g = Graph(directed=True)
        g.add_edge("s", "a", 1.0)
        g.add_edge("a", "a", -0.5)
        result = bellman_ford(g, "s")
        self.assertTrue(result.has_negative_cycle)
        self.assertIn("a", result.negative_cycle)

    def test_cycle_behind_unreachable_subgraph_not_reported(self):
        g = Graph(directed=True)
        g.add_edge("s", "ok", 1.0)
        g.add_edge("u", "v", -5.0)
        g.add_edge("v", "u", -5.0)
        result = bellman_ford(g, "s")
        self.assertFalse(result.has_negative_cycle)
        self.assertEqual(result.dist["ok"], 1.0)

    def test_undirected_negative_edge_is_a_negative_cycle(self):
        # an undirected edge usable both ways with negative weight = cycle
        g = Graph(directed=False)
        g.add_edge("s", "a", -2.0)
        result = bellman_ford(g, "s")
        self.assertTrue(result.has_negative_cycle)

    def test_unknown_source_raises(self):
        with self.assertRaises(GraphError):
            bellman_ford(Graph(directed=True), "nope")


class RandomizedCrossCheckTests(unittest.TestCase):
    """Seeded random graphs where algorithms must agree with each other."""

    def _random_digraph(self, rng: random.Random, n: int, p: float) -> Graph:
        g = Graph(directed=True)
        for i in range(n):
            g.add_node(i)
        for u in range(n):
            for v in range(n):
                if u != v and rng.random() < p:
                    g.add_edge(u, v, rng.randint(1, 20))
        return g

    def test_dijkstra_equals_bellman_ford_on_random_graphs(self):
        rng = random.Random(20261010)
        for trial in range(8):
            g = self._random_digraph(rng, rng.randint(3, 9), 0.35)
            src = rng.choice(g.nodes())
            dist_d, _ = dijkstra(g, src)
            bf = bellman_ford(g, src)
            for v in g.nodes():
                expected = dist_d.get(v, math.inf)
                self.assertEqual(
                    bf.dist[v], expected,
                    f"trial {trial}: node {v} diverges",
                )

    def test_astar_zero_equals_dijkstra_on_random_graphs(self):
        rng = random.Random(777)
        for trial in range(8):
            g = self._random_digraph(rng, rng.randint(3, 9), 0.4)
            src, dst = rng.sample(g.nodes(), 2)
            dist_d, _ = dijkstra(g, src)
            cost, _ = astar(g, src, dst, lambda _u: 0.0)
            self.assertEqual(cost, dist_d.get(dst, math.inf), f"trial {trial}")

    def test_dijkstra_distances_are_triangle_consistent(self):
        rng = random.Random(4242)
        for _ in range(5):
            g = self._random_digraph(rng, 8, 0.5)
            src = 0
            dist, _ = dijkstra(g, src)
            for u in g.nodes():
                for v, w in g.neighbors(u):
                    if u in dist and v in dist:
                        self.assertLessEqual(dist[v], dist[u] + w + 1e-9)


if __name__ == "__main__":
    unittest.main()
