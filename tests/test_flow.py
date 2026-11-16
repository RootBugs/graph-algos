"""Tests for Edmonds-Karp max flow and minimum-cut extraction."""

from __future__ import annotations

import itertools
import random
import unittest

from flow import max_flow
from graph import Graph, GraphError


def _clrs_style_network() -> Graph:
    """Hand-computed network: max flow 16.

    s->a 10, s->b 10, a->c 8, b->c 4, b->t 10, c->t 6.
    The cut {s,a,c} | {b,t} has capacity 10+6 = 16 and a matching feasible
    flow of 16 exists (s->b->t = 10, s->a->c->t = 6), so the optimum is 16.
    """
    g = Graph(directed=True)
    for u, v, w in [
        ("s", "a", 10), ("s", "b", 10), ("a", "c", 8),
        ("b", "c", 4), ("b", "t", 10), ("c", "t", 6),
    ]:
        g.add_edge(u, v, w)
    return g


def _brute_min_cut(g: Graph, source, sink) -> float:
    """Enumerate every s/t cut (small graphs only) and take the minimum."""
    nodes = [u for u in g.nodes() if u not in (source, sink)]
    best = float("inf")
    for r in range(len(nodes) + 1):
        for combo in itertools.combinations(nodes, r):
            side = {source, *combo}
            cap = sum(
                w for u, v, w in g.edges() if u in side and v not in side
            )
            best = min(best, cap)
    return best


class MaxFlowTests(unittest.TestCase):
    def test_hand_computed_value_and_cut(self):
        result = max_flow(_clrs_style_network(), "s", "t")
        self.assertEqual(result.value, 16.0)
        cut_cap = sum(w for _, _, w in result.cut)
        self.assertEqual(cut_cap, 16.0)
        # cut endpoints straddle the partition
        for u, v, _ in result.cut:
            self.assertIn(u, result.source_side)
            self.assertNotIn(v, result.source_side)

    def test_flow_conservation_and_capacity(self):
        g = _clrs_style_network()
        result = max_flow(g, "s", "t")
        out_s = sum(f for (u, _), f in result.flow.items() if u == "s")
        into_t = sum(f for (_, v), f in result.flow.items() if v == "t")
        self.assertAlmostEqual(out_s, result.value)
        self.assertAlmostEqual(into_t, result.value)
        for u, v, cap in g.edges():
            used = result.flow.get((u, v), 0.0)
            self.assertGreaterEqual(used, -1e-9)
            self.assertLessEqual(used, cap + 1e-9)
        # intermediate nodes conserve flow
        for node in g.nodes():
            if node in ("s", "t"):
                continue
            inflow = sum(f for (_, v), f in result.flow.items() if v == node)
            outflow = sum(f for (u, _), f in result.flow.items() if u == node)
            self.assertAlmostEqual(inflow, outflow, msg=f"conservation at {node}")

    def test_bottleneck_path_found_over_multiple_routes(self):
        g = Graph(directed=True)
        for u, v, w in [
            ("s", "a", 3), ("s", "b", 2), ("a", "c", 2), ("b", "c", 3),
            ("c", "t", 4),
        ]:
            g.add_edge(u, v, w)
        # paths: s-a-c-t (2) + s-b-c-t (2) = 4; s->a has 1 unit idle, c->t 0 idle
        self.assertEqual(max_flow(g, "s", "t").value, 4.0)

    def test_disconnected_returns_zero_and_empty_cut_flow(self):
        g = Graph(directed=True)
        g.add_edge("s", "a", 5.0)
        g.add_edge("b", "t", 5.0)
        result = max_flow(g, "s", "t")
        self.assertEqual(result.value, 0.0)
        self.assertEqual(result.flow, {})

    def test_rejects_undirected_source_sink_and_negatives(self):
        u = Graph(directed=False)
        u.add_edge("s", "t", 1.0)
        with self.assertRaises(GraphError):
            max_flow(u, "s", "t")
        d = Graph(directed=True)
        d.add_edge("s", "t", 1.0)
        with self.assertRaises(GraphError):
            max_flow(d, "s", "s")
        with self.assertRaises(GraphError):
            max_flow(d, "s", "zz")
        neg = Graph(directed=True)
        neg.add_edge("s", "t", -1.0)
        with self.assertRaises(GraphError):
            max_flow(neg, "s", "t")

    def test_parallel_arcs_are_summed(self):
        g = Graph(directed=True)
        g.add_edge("s", "t", 2.0)
        g.add_edge("s", "t", 3.0)
        self.assertEqual(max_flow(g, "s", "t").value, 5.0)

    def test_self_loop_ignored(self):
        g = Graph(directed=True)
        g.add_edge("s", "t", 4.0)
        g.add_edge("s", "s", 99.0)
        self.assertEqual(max_flow(g, "s", "t").value, 4.0)

    def test_max_equals_brute_min_cut_on_random_networks(self):
        rng = random.Random(31337)
        for trial in range(10):
            n = rng.randint(4, 6)
            g = Graph(directed=True)
            for i in range(n):
                g.add_node(str(i))
            src, snk = "0", str(n - 1)
            for u in range(n):
                for v in range(n):
                    if u != v and rng.random() < 0.4:
                        g.add_edge(str(u), str(v), rng.randint(1, 9))
            result = max_flow(g, src, snk)
            brute = _brute_min_cut(g, src, snk)
            self.assertEqual(
                round(result.value, 6), brute, f"trial {trial}: flow vs cut"
            )
            self.assertEqual(
                round(sum(w for _, _, w in result.cut), 6), brute
            )


if __name__ == "__main__":
    unittest.main()
