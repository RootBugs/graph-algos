"""Tests for Kahn topological sort and Tarjan SCC."""

from __future__ import annotations

import random
import unittest

from graph import Graph, GraphError
from traversal import CycleError, tarjan_scc, topological_sort


class TopologicalSortTests(unittest.TestCase):
    def test_hand_computed_order_respects_all_edges(self):
        g = Graph(directed=True)
        for u, v in [("shirt", "belt"), ("shirt", "pants"),
                     ("pants", "belt"), ("pants", "shoes"),
                     ("belt", "jacket"), ("shoes", "jacket")]:
            g.add_edge(u, v)
        order = topological_sort(g)
        self.assertEqual(len(order), 5)  # shirt, pants, belt, shoes, jacket
        pos = {node: i for i, node in enumerate(order)}
        for u, v, _ in g.edges():
            self.assertLess(pos[u], pos[v], f"order violates {u}->{v}")

    def test_cycle_reported_with_concrete_cycle(self):
        g = Graph(directed=True)
        for u, v in [("a", "b"), ("b", "c"), ("c", "a"), ("c", "d")]:
            g.add_edge(u, v)
        with self.assertRaises(CycleError) as ctx:
            topological_sort(g)
        cycle = ctx.exception.cycle
        self.assertEqual(cycle[0], cycle[-1])
        self.assertEqual(set(cycle), {"a", "b", "c"})
        for a, b in zip(cycle, cycle[1:]):
            self.assertIsNotNone(g.weight(a, b), f"missing {a}->{b}")

    def test_cycle_downstream_of_source_reported(self):
        # e -> f forms a cycle; s reaches it, but the cycle must still surface
        g = Graph(directed=True)
        for u, v in [("s", "e"), ("e", "f"), ("f", "e")]:
            g.add_edge(u, v)
        with self.assertRaises(CycleError) as ctx:
            topological_sort(g)
        self.assertEqual(set(ctx.exception.cycle), {"e", "f"})

    def test_repeated_edge_does_not_false_positive(self):
        g = Graph(directed=True)
        g.add_edge("a", "b", 1.0)
        g.add_edge("a", "b", 2.0)  # parallel edge inflates indegree twice
        g.add_edge("b", "c", 1.0)
        order = topological_sort(g)
        self.assertEqual(order, ["a", "b", "c"])

    def test_undirected_rejected(self):
        g = Graph(directed=False)
        g.add_edge("a", "b")
        with self.assertRaises(GraphError):
            topological_sort(g)

    def test_random_dags_always_sort(self):
        rng = random.Random(2026)
        for _ in range(10):
            n = rng.randint(3, 12)
            g = Graph(directed=True)
            for i in range(n):
                g.add_node(i)
            # random edges only from low to high index => acyclic by design
            for u in range(n):
                for v in range(u + 1, n):
                    if rng.random() < 0.3:
                        g.add_edge(u, v)
            order = topological_sort(g)
            pos = {node: i for i, node in enumerate(order)}
            for u, v, _ in g.edges():
                self.assertLess(pos[u], pos[v])

    def test_random_graph_with_forced_cycle_detected(self):
        rng = random.Random(55)
        for _ in range(6):
            n = rng.randint(3, 8)
            g = Graph(directed=True)
            for i in range(n):
                g.add_node(i)
            # an index chain plus a back edge guarantees a cycle exists
            for i in range(n - 1):
                g.add_edge(i, i + 1)
            g.add_edge(n - 1, 0)
            # sprinkle extra random edges (may add more cycles)
            for u in range(n):
                for v in range(n):
                    if u != v and rng.random() < 0.2:
                        g.add_edge(u, v)
            with self.assertRaises(CycleError):
                topological_sort(g)


class TarjanSCCTests(unittest.TestCase):
    def test_classic_three_component_partition(self):
        g = Graph(directed=True)
        for u, v in [
            ("a", "b"), ("b", "c"), ("c", "a"),
            ("d", "e"), ("e", "d"),
            ("f", "g"), ("g", "f"),
            ("c", "d"), ("e", "f"),
        ]:
            g.add_edge(u, v)
        comps = set(map(frozenset, tarjan_scc(g)))
        self.assertEqual(
            comps,
            {frozenset({"a", "b", "c"}), frozenset({"d", "e"}),
             frozenset({"f", "g"})},
        )
        self.assertEqual(len(comps), 3)

    def test_partition_matches_reachability_reference(self):
        rng = random.Random(808)
        for _ in range(8):
            n = rng.randint(3, 8)
            g = Graph(directed=True)
            for i in range(n):
                g.add_node(i)
            for u in range(n):
                for v in range(n):
                    if u != v and rng.random() < 0.3:
                        g.add_edge(u, v)
            comps = tarjan_scc(g)
            # every node appears exactly once
            flat = [x for c in comps for x in c]
            self.assertEqual(sorted(flat), sorted(g.nodes()))
            # membership matches mutual reachability (BFS reference)
            reach = {}
            for src in g.nodes():
                seen = {src}
                stack = [src]
                while stack:
                    u = stack.pop()
                    for v, _ in g.neighbors(u):
                        if v not in seen:
                            seen.add(v)
                            stack.append(v)
                reach[src] = seen
            for u in g.nodes():
                for v in g.nodes():
                    same = any(u in c and v in c for c in comps)
                    self.assertEqual(
                        same, v in reach[u] and u in reach[v],
                        f"pair ({u},{v}) mismatch",
                    )

    def test_single_node_components(self):
        g = Graph(directed=True)
        g.add_node("solo")
        self.assertEqual(tarjan_scc(g), [["solo"]])

    def test_linear_chain_is_one_component_each(self):
        g = Graph(directed=True)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        comps = tarjan_scc(g)
        self.assertEqual(sorted(sorted(c) for c in comps), [[1], [2], [3]])

    def test_directed_required(self):
        with self.assertRaises(GraphError):
            tarjan_scc(Graph(directed=False))

    def test_large_cycle_needs_no_recursion(self):
        # 5000-node cycle would overflow a recursive Tarjan
        g = Graph(directed=True)
        n = 5000
        for i in range(n):
            g.add_edge(i, (i + 1) % n)
        comps = tarjan_scc(g)
        self.assertEqual(len(comps), 1)
        self.assertEqual(len(comps[0]), n)


if __name__ == "__main__":
    unittest.main()
