"""Minimum spanning trees: Kruskal (with an internal DSU) and Prim."""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass

from graph import Graph, GraphError


class _DSU:
    """Union-find with union by rank and iterative path compression.

    Used internally by Kruskal; kept private to this module because graph-algos
    must not import sibling projects.
    """

    def __init__(self, items: list[object]) -> None:
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x: object) -> object:
        root = x
        while self.parent[root] != root:  # iterative: no recursion limit
            root = self.parent[root]
        while self.parent[x] != root:  # path compression pass
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, a: object, b: object) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


@dataclass
class SpanningTree:
    """Result of a minimum-spanning-tree computation.

    ``total`` is the summed edge weight; ``edges`` lists ``(u, v, weight)``.
    """

    total: float
    edges: list[tuple[object, object, float]]

    def edge_pairs(self) -> list[tuple[object, object]]:
        return [(u, v) for u, v, _ in self.edges]


def kruskal(g: Graph) -> SpanningTree:
    """Kruskal's algorithm via union-find.

    Requires an undirected, connected graph; raises :class:`GraphError`
    otherwise.  Ties between equal weights are broken by edge insertion order,
    which makes the result deterministic.
    """
    if g.directed:
        raise GraphError("kruskal requires an undirected graph")
    nodes = g.nodes()
    if not nodes:
        return SpanningTree(0.0, [])
    dsu = _DSU(nodes)
    chosen: list[tuple[object, object, float]] = []
    # stable sort: equal weights keep insertion order (deterministic ties)
    for u, v, w in sorted(g.edges(), key=lambda e: e[2]):
        if dsu.union(u, v):
            chosen.append((u, v, w))
            if len(chosen) == len(nodes) - 1:
                break
    if len(chosen) != max(len(nodes) - 1, 0):
        raise GraphError("graph is not connected; spanning tree undefined")
    total = sum(w for _, _, w in chosen)
    return SpanningTree(total, chosen)


def prim(g: Graph, start: object | None = None) -> SpanningTree:
    """Prim's algorithm with a binary heap over an undirected connected graph.

    ``start`` defaults to the first node in insertion order.  Ties are broken
    by insertion counters so mixed-type nodes never compare.  Raises
    :class:`GraphError` for directed graphs or disconnected inputs.
    """
    if g.directed:
        raise GraphError("prim requires an undirected graph")
    nodes = g.nodes()
    if not nodes:
        return SpanningTree(0.0, [])
    if start is None:
        start = nodes[0]
    if not g.has_node(start):
        raise GraphError(f"start {start!r} not in graph")
    counter = itertools.count()
    chosen: list[tuple[object, object, float]] = []
    in_tree: set[object] = {start}
    heap: list[tuple[float, int, object, object]] = []
    for v, w in g.neighbors(start):
        heapq.heappush(heap, (w, next(counter), v, start))
    while heap and len(in_tree) < len(nodes):
        w, _, v, u = heapq.heappop(heap)
        if v in in_tree:
            continue
        in_tree.add(v)
        chosen.append((u, v, w))
        for nxt, nw in g.neighbors(v):
            if nxt not in in_tree:
                heapq.heappush(heap, (nw, next(counter), nxt, v))
    if len(in_tree) != len(nodes):
        raise GraphError("graph is not connected; spanning tree undefined")
    total = sum(w for _, _, w in chosen)
    return SpanningTree(total, chosen)
