"""Shortest-path algorithms: Dijkstra, A*, Bellman-Ford.

All three return predecessor maps so callers can reconstruct paths.  Unreachable
nodes are simply absent from the distance dict (Dijkstra) or carry ``math.inf``
(Bellman-Ford); :func:`reconstruct_path` turns ``(dist, pred)`` into a node
list or ``None``.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from graph import Graph, GraphError


def reconstruct_path(
    pred: dict[object, object | None], source: object, target: object
) -> list[object] | None:
    """Walk ``pred`` from ``target`` back to ``source``.

    Returns ``None`` when ``target`` was never reached.
    """
    if target not in pred and target != source:
        return None
    path = [target]
    seen = {target}
    cur = target
    while cur != source:
        cur = pred.get(cur)
        if cur is None or cur in seen:
            return None
        seen.add(cur)
        path.append(cur)
    path.reverse()
    return path


def dijkstra(g: Graph, source: object) -> tuple[dict[object, float], dict[object, object | None]]:
    """Dijkstra with a binary heap.

    Returns ``(dist, pred)``: ``dist[v]`` is the shortest distance from
    ``source`` to every *reachable* node; ``pred[v]`` is its predecessor on a
    shortest path (``pred[source]`` is ``None``).

    Raises :class:`GraphError` on negative edge weights.
    """
    if not g.has_node(source):
        raise GraphError(f"source {source!r} not in graph")
    for _, _, w in g.edges():
        if w < 0:
            raise GraphError("dijkstra does not accept negative weights; use bellman_ford")
    dist: dict[object, float] = {source: 0.0}
    pred: dict[object, object | None] = {source: None}
    counter = itertools.count()  # tie-break so mixed-type nodes never compare
    heap: list[tuple[float, int, object]] = [(0.0, next(counter), source)]
    settled: set[object] = set()
    while heap:
        d, _, u = heapq.heappop(heap)
        if u in settled:
            continue
        settled.add(u)
        for v, w in g.neighbors(u):
            nd = d + w
            if v not in settled and nd < dist.get(v, math.inf):
                dist[v] = nd
                pred[v] = u
                heapq.heappush(heap, (nd, next(counter), v))
    return dist, pred


def astar(
    g: Graph,
    source: object,
    target: object,
    heuristic: Callable[[object], float],
) -> tuple[float, list[object] | None]:
    """A* search with a pluggable heuristic ``h(node) -> estimate``.

    The heuristic must be *admissible* (never overestimates) for optimality;
    ``lambda _: 0`` degrades to Dijkstra.  Returns ``(distance, path)``; when
    ``target`` is unreachable the result is ``(math.inf, None)``.
    """
    if not g.has_node(source):
        raise GraphError(f"source {source!r} not in graph")
    if not g.has_node(target):
        raise GraphError(f"target {target!r} not in graph")
    counter = itertools.count()
    g_score: dict[object, float] = {source: 0.0}
    pred: dict[object, object | None] = {source: None}
    open_heap: list[tuple[float, int, object]] = [
        (heuristic(source), next(counter), source)
    ]
    closed: set[object] = set()
    while open_heap:
        _, _, u = heapq.heappop(open_heap)
        if u in closed:
            continue
        if u == target:
            return g_score[target], reconstruct_path(pred, source, target)
        closed.add(u)
        for v, w in g.neighbors(u):
            tentative = g_score[u] + w
            if tentative < g_score.get(v, math.inf):
                g_score[v] = tentative
                pred[v] = u
                f = tentative + heuristic(v)
                heapq.heappush(open_heap, (f, next(counter), v))
    return math.inf, None


@dataclass
class BellmanFordResult:
    """Outcome of a Bellman-Ford run.

    ``dist`` maps every node to its distance (``math.inf`` when unreachable).
    When a negative cycle is reachable from ``source``, ``negative_cycle`` is a
    list of nodes forming the cycle (first node repeated at the end) and
    ``has_negative_cycle`` is ``True``; distances are then unreliable.
    """

    dist: dict[object, float]
    pred: dict[object, object | None]
    has_negative_cycle: bool = False
    negative_cycle: list[object] = field(default_factory=list)


def bellman_ford(g: Graph, source: object) -> BellmanFordResult:
    """Bellman-Ford with negative-cycle detection.

    Relaxes all edges ``|V| - 1`` times, then performs one extra pass: any
    distance still improving implies a reachable negative cycle, which is then
    extracted from the predecessor graph.
    """
    if not g.has_node(source):
        raise GraphError(f"source {source!r} not in graph")
    nodes = g.nodes()
    dist: dict[object, float] = {u: math.inf for u in nodes}
    pred: dict[object, object | None] = {u: None for u in nodes}
    dist[source] = 0.0
    edges = [(u, v, w) for u, v, w in g.edges()]
    if not g.directed:
        edges += [(v, u, w) for u, v, w in g.edges()]
    for _ in range(max(len(nodes) - 1, 0)):
        changed = False
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                pred[v] = u
                changed = True
        if not changed:
            break
    # detection pass: an edge that still relaxes after |V|-1 rounds implies a
    # negative cycle reachable from source
    offenders: list[object] = []
    for u, v, w in edges:
        if dist[u] + w < dist[v]:
            offenders.append(v)
    if not offenders:
        return BellmanFordResult(dist, pred, False, [])
    cycle = _find_negative_cycle(g, pred, offenders, nodes, edges)
    return BellmanFordResult(dist, pred, True, cycle or [])


def _edge_weight(g: Graph, u: object, v: object) -> float | None:
    w = g.weight(u, v)
    if w is None and not g.directed:
        w = g.weight(v, u)
    return w


def _cycle_weight(cycle: list[object], g: Graph) -> float:
    """Total weight of a closed node walk ``n0, n1, ..., n0``.

    Every consecutive pair ``(a, b)`` must be joined by an edge; either
    direction is accepted so undirected graphs and reversed predecessor walks
    both verify.
    """
    total = 0.0
    for a, b in zip(cycle, cycle[1:]):
        w = _edge_weight(g, a, b)
        if w is None:
            w = _edge_weight(g, b, a)
        total += w if w is not None else 0.0
    return total


def _find_negative_cycle(
    g: Graph,
    pred: dict[object, object | None],
    offenders: list[object],
    nodes: list[object],
    edges: list[tuple[object, object, float]],
) -> list[object] | None:
    """Extract an actual negative cycle, or ``None`` if not found.

    Strategy 1 — predecessor walk: follow ``pred`` from each offending node;
    a repeat in the functional predecessor graph *is* a cycle, verified by
    summing its edge weights.  Strategy 2 — bounded DFS over the reachable
    subgraph searching for any closed walk whose total weight is negative
    (provably finds one whenever detection fired).  The DFS carries a step
    budget so pathological graphs degrade to a flag-only result instead of
    hanging.
    """
    for start in offenders:
        cycle = _pred_walk_cycle(pred, start, len(nodes))
        if cycle is not None and _cycle_weight(cycle, g) < 0:
            return cycle
    return _dfs_negative_cycle(g, nodes, edges)


def _pred_walk_cycle(
    pred: dict[object, object | None], start: object, limit: int
) -> list[object] | None:
    seen: dict[object, int] = {}
    path: list[object] = []
    cur = start
    for _ in range(max(limit, 1) + 1):
        if cur in seen:
            cyc = path[seen[cur]:]
            # reverse so consecutive pairs follow edge direction a -> b
            return [cyc[0]] + list(reversed(cyc[1:])) + [cyc[0]]
        seen[cur] = len(path)
        path.append(cur)
        nxt = pred.get(cur)
        if nxt is None:
            return None
        cur = nxt
    return None


def _dfs_negative_cycle(
    g: Graph,
    nodes: list[object],
    edges: list[tuple[object, object, float]],
    step_budget: int = 200_000,
) -> list[object] | None:
    adjacency: dict[object, list[tuple[object, float]]] = {u: [] for u in nodes}
    if not g.directed:
        for u, v, w in edges:
            adjacency[u].append((v, w))
            adjacency[v].append((u, w))
    else:
        for u, v, w in edges:
            adjacency[u].append((v, w))
    steps = 0
    for root in nodes:
        if not adjacency[root]:
            continue
        path: list[object] = []
        on_path: set[object] = set()
        depth_iter: list[tuple[object, iter, float]] = []
        path_weight: dict[object, float] = {root: 0.0}
        path.append(root)
        on_path.add(root)
        depth_iter.append((root, iter(adjacency[root]), 0.0))
        while depth_iter:
            steps += 1
            if steps > step_budget:
                return None
            node, it, _ = depth_iter[-1]
            advanced = False
            for nbr, w in it:
                if nbr in on_path:
                    cycle_weight = path_weight[node] + w - path_weight[nbr]
                    if cycle_weight < 0:
                        idx = path.index(nbr)
                        cyc = path[idx:] + [nbr]
                        return cyc
                elif len(path) < len(nodes):
                    path_weight[nbr] = path_weight[node] + w
                    path.append(nbr)
                    on_path.add(nbr)
                    depth_iter.append((nbr, iter(adjacency[nbr]), 0.0))
                    advanced = True
                    break
            if advanced:
                continue
            depth_iter.pop()
            done = path.pop()
            on_path.discard(done)
            path_weight.pop(done, None)
    return None
