"""Maximum flow (Edmonds-Karp) with minimum-cut extraction."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from graph import Graph, GraphError


@dataclass
class FlowResult:
    """Outcome of a max-flow computation.

    ``value`` is the maximum flow.  ``flow`` maps ``(u, v)`` to the net flow
    on that arc (only non-zero arcs listed).  ``cut`` lists the original arcs
    ``(u, v, capacity)`` crossing from the source side of a minimum cut to the
    sink side, and ``source_side`` is the set of nodes reachable from the
    source in the residual graph.
    """

    value: float
    flow: dict[tuple[object, object], float] = field(default_factory=dict)
    cut: list[tuple[object, object, float]] = field(default_factory=list)
    source_side: set[object] = field(default_factory=set)


def max_flow(g: Graph, source: object, sink: object) -> FlowResult:
    """Edmonds-Karp max-flow (BFS augmenting paths, O(V * E^2)).

    Requires a directed graph; capacity of arc ``u -> v`` is the edge weight
    (negative or zero capacities are rejected).  Returns the flow value, the
    per-arc flows, and a minimum cut whose capacity provably equals the flow
    value.
    """
    if not g.directed:
        raise GraphError("max_flow requires a directed graph (capacity network)")
    if source == sink:
        raise GraphError("source and sink must differ")
    if not g.has_node(source):
        raise GraphError(f"source {source!r} not in graph")
    if not g.has_node(sink):
        raise GraphError(f"sink {sink!r} not in graph")

    # net capacity per arc, summed when the edge list repeats an arc
    cap: dict[tuple[object, object], float] = {}
    for u, v, w in g.edges():
        if w < 0:
            raise GraphError(f"negative capacity on arc {u!r} -> {v!r}")
        if u == v:
            continue  # self-loops cannot carry useful flow
        cap[(u, v)] = cap.get((u, v), 0.0) + w

    residual: dict[tuple[object, object], float] = dict(cap)
    flow: dict[tuple[object, object], float] = {}
    adj: dict[object, list[object]] = {u: [] for u in g.nodes()}
    for u, v in cap:
        if v not in adj[u]:
            adj[u].append(v)
        if u not in adj[v]:
            adj[v].append(u)

    def bfs_parent() -> dict[object, object | None]:
        """BFS in the residual graph; returns a parent map to the sink."""
        parent: dict[object, object | None] = {source: None}
        queue: deque[object] = deque([source])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v in parent:
                    continue
                if residual.get((u, v), 0.0) > 1e-12:
                    parent[v] = u
                    if v == sink:
                        return parent
                    queue.append(v)
        return parent

    total = 0.0
    while True:
        parent = bfs_parent()
        if sink not in parent:
            break
        # bottleneck along the discovered path
        bottleneck: float | None = None
        cur = sink
        while cur != source:
            prev = parent[cur]
            r = residual[(prev, cur)]
            bottleneck = r if bottleneck is None else min(bottleneck, r)
            cur = prev
        assert bottleneck is not None  # path exists, so a bottleneck does too
        total += bottleneck
        cur = sink
        while cur != source:
            prev = parent[cur]
            residual[(prev, cur)] -= bottleneck
            residual[(cur, prev)] = residual.get((cur, prev), 0.0) + bottleneck
            cur = prev

    # net flow per original arc: capacity - residual (never negative, never
    # above capacity — backward pushes only cancel prior forward pushes)
    flow = {
        key: cap[key] - residual.get(key, 0.0)
        for key in cap
        if abs(cap[key] - residual.get(key, 0.0)) > 1e-12
    }

    # min cut: nodes still reachable from source in the residual graph
    reachable: set[object] = {source}
    queue: deque[object] = deque([source])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v not in reachable and residual.get((u, v), 0.0) > 1e-12:
                reachable.add(v)
                queue.append(v)
    cut = [
        (u, v, w)
        for u, v, w in g.edges()
        if u in reachable and v not in reachable
    ]
    return FlowResult(total, flow, cut, reachable)
