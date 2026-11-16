"""Topological ordering (Kahn) and strongly connected components (Tarjan)."""

from __future__ import annotations

from graph import Graph, GraphError


class CycleError(ValueError):
    """Raised by :func:`topological_sort` when the digraph has a cycle.

    ``cycle`` holds the offending node sequence with the first node repeated
    at the end (e.g. ``["a", "b", "a"]``).
    """

    def __init__(self, cycle: list[object]) -> None:
        self.cycle = cycle
        super().__init__(f"graph contains a cycle: {' -> '.join(map(str, cycle))}")


def topological_sort(g: Graph) -> list[object]:
    """Kahn's algorithm: BFS over zero-indegree nodes.

    Returns an ordering where every edge ``u -> v`` places ``u`` before ``v``.
    When the graph is not a DAG, raises :class:`CycleError` whose ``cycle``
    attribute names a concrete cycle found in the leftover subgraph.
    """
    if not g.directed:
        raise GraphError("topological sort requires a directed graph")
    indegree: dict[object, int] = {u: 0 for u in g.nodes()}
    for u in g.nodes():
        for v, _ in g.neighbors(u):
            indegree[v] = indegree.get(v, 0) + 1
    queue = [u for u in g.nodes() if indegree[u] == 0]
    order: list[object] = []
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        order.append(u)
        for v, _ in g.neighbors(u):
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)
    if len(order) != len(indegree):
        chosen = set(order)
        remaining = [u for u in g.nodes() if u not in chosen]
        raise CycleError(_find_cycle(g, remaining))
    return order


def _find_cycle(g: Graph, remaining: list[object]) -> list[object]:
    """Locate one concrete cycle inside the subgraph induced by ``remaining``.

    Kahn's leftover nodes each sit on or downstream of a cycle; an iterative
    DFS tracking the current path finds a back edge and reports the loop, so
    arbitrarily long cycles never touch Python's recursion limit.
    """
    allowed = set(remaining)
    color: dict[object, int] = {}  # 0 = visiting, 1 = done
    for start in remaining:
        if color.get(start) is not None:
            continue
        path: list[object] = []
        on_path: set[object] = set()
        call_stack: list[tuple[object, iter]] = [(start, iter(g.neighbors(start)))]
        color[start] = 0
        path.append(start)
        on_path.add(start)
        while call_stack:
            u, it = call_stack[-1]
            advanced = False
            for v, _ in it:
                if v not in allowed:
                    continue
                if color.get(v) is None:
                    color[v] = 0
                    path.append(v)
                    on_path.add(v)
                    call_stack.append((v, iter(g.neighbors(v))))
                    advanced = True
                    break
                if v in on_path:
                    idx = path.index(v)
                    return path[idx:] + [v]
            if advanced:
                continue
            call_stack.pop()
            path.pop()
            on_path.discard(u)
            color[u] = 1
    # unreachable when Kahn failed, but never raise without a payload
    first = remaining[0] if remaining else "?"
    return [first, first]


def tarjan_scc(g: Graph) -> list[list[object]]:
    """Tarjan's strongly connected components, iterative (no recursion limit).

    Returns components as lists of nodes.  Discovery order follows node
    insertion order and each component's members appear in stack pop order,
    making the output fully deterministic.  Works for directed and undirected
    graphs (an undirected graph yields its connected components).
    """
    if not g.directed:
        raise GraphError("tarjan_scc requires a directed graph")
    index: dict[object, int] = {}
    low: dict[object, int] = {}
    on_stack: set[object] = set()
    stack: list[object] = []
    components: list[list[object]] = []
    counter = 0

    for root in g.nodes():
        if root in index:
            continue
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        call_stack: list[tuple[object, iter]] = [(root, iter(g.neighbors(root)))]
        while call_stack:
            node, it = call_stack[-1]
            advanced = False
            for v, _ in it:
                if v not in index:
                    index[v] = low[v] = counter
                    counter += 1
                    stack.append(v)
                    on_stack.add(v)
                    call_stack.append((v, iter(g.neighbors(v))))
                    advanced = True
                    break
                if v in on_stack:
                    low[node] = min(low[node], index[v])
            if advanced:
                continue
            call_stack.pop()
            if call_stack:
                parent = call_stack[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                component: list[object] = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    component.append(w)
                    if w == node:
                        break
                components.append(component)
    return components
