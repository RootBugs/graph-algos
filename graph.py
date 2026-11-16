"""Graph representation, edge-list loading, and DOT export.

The :class:`Graph` is an adjacency-list structure that also knows how to
materialize an adjacency matrix, so both representations required by the
manifest live behind one API.  Edges carry weights (default ``1.0``), may be
negative (Bellman-Ford needs them), and the structure supports directed and
undirected weighted graphs.
"""

from __future__ import annotations

import math


class GraphError(ValueError):
    """Raised for invalid graph operations (bad endpoints, bad files, ...)."""


class Graph:
    """Directed or undirected weighted graph over hashable nodes.

    Adjacency is stored as ``{node: [(neighbor, weight), ...]}`` preserving
    insertion order of neighbors, which keeps every algorithm deterministic.
    An edge list is kept separately so undirected edges are enumerated once.
    """

    def __init__(self, directed: bool = False) -> None:
        self.directed = directed
        self._adj: dict[object, list[tuple[object, float]]] = {}
        self._edges: list[tuple[object, object, float]] = []

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def add_node(self, u: object) -> object:
        """Add ``u`` if new and return it."""
        if u not in self._adj:
            self._adj[u] = []
        return u

    def add_edge(self, u: object, v: object, weight: float = 1.0) -> None:
        """Add edge ``u -> v`` with ``weight`` (both directions if undirected)."""
        self.add_node(u)
        self.add_node(v)
        self._adj[u].append((v, weight))
        if not self.directed:
            self._adj[v].append((u, weight))
        self._edges.append((u, v, weight))

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------
    def nodes(self) -> list[object]:
        """Nodes in insertion order."""
        return list(self._adj)

    def neighbors(self, u: object) -> list[tuple[object, float]]:
        """``[(neighbor, weight), ...]`` for ``u`` (empty if unknown)."""
        return list(self._adj.get(u, ()))

    def has_node(self, u: object) -> bool:
        return u in self._adj

    def edges(self) -> list[tuple[object, object, float]]:
        """Edge list as inserted; undirected edges appear once."""
        return list(self._edges)

    def weight(self, u: object, v: object) -> float | None:
        """Weight of edge ``u -> v`` or ``None`` if absent."""
        for nbr, w in self._adj.get(u, ()):
            if nbr == v:
                return w
        return None

    def degree(self, u: object) -> int:
        return len(self._adj.get(u, ()))

    def node_count(self) -> int:
        return len(self._adj)

    def edge_count(self) -> int:
        return len(self._edges)

    # ------------------------------------------------------------------
    # adjacency matrix representation
    # ------------------------------------------------------------------
    def to_matrix(self) -> tuple[list[object], list[list[float | None]]]:
        """Materialize ``(nodes, matrix)`` where ``matrix[i][j]`` is the weight
        of ``nodes[i] -> nodes[j]`` or ``None`` when no edge exists.

        For undirected graphs the matrix is symmetric.  Duplicate edges keep
        the last weight, matching :meth:`weight` semantics only if the caller
        prefers the first — documented policy: **last write wins**.
        """
        nodes = self.nodes()
        index = {u: i for i, u in enumerate(nodes)}
        n = len(nodes)
        matrix: list[list[float | None]] = [[None] * n for _ in range(n)]
        for u in nodes:
            matrix[index[u]][index[u]] = None
        for u, v, w in self._edges:
            matrix[index[u]][index[v]] = w
            if not self.directed:
                matrix[index[v]][index[u]] = w
        return nodes, matrix

    @classmethod
    def from_matrix(
        cls,
        matrix: list[list[float | None]],
        directed: bool = False,
        nodes: list[object] | None = None,
    ) -> "Graph":
        """Build a graph from an adjacency matrix (``None`` = no edge).

        Node labels default to ``0..n-1`` or the supplied ``nodes`` list.
        """
        n = len(matrix)
        if nodes is None:
            nodes = list(range(n))
        if len(nodes) != n:
            raise GraphError("nodes list length must match matrix size")
        g = cls(directed=directed)
        for u in nodes:
            g.add_node(u)
        for i in range(n):
            row = matrix[i]
            if len(row) != n:
                raise GraphError("adjacency matrix must be square")
            for j in range(n):
                w = row[j]
                if w is None:
                    continue
                if not directed and i == j:
                    raise GraphError("undirected matrices cannot carry self-loops")
                g.add_edge(nodes[i], nodes[j], float(w))
        return g

    def is_undirected(self) -> bool:
        return not self.directed


# ----------------------------------------------------------------------
# edge-list loader
# ----------------------------------------------------------------------
def load_edgelist(path: str, directed: bool = False) -> Graph:
    """Load a whitespace-separated edge list.

    File format::

        # comments and blank lines are ignored
        u v            # unweighted edge (weight 1.0)
        u v 2.5        # weighted edge

    Nodes are strings.  Malformed lines raise :class:`GraphError` naming the
    line number.
    """
    g = Graph(directed=directed)
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) not in (2, 3):
                raise GraphError(
                    f"{path}:{lineno}: expected 'u v' or 'u v weight', got {line!r}"
                )
            u, v = parts[0], parts[1]
            if len(parts) == 3:
                try:
                    w = float(parts[2])
                except ValueError:
                    raise GraphError(
                        f"{path}:{lineno}: bad weight {parts[2]!r}"
                    ) from None
            else:
                w = 1.0
            g.add_edge(u, v, w)
    return g


# ----------------------------------------------------------------------
# DOT export
# ----------------------------------------------------------------------
def to_dot(g: Graph, name: str = "G") -> str:
    """Export the graph in Graphviz DOT syntax.

    Directed graphs use ``digraph`` with ``->`` edges; undirected graphs use
    ``graph`` with ``--`` edges.  Weights become ``label`` attributes and node
    identifiers are quoted.
    """
    keyword = "digraph" if g.directed else "graph"
    arrow = "->" if g.directed else "--"
    lines = [f'{keyword} "{_dot_escape(name)}" {{']
    for u in g.nodes():
        lines.append(f'  "{_dot_escape(str(u))}";')
    for u, v, w in g.edges():
        if w == 1.0:
            lines.append(f'  "{_dot_escape(str(u))}" {arrow} "{_dot_escape(str(v))}";')
        else:
            label = _dot_number(w)
            lines.append(
                f'  "{_dot_escape(str(u))}" {arrow} "{_dot_escape(str(v))}" '
                f'[label="{label}"];'
            )
    lines.append("}")
    return "\n".join(lines) + "\n"


def visualize(g: Graph, path: str, name: str = "G") -> str:
    """Write the DOT rendering of ``g`` to ``path`` and return the text."""
    text = to_dot(g, name=name)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return text


def _dot_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _dot_number(w: float) -> str:
    if w == math.trunc(w) and abs(w) < 1e15:
        return str(int(w))
    return repr(w)
