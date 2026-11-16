"""Command-line interface for graph-algos.

Usage::

    python cli.py info   <edge-list> [--directed]
    python cli.py dijkstra <edge-list> <source> [--directed]
    python cli.py astar  <edge-list> <source> <target> [--directed]
    python cli.py bellman <edge-list> <source> [--directed]
    python cli.py mst    <edge-list> [--prim]
    python cli.py topo   <edge-list>
    python cli.py scc    <edge-list>
    python cli.py flow   <edge-list> <source> <sink>
    python cli.py dot    <edge-list> [--directed] [-o OUT.dot]
"""

from __future__ import annotations

import argparse
import math
import sys

from flow import FlowResult, max_flow
from graph import load_edgelist, to_dot, visualize
from mst import kruskal, prim
from shortest import astar, bellman_ford, dijkstra, reconstruct_path
from traversal import tarjan_scc, topological_sort


def _format_path(path: list[object] | None, dist: float | None = None) -> str:
    if path is None:
        return "unreachable"
    text = " -> ".join(str(n) for n in path)
    if dist is not None and not math.isinf(dist):
        text += f"  (distance {dist:g})"
    return text


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="graph-algos", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    def add(name: str, *args: str, **kwargs) -> argparse.ArgumentParser:
        p = sub.add_parser(name, **kwargs)
        p.add_argument("edge_list", help="path to whitespace-separated edge list")
        for a in args:
            p.add_argument(a)
        return p

    add("info")
    add("dijkstra", "source")
    add("astar", "source", "target")
    add("bellman", "source")
    mst_p = add("mst")
    mst_p.add_argument("--prim", action="store_true", help="use Prim instead of Kruskal")
    add("topo")
    add("scc")
    add("flow", "source", "sink")
    dot_p = add("dot")
    dot_p.add_argument("-o", "--out", default=None, help="write DOT to this file")
    for name in ("info", "dijkstra", "astar", "bellman", "dot"):
        sub.choices[name].add_argument(
            "--directed", action="store_true", help="load as a directed graph"
        )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return _dispatch(args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace) -> int:
    directed = getattr(args, "directed", False)
    if args.command == "topo":
        directed = True
    if args.command == "scc":
        directed = True
    if args.command == "flow":
        directed = True

    if args.command == "dot":
        g = load_edgelist(args.edge_list, directed=directed)
        if args.out:
            visualize(g, args.out)
            print(f"wrote {args.out}")
        else:
            sys.stdout.write(to_dot(g))
        return 0

    g = load_edgelist(args.edge_list, directed=directed)

    if args.command == "info":
        kind = "directed" if g.directed else "undirected"
        print(f"{kind} graph: {g.node_count()} nodes, {g.edge_count()} edges")
        print("nodes: " + " ".join(str(u) for u in g.nodes()))
        return 0

    if args.command == "dijkstra":
        dist, pred = dijkstra(g, args.source)
        for v in g.nodes():
            if v in dist:
                path = reconstruct_path(pred, args.source, v)
                print(f"{v}: {_format_path(path, dist[v])}")
            else:
                print(f"{v}: unreachable")
        return 0

    if args.command == "astar":
        # a zero heuristic is always admissible, so the printed path is optimal
        dist, path = astar(g, args.source, args.target, lambda _u: 0.0)
        print(_format_path(path, dist))
        return 0 if path is not None else 2

    if args.command == "bellman":
        result = bellman_ford(g, args.source)
        if result.has_negative_cycle:
            cycle = " -> ".join(str(n) for n in result.negative_cycle) or "(unknown)"
            print(f"negative cycle detected: {cycle}")
            return 3
        for v in g.nodes():
            d = result.dist.get(v, math.inf)
            if math.isinf(d):
                print(f"{v}: unreachable")
            else:
                path = reconstruct_path(result.pred, args.source, v)
                print(f"{v}: {_format_path(path, d)}")
        return 0

    if args.command == "mst":
        tree = prim(g) if args.prim else kruskal(g)
        for u, v, w in tree.edges:
            print(f"{u} -- {v}  {w:g}")
        print(f"total: {tree.total:g}")
        return 0

    if args.command == "topo":
        order = topological_sort(g)
        print(" -> ".join(str(u) for u in order))
        return 0

    if args.command == "scc":
        for i, comp in enumerate(tarjan_scc(g), start=1):
            members = " ".join(str(u) for u in comp)
            print(f"scc {i}: {members}")
        return 0

    if args.command == "flow":
        result: FlowResult = max_flow(g, args.source, args.sink)
        print(f"max flow: {result.value:g}")
        print("min cut:")
        for u, v, w in result.cut:
            print(f"  {u} -> {v}  cap {w:g}")
        return 0

    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
