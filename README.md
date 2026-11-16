# graph-algos — Graph Algorithm Library

The classics, one import away. `graph-algos` implements the standard
graph-algorithm toolkit **from scratch on top of one `Graph` type**: adjacency
list *and* adjacency matrix representations, Dijkstra, A* with a pluggable
heuristic, Bellman-Ford with negative-cycle detection, Kruskal and Prim
minimum spanning trees, Kahn topological sort with a concrete cycle report,
iterative Tarjan SCC, and Edmonds-Karp max-flow with minimum-cut extraction —
plus an edge-list loader and Graphviz DOT export.

## Why it's special

- **Both representations are first-class.** `Graph` stores an adjacency list
  for algorithms and can materialize (`to_matrix`) or be built from
  (`from_matrix`) an adjacency matrix without a second data type.
- **Every algorithm is checked against a hand-computed answer** — the MST tests
  use the CLRS Figure 23.4 graph (weight exactly 37), the flow tests assert a
  max-flow/min-cut pair of 16, and the shortest-path tests trace exact paths —
  and then again by randomized cross-checks: Dijkstra ≡ Bellman-Ford,
  A*(h=0) ≡ Dijkstra, Kruskal ≡ Prim, and Edmonds-Karp ≡ brute-force cut
  enumeration on seeded random graphs.
- **Failures are explicit.** Negative weights to Dijkstra, undirected graphs
  to max-flow, disconnected graphs to MST — each raises a precise
  `GraphError`; Kahn raises `CycleError` carrying an actual cycle
  (`a -> b -> c -> a`), not just a boolean.
- **Min-cut comes with the flow.** `max_flow` returns the flow value, per-arc
  flows, the reachable set of the residual graph, and the crossing arcs whose
  capacity provably equals the flow.
- **No recursion anywhere it matters.** Tarjan and the cycle-extraction DFS
  are iterative, so a 5 000-node cycle is an ordinary test case.

## Architecture

| Module | Responsibility |
|---|---|
| `graph.py` | `Graph` (adjacency list + matrix materialization), edge-list loader, DOT `to_dot` / `visualize` |
| `shortest.py` | `dijkstra`, `astar` (pluggable heuristic), `bellman_ford` with negative-cycle extraction, path reconstruction |
| `mst.py` | `kruskal` (internal union-find) and `prim`, both returning `SpanningTree(total, edges)` |
| `traversal.py` | `topological_sort` (Kahn) with `CycleError` cycle report, iterative `tarjan_scc` |
| `flow.py` | `max_flow` (Edmonds-Karp) returning `FlowResult(value, flow, cut, source_side)` |
| `cli.py` | `info`, `dijkstra`, `astar`, `bellman`, `mst`, `topo`, `scc`, `flow`, `dot` subcommands |

## Runnable example

```console
$ python -c "
from graph import Graph
from shortest import dijkstra, astar, bellman_ford, reconstruct_path
from mst import kruskal
from flow import max_flow

g = Graph(directed=True)
for u, v, w in [('A','B',4),('A','C',2),('C','B',1),('B','D',5),
                ('C','D',8),('C','E',10),('D','E',2)]:
    g.add_edge(u, v, w)
dist, pred = dijkstra(g, 'A')
print('dijkstra E:', reconstruct_path(pred, 'A', 'E'), dist['E'])
print('astar     E:', astar(g, 'A', 'E', lambda n: 0.0))

net = Graph(directed=True)
for u, v, w in [('s','a',10),('s','b',10),('a','c',8),
                ('b','c',4),('b','t',10),('c','t',6)]:
    net.add_edge(u, v, w)
r = max_flow(net, 's', 't')
print('max-flow:', r.value, 'cut cap:', sum(w for _,_,w in r.cut))

bf = Graph(directed=True)
for u, v, w in [('s','x',1),('x','y',-2),('y','x',-1)]:
    bf.add_edge(u, v, w)
res = bellman_ford(bf, 's')
print('bellman neg cycle:', res.has_negative_cycle, res.negative_cycle)
"
dijkstra E: ['A', 'C', 'B', 'D', 'E'] 10.0
astar     E: (10.0, ['A', 'C', 'B', 'D', 'E'])
max-flow: 16.0 cut cap: 16
bellman neg cycle: True ['y', 'x', 'y']
```

The CLI works over any edge-list file (`u v [weight]`, `#` comments):

```console
$ printf "A B 4\nA C 2\nC B 1\nB D 5\nC D 8\nC E 10\nD E 2\n" > edges.txt
$ python cli.py dijkstra edges.txt A --directed
A: A  (distance 0)
B: A -> C -> B  (distance 3)
C: A -> C  (distance 2)
D: A -> C -> B -> D  (distance 8)
E: A -> C -> B -> D -> E  (distance 10)
$ python cli.py dot edges.txt --directed -o graph.dot
wrote graph.dot
```

## Edge-list format

```
# comment lines and blank lines are ignored
u v            # unweighted edge (weight 1.0)
u v 2.5        # weighted edge
```

`--directed` (CLI) / `Graph(directed=True)` (library) switch directionality;
`topo`, `scc`, and `flow` always load directed graphs because they require them.

## Test command

```
python -m unittest discover -s tests -v
```

Run from the project root. 77 tests cover hand-computed graphs, error paths,
and seeded randomized cross-checks between algorithms.

## License

MIT — see `LICENSE`.
