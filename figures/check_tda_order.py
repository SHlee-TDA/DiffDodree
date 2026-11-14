"""Does the TDA note-graph barcode depend on the order in which nodes are recorded?

Distance as in Tran, Park & Jung (arXiv:2103.06620, eq. 2): for i < j take the path with the
fewest edges found by Dijkstra (ties broken by the search order, i.e. by node index) and add
1/w along it; for i > j use d_ji.  We relabel the nodes at random and recompute the barcode.
"""
import heapq, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
from ripser import ripser

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / "figure_data.json").read_text())
notes = [tuple(e) for ph in D["phrases"] for e in ph]
nodes = sorted(set(notes), key=lambda x: (x[0] is None, x[0] or 0, x[1]))
W = Counter()
directed = Counter()
for a, b in zip(notes[:-1], notes[1:]):
    if a != b:
        W[frozenset((a, b))] += 1
        directed[(a, b)] += 1

def fewest_edge_path(adj, s, t):
    """Dijkstra on unit weights; ties go to the smaller node index (the order the data were entered)."""
    dist = {s: 0}; prev = {}; heap = [(0, s)]; done = set()
    while heap:
        d, u = heapq.heappop(heap)
        if u in done: continue
        done.add(u)
        if u == t: break
        for v in sorted(adj[u]):
            if v not in dist or d + 1 < dist[v]:
                dist[v] = d + 1; prev[v] = u; heapq.heappush(heap, (d + 1, v))
    path = [t]
    while path[-1] != s: path.append(prev[path[-1]])
    return path[::-1]

def barcode_for_order(order):
    idx = {n: i for i, n in enumerate(order)}
    n = len(order); adj = {i: set() for i in range(n)}; w = {}
    for e, c in W.items():
        a, b = tuple(e); i, j = idx[a], idx[b]
        adj[i].add(j); adj[j].add(i); w[(i, j)] = w[(j, i)] = c
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            p = fewest_edge_path(adj, i, j)
            M[i, j] = M[j, i] = sum(1 / w[(a, b)] for a, b in zip(p[:-1], p[1:]))
    dgm = ripser(M, distance_matrix=True, maxdim=1)["dgms"][1]
    return np.array(sorted(map(tuple, np.round(dgm, 6))))

def bottleneck(a, b):
    from scipy.optimize import linear_sum_assignment
    def pts(x): return [tuple(p) for p in x]
    A, B = pts(a), pts(b); m, k = len(A), len(B)
    C = np.zeros((m + k, m + k))
    for i in range(m + k):
        for j in range(m + k):
            if i < m and j < k: C[i, j] = max(abs(A[i][0] - B[j][0]), abs(A[i][1] - B[j][1]))
            elif i < m: C[i, j] = (A[i][1] - A[i][0]) / 2
            elif j < k: C[i, j] = (B[j][1] - B[j][0]) / 2
    vals = sorted(set(C.ravel()))
    for v in vals:
        r, c = linear_sum_assignment((C > v + 1e-12).astype(float))
        if ((C > v + 1e-12)[r, c]).sum() == 0: return v
    return vals[-1]

base = barcode_for_order(nodes)
rng = np.random.default_rng(0)
examples = [base.tolist()]
counts, dists, sizes = Counter(), [], []
for k in range(200):
    order = list(nodes); rng.shuffle(order)
    bc = barcode_for_order(order)
    counts[bc.tobytes()] += 1; sizes.append(len(bc)); dists.append(bottleneck(base, bc))
    if k < 2: examples.append(bc.tolist())
pairs = sum(1 for (a, b) in directed if (b, a) not in directed)
print(f"nodes {len(nodes)}, undirected edges {len(W)}, directed pairs {len(directed)}, one-way {pairs} ({pairs/len(directed):.0%})")
print("base H1 bars:", len(base))
print("distinct barcodes over 200 orderings:", len(counts), "| bars range", min(sizes), "-", max(sizes),
      "| bottleneck to base: median", round(float(np.median(dists)), 3), "max", round(max(dists), 3),
      "| share identical to base", round(float(np.mean(np.array(dists) < 1e-9)), 3))
out = {"nodes": len(nodes), "edges": len(W), "one_way_share": pairs / len(directed), "base_bars": len(base),
       "distinct_barcodes": len(counts), "bars_min": min(sizes), "bars_max": max(sizes),
       "bottleneck_median": float(np.median(dists)), "bottleneck_max": float(max(dists)),
       "identical_share": float(np.mean(np.array(dists) < 1e-9)), "orders": 200, "examples": examples}
(HERE / "tda_order_check.json").write_text(json.dumps(out, indent=1))
