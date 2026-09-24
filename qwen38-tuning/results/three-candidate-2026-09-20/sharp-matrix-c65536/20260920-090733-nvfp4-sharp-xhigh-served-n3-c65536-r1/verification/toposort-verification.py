import heapq

def topo_sort(graph):
    adj = {u: (list(v) if v is not None else []) for u, v in graph.items()}

    nodes = set(adj)
    for edges in adj.values():
        nodes.update(edges)

    indegree = {node: 0 for node in nodes}
    for edges in adj.values():
        for v in edges:
            indegree[v] += 1

    available = [node for node, degree in indegree.items() if degree == 0]
    heapq.heapify(available)

    order = []
    while available:
        u = heapq.heappop(available)
        order.append(u)
        for v in adj.get(u, []):
            indegree[v] -= 1
            if indegree[v] == 0:
                heapq.heappush(available, v)

    if len(order) != len(nodes):
        raise ValueError("Cycle detected")

    return order


# ---- verification ----

assert topo_sort({'a': ['b'], 'b': ['c'], 'c': []}) == ['a','b','c']
assert topo_sort({'b': [], 'a': []}) == ['a','b']          # tie-break
assert topo_sort({'a': ['c'], 'b': ['c'], 'c': []}) == ['a','b','c']
try:
    topo_sort({'a': ['b'], 'b': ['a']})
    raise AssertionError("cycle not detected")
except ValueError:
    pass
assert topo_sort({}) == []
r = topo_sort({'x': ['y','z'], 'y': ['z'], 'z': []})
assert r == ['x','y','z']
