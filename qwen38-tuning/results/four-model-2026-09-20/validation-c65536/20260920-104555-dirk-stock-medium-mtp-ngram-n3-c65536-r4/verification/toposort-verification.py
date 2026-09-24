import heapq

def topo_sort(graph):
    nodes = set(graph.keys())
    for adj in graph.values():
        nodes.update(adj)

    indegree = {node: 0 for node in nodes}
    for u, adj in graph.items():
        for v in adj:
            indegree[v] += 1

    heap = [node for node, deg in indegree.items() if deg == 0]
    heapq.heapify(heap)

    order = []
    while heap:
        u = heapq.heappop(heap)
        order.append(u)

        for v in graph.get(u, []):
            indegree[v] -= 1
            if indegree[v] == 0:
                heapq.heappush(heap, v)

    if len(order) != len(indegree):
        raise ValueError("Graph contains a cycle")

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
