import heapq


def topo_sort(graph):
    indegree = {node: 0 for node in graph}
    adjacency = {node: list(neighbors) for node, neighbors in graph.items()}

    for neighbors in adjacency.values():
        for node in neighbors:
            if node not in indegree:
                indegree[node] = 0
            indegree[node] += 1

    heap = [node for node, degree in indegree.items() if degree == 0]
    heapq.heapify(heap)

    order = []
    while heap:
        current = heapq.heappop(heap)
        order.append(current)

        for nxt in adjacency.get(current, []):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                heapq.heappush(heap, nxt)

    if len(order) != len(indegree):
        raise ValueError("graph contains a cycle")

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
