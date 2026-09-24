import heapq

def topo_sort(graph):
    indegree = {node: 0 for node in graph}
    for u, neighbors in graph.items():
        for v in neighbors:
            if v not in indegree:
                indegree[v] = 0
            indegree[v] += 1

    heap = [node for node, deg in indegree.items() if deg == 0]
    heapq.heapify(heap)

    result = []
    while heap:
        node = heapq.heappop(heap)
        result.append(node)
        for neighbor in graph.get(node, []):
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                heapq.heappush(heap, neighbor)

    if len(result) != len(indegree):
        raise ValueError("Cycle detected")
    return result


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
