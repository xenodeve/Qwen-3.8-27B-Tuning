import heapq
from collections import defaultdict


def topo_sort(graph):
    nodes = set(graph)
    for neighbors in graph.values():
        nodes.update(neighbors)

    indegree = defaultdict(int)
    adjacency = defaultdict(list)

    for node, neighbors in graph.items():
        for neighbor in neighbors:
            adjacency[node].append(neighbor)
            indegree[neighbor] += 1

    heap = [node for node in nodes if indegree[node] == 0]
    heapq.heapify(heap)

    result = []
    while heap:
        node = heapq.heappop(heap)
        result.append(node)

        for neighbor in adjacency[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                heapq.heappush(heap, neighbor)

    if len(result) != len(nodes):
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
