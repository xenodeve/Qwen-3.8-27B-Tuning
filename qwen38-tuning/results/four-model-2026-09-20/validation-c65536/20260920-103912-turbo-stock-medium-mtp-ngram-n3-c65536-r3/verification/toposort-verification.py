import heapq

def topo_sort(graph):
    nodes = set(graph.keys())
    for neighbors in graph.values():
        nodes.update(neighbors)

    in_degree = {n: 0 for n in nodes}
    for n in graph:
        for neighbor in graph[n]:
            in_degree[neighbor] += 1

    heap = [n for n in nodes if in_degree[n] == 0]
    heapq.heapify(heap)

    result = []
    while heap:
        n = heapq.heappop(heap)
        result.append(n)
        for neighbor in graph.get(n, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                heapq.heappush(heap, neighbor)

    if len(result) != len(nodes):
        raise ValueError("Graph contains a cycle")

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
