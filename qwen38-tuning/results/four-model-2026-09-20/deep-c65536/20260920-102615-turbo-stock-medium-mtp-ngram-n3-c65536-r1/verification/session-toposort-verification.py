import heapq


def topo_sort(graph):
    nodes = set(graph.keys())
    for targets in graph.values():
        nodes.update(targets)

    in_degree = {n: 0 for n in nodes}
    for src, targets in graph.items():
        for t in targets:
            in_degree[t] += 1

    heap = [n for n in nodes if in_degree[n] == 0]
    heapq.heapify(heap)

    order = []
    while heap:
        node = heapq.heappop(heap)
        order.append(node)
        for neighbor in graph.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                heapq.heappush(heap, neighbor)

    if len(order) != len(nodes):
        raise ValueError("cycle detected in graph")
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
