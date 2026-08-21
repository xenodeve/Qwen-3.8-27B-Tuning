"""Coding task corpus with executable verification.

Each task is a prompt plus a test module. A task counts as VERIFIED only when the
model's emitted code passes every assertion -- no partial credit, no human
judgement, no LLM-as-judge. That keeps the primary metric ("verified successful
coding tasks per unit time") measurable rather than asserted.

Tasks are chosen to discriminate quantization damage, so they lean on details a
degraded model tends to drop: eviction order, tie-breaking, boundary conditions,
cycle detection, operator precedence. Tasks that any 3-bit model solves are
useless for a Q3-vs-Q4 decision.
"""

TASKS = [
    dict(
        id="lru_cache",
        difficulty="easy",
        prompt=(
            "Write a Python class `LRUCache` with `__init__(self, capacity)`, "
            "`get(self, key)` returning the value or -1 if absent, and "
            "`put(self, key, value)`. On overflow it must evict the LEAST recently "
            "used entry. Both get and put count as a use. Output only the code."
        ),
        test="""
c = LRUCache(2)
c.put(1, 1); c.put(2, 2)
assert c.get(1) == 1
c.put(3, 3)                 # 2 is LRU -> evicted
assert c.get(2) == -1
assert c.get(3) == 3
c.put(4, 4)                 # 1 is LRU -> evicted
assert c.get(1) == -1
assert c.get(3) == 3 and c.get(4) == 4
c2 = LRUCache(1)
c2.put(5, 5); c2.put(6, 6)
assert c2.get(5) == -1 and c2.get(6) == 6
c3 = LRUCache(2)
c3.put(1, 1); c3.put(1, 10)   # update must not grow size
assert c3.get(1) == 10
""",
    ),
    dict(
        id="merge_intervals",
        difficulty="easy",
        prompt=(
            "Write a Python function `merge_intervals(intervals)` taking a list of "
            "[start, end] pairs and returning the merged, sorted list. Intervals "
            "that merely touch (e.g. [1,2] and [2,3]) must merge. Output only the code."
        ),
        test="""
assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]
assert merge_intervals([[1,4],[4,5]]) == [[1,5]]
assert merge_intervals([]) == []
assert merge_intervals([[1,4],[0,4]]) == [[0,4]]
assert merge_intervals([[1,4],[2,3]]) == [[1,4]]
assert merge_intervals([[5,6],[1,2]]) == [[1,2],[5,6]]
""",
    ),
    dict(
        id="bracket_matching",
        difficulty="easy",
        prompt=(
            "Write a Python function `is_balanced(s)` returning True iff brackets "
            "(), [], {} are balanced and correctly nested. Characters inside single "
            "or double quoted string literals must be IGNORED, and a backslash "
            "escapes the next character inside a literal. Output only the code."
        ),
        test="""
assert is_balanced("([]{})") is True
assert is_balanced("([)]") is False
assert is_balanced("(") is False
assert is_balanced("") is True
assert is_balanced("('(')") is True          # bracket inside a string literal
assert is_balanced('("]")') is True
assert is_balanced("('\\\\')") is True        # escaped quote inside literal
assert is_balanced("(']')") is True
assert is_balanced(")(") is False
""",
    ),
    dict(
        id="toposort",
        difficulty="medium",
        prompt=(
            "Write a Python function `topo_sort(graph)` where graph is a dict "
            "mapping node -> list of nodes it points to. Return a topologically "
            "sorted list. If a cycle exists, raise ValueError. When several "
            "orderings are valid, break ties by choosing the smallest available "
            "node (lexicographic). Output only the code."
        ),
        test="""
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
""",
    ),
    dict(
        id="expr_eval",
        difficulty="medium",
        prompt=(
            "Write a Python function `evaluate(expr)` that evaluates an arithmetic "
            "expression string containing non-negative integers, + - * /, and "
            "parentheses, honouring normal precedence. Division is integer division "
            "truncating toward zero. Do not use eval(). Output only the code."
        ),
        test="""
assert evaluate("1+2*3") == 7
assert evaluate("(1+2)*3") == 9
assert evaluate("10/3") == 3
assert evaluate("7-3-2") == 2
assert evaluate("2*(3+4)-5") == 9
assert evaluate("100/10/2") == 5
assert evaluate("((2))") == 2
assert evaluate("1+2*3-4/2") == 5
""",
    ),
    dict(
        id="rotated_search",
        difficulty="medium",
        prompt=(
            "Write a Python function `search_rotated(nums, target)` returning the "
            "index of target in a rotated sorted array of DISTINCT integers, or -1. "
            "It must run in O(log n). Output only the code."
        ),
        test="""
assert search_rotated([4,5,6,7,0,1,2], 0) == 4
assert search_rotated([4,5,6,7,0,1,2], 3) == -1
assert search_rotated([1], 1) == 0
assert search_rotated([], 1) == -1
assert search_rotated([3,1], 1) == 1
assert search_rotated([5,1,3], 3) == 2
assert search_rotated([1,2,3,4,5], 5) == 4
""",
    ),
    dict(
        id="lfu_cache",
        difficulty="hard",
        prompt=(
            "Write a Python class `LFUCache` with `__init__(self, capacity)`, "
            "`get(self, key)` returning the value or -1, and `put(self, key, value)`. "
            "Evict the LEAST FREQUENTLY used entry; break frequency ties by evicting "
            "the least recently used among them. get and put both increment "
            "frequency. Output only the code."
        ),
        test="""
c = LFUCache(2)
c.put(1,1); c.put(2,2)
assert c.get(1) == 1          # freq: 1->2, 2->1
c.put(3,3)                    # evicts key 2
assert c.get(2) == -1
assert c.get(3) == 3
c.put(4,4)                    # 1 and 3 both freq 2 -> evict LRU among them (1)
assert c.get(1) == -1
assert c.get(3) == 3 and c.get(4) == 4
c0 = LFUCache(0)
c0.put(1,1)
assert c0.get(1) == -1
""",
    ),
    dict(
        id="damerau",
        difficulty="hard",
        prompt=(
            "Write a Python function `damerau_levenshtein(a, b)` returning the "
            "optimal string alignment distance: insertions, deletions, "
            "substitutions each cost 1, and a transposition of two ADJACENT "
            "characters also costs 1. Output only the code."
        ),
        test="""
assert damerau_levenshtein("", "") == 0
assert damerau_levenshtein("abc", "abc") == 0
assert damerau_levenshtein("ca", "ac") == 1        # transposition
assert damerau_levenshtein("abc", "acb") == 1
assert damerau_levenshtein("kitten", "sitting") == 3
assert damerau_levenshtein("", "abc") == 3
assert damerau_levenshtein("a", "") == 1
assert damerau_levenshtein("teh", "the") == 1
""",
    ),
    dict(
        id="tree_codec",
        difficulty="hard",
        prompt=(
            "Write a Python class `Node` with attributes val, left, right (a binary "
            "tree node, constructor `Node(val, left=None, right=None)`), plus two "
            "functions `serialize(root)` returning a string and `deserialize(s)` "
            "rebuilding the tree. Round-tripping any tree must preserve structure, "
            "including None children and negative values. Output only the code."
        ),
        test="""
def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.val == b.val and same(a.left, b.left) and same(a.right, b.right)

t = Node(1, Node(-2), Node(3, Node(4), None))
assert same(deserialize(serialize(t)), t)
assert same(deserialize(serialize(None)), None)
single = Node(0)
assert same(deserialize(serialize(single)), single)
deep = Node(1, Node(2, Node(3, Node(4))))
assert same(deserialize(serialize(deep)), deep)
""",
    ),
    dict(
        id="text_wrap",
        difficulty="medium",
        prompt=(
            "Write a Python function `wrap_text(text, width)` that greedily wraps "
            "text into lines of at most `width` characters, splitting only on "
            "spaces, collapsing runs of whitespace, and returning a list of lines. "
            "A word longer than width goes on its own line unbroken. Empty or "
            "whitespace-only input returns []. Output only the code."
        ),
        test="""
assert wrap_text("the quick brown fox", 10) == ["the quick", "brown fox"]
assert wrap_text("", 5) == []
assert wrap_text("   ", 5) == []
assert wrap_text("supercalifragilistic", 5) == ["supercalifragilistic"]
assert wrap_text("a  b   c", 3) == ["a b", "c"]
assert wrap_text("aa bb cc", 5) == ["aa bb", "cc"]
assert wrap_text("one two", 100) == ["one two"]
""",
    ),
]
