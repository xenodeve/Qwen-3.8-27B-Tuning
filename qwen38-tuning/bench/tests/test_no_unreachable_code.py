"""The incident: `warm = [rate(t)]` sat AFTER `return` inside `rate()`.

It never ran, so `warm` was never bound, and `run()` raised UnboundLocalError at
the end of a measurement that had already cost minutes of GPU time. Python
compiles it happily, so `python -m compileall` -- the repo typecheck gate -- was
green the whole time. That gate was written the same day and did not catch the
defect it was written for.

This is the cheapest guard that would have: no statement may follow a `return`,
`raise`, `break` or `continue` in the same block, anywhere in the measurement
modules. It costs no GPU and it is not fooled by a syntactically valid file.
"""
import ast
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).parent.parent
TERMINAL = (ast.Return, ast.Raise, ast.Break, ast.Continue)
MODULES = sorted(p for p in BENCH.glob("*.py"))


def unreachable_statements(source, filename):
    """Every statement that can never run because a terminator precedes it."""
    found = []
    for node in ast.walk(ast.parse(source, filename)):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if not isinstance(block, list):
                continue
            for i, stmt in enumerate(block[:-1]):
                if isinstance(stmt, TERMINAL):
                    nxt = block[i + 1]
                    found.append((nxt.lineno, type(nxt).__name__,
                                  type(stmt).__name__))
    return found


@pytest.mark.parametrize("module", MODULES, ids=lambda p: p.name)
def test_no_statement_follows_a_terminator(module):
    hits = unreachable_statements(module.read_text(encoding="utf-8"), str(module))
    assert not hits, (
        f"{module.name} has unreachable code: "
        + "; ".join(f"line {ln}: {what} after {term}" for ln, what, term in hits)
    )


def test_the_guard_actually_catches_the_original_defect():
    """Mutate on purpose. A detector nobody has seen find anything has not been
    shown to find anything -- this pins the guard to the real incident."""
    defect = (
        "def rate(t):" + chr(10)
        + "    return t" + chr(10)
        + "    warm = [rate(t)]" + chr(10)
    )
    hits = unreachable_statements(defect, "<the 2026-08-21 incident>")
    assert len(hits) == 1
    assert hits[0][1] == "Assign" and hits[0][2] == "Return"
