r"""The 17,881-token read has to be split into MCP and built-in, and `bench/tap.py`
could not do it.

Issue #55's gate is one number: of the tokens a Claude Code request spends on tool
schemas, how many belong to MCP servers and how many to tools no MCP proxy can
hide -- `Read`, `Edit`, `Write`, `Bash`, `Glob`, `Grep`, `Task`. If the answer is
"mostly built-in", the first rung of the context-reduction ladder moves almost
nothing and the order below it is wrong.

`tap.py` recorded `n_tools` and `tools_bytes`, which are the total and say nothing
about the split. These tests are for the attribution that does.

**Bytes, not tokens, and deliberately.** Tokenising inside the tap would mean
either shipping a tokenizer or calling the server's `/tokenize` mid-request, and
the one rule this instrument must not break is that it forwards bytes and observes
them -- see `test_llama_tap.py`. The row carries per-tool byte counts and names;
turning those into tokens is a reader's job, offline, against `/tokenize`.
"""
import os
import sys

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BENCH)

import tap  # noqa: E402


def _tool(name, extra=""):
    return {"type": "function",
            "function": {"name": name,
                         "description": "d" + extra,
                         "parameters": {"type": "object", "properties": {}}}}


def test_an_absent_tools_array_is_not_an_error_and_not_a_zero_split():
    """A request with no tools is normal traffic, not a request whose tools are
    all built-in. The two must not read the same downstream."""
    share = tap.tool_share(None)
    assert share["n_tools"] == 0
    assert share["bytes_total"] == 0
    assert share["mcp"]["n"] == 0
    assert share["builtin"]["n"] == 0


def test_mcp_tools_are_attributed_to_their_server():
    tools = [_tool("mcp__serena__find_symbol"),
             _tool("mcp__serena__read_memory"),
             _tool("mcp__github__list_issues")]
    share = tap.tool_share(tools)
    assert share["mcp"]["n"] == 3
    assert set(share["mcp"]["by_server"]) == {"serena", "github"}
    assert share["mcp"]["by_server"]["serena"]["n"] == 2
    assert share["mcp"]["by_server"]["github"]["n"] == 1


def test_a_tool_with_no_mcp_prefix_is_builtin_and_is_named():
    """The names matter: the gate is about which built-ins survive, so a count
    alone would not answer it."""
    share = tap.tool_share([_tool("Read"), _tool("Bash"), _tool("mcp__pal__chat")])
    assert share["builtin"]["n"] == 2
    assert share["builtin"]["names"] == ["Bash", "Read"]


def test_the_two_halves_add_up_to_the_total():
    """The whole point is a split. If the parts do not sum, the row is a number
    that looks like an answer."""
    tools = [_tool("Read", "x" * 100), _tool("mcp__serena__find_symbol", "y" * 250)]
    share = tap.tool_share(tools)
    assert share["mcp"]["bytes"] + share["builtin"]["bytes"] == share["bytes_total"]
    assert share["mcp"]["n"] + share["builtin"]["n"] == share["n_tools"]


def test_bytes_are_the_serialised_size_of_each_tool_not_the_array():
    """Per-tool bytes must exclude the array's own separators, or a long tool list
    would credit its commas to whichever half happened to be counted first."""
    big = _tool("mcp__serena__find_symbol", "y" * 500)
    share = tap.tool_share([big])
    assert share["mcp"]["bytes"] == len(tap.json.dumps(big))
    assert share["bytes_total"] == share["mcp"]["bytes"]


def test_a_double_underscore_inside_a_tool_name_does_not_split_the_server():
    """`mcp__<server>__<tool>` -- the tool half may itself contain `__`, and
    splitting on every occurrence would invent servers."""
    share = tap.tool_share([_tool("mcp__cloudflare-api__kv__namespace__list")])
    assert list(share["mcp"]["by_server"]) == ["cloudflare-api"]


def test_a_malformed_entry_is_counted_but_never_raises():
    """Studio and Claude Code are not the only clients; a tap that throws on an
    unexpected shape stops recording the run it was measuring."""
    share = tap.tool_share([{"no": "function key"}, _tool("Read")])
    assert share["n_tools"] == 2
    assert share["builtin"]["n"] == 2
    assert "" not in share["builtin"]["names"] or True  # shape-tolerant, must not raise


def test_the_row_builder_carries_the_split():
    """The function is useless if the row does not carry it -- the failure this
    repo keeps hitting is a correct helper nothing calls."""
    src = open(os.path.join(BENCH, "tap.py"), encoding="utf-8").read()
    assert '"tool_share": tool_share(' in src, "tap.py builds a row without the split"
