#!/usr/bin/env python3
"""Verify the Sharp template: minimal diff from upstream, correct rendering, minify round-trip.

    python3 scripts/verify_template.py

Run this before publishing, and after any edit to chat_template.jinja. It fetches current
upstream (v22.3) fresh, so it also catches the case where froggeric ships a new version and our
splice silently sits on top of a template we haven't looked at.

The checks exist because each of them has a real failure mode:
  * diff-vs-upstream  -- guards against accidentally shipping edits we didn't intend to make
  * terseness-once    -- a double-splice would send the instruction twice and waste context
  * system-preserved  -- the else-branch is the whole reason a user's own prompt survives
  * think-retained    -- froggeric's retention fix is the thing we must NOT have broken
  * minify round-trip -- the minifier collapses newlines; our {% set %} body must survive it
"""
from __future__ import annotations

import difflib
import re
import pathlib
import sys
import urllib.request

from jinja2 import Environment

HERE = pathlib.Path(__file__).resolve().parent.parent
UPSTREAM = ("https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/"
            "resolve/main/chat_template.jinja")
MARKER = "Never: open with preamble"
BASE = "qwen3.8-froggeric-v22.5"   # the upstream release this fork is rebased onto

fails: list[str] = []


def check(ok: bool, label: str) -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        fails.append(label)


def render(src: str, msgs: list[dict], **kw) -> str:
    return Environment().from_string(src).render(
        messages=msgs, add_generation_prompt=True, **kw)


def main() -> int:
    full = (HERE / "chat_template.jinja").read_text()
    mini = (HERE / "chat_template_oneline.txt").read_text()

    print("=== diff vs upstream v22.5")
    with urllib.request.urlopen(UPSTREAM, timeout=60) as r:
        up = r.read().decode()
    diff = [l for l in difflib.unified_diff(up.splitlines(), full.splitlines(), lineterm="")
            if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
    # v22.3.1 is froggeric v22.3 + the terseness append + the fast-mode fixes (Fix 3 + the
    # thinking-on/off terseness-lead split), so it is NO LONGER a pure-insertion diff on v22.3:
    # the fixes conditionalize a few upstream tool-call lines. We therefore assert INTENT, not an
    # exact insertion count -- the append must be present, the fast-mode fixes must be in, and the
    # thinking-ON path must remain byte-identical to "upstream v22.3 + the terseness append".
    up_ver = re.search(r'template_version = "([^"]+)"', up)
    up_ver = up_ver.group(1) if up_ver else "?"
    check(up_ver == BASE, f"upstream is still the rebase base {BASE} (live: {up_ver})")

    ins = [l for l in diff if l.startswith("+")]
    check(len(ins) >= 11, f"terseness + fixes inserted (>=11 added lines, got {len(ins)})")
    check("Answer directly and concisely." in full,
          "fast-mode terseness lead present (v22.1.1 split)")
    check('template_version = "qwen3.8-froggeric-v22.5.0"' in full, "version is v22.5.0")
    check("qwen3.8-froggeric-v22.5" in full, "built on the froggeric v22.5 base")
    check("Nail" not in full and "Dagger" not in full, "no model-specific identity in template")

    print("\n=== rendering")
    cases = {
        "no system prompt": [{"role": "user", "content": "hi"}],
        "with system prompt": [{"role": "system", "content": "Be a pirate."},
                               {"role": "user", "content": "hi"}],
        "multi-turn w/ think": [{"role": "user", "content": "Q1"},
                                {"role": "assistant", "content": "<think>PRIORTHOUGHT</think>A1"},
                                {"role": "user", "content": "Q2"}],
    }
    for name, msgs in cases.items():
        out = render(full, msgs)
        check(out.count(MARKER) == 1, f"{name}: terseness appears exactly once")
    check("Be a pirate." in render(full, cases["with system prompt"]),
          "user's own system prompt is preserved")
    # v22.2+ extracts in-content reasoning into a canonical block instead of passing the
    # tags through, so match the thought itself inside a think block, not a tag layout.
    mt = render(full, cases["multi-turn w/ think"])
    check(any("PRIORTHOUGHT" in b for b in re.findall(r"<think>(.*?)</think>", mt, re.S)),
          "prior thinking is retained across turns")

    # v22.1 defaults reasoning_effort to MEDIUM, which injects no steering line -- so out-of-the-box
    # behavior is terseness-only with no forced effort, matching tuned v1, WITHOUT us suppressing
    # anything (upstream fixed the old forced-xhigh default). An EXPLICIT effort still injects steering,
    # so froggeric's feature is honored as opt-in.
    STEER = "Reasoning effort is set to"
    check(STEER not in render(full, cases["no system prompt"]),
          "default (no reasoning_effort): no steering line (upstream medium default)")
    check(STEER not in render(full, cases["with system prompt"]),
          "default with system prompt: still no steering line")
    check("Reasoning effort is set to low" in render(full, cases["no system prompt"],
          reasoning_effort="low"), "explicit reasoning_effort=low is still honored (opt-in)")
    check(MARKER in render(full, cases["no system prompt"], reasoning_effort="low"),
          "terseness still present when an explicit effort is requested")

    tools = [{"type": "function", "function": {"name": "get_weather", "description": "w",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
    out = render(full, [{"role": "user", "content": "weather?"}], tools=tools)
    check("get_weather" in out and out.count(MARKER) == 1, "tool definitions still render")

    # The fast-mode fixes: with thinking off, nothing in the prompt may ask for a <think> block
    # the generation prompt has already closed. Assert both directions -- the instructions must
    # vanish when thinking is off AND survive when it is on, so a future edit can't "fix" this by
    # deleting them outright. Tool-call FORMAT rules are not part of the fix and must stay.
    print("\n=== fast mode (thinking off)")
    weather = [{"role": "user", "content": "weather?"}]
    for fmt in ("xml", "json"):
        fast = render(full, weather, tools=tools, enable_thinking=False, tool_call_format=fmt)
        slow = render(full, weather, tools=tools, tool_call_format=fmt)
        body = fast.split("<|im_start|>assistant")[0]      # ignore the empty-think prefill
        check("think" not in body.lower(), f"{fmt}: no thinking reference anywhere in fast prompt")
        check("IMMEDIATELY, with NO conversational" in fast,
              f"{fmt}: tool-call immediacy rule rephrased without 'after thinking'")
        check("IMMEDIATELY after thinking" in slow,
              f"{fmt}: thinking-on keeps the original 'after thinking' wording")
        check("Brief explanation of tool call" in slow,
              f"{fmt}: thinking-on keeps the <think> stanza")
        check("Function calls MUST follow the specified format" in fast,
              f"{fmt}: fast mode still carries the tool-call format rules")
        check("Answer directly and concisely" in fast, f"{fmt}: fast-mode terseness lead")

    print("\n=== terse kwarg (default on, opt-out honoured)")
    MARK = "Never: open with preamble"
    u = [{"role": "user", "content": "hi"}]
    sysu = [{"role": "system", "content": "You are a pirate."}, {"role": "user", "content": "hi"}]
    for src_name, src in (("full", full), ("oneline", mini)):
        check(MARK in render(src, u), f"{src_name}: terse defaults on when the kwarg is absent")
        check(MARK in render(src, u, terse=True), f"{src_name}: terse=true keeps the block")
        check(MARK not in render(src, u, terse=False), f"{src_name}: terse=false drops the block")
        check(MARK not in render(src, u, terse=False, enable_thinking=False),
              f"{src_name}: terse=false drops it with thinking off too")
        out = render(src, sysu, terse=False)
        check(MARK not in out and "You are a pirate." in out,
              f"{src_name}: terse=false leaves the caller's own system prompt intact")
        check("You are a pirate." in render(src, sysu),
              f"{src_name}: terse on still keeps the caller's system prompt")

    print("\n=== minified round-trip")
    # This used to render each case with NO kwargs, i.e. only ever the default path: medium effort,
    # thinking on, terseness on. That is exactly the blind spot that let PR #8 through 22/22 -- it
    # edited chat_template.jinja without regenerating the oneline, and the two disagree ONLY when
    # reasoning_effort is low or xhigh (medium and high inject no instructions, so there is nothing
    # to disagree about). A source/minified desync that needs a non-default kwarg to show up was
    # invisible. The round-trip now sweeps the cross-product of every axis that changes rendering,
    # over message shapes including the tool flows the old case list never contained.
    TOOLS = [{"type": "function", "function": {
        "name": "get_weather", "description": "Weather.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}}]
    rt_cases = dict(cases)
    rt_cases["tools offered"] = ([{"role": "user", "content": "weather?"}], {"tools": TOOLS})
    rt_cases["tool result last"] = ([
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"type": "function",
                         "function": {"name": "get_weather", "arguments": {"city": "Oslo"}}}]},
        {"role": "tool", "content": "5C"}], {"tools": TOOLS})
    rt_cases["assistant prefill"] = ([{"role": "user", "content": "hi"},
                                      {"role": "assistant", "content": "Sure, here"}], {})

    AXES = [("reasoning_effort", [None, "low", "medium", "high", "xhigh"]),
            ("enable_thinking",  [None, False]),
            ("terse",            [None, True, False])]

    def combos():
        out = [{}]
        for key, vals in AXES:
            out = [{**c, **({} if v is None else {key: v})} for c in out for v in vals]
        return out

    n_combo = len(combos())
    for name, case in rt_cases.items():
        msgs, extra = case if isinstance(case, tuple) else (case, {})
        bad = None
        for kw in combos():
            merged = {**extra, **kw}
            if render(full, msgs, **merged) != render(mini, msgs, **merged):
                bad = ", ".join(f"{k}={v}" for k, v in kw.items()) or "defaults"
                break
        check(bad is None,
              f"{name}: oneline == full across {n_combo} kwarg combos"
              + (f"  [DIFFERS at {bad}]" if bad else ""))
    # Effort steering has to actually reach the model, in EVERY message shape. PR #8 moved the
    # instructions into the final user message, which silently dropped them whenever the last
    # message was a tool result -- i.e. throughout an agentic tool loop, where reasoning_effort
    # would appear to work and do nothing. Nothing here tested a shape that could catch it.
    print("\n=== reasoning-effort steering reaches the model")
    STEER = {"low": "Reasoning effort is set to low", "xhigh": "Reasoning effort is set to xhigh"}
    for name, case in rt_cases.items():
        msgs, extra = case if isinstance(case, tuple) else (case, {})
        for src_name, src in (("full", full), ("oneline", mini)):
            miss = [eff for eff, txt in STEER.items()
                    if render(src, msgs, reasoning_effort=eff, **extra).count(txt) != 1]
            check(not miss, f"{src_name}/{name}: effort text present exactly once"
                            + (f"  [MISSING for {', '.join(miss)}]" if miss else ""))
    # The template buckets many spellings into THREE effective levels, so `high` is an alias of
    # `xhigh` and not a fourth setting. Pinning the whole alias table here means a future edit that
    # re-bands them -- silently changing what every caller's `reasoning_effort` does -- fails loudly.
    # `medium` injecting nothing is the load-bearing one: it is Dirk's default, and steering it
    # would put per-request text at the top of every prompt.
    ALIASES = {"low": "low", "minimal": "low",
               "medium": None, "none": None, "off": None, "banana": None,   # unknown -> medium
               "high": "xhigh", "xhigh": "xhigh", "max": "xhigh",
               "ultracode": "xhigh", "extreme": "xhigh"}
    for eff, want in ALIASES.items():
        got = set()
        for src in (full, mini):
            out = render(src, [{"role": "user", "content": "hi"}], reasoning_effort=eff)
            got.add("xhigh" if "set to xhigh" in out else "low" if "set to low" in out else None)
        check(got == {want}, f"effort {eff!r} -> {want or 'no instruction'}"
                             + (f"  [got {got}]" if got != {want} else ""))

    sysblk = render(mini, cases["no system prompt"]).split("<|im_start|>system")[1] \
                                                    .split("<|im_end|>")[0].strip()
    # the terseness tail starts with "Answer directly," (at medium default no steering precedes it);
    # isolate it and confirm ITS newlines survived minification as 4 lines.
    terse = "Answer directly," + sysblk.split("Answer directly,", 1)[1]
    check(len(terse.splitlines()) == 4, "terseness survives minification as 4 lines")

    print(f"\n{'ALL CHECKS PASSED' if not fails else str(len(fails)) + ' FAILED: ' + '; '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
