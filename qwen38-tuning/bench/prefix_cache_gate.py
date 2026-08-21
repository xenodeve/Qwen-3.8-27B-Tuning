"""Prefix-cache gate — does llama-server reuse KV across agent turns on THIS model?

Why this is a gate and not a micro-optimization: Qwen3.8 is a hybrid
recurrent/attention architecture (this machine allocates separate RS buffers).
Recent llama.cpp issues report hybrid-memory models logging
"forcing full prompt re-processing due to lack of cache data". If that happens
here, every agent turn pays a full cold prefill and prefill — not decode —
becomes the bottleneck, which reorders every remaining optimization.

Method: build an OpenCode-shaped conversation that only ever APPENDS, and read
`cache_n` and `prompt_n` from /completion timings.

    cache_n     tokens served from cache
    prompt_n    tokens actually evaluated this request

Healthy reuse:   prompt_n ~= the newly appended suffix, cache_n ~= prior length
Broken reuse:    prompt_n ~= the whole prompt every turn, cache_n ~= 0

Then deliberately perturb one dimension at a time — reorder tool schemas, edit
the system prompt, inject a skill block at the front — because Xeno injects
skills into context and OpenCode may reserialize tools between turns. Knowing
WHICH edit invalidates the cache is what makes the finding actionable.

Uses /completion (not /v1/chat/completions) because only the raw endpoint
returns the timings block with cache_n.
"""
import json, sys, time, urllib.request
from pathlib import Path

ENDPOINT = "http://127.0.0.1:8080/completion"
OUT = Path(r"C:\AI\qwen38-tuning\results\prefix-cache.jsonl")


def gen(prompt, n_predict=8, cache_prompt=True):
    body = json.dumps({
        "prompt": prompt, "n_predict": n_predict,
        "temperature": 0.0, "top_k": 1, "seed": 42,
        "cache_prompt": cache_prompt,
    }).encode()
    req = urllib.request.Request(ENDPOINT, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read().decode())
    return d["timings"], time.time() - t0


# ── Build an agent-shaped conversation ────────────────────────────────────────
# Roughly OpenCode's shape: a system block, a tool schema block, repo context,
# then an append-only run of user/assistant/tool messages.

SYSTEM = ("You are a coding agent operating inside a repository. Inspect before "
          "editing. Prefer minimal diffs. Always run the tests after a change.\n")

TOOLS = "\n".join(
    f'<tool name="{n}">{{"type":"object","properties":{{"path":{{"type":"string"}},'
    f'"content":{{"type":"string"}},"opts":{{"type":"object"}}}}}}</tool>'
    for n in ["read_file", "write_file", "run_tests", "grep", "list_dir",
              "apply_patch", "git_status", "git_diff"]
)

REPO = "\n".join(
    f"# src/module_{i:02d}.py\n"
    f"class Handler{i:02d}:\n"
    f"    def __init__(self, config):\n"
    f"        self.config = config\n"
    f"        self.cache = {{}}\n"
    f"    def process(self, item):\n"
    f"        key = item.get('id')\n"
    f"        if key in self.cache:\n"
    f"            return self.cache[key]\n"
    f"        result = self.transform(item)\n"
    f"        self.cache[key] = result\n"
    f"        return result\n"
    for i in range(40)
)

BASE = f"<system>\n{SYSTEM}</system>\n<tools>\n{TOOLS}\n</tools>\n<repo>\n{REPO}\n</repo>\n"

TURNS = [
    "<user>Find the caching bug in module_07.</user>\n<assistant>",
    "</assistant>\n<tool_result>module_07 read: cache never evicts, unbounded growth</tool_result>\n<assistant>",
    "</assistant>\n<tool_result>tests: FAILED test_memory_bound</tool_result>\n<assistant>",
    "</assistant>\n<user>Now apply the same fix to module_12.</user>\n<assistant>",
]

rows = []


def record(label, timings, wall, note=""):
    row = dict(
        label=label,
        prompt_n=timings.get("prompt_n"),
        cache_n=timings.get("cache_n"),
        prompt_ms=round(timings.get("prompt_ms", 0), 1),
        pp_tok_s=round(timings.get("prompt_per_second", 0), 1),
        wall_s=round(wall, 1),
        note=note,
    )
    rows.append(row)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(f"  {label:<28} prompt_n={row['prompt_n']:<6} cache_n={row['cache_n']:<6} "
          f"pp={row['pp_tok_s']:<7} wall={row['wall_s']}s  {note}", flush=True)
    return row


print("=== A. append-only agent turns (the healthy path) ===", flush=True)
convo = BASE
for i, turn in enumerate(TURNS, 1):
    convo += turn
    t, w = gen(convo)
    record(f"turn-{i}", t, w)
    convo += " Understood. Proceeding with the next step."

full_len = rows[-1]["prompt_n"] + rows[-1]["cache_n"]

print("\n=== B. perturbations (one dimension at a time) ===", flush=True)

# Each perturbation restarts from the SAME accumulated conversation, changing
# exactly one thing, so the drop in cache_n is attributable.
tools_reordered = "\n".join(
    f'<tool name="{n}">{{"type":"object","properties":{{"path":{{"type":"string"}},'
    f'"content":{{"type":"string"}},"opts":{{"type":"object"}}}}}}</tool>'
    for n in ["grep", "read_file", "list_dir", "write_file", "git_diff",
              "run_tests", "apply_patch", "git_status"]
)

perturbations = [
    ("reorder-tool-schemas",
     convo.replace(TOOLS, tools_reordered),
     "OpenCode may reserialize tools between turns"),
    ("edit-system-prompt",
     convo.replace("Prefer minimal diffs.", "Prefer small, surgical diffs."),
     "one sentence changed near the very front"),
    ("prepend-skill-block",
     "<skill>Follow the repository conventions in CLAUDE.md.</skill>\n" + convo,
     "Xeno injects skills ahead of everything"),
    ("append-only-control",
     convo + "\n<user>And module_19 as well.</user>\n<assistant>",
     "control: pure append, should stay cached"),
]

for name, prompt, note in perturbations:
    t, w = gen(prompt)
    record(name, t, w, note)

print("\n=== C. cache disabled (worst-case reference) ===", flush=True)
t, w = gen(convo, cache_prompt=False)
record("cache_prompt=false", t, w, "full re-prefill by request")

# ── Verdict ──────────────────────────────────────────────────────────────────
print("\n=== VERDICT ===")
appends = [r for r in rows if r["label"].startswith("turn-") and r["label"] != "turn-1"]
reused = [r for r in appends if r["cache_n"] and r["cache_n"] > 0.5 * full_len]
print(f"  conversation length at end : ~{full_len} tokens")
print(f"  append-only turns reusing cache : {len(reused)}/{len(appends)}")
if len(reused) == len(appends):
    print("  PASS — llama-server serves agent turns from cache; prefill is NOT the bottleneck.")
else:
    print("  FAIL — turns are re-prefilling. Prefill dominates agent cost; reprioritize.")

print("\n  perturbation sensitivity (cache_n retained):")
for r in rows:
    if r["label"] in [p[0] for p in perturbations]:
        pct = 100.0 * (r["cache_n"] or 0) / full_len if full_len else 0
        print(f"    {r['label']:<24} {pct:5.1f}%  {r['note']}")
print(f"\n-> {OUT}")
