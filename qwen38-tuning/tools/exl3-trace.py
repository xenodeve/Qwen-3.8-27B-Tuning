"""Render what the EXL3 server streamed to Claude Code, event by event, with
timing -- both sides of the "thinking shows for a second, then the counter
jumps" question (2026-09-04).

Server side: start the server with EXL3_TRACE_SSE=C:\\AI\\qwen38-tuning\\logs\\exl3-sse.jsonl
(anthropic_routes writes one line per outgoing SSE event). Then:

    python qwen38-tuning\\tools\\exl3-trace.py                # last request
    python qwen38-tuning\\tools\\exl3-trace.py --all          # every request in the file
    python qwen38-tuning\\tools\\exl3-trace.py --gaps 2       # flag silences longer than 2 s

Claude Code side: it keeps no event log for interactive sessions; the only
observable stream is `claude -p ... --output-format stream-json
--include-partial-messages`, and `--claude-p "<prompt>"` runs exactly that
against the server and prints the same kind of timeline from Claude Code's own
output, so the two can be laid side by side.
"""
import argparse, json, os, subprocess, sys, time, shutil

DEFAULT = r"C:\AI\qwen38-tuning\logs\exl3-sse.jsonl"


def load(path):
    rows = []
    with open(path, encoding = "utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    by = {}
    for r in rows:
        by.setdefault(r["rid"], []).append(r)
    return by


def render(rows, gaps):
    """Collapse runs of same-kind deltas; print one line per event kind change,
    with the time span, the count and the token figures the UI could read."""
    out, run = [], None
    for r in rows:
        key = (r["ev"], r.get("dt"))
        if run and run["key"] == key:
            run["n"] += 1; run["t1"] = r["t"]; run["chars"] += r.get("n") or 0
            run["est"] = r.get("est") if r.get("est") is not None else run["est"]
            run["out"] = r.get("out") if r.get("out") is not None else run["out"]
            continue
        if run:
            out.append(run)
        run = {"key": key, "t0": r["t"], "t1": r["t"], "n": 1, "chars": r.get("n") or 0,
               "est": r.get("est"), "out": r.get("out"), "stop": r.get("stop"), "idx": r.get("idx")}
    if run:
        out.append(run)
    prev_end = None
    for x in out:
        ev, dt = x["key"]
        gap = "" if prev_end is None or x["t0"] - prev_end < gaps else f"   <-- silence {x['t0'] - prev_end:.1f} s"
        label = ev if not dt else f"{ev}/{dt}"
        extra = []
        if x["n"] > 1: extra.append(f"x{x['n']}")
        if x["chars"]: extra.append(f"{x['chars']} chars")
        if x["est"] is not None: extra.append(f"est={x['est']}")
        if x["out"] is not None: extra.append(f"out={x['out']}")
        if x["stop"]: extra.append(f"stop={x['stop']}")
        print(f"{x['t0']:8.2f}-{x['t1']:8.2f}s  idx={x['idx'] if x['idx'] is not None else '-':<3} {label:<40} {' '.join(extra)}{gap}")
    print(f"total {rows[-1]['t']:.1f} s, {len(rows)} events")


def claude_p(prompt, base, model):
    exe = shutil.which("claude.cmd") or shutil.which("claude") or "claude"
    env = os.environ.copy(); env.update(ANTHROPIC_BASE_URL = base, ANTHROPIC_AUTH_TOKEN = "sk-local")
    cmd = [exe, "-p", prompt, "--model", model, "--strict-mcp-config", "--effort", "low",
           "--output-format", "stream-json", "--verbose", "--include-partial-messages"]
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, env = env, stdout = subprocess.PIPE, stderr = subprocess.DEVNULL, text = True, encoding = "utf-8", errors = "replace")
    rows = []
    for line in proc.stdout:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        ev = obj.get("event") if obj.get("type") == "stream_event" else None
        if not ev:
            continue
        d = ev.get("delta") or {}
        rows.append({"rid": "claude", "t": round(time.perf_counter() - t0, 3), "ev": ev.get("type"), "idx": ev.get("index"),
                     "dt": d.get("type"), "n": len(d.get("thinking") or d.get("text") or d.get("partial_json") or ""),
                     "est": d.get("estimated_tokens"), "out": (ev.get("usage") or {}).get("output_tokens"), "stop": d.get("stop_reason")})
    proc.wait()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default = DEFAULT)
    ap.add_argument("--all", action = "store_true")
    ap.add_argument("--gaps", type = float, default = 2.0, help = "flag silences longer than this many seconds")
    ap.add_argument("--claude-p", metavar = "PROMPT", help = "run claude -p with stream-json and render what Claude Code received")
    ap.add_argument("--base", default = "http://127.0.0.1:8000")
    ap.add_argument("--model", default = None)
    a = ap.parse_args()
    if a.claude_p:
        model = a.model
        if not model:
            import urllib.request
            with urllib.request.urlopen(a.base + "/health", timeout = 5) as r:
                model = json.load(r)["model"]
        rows = claude_p(a.claude_p, a.base, model)
        print(f"== Claude Code side (stream-json), model {model}")
        render(rows, a.gaps) if rows else print("no stream events received")
        return
    if not os.path.exists(a.file):
        sys.exit(f"no trace at {a.file}; start the server with EXL3_TRACE_SSE={a.file}")
    by = load(a.file)
    rids = list(by) if a.all else list(by)[-1:]
    for rid in rids:
        print(f"== server side, request {rid}")
        render(by[rid], a.gaps)


if __name__ == "__main__":
    main()
