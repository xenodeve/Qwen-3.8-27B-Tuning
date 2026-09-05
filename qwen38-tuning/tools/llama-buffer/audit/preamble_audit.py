"""What Claude Code actually sends llama-server, read from real traffic.

Issue #55's GATE asked: split the first request of a session into what MCP
contributes and what Claude Code's built-ins contribute, so that the first rung
of the plan (a tool-schema router) is spent on a measured share, not a guessed
one. `llama-tap` was the planned instrument and has never captured a real
request. This reads two things that already exist instead:

  * the serve log   -- every request's prompt size, cache hit and prefill time
  * the transcripts -- ~/.claude/projects/**/*.jsonl: per-reply usage as the
                       server reported it, every tool call, and every
                       attachment Claude Code injected (hooks, skill listing)

Real data, two limits: a transcript holds neither the system prompt nor the
tool schemas, so the fixed preamble is inferred as (first request) minus (what
the transcript does hold); and the token counts for attachments come from the
served model's /tokenize, which needs the server up.

    python preamble_audit.py --log ..\..\..\logs\serve-20260902-160749.log \
        --boot "2026-09-02 16:07:55" --project D--Github-Agentic-Framework
"""
import argparse, collections, datetime as dt, glob, json, os, re, statistics, sys, urllib.request

ANSI = re.compile(r"\x1b\[[0-9;]*m")
PROMPT_EVAL = re.compile(r"(\d+)\.(\d+)\.(\d+)\.(\d+) .*task (\d+) \| prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens")
EVAL = re.compile(r"task (\d+) \|\s+eval time =\s+([\d.]+) ms /\s+(\d+) tokens")
CACHED = re.compile(r"task (\d+) \| cached n_tokens = (\d+)")


def tokenize(text, server):
    n = 0
    for i in range(0, len(text), 60000):
        req = urllib.request.Request(server + "/tokenize", data=json.dumps({"content": text[i:i + 60000]}).encode(),
                                     headers={"Content-Type": "application/json"})
        n += len(json.load(urllib.request.urlopen(req, timeout=120))["tokens"])
    return n


def read_log(path, boot):
    cached, pe, de, stamp = {}, {}, {}, {}
    for ln in open(path, encoding="utf-8", errors="replace"):
        ln = ANSI.sub("", ln)
        m = CACHED.search(ln)
        if m:
            cached.setdefault(int(m.group(1)), int(m.group(2)))
        m = PROMPT_EVAL.match(ln)
        if m:
            t = int(m.group(5))
            pe[t] = (int(m.group(7)), float(m.group(6)))
            stamp[t] = boot + dt.timedelta(minutes=int(m.group(1)), seconds=int(m.group(2)), milliseconds=int(m.group(3)))
        m = EVAL.search(ln)
        if m:
            de[int(m.group(1))] = (int(m.group(3)), float(m.group(2)))
    return cached, pe, de, stamp


def transcript_events(projects_dir, since):
    """Every reply whose prompt was prefilled from an empty cache, main and subagent."""
    ev = []
    for f in glob.glob(os.path.join(projects_dir, "*", "**", "*.jsonl"), recursive=True):
        if os.path.getmtime(f) < since:
            continue
        proj = os.path.relpath(f, projects_dir).split(os.sep)[0]
        sub = "subagents" in f
        for ln in open(f, encoding="utf-8", errors="replace"):
            if '"cache_read_input_tokens":0' not in ln:
                continue
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            if d.get("type") != "assistant":
                continue
            u = d["message"].get("usage") or {}
            if u.get("input_tokens", 0) > 500:
                ev.append((dt.datetime.fromisoformat(d["timestamp"].replace("Z", "+00:00")), u["input_tokens"],
                           proj, "sub" if sub else "main", os.path.basename(f)[:14], d["message"].get("model")))
    return ev


def first_request(path, server):
    """Token cost of everything the transcript holds before the first reply.

    hook_success rows are the hook's raw stdout and hook_additional_context is
    what the model received, so only the latter is counted for hooks."""
    rows, total = [], 0
    for ln in open(path, encoding="utf-8", errors="replace"):
        d = json.loads(ln)
        t = d.get("type")
        if t == "assistant":
            break
        if t == "attachment":
            a = d["attachment"]
            kind = a.get("type")
            if kind == "hook_success":
                continue
            body = a.get("content") or a.get("stdout") or a.get("text") or ""
            if isinstance(body, list):
                body = json.dumps(body, ensure_ascii=False)
            n = tokenize(body, server) if body else 0
            rows.append((kind, n)); total += n
        elif t == "user":
            c = d["message"].get("content")
            s = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
            n = tokenize(s, server); rows.append(("user", n)); total += n
    return rows, total


def tool_mix(path):
    tools, msgs, first = collections.Counter(), 0, None
    for ln in open(path, encoding="utf-8", errors="replace"):
        d = json.loads(ln)
        if d.get("type") != "assistant":
            continue
        m = d["message"]; msgs += 1
        u = m.get("usage") or {}
        if first is None and u:
            first = u.get("input_tokens")
        for c in m.get("content", []) if isinstance(m.get("content"), list) else []:
            if c.get("type") == "tool_use":
                tools[c["name"]] += 1
    return msgs, first, tools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--boot", required=True, help="server start, local time, 'YYYY-MM-DD HH:MM:SS'")
    ap.add_argument("--utc-offset", type=int, default=7)
    ap.add_argument("--project", required=True, help="folder name under ~/.claude/projects")
    ap.add_argument("--server", default="http://localhost:8080")
    ap.add_argument("--projects-dir", default=os.path.expanduser("~/.claude/projects"))
    a = ap.parse_args()
    boot = dt.datetime.strptime(a.boot, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone(dt.timedelta(hours=a.utc_offset))).astimezone(dt.timezone.utc)

    cached, pe, de, stamp = read_log(a.log, boot)
    cold = {t for t in pe if cached.get(t, 0) == 0}
    big = {t for t in pe if t not in cold and pe[t][0] > 20000}
    ms = lambda s: sum(pe[t][1] for t in s)
    tot = ms(pe); dec = sum(v[1] for v in de.values()); dtok = sum(v[0] for v in de.values())
    print(f"== {os.path.basename(a.log)}: {len(pe)} requests")
    print(f"prefill {tot/1000:7.0f} s = cold {ms(cold)/1000:.0f} s ({100*ms(cold)/tot:.0f} %, n={len(cold)})"
          f" + re-prefill>20K {ms(big)/1000:.0f} s ({100*ms(big)/tot:.0f} %, n={len(big)})"
          f" + incremental {(tot-ms(cold)-ms(big))/1000:.0f} s")
    print(f"decode  {dec/1000:7.0f} s for {dtok} tokens = {dtok/(dec/1000):.1f} tok/s;  prefill is {100*tot/(tot+dec):.0f} % of model time")

    ev = transcript_events(a.projects_dir, boot.timestamp() - 3600)
    print(f"\n== cold prefills matched to transcripts ({len(ev)} cold usage rows on disk)")
    for t in sorted(cold, key=lambda t: stamp[t]):
        s, n = stamp[t], pe[t][0]
        best = min((e for e in ev if abs(e[1] - n) <= 60 and abs((e[0] - s).total_seconds()) < 600),
                   key=lambda e: abs((e[0] - s).total_seconds()), default=None)
        tag = f"{best[3]:4s} {best[2]} {best[4]} {best[5]}" if best else "-- no transcript match --"
        print(f"  {s.strftime('%m-%d %H:%M:%S')} UTC  {n:7d} tok {pe[t][1]/1000:6.1f} s  {tag}")

    pdir = os.path.join(a.projects_dir, a.project)
    print(f"\n== sessions in {a.project} since boot")
    for f in sorted(glob.glob(os.path.join(pdir, "*.jsonl")), key=os.path.getmtime):
        if os.path.getmtime(f) < boot.timestamp():
            continue
        msgs, first, tools = tool_mix(f)
        if not first:
            continue
        mcp = sum(v for k, v in tools.items() if k.startswith("mcp__"))
        builtin = sum(tools.values()) - mcp
        print(f"  {os.path.basename(f)[:14]} replies={msgs:4d} first_request={first:6d} tool_calls builtin={builtin:3d} mcp={mcp:2d}"
              f" {sorted(k for k in tools if k.startswith('mcp__'))}")

    latest = max(glob.glob(os.path.join(pdir, "*.jsonl")), key=os.path.getmtime)
    try:
        rows, total = first_request(latest, a.server)
        print(f"\n== what the transcript holds before the first reply of {os.path.basename(latest)[:14]} (server tokenizer)")
        for kind, n in rows:
            print(f"  {kind:26s} {n:6d}")
        print(f"  {'TOTAL':26s} {total:6d}   -> the rest of the first request is system prompt + tool schemas + CLAUDE.md")
    except Exception as e:
        print(f"\n(attachment tokenizing skipped: {e})")


if __name__ == "__main__":
    main()
