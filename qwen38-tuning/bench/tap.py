"""A recording proxy between a harness and llama-server.

**Everything a run can tell us, captured without changing either side.**

The pieces are otherwise scattered and each is missing something:

- the **server log** has prefill and decode rates but no idea which task a
  request belonged to, and drops the request body entirely;
- the **`/slots`** endpoint shows sampling params but only for the request in
  flight, so a poll almost always misses;
- the **harness transcript** shows tool calls and wall time but nothing about
  tokens;
- `--metrics` is cumulative, so it cannot attribute anything to one request.

This sits in the middle and writes one row per request with all of it joined.
Point the harness at `--port` and it forwards to `--upstream`.

    python bench/tap.py --port 8081 --upstream 127.0.0.1:8080 --out results/tap.jsonl

**Streaming is passed through chunk by chunk, never buffered.** Holding a
response to parse it would change the very decode timing being measured, which
is the failure this whole file exists to avoid. Chunks are forwarded first and
inspected after.

`--mark` writes a marker row, so a harness can label which task follows:

    curl -X POST localhost:8081/_tap/mark -d '{"task":"lru_cache"}'
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(r"C:\AI\qwen38-tuning")


def tool_share(tools):
    """Split the tool schemas into the MCP half and the half no proxy can hide.

    Issue #55's gate needs that split, and `tools_bytes` alone cannot give it.
    A Super-MCP style proxy can collapse `mcp__*` tools behind one entry; it
    cannot touch `Read`, `Edit`, `Write`, `Bash`, `Glob`, `Grep` or `Task`, so if
    the bytes are mostly built-in the first rung of the ladder moves almost
    nothing.

    Bytes per tool, not tokens, and each tool serialised alone so the array's own
    separators are credited to neither half. Turning bytes into tokens is a
    reader's job against the server's `/tokenize`; doing it here would mean
    holding or re-serialising a request, which is the one thing this instrument
    must never do.
    """
    out = {"n_tools": 0, "bytes_total": 0,
           "mcp": {"n": 0, "bytes": 0, "by_server": {}},
           "builtin": {"n": 0, "bytes": 0, "names": []}}
    for t in tools or []:
        size = len(json.dumps(t))
        out["n_tools"] += 1
        out["bytes_total"] += size
        try:
            name = (t.get("function") or {}).get("name") or t.get("name") or ""
        except AttributeError:  # not a mapping at all; still counted, never fatal
            name = ""
        parts = name.split("__")
        # mcp__<server>__<tool>, and the tool half may itself contain "__", so
        # take parts[1] rather than splitting the whole name into two.
        if name.startswith("mcp__") and len(parts) >= 3 and parts[1]:
            server = parts[1]
            slot = out["mcp"]["by_server"].setdefault(server, {"n": 0, "bytes": 0})
            slot["n"] += 1
            slot["bytes"] += size
            out["mcp"]["n"] += 1
            out["mcp"]["bytes"] += size
        else:
            out["builtin"]["n"] += 1
            out["builtin"]["bytes"] += size
            out["builtin"]["names"].append(name)
    out["builtin"]["names"].sort()
    return out


def _roles(messages):
    out = {}
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, list):
            c = json.dumps(c)
        out[m.get("role", "?")] = out.get(m.get("role", "?"), 0) + len(c or "")
    return out


class Tap(BaseHTTPRequestHandler):
    upstream = None
    out_path = None
    seq = 0
    mark = None

    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _emit(self, row):
        if Tap.out_path:
            with Tap.out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

    def _relay(self, method):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""

        if self.path.startswith("/_tap/mark"):
            try:
                Tap.mark = json.loads(body or b"{}")
            except ValueError:
                Tap.mark = {"raw": body.decode("utf-8", "replace")[:200]}
            self._emit({"kind": "MARK", "mark": Tap.mark, "t": time.time()})
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")
            return

        req = {}
        if body:
            try:
                req = json.loads(body)
            except ValueError:
                pass

        Tap.seq += 1
        seq = Tap.seq
        started = time.time()

        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in ("host", "content-length", "connection")}
        url = f"http://{Tap.upstream}{self.path}"
        r = urllib.request.Request(url, data=body or None, headers=headers,
                                   method=method)

        try:
            resp = urllib.request.urlopen(r, timeout=3600)
        except urllib.error.HTTPError as e:
            resp = e
        except Exception as e:
            self._emit({"kind": "REQ", "seq": seq, "path": self.path,
                        "error": f"{type(e).__name__}: {e}",
                        "mark": Tap.mark})
            self.send_response(502)
            self.end_headers()
            return

        self.send_response(resp.status)
        streamed = False
        for k, v in resp.headers.items():
            if k.lower() in ("transfer-encoding", "connection", "content-length"):
                continue
            if k.lower() == "content-type" and "event-stream" in v.lower():
                streamed = True
            self.send_header(k, v)

        row = {"kind": "REQ", "seq": seq, "path": self.path,
               "mark": Tap.mark, "status": resp.status,
               "model": req.get("model"),
               "stream": bool(req.get("stream")),
               "n_messages": len(req.get("messages") or []),
               "n_tools": len(req.get("tools") or []),
               "tools_bytes": len(json.dumps(req.get("tools"))) if req.get("tools") else 0,
               "tool_share": tool_share(req.get("tools")),
               "chars_by_role": _roles(req.get("messages")),
               "sampling": {k: req[k] for k in
                            ("temperature", "top_p", "top_k", "min_p", "seed",
                             "max_tokens", "n_predict", "stop", "reasoning_effort")
                            if k in req},
               "tool_choice": req.get("tool_choice")}

        if streamed:
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            text, timings, usage, finish, tool_calls = [], None, None, None, 0
            first_byte = None
            # Forward first, parse after: buffering to inspect would change the
            # decode timing this file exists to measure.
            for raw in resp:
                if first_byte is None:
                    first_byte = time.time()
                self.wfile.write(b"%x\r\n" % len(raw) + raw + b"\r\n")
                self.wfile.flush()
                if not raw.startswith(b"data: "):
                    continue
                payload = raw[6:].strip()
                if payload in (b"[DONE]", b""):
                    continue
                try:
                    ch = json.loads(payload)
                except ValueError:
                    continue
                timings = ch.get("timings") or timings
                usage = ch.get("usage") or usage
                for c in ch.get("choices") or []:
                    d = c.get("delta") or {}
                    if d.get("content"):
                        text.append(d["content"])
                    if d.get("tool_calls"):
                        tool_calls += len(d["tool_calls"])
                    finish = c.get("finish_reason") or finish
            self.wfile.write(b"0\r\n\r\n")
            row.update(ttfb_s=round((first_byte or started) - started, 3),
                       content_chars=sum(len(t) for t in text),
                       finish_reason=finish, tool_call_deltas=tool_calls)
        else:
            payload = resp.read()
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            try:
                d = json.loads(payload)
            except ValueError:
                d = {}
            timings = d.get("timings")
            usage = d.get("usage")
            ch = (d.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            row.update(content_chars=len(msg.get("content") or ""),
                       finish_reason=ch.get("finish_reason"),
                       tool_call_deltas=len(msg.get("tool_calls") or []))

        row["wall_s"] = round(time.time() - started, 3)
        # llama-server's own numbers, per request: prompt_n, prompt_ms,
        # prompt_per_second, predicted_n, predicted_ms, predicted_per_second,
        # and draft_n / draft_n_accepted when speculation is on.
        if timings:
            row["timings"] = timings
            if timings.get("draft_n"):
                row["acceptance_pct"] = round(
                    100.0 * timings.get("draft_n_accepted", 0) / timings["draft_n"], 1)
        if usage:
            row["usage"] = usage
        self._emit(row)

        print("  #%-4d %-26s %6.1fs  prompt %-6s decode %-6s acc %-6s %s"
              % (seq, (Tap.mark or {}).get("task", "-")[:26], row["wall_s"],
                 (timings or {}).get("prompt_n", "-"),
                 (timings or {}).get("predicted_n", "-"),
                 row.get("acceptance_pct", "-"), row.get("finish_reason", "-")),
              flush=True)

    def do_POST(self):
        self._relay("POST")

    def do_GET(self):
        self._relay("GET")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--upstream", default="127.0.0.1:8080")
    ap.add_argument("--out", default="results/tap.jsonl")
    args = ap.parse_args()

    Tap.upstream = args.upstream
    Tap.out_path = ROOT / args.out
    Tap.out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"tap :{args.port} -> {args.upstream}   -> {Tap.out_path}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Tap).serve_forever()


if __name__ == "__main__":
    main()
