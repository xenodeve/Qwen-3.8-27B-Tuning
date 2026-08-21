"""Measure exactly what a harness sends before any work starts.

The question this answers: **how much of the context window does the harness
spend on itself?** On a 12 GB card serving 131,072 tokens, a 40 K prefix is 31 %
of the window gone before the task is read, and it is paid on every call.

Measured this way rather than asked, because the number a harness reports is not
necessarily the number it sent -- `claude-9arm` reported `input_tokens` of
39,762-40,648 across four calls with no breakdown, and the obvious explanation
(a 324-skill catalogue, 26.5 K of descriptions) turned out to be wrong when the
worker was asked directly and said no skill list was present.

This is an OpenAI-compatible endpoint that **records the request and returns a
stub**. No model, no GPU, no tokens burned. Point a harness at it, send one
trivial prompt, and read what it actually put on the wire.

    python bench/prefix_probe.py --port 8099 --out logs/prefix-probe.jsonl

Then, in a directory containing an `opencode.json` that names this endpoint:

    opencode run -m local/probe "say READY"

Each row records the byte size, the message count, the tool count, and the role
breakdown, so the prefix can be attributed rather than just totalled.

**Byte counts are not token counts.** Divide by ~4 for English prose for a rough
figure, or feed `raw_path` to `llama-server /tokenize` for the exact number
under the tokenizer that will actually serve it.
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(r"C:\AI\qwen38-tuning")
STUB = "READY"


class Probe(BaseHTTPRequestHandler):
    out_path = None
    raw_dir = None
    seq = 0

    def log_message(self, *a):          # silence the default access log
        pass

    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._json(200, {"object": "list", "data": [
                {"id": "probe", "object": "model", "owned_by": "prefix_probe"}]})
        else:
            self._json(404, {"error": "probe: only /v1/models and "
                                      "/v1/chat/completions are implemented"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n)
        try:
            req = json.loads(raw or b"{}")
        except ValueError:
            req = {}

        Probe.seq += 1
        msgs = req.get("messages") or []
        tools = req.get("tools") or []
        roles = {}
        for m in msgs:
            r = m.get("role", "?")
            c = m.get("content")
            if isinstance(c, list):     # multimodal / block form
                c = json.dumps(c)
            roles[r] = roles.get(r, 0) + len(c or "")

        row = {
            "seq": Probe.seq,
            "path": self.path,
            "model": req.get("model"),
            "request_bytes": len(raw),
            "n_messages": len(msgs),
            "n_tools": len(tools),
            "tools_bytes": len(json.dumps(tools)) if tools else 0,
            "chars_by_role": roles,
            "chars_total": sum(roles.values()),
            "stream": bool(req.get("stream")),
            "tool_names": [((t.get("function") or {}).get("name") or t.get("name"))
                           for t in tools][:40],
        }

        if Probe.raw_dir:
            p = Probe.raw_dir / f"request-{Probe.seq:03d}.json"
            p.write_bytes(raw)
            row["raw_path"] = str(p)

        if Probe.out_path:
            with Probe.out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

        print("  #%d  %d bytes  %d messages  %d tools  roles=%s"
              % (row["seq"], row["request_bytes"], row["n_messages"],
                 row["n_tools"], row["chars_by_role"]), flush=True)

        created = int(time.time())
        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in (
                {"choices": [{"index": 0, "delta": {"role": "assistant",
                                                    "content": STUB},
                              "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {},
                              "finish_reason": "stop"}]},
            ):
                chunk.update(id="probe", object="chat.completion.chunk",
                             created=created, model=req.get("model", "probe"))
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            return

        self._json(200, {
            "id": "probe", "object": "chat.completion", "created": created,
            "model": req.get("model", "probe"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": STUB}}],
            "usage": {"prompt_tokens": row["chars_total"] // 4,
                      "completion_tokens": 1,
                      "total_tokens": row["chars_total"] // 4 + 1},
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099,
                    help="not 8080 -- that is the real server's port")
    ap.add_argument("--out", default="logs/prefix-probe.jsonl")
    ap.add_argument("--raw-dir", default="logs/prefix-probe-raw")
    args = ap.parse_args()

    Probe.out_path = ROOT / args.out
    Probe.out_path.parent.mkdir(parents=True, exist_ok=True)
    Probe.raw_dir = ROOT / args.raw_dir
    Probe.raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"prefix probe on http://127.0.0.1:{args.port}/v1  ->  {Probe.out_path}",
          flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Probe).serve_forever()


if __name__ == "__main__":
    main()
