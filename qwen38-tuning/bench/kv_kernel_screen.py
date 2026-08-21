"""Which KV cache types actually have a fast kernel in THIS build?

Motivation, from a run that had to be abandoned: at 128K with `-ctk q5_1 -ctv
q5_1`, prompt processing came in at **43.8 tok/s and fell to 22.5** as depth
grew, against **468.9 tok/s** for q8_0 on the same artifact and context. That is
roughly 20x slower and degrading -- the "quantized-KV Flash-Attention kernel that
was not compiled falls back to a very slow path" failure the deep-research
document warns about. One arm would have taken over ninety minutes to fill its
window and produced a number nobody needed.

So: screen cheaply, at shallow depth, before spending a deep run on a type that
has no kernel. A 2,000-token prompt is enough -- a type with a kernel finishes it
in seconds, one without takes a minute or more, and the gap is three orders of
magnitude wider than any tuning effect this machine can resolve.

This is a SCREEN, not a benchmark. Its output is "has a usable kernel / does
not". Throughput comparisons between survivors belong at real depth, because the
cache is not the bottleneck at 16K.

Usage:
    python kv_kernel_screen.py
    python kv_kernel_screen.py --types f16,q8_0,q4_0 --ctx 16384
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness import median

ROOT = Path(r"C:\AI\qwen38-tuning")
EXE = r"C:\AI\llama.cpp-cuda\llama-server.exe"
BASE = "http://127.0.0.1:8080"
MODEL = "unsloth/Qwen3.8-27B-GGUF:UD-IQ2_XXS"

# ~55 tokens per repeat
_BLOCK = """
class Handler{i:04d}:
    def __init__(self, config):
        self.config = config
        self.cache = {{}}
    def process(self, item):
        key = item.get('id')
        if key in self.cache:
            return self.cache[key]
        return self.transform(item)
"""


def filler(target_tokens):
    out, n = [], 0
    i = 0
    while n < target_tokens:
        out.append(_BLOCK.format(i=i))
        n += 55
        i += 1
    return "".join(out)


def kill():
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    "$c=Get-NetTCPConnection -LocalPort 8080 -State Listen "
                    "-ErrorAction SilentlyContinue; if($c){Stop-Process -Id "
                    "$c.OwningProcess -Force}"], capture_output=True)
    time.sleep(6)


def start(ctx, kvtype, tag):
    kill()
    log = ROOT / "logs" / ("kvscreen-%s.log" % tag)
    args = [EXE, "-hf", MODEL, "--alias", "kvscreen", "-c", str(ctx),
            "-ngl", "auto", "--fit", "on", "--fit-target", "768", "-fa", "on",
            "-np", "1", "-t", "18", "-b", "2048", "-ub", "256",
            "--no-mmproj-auto", "--host", "127.0.0.1", "--port", "8080"]
    if kvtype != "f16":
        args += ["-ctk", kvtype, "-ctv", kvtype]
    fh = log.open("w", encoding="utf-8", errors="replace")
    p = subprocess.Popen(args, stdout=fh, stderr=subprocess.STDOUT)
    for _ in range(60):
        time.sleep(4)
        if p.poll() is not None:
            fh.close()
            return None, None, log
        try:
            urllib.request.urlopen(BASE + "/health", timeout=3).read()
            time.sleep(2)
            return p, fh, log
        except Exception:
            pass
    fh.close()
    return None, None, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="f16,q8_0,q5_1,q5_0,q4_1,q4_0,iq4_nl,bf16")
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--probe-tokens", type=int, default=2000)
    # Measured 2026-08-19 on IQ2_XXS: the results form two clean clusters with
    # nothing between them --
    #     f16 1183 | bf16 1182 | q4_0 1176 | q8_0 1174     <- fast kernel
    #     q4_1  170 | q5_0  161 | q5_1  157 | iq4_nl 144    <- fallback path
    # The first run of this screen used 150, which sat INSIDE the slow cluster
    # and passed q5_1/q5_0/q4_1 as usable. 500 sits in the empty gap.
    ap.add_argument("--threshold", type=float, default=500.0)
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--out", default="kv-kernel-screen.jsonl")
    args = ap.parse_args()

    out = ROOT / "results" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    prompt = filler(args.probe_tokens)
    results = []

    for kvtype in [t.strip() for t in args.types.split(",")]:
        p, fh, log = start(args.ctx, kvtype, kvtype)
        if p is None:
            row = dict(kv=kvtype, loaded=False, note="server failed to start")
            print("  %-8s FAILED TO START -- see %s" % (kvtype, log.name), flush=True)
        else:
            try:
                t0 = time.time()
                req = urllib.request.Request(
                    BASE + "/completion",
                    data=json.dumps({"prompt": prompt, "n_predict": 16,
                                     "temperature": 0.0, "top_k": 1,
                                     "cache_prompt": False}).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=args.timeout) as r:
                    d = json.loads(r.read().decode())
                wall = time.time() - t0
                t = d["timings"]
                pp = round(t["prompt_per_second"], 1)
                row = dict(kv=kvtype, loaded=True, prompt_n=t["prompt_n"],
                           pp_tok_s=pp, tg_tok_s=round(t["predicted_per_second"], 2),
                           wall_s=round(wall, 1),
                           has_kernel=pp >= args.threshold)
                print("  %-8s pp=%-8s tg=%-7s wall=%-6ss  %s"
                      % (kvtype, pp, row["tg_tok_s"], row["wall_s"],
                         "OK" if row["has_kernel"] else "NO FAST KERNEL"),
                      flush=True)
            except Exception as e:
                row = dict(kv=kvtype, loaded=True, error=str(e),
                           has_kernel=False)
                print("  %-8s PROBE FAILED after %ss: %s -- treating as no kernel"
                      % (kvtype, args.timeout, e), flush=True)
            p.kill()
            fh.close()
        results.append(row)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    kill()
    ok = [r["kv"] for r in results if r.get("has_kernel")]
    bad = [r["kv"] for r in results if r.get("loaded") and not r.get("has_kernel")]
    print("\n=== KV KERNEL SCREEN (ctx %d, %d-token probe, threshold %s tok/s) ==="
          % (args.ctx, args.probe_tokens, args.threshold))
    print("  usable at depth : %s" % (", ".join(ok) or "none"))
    print("  no fast kernel  : %s" % (", ".join(bad) or "none"))
    print("\n  Only the first list is worth a 128K run; the second would spend")
    print("  over an hour per arm filling its window.")


if __name__ == "__main__":
    main()
