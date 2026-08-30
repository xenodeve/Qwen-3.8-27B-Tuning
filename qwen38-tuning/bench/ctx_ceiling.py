"""What is the deepest context this artifact can hold WITHOUT leaving the GPU?

The project's governing measurement is the residency cliff: on this card the
last four CPU-resident layers cost about half the decode throughput, and every
depth result so far has been the same story told backwards -- context spends
exactly the VRAM that quantization frees.

    16K    65 + 0    42.4 tok/s      KV ~0.5 GB
    64K    61 + 4    15.8 tok/s      KV  2.0 GB
    128K   47 + 18    5.2 tok/s      KV  3.3 GB
    256K   31 + 34    1.7 tok/s      KV  4.4 GB

So "how deep can we go" is not a throughput question first. It is: at what
context does the split break, and how much headroom is left just below that.
This finds the ceiling by bisecting the context ladder and reading the layer
split out of the load report -- no generation, no prefill, about a minute per
boot instead of the ten a 256K cold prefill costs.

The split and the KV allocation are deterministic properties of the load, so
unlike decode they are safe to measure while other work shares the machine.

Usage:
    python ctx_ceiling.py --quant v3-iq1s
    python ctx_ceiling.py --quant ornith9b --kv q4_0 --ladder 131072,196608,262144
"""
import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gpu_device
from provenance import resolve_exe
from harness import parse_layer_split
import depth_sweep as D

ROOT = Path(r"C:\AI\qwen38-tuning")
EXE = resolve_exe(r"C:\AI\llama.cpp-cuda\llama-server.exe")
BASE = "http://127.0.0.1:8080"


def kill():
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    "$c=Get-NetTCPConnection -LocalPort 8080 -State Listen "
                    "-ErrorAction SilentlyContinue; if($c){Stop-Process -Id "
                    "$c.OwningProcess -Force}"], capture_output=True)
    time.sleep(6)


def vram():
    """[used, free] summed over the cards this process was pinned to.

    Not the served card alone. This script is launched with
    CUDA_VISIBLE_DEVICES already exported, so on a two-card run the served
    card's free memory is a fraction of the headroom the run actually had --
    which is what made the Q4 ladder print `free 4130` at ctx 131,072 when
    there was more (issue #50, #51).

    The sum is a CEILING: a layer cannot straddle two cards, so free memory
    does not really add. Residency comes from the layer split, not from here.
    """
    used, free = gpu_device.visible_vram()
    return [used, free]


def boot(model, ctx, kv, tag, extra=()):
    kill()
    free_before = vram()[1]
    log = ROOT / "logs" / ("ceil-%s.log" % tag)
    args = [EXE, "-m", model, "--alias", "ceil", "-c", str(ctx),
            "-ngl", "auto", "--fit", "on", "--fit-target", "768", "-fa", "on",
            "-np", "1", "-t", "18", "-b", "2048", "-ub", "256",
            "--no-mmproj-auto", "-lv", "5",
            "--host", "127.0.0.1", "--port", "8080"]
    if kv != "f16":
        args += ["-ctk", kv, "-ctv", kv]
    # Passthrough for flags that change the VRAM budget without changing the
    # cache TYPE -- e.g. --ctx-checkpoints, which defaults to 32 per slot and is
    # speculative VRAM an append-only agent never rewinds into. Residency is the
    # thing being measured, so anything that moves the budget belongs here.
    args += list(extra)
    fh = log.open("w", encoding="utf-8", errors="replace")
    p = subprocess.Popen(args, stdout=fh, stderr=subprocess.STDOUT)
    for _ in range(150):                 # deep contexts allocate slowly
        time.sleep(4)
        if p.poll() is not None:
            fh.close()
            return None, None, log, free_before
        try:
            urllib.request.urlopen(BASE + "/health", timeout=3).read()
            time.sleep(2)
            fh.flush()
            return p, fh, log, free_before
        except Exception:
            pass
    fh.close()
    return None, None, log, free_before


def kv_mib(text):
    best = 0.0
    for m in re.finditer(r"KV self size\s*=\s*([\d.]+)\s*MiB", text):
        best = max(best, float(m.group(1)))
    for m in re.finditer(r"kv_unified:.*?size\s*=\s*([\d.]+)\s*MiB", text):
        best = max(best, float(m.group(1)))
    return round(best, 1) or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", required=True, help="key in depth_sweep.QUANTS")
    ap.add_argument("--kv", default="q4_0",
                    help="only f16/bf16/q8_0/q4_0 have a fast kernel in b10472")
    ap.add_argument("--ladder", default="131072,163840,196608,229376,262144")
    ap.add_argument("--out", default="ctx-ceiling.jsonl")
    ap.add_argument("--extra", default="",
                    help="extra server flags, space separated, e.g. "
                         "\"--ctx-checkpoints 8\"")
    ap.add_argument("--tag", default="", help="suffix for log names and the row")
    args = ap.parse_args()

    model = D.QUANTS.get(args.quant)
    if not model:
        sys.exit("unknown or not-downloaded quant %r; known: %s"
                 % (args.quant, ", ".join(k for k, v in D.QUANTS.items() if v)))

    out = ROOT / "results" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    ladder = [int(x) for x in args.ladder.split(",")]
    extra = args.extra.split()
    rows, ceiling = [], None

    print("quant %s  KV %s" % (args.quant, args.kv), flush=True)
    for ctx in ladder:
        p, fh, log, free_before = boot(model, ctx, args.kv,
                                       "%s-%s-%d%s" % (args.quant, args.kv, ctx,
                                                       args.tag and "-" + args.tag),
                                       extra)
        if p is None:
            row = dict(quant=args.quant, kv=args.kv, extra=args.extra, tag=args.tag, ctx=ctx, loaded=False,
                       free_before=free_before)
            print("  ctx %-7d FAILED TO LOAD" % ctx, flush=True)
            rows.append(row)
            break

        used, free = vram()
        text = log.read_text(encoding="utf-8", errors="replace")
        try:
            gpu, cpu = parse_layer_split(text)
        except ValueError as e:
            gpu = cpu = None
            print("    split unavailable: %s" % e, flush=True)
        resident = (cpu == 0)
        row = dict(quant=args.quant, kv=args.kv, extra=args.extra, tag=args.tag, ctx=ctx, loaded=True,
                   gpu_layers=gpu, cpu_layers=cpu, resident=resident,
                   free_before=free_before, vram_used=used, vram_free=free,
                   kv_mib=kv_mib(text))
        rows.append(row)
        if resident:
            ceiling = ctx
        print("  ctx %-7d split %s+%s  free %-6s KV %-8s %s"
              % (ctx, gpu, cpu, free, row["kv_mib"],
                 "RESIDENT" if resident else "spilled to CPU"), flush=True)
        p.kill()
        fh.close()
        if not resident:
            # Deeper contexts only allocate more cache; once the split breaks it
            # stays broken, so there is nothing above this worth a ten-minute
            # boot. Stop rather than walk the rest of the ladder.
            break

    kill()
    verdict = dict(quant=args.quant, kv=args.kv, extra=args.extra, tag=args.tag, kind="SUMMARY",
                   deepest_resident_ctx=ceiling,
                   ladder=ladder)
    with out.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps(verdict) + "\n")

    print("\n=== CONTEXT CEILING %s (%s KV) ===" % (args.quant, args.kv))
    if ceiling:
        print("  deepest fully-resident context: %d (%.0fK)" % (ceiling, ceiling / 1024))
    else:
        print("  no context in the ladder held full residency")
    print("\n  Residency is necessary, not sufficient: this says the weights stay")
    print("  on the card, not that the model answers well there. Retrieval")
    print("  quality at depth is still verified on Q4 alone.")


if __name__ == "__main__":
    main()
