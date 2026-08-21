"""Context-depth sweep, with Q8 KV as an arm.

This is the phase every planning document names as the one that can overturn the
Q3-vs-Q4 verdict: KV grows with context, `--fit` then has less room for weights,
and the layer split moves. Effects here are expected in tens of percent, which is
above the 13.6% restart-drift floor measured in E9 — unlike the remaining flag
tuning at 16K.

Design notes:

- **Cold prefill is measured once per depth** (a 256K prefill costs minutes; N=3
  would cost an hour for one row). Decode is measured N=3 using `cache_prompt`,
  so the deep prefill is paid once and the three generations reuse it. That also
  matches how an agent actually behaves — one cold turn, then appends.

- **Q8 KV is tested on the CURRENT build.** The deep-research report warns that a
  quantized-KV Flash-Attention kernel that was not compiled falls back to a very
  slow path. Rather than assume, measure: a catastrophic Q8 result IS the
  evidence that the FA_ALL_QUANTS build is required, and a good one means it is not.

- **Loading is allowed to fail.** At deep context the model may not fit at all.
  A failure to start is a result, recorded as such, not an error to hide.
"""
import hashlib, json, subprocess, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness import (median, parse_layer_split, project_prefill_seconds,
                     completion_timeout_s, vram_settled, VRAM_MIN_RISE_MIB,
                     draft_acceptance)

ROOT = Path(r"C:\AI\qwen38-tuning")
EXE = r"C:\AI\llama.cpp-cuda\llama-server.exe"
import glob as _glob, os as _os
def _c(repo, fn, size=None):
    """Resolve a cached artifact to a path. See the note in model_arena.ARMS:
    `-hf` does an online etag check per launch and can stall on a busy link."""
    h = _glob.glob(_os.path.expanduser("~/.cache/huggingface/hub/models--%s/snapshots/*/%s"
                                       % (repo.replace("/", "--"), fn)))
    if size is not None:
        h = [x for x in h if _os.path.getsize(x) == size]
    if len(h) > 1:
        raise ValueError("%s/%s is ambiguous across %d snapshots; pin the byte "
                         "count (the repo was republished in place)" % (repo, fn, len(h)))
    return _os.path.abspath(h[0]) if h else None

QUANTS = {"q4": _c("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-Q4_K_XL.gguf", 17923394624),
          "q3": _c("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-Q3_K_XL.gguf", 13441059904),
          "q2kxl": _c("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-Q2_K_XL.gguf", 10676423744),
          "iq2xxs": _c("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-IQ2_XXS.gguf", 9010048064),
          # Paths, not repo:tag -- see the note in model_arena.ARMS.
          "iq1m": 'C:\\Users\\xenod\\.cache\\huggingface\\hub\\models--AtomicChat--Qwen3.8-27B-GGUF\\snapshots\\ca10ebceb1887be9d33b838770a36b39d75a8a4c\\Qwen3.8-27B-AD-IQ1_M.gguf',
          "ornith9b": 'C:\\Users\\xenod\\.cache\\huggingface\\hub\\models--ornith-ai--Ornith-1.0-9B-GGUF\\snapshots\\3296bc7a404871a72ac3f1903f561459c09b5c17\\ornith-1.0-9b-Q6_K.gguf',
          "bonsai-g64": 'C:\\Users\\xenod\\.cache\\huggingface\\hub\\models--prism-ml--Ternary-Bonsai-27B-gguf\\snapshots\\abbae723028d71be674e71e1a71201a6f43fab22\\Ternary-Bonsai-27B-Q2_g64.gguf',
          # Dynamic V3. Pinned by byte count: the repo was republished in place
          # on 2026-08-19 with identical filenames, so name alone is ambiguous.
          "v3-iq1s": _c("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-IQ1_S.gguf"),
          "v3-iq1m": _c("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-IQ1_M.gguf"),
          "v3-iq2xxs": _c("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-IQ2_XXS.gguf", 7266070528),
          # The missing rung of the bits-per-weight ladder. 8.37 GB, in the
          # cache since 2026-08-20 and never loaded. It sits between
          # UD-IQ2_XXS (6.77 GiB, 2.16 bpw, 58.3 % contract) and pre-V3
          # UD-IQ2_XXS (8.39 GiB, 2.64 bpw, 90 % accept) -- exactly the gap
          # between "fails on format" and "works". See plan 04 section 0.
          "v3-iq2s": _c("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-IQ2_S.gguf", 8371970048),
          "v3-q2kxl": _c("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-Q2_K_XL.gguf", 9828981664),
          "bonsai-1bit": r"C:\Users\xenod\.cache\huggingface\hub\models--prism-ml--Bonsai-27B-gguf\snapshots\f10afb355f104535e3e3e98cf7ab7795c72bd292\Bonsai-27B-Q1_0.gguf",
          "ornith9b-q8": r"C:\Users\xenod\.cache\huggingface\hub\models--ornith-ai--Ornith-1.0-9B-GGUF\snapshots\3296bc7a404871a72ac3f1903f561459c09b5c17\ornith-1.0-9b-Q8_0.gguf"}

# MTP is a net LOSS on a resident target (report 10 s1: -7% on Q2_K_XL, because
# the draft head's VRAM pushes six target layers back to the CPU). But KV grows
# with depth and pushes layers off anyway, so the trade may invert again out
# there. Controlled rather than hardcoded, so both can be measured.
USE_MTP = True

# Sampling for the TIMED generations.
#
# Default False = temperature 0.7, which is what a real turn looks like. That
# is right for anything whose cost does not depend on what is being written.
#
# It is WRONG for n-gram speculation, whose whole mechanism is replaying token
# sequences already present in the context: at 0.7 every round writes different
# text, the hit rate follows the text, and the same arm measured twice on
# 2026-08-20 returned +80.79 % and -30.56 %. Both passed the paired test,
# because the 13.6 % floor was built for boot-to-boot VRAM drift and cannot
# see variance that comes from the content.
#
# Set True to pin temperature 0 and a fixed seed so every round writes the SAME
# text. Comparable only against other runs that also set it.
FIXED_TEXT = False

# How many tokens each TIMED generation produces. 160 since the harness was
# written, and never questioned until 2026-08-21, when an external review of
# this model reported speculation warming up over a long generation:
#
#   "By the time it came to output the actual response, the MTP had gotten
#    extremely fast (91 tk/s vs 62 tk/s starting rate)"
#
# If a speculative decoder needs more than 160 tokens to reach its rate, then
# every decoder number this project holds is understated -- and `draft-mtp`,
# `draft-dflash`, eagle3 and dspark were all ELIMINATED on those numbers. The
# value is recorded on every row so a short run can never be compared against a
# long one by accident.
N_PREDICT = 160
MODEL = QUANTS["q4"]          # overridden by argv
BASE = "http://127.0.0.1:8080"
OUT = ROOT / "results" / "depth-sweep.jsonl"

# ~55 tokens per repeat, so the filler scales predictably with the target depth.
_BLOCK = '''
class Handler{i:04d}:
    """Processes records for shard {i}."""
    def __init__(self, config, registry):
        self.config = config
        self.registry = registry
        self.cache = {{}}

    def process(self, item):
        key = item.get("id")
        if key in self.cache:
            return self.cache[key]
        result = self.registry.lookup(key).transform(item, self.config)
        self.cache[key] = result
        return result
'''


# Which filler the timed prompt is built from. Instrument fault 8, 2026-08-21:
# the original repeats one class with a four-digit index -- 962 blocks at
# 147,456, adjacent blocks 99.5 % identical, 84.5 % of non-blank lines exact
# duplicates. An n-gram decoder drafts from context, so that is the best case
# that can be constructed for it, and EVERY n-gram figure this project holds was
# measured on it (acceptance 99-100 % at every depth is the tell).
#
#   "high"  the historic filler. Comparable to every earlier result.
#   "low"   varied blocks, for an estimate that is not a synthetic best case.
#
# Recorded on every row so the two can never be silently compared.
FILLER = "high"

_NOUNS = ("record", "shard", "packet", "ledger", "token", "frame", "chunk",
          "entry", "batch", "slice", "column", "digest", "segment", "envelope")
_VERBS = ("resolve", "flush", "compact", "reconcile", "hydrate", "prune",
          "annotate", "dispatch", "coalesce", "rewind", "seal", "probe")
_TYPES = ("dict", "list", "set", "tuple", "bytes", "int", "float", "str")


def _varied_block(i):
    """A code-like block whose lines mostly do not recur.

    Deterministic -- no RNG, because `--fixed-text` exists to make a sweep
    reproducible and a random prompt would undo it. Variety comes from indexing
    three word lists at different strides, so consecutive blocks share structure
    (as real code does) without sharing lines.
    """
    n = _NOUNS[i % len(_NOUNS)]
    v = _VERBS[(i * 5) % len(_VERBS)]
    v2 = _VERBS[(i * 7 + 3) % len(_VERBS)]
    t = _TYPES[(i * 3) % len(_TYPES)]
    nl = chr(10)
    q3 = chr(34) * 3
    lines = [
        f"def {v}_{n}_{i}(source, limit={i % 97 + 3}):",
        f"    {q3}{v.capitalize()} every {n} in source, keeping at most limit.{q3}",
        f"    staged: {t} = {t}()",
        f"    for offset, item in enumerate(source):",
        f"        if offset >= limit or item is None:",
        f"            break",
        f"        key = ({i} * offset) ^ hash(item)",
        f"        staged = _{v2}(staged, key, item, depth={(i * 13) % 31})",
        f"    return staged, offset if source else 0",
        "",
    ]
    return nl.join(lines) + nl


def filler(target_tokens):
    """Roughly `target_tokens` of realistic source text (~3.6 chars/token)."""
    out, approx = [], 0
    i = 0
    while approx < target_tokens * 3.6:
        block = _varied_block(i) if FILLER == "low" else _BLOCK.format(i=i)
        out.append(block)
        approx += len(block)
        i += 1
    return "".join(out)


# No default timeout. A flat one was the whole of instrument fault 6: 3600 s
# is simultaneously absurd at 16K and arbitrary at 131,072, and on 2026-08-21
# it sat exactly one hour on an arm whose prefill had already collapsed to
# 8.56 tok/s. Callers size it with harness.completion_timeout_s(ctx).
def post(path, payload, timeout):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def kill():
    """Stop whatever listens on 8080, then wait for the GPU to actually free."""
    free_before = vram()[1]
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
                        "$c=Get-NetTCPConnection -LocalPort 8080 -State Listen "
                        "-ErrorAction SilentlyContinue; if($c){Stop-Process -Id "
                        "$c.OwningProcess -Force; 'KILLED'}"],
                       capture_output=True, text=True)
    if "KILLED" not in (r.stdout or ""):
        return                      # nothing was listening; nothing to wait for
    # Something WAS resident, so free VRAM must rise. Without this floor a poll
    # taken before the driver starts releasing looks identical to one taken
    # after it finishes -- see harness.vram_settled.
    wait_for_vram_release(floor_mib=free_before + VRAM_MIN_RISE_MIB)


def wait_for_vram_release(floor_mib=None, limit_s=120, poll_s=3):
    """Block until free VRAM stops moving, or `limit_s` elapses.

    The flat `time.sleep(5)` this replaces is instrument fault 7. WDDM releases
    a 12 GB allocation in stages; on 2026-08-21 the next arm started 5 s after
    the kill, passed /health against VRAM the driver still held, and died on
    its first /completion with ConnectionResetError -- destroying a queue step
    that had nothing to do with the arm that was slow.

    Returns the readings so a caller can record what it waited for. Timing out
    is reported, never silent: a still-moving GPU is a fact the row should
    carry, not something to paper over with another sleep.
    """
    readings = []
    for _ in range(int(limit_s / poll_s)):
        time.sleep(poll_s)
        readings.append(vram()[1])
        if vram_settled(readings, floor_mib=floor_mib):
            return readings
    print(f"    VRAM still moving after {limit_s}s: {readings[-4:]}", flush=True)
    return readings


def vram():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip()
    return [int(x) for x in o.split(",")]


def start(ctx, extra, tag):
    kill()
    free_before = vram()[1]
    log = ROOT / "logs" / f"depth-{tag}.log"
    ref = ["-m", MODEL] if MODEL.endswith(".gguf") else ["-hf", MODEL]
    args = [EXE] + ref + ["--alias", "qwen38-q4", "-c", str(ctx),
            "-ngl", "auto", "--fit", "on", "--fit-target", "768", "-fa", "on",
            "-np", "1", "-t", "18", "-b", "2048", "-ub", "256",
            "--no-mmproj-auto", "-lv", "5",
            "--host", "127.0.0.1", "--port", "8080"] + (
            ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"] if USE_MTP else []
            ) + extra
    fh = log.open("w", encoding="utf-8", errors="replace")
    p = subprocess.Popen(args, stdout=fh, stderr=subprocess.STDOUT)
    for _ in range(90):                      # deep contexts allocate slowly
        time.sleep(4)
        if p.poll() is not None:
            fh.close()
            return None, None, log, free_before
        try:
            urllib.request.urlopen(BASE + "/health", timeout=3).read()
            time.sleep(2); fh.flush()
            return p, fh, log, free_before
        except Exception:
            pass
    fh.close()
    return None, None, log, free_before


def kv_buffers(log):
    """Pull the actual KV / recurrent-state allocations out of the load report."""
    out = {}
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        for key, needle in (("kv_mib", "KV buffer size"),
                            ("rs_gpu_mib", "CUDA0 RS buffer size"),
                            ("rs_cpu_mib", "CPU RS buffer size")):
            if needle in line and "=" in line:
                try:
                    v = float(line.split("=")[-1].strip().split()[0])
                except ValueError:
                    continue
                out[key] = max(out.get(key, 0), v)
    return out


def run(ctx, extra, label, tag):
    p, fh, log, free_before = start(ctx, extra, tag)
    row = dict(ctx=ctx, label=label, args=" ".join(extra),
               fixed_text=FIXED_TEXT, n_predict=N_PREDICT, filler=FILLER,
               free_before=free_before)

    if p is None:
        row.update(loaded=False, note="server failed to start")
        print(f"  {label:<26} ctx={ctx:<7} FAILED TO LOAD", flush=True)
    else:
        used, free = vram()
        try:
            gpu, cpu = parse_layer_split(log.read_text(encoding="utf-8", errors="replace"))
        except ValueError as e:
            gpu = cpu = None
            print(f"    layer split unavailable: {e}")

        # Everything below can raise: a timeout, a reset connection, a server
        # that dies mid-request. Before 2026-08-21 the teardown sat at the end
        # of the happy path, so ANY raise left a 12 GB server resident and the
        # next queue step started into a GPU that was still full. The arm that
        # failed is one lost row; the arm it took down with it was avoidable.
        budget = completion_timeout_s(ctx)
        try:
            # Fill ~80% of the window; the rest is headroom for generation.
            prompt = filler(int(ctx * 0.8))

            t0 = time.time()
            samp = ({"temperature": 0.0, "top_k": 1, "seed": 42} if FIXED_TEXT
                    else {"temperature": 0.7})
            r = post("/completion", dict({"prompt": prompt, "n_predict": N_PREDICT,
                                          "cache_prompt": True}, **samp), timeout=budget)
            cold_wall = time.time() - t0
            t = r["timings"]

            # Warm decode: same prefix + a short suffix, so the deep prefill is reused.
            #
            # A request that generated nothing reports 0.0 tok/s. That is missing
            # data, not a slow result, and folding it into the sample is the same
            # plausible-wrong-number failure the harness exists to prevent: with 3
            # samples one zero survives, with two the median becomes 0. Take 5
            # attempts and drop the duds, counting them.
            def rate(tim):
                # Both guards are needed. Checking predicted_n alone let a 0.0 tok/s
                # sample through at ctx=65536 while empty_generations still read 0:
                # a rate of zero is missing data whatever the token count says.
                r = tim.get("predicted_per_second")
                return r if tim.get("predicted_n") and r and r > 0 else None

                warm = [rate(t)]
            # Fault 9: acceptance used to come from `t` alone -- the FIRST of
            # five generations -- while tg_med is the median of all five. The
            # two columns described different requests.
            all_timings = [t]
            for i in range(4):
                r2 = post("/completion",
                          dict({"prompt": prompt + chr(10) + "# note %d" % i + chr(10),
                                "n_predict": N_PREDICT, "cache_prompt": True}, **samp), timeout=budget)
                all_timings.append(r2["timings"])
                warm.append(rate(r2["timings"]))
            dropped = sum(1 for v in warm if v is None)
            warm = [v for v in warm if v is not None]
            if not warm:
                raise RuntimeError(f"ctx={ctx}: every generation produced 0 tokens")

            g = post("/completion", {"prompt": "def fibonacci(n):", "n_predict": 60,
                                     "temperature": 0.0, "top_k": 1, "seed": 42,
                                     "cache_prompt": False}, timeout=budget)
            row.update(
                loaded=True, gpu_layers=gpu, cpu_layers=cpu,
                vram_used=used, vram_free=free,
                prompt_n=t["prompt_n"], pp_tok_s=round(t["prompt_per_second"], 1),
                cold_prefill_s=round(t["prompt_ms"] / 1000, 1),
                cold_wall_s=round(cold_wall, 1),
                tg_med=round(median(warm), 2), tg_all=[round(v, 2) for v in warm],
                empty_generations=dropped,
                acceptance=draft_acceptance(all_timings),
                # kept so rows written before 2026-08-21 06:12 stay comparable
                acceptance_cold=(round(100.0 * t.get("draft_n_accepted", 0) / t["draft_n"], 1)
                                 if t.get("draft_n") else None),
                greedy_hash=hashlib.sha256(g["content"].encode()).hexdigest()[:16].upper(),
                **kv_buffers(log))
            print(f"  {label:<26} ctx={ctx:<7} gpu={gpu}/{cpu} pp={row['pp_tok_s']:<7} "
                  f"prefill={row['cold_prefill_s']}s tg={row['tg_med']:<6} "
                  f"kv={row.get('kv_mib','?')}MiB free={free}", flush=True)
        except (TimeoutError, ConnectionError, OSError) as e:
            # Record the abandonment as data. An arm too slow to finish inside
            # its depth budget is a RESULT -- that is exactly what `-ot` on
            # AD-IQ1_M was -- and a traceback that kills the queue is not.
            row.update(loaded=True, gpu_layers=gpu, cpu_layers=cpu,
                       vram_used=used, vram_free=free,
                       note="abandoned after %.0fs: %s: %s"
                            % (budget, type(e).__name__, e))
            print(f"  {label:<26} ctx={ctx:<7} ABANDONED "
                  f"({type(e).__name__} after {budget:.0f}s)", flush=True)
        finally:
            p.kill(); fh.close()

    with OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "f16"
    if len(sys.argv) > 2:                      # e.g. `depth_sweep.py f16 q3`
        MODEL = QUANTS[sys.argv[2]]
        print(f"quant: {sys.argv[2]} -> {MODEL}")
    quant = sys.argv[2] if len(sys.argv) > 2 else "q4"
    if "nospec" in sys.argv:
        USE_MTP = False
        print("speculation: OFF")
    spec = "mtp2" if USE_MTP else "nospec"

    if which == "deep":
        print("=== deep, F16 KV ===", flush=True)
        for ctx in (65536, 131072):
            run(ctx, [], f"F16 KV {quant}", f"{quant}-{spec}-deep-{ctx}")
    elif which == "f16":
        print("=== context depth, F16 KV ===", flush=True)
        for ctx in (65536, 131072, 262144):
            r = run(ctx, [], "F16 KV", f"{quant}-{spec}-f16-{ctx}")
            if not r.get("loaded"):
                print(f"  stopping: {ctx} did not load", flush=True)
                break
    else:
        print("=== context depth, Q8_0 KV ===", flush=True)
        for ctx in (65536, 131072, 262144):
            r = run(ctx, ["-ctk", "q8_0", "-ctv", "q8_0"], "Q8_0 KV",
                    f"{quant}-{spec}-q8-{ctx}")
            if not r.get("loaded"):
                print(f"  stopping: {ctx} did not load", flush=True)
                break
