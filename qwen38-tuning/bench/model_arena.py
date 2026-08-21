"""Cross-model arena: compare quantization ARMS under the paired-boot design.

Why a new script rather than another sweep_runtime config: sweep_runtime varies
FLAGS on one model, so every arm shares a boot. Two different quantizations
cannot share a boot -- the weights differ -- so each arm change costs a restart,
and report 04 measured a 13.6 % peak-to-peak spread across restarts of an
UNCHANGED config. A control-first ordering here would measure that drift and
call it quantization.

The design alternates arms and pairs by round:

    round 1:  A  B  C
    round 2:  C  B  A      <- order reversed, so no arm always runs in the
    round 3:  A  B  C         same position within a round

Per-round representatives go to harness.paired_deltas, which refuses to call an
effect real unless it clears the drift floor AND keeps its sign in every round.

Residency is recorded, not assumed: the hypothesis under test is that a smaller
artifact crosses the VRAM threshold, and --fit on derives the split from free
VRAM at boot, which itself moved 9326-10530 MiB across recorded launches.

Usage:
    python model_arena.py --rounds 3
    python model_arena.py --rounds 1 --arms q4-tuned,q2kxl-nomtp
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness import median, parse_layer_split, paired_deltas

ROOT = Path(r"C:\AI\qwen38-tuning")
EXE = r"C:\AI\llama.cpp-cuda\llama-server.exe"
PORT = 8080
BASE_URL = "http://127.0.0.1:%d" % PORT

# Flags every arm shares. Each won its own sweep on Q4 (report 01); they are held
# CONSTANT here so the only variable across arms is the artifact itself.
COMMON = ["-c", "16384", "-ngl", "auto", "--fit", "on", "--fit-target", "768",
          "-fa", "on", "-np", "1", "-t", "18", "-b", "2048", "-ub", "256",
          "--no-mmproj-auto", "-lv", "5"]

MTP = ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]

def cached(repo, filename, size=None):
    """Resolve a repo + exact filename to its path in the Hugging Face cache.

    `size` is the exact byte count and it is NOT optional in spirit. Unsloth
    replaced every file in Qwen3.8-27B-GGUF in place on 2026-08-19 -- same
    filenames, new contents, new sizes -- so the cache now holds two snapshot
    directories with identical filenames inside. Returning `hits[0]` would let
    the pre-V3 arm and the V3 arm resolve to the same file and produce a
    beautifully paired comparison of an artifact against itself, with nothing in
    the output to show for it. This is the `PQ2_0` failure again, one directory
    level up.

    Raises on ambiguity rather than guessing. Returns None when the file is not
    downloaded yet, so an arm for a pending candidate reports FAILED TO START
    instead of crashing the sweep.
    """
    import glob
    import os
    root = os.path.expanduser("~/.cache/huggingface/hub/models--%s"
                              % repo.replace("/", "--"))
    hits = [h for h in glob.glob(os.path.join(root, "snapshots", "*", filename))
            if os.path.exists(h)]
    if size is not None:
        hits = [h for h in hits if os.path.getsize(h) == size]
    if not hits:
        return None
    if len(hits) > 1:
        raise ValueError(
            "%s/%s matches %d cached files (%s). Pass the exact byte count to "
            "disambiguate -- the repo was re-published with the same filenames."
            % (repo, filename, len(hits),
               ", ".join("%s=%d" % (os.path.basename(os.path.dirname(h)),
                                    os.path.getsize(h)) for h in hits)))
    return os.path.abspath(hits[0])


QWEN = "unsloth/Qwen3.8-27B-GGUF"
BONSAI = "prism-ml/Ternary-Bonsai-27B-gguf"

# The drafter is passed as a PATH. `-hfd repo:tag` resolved to an empty string
# and the server exited with `common_speculative_init_result: loading draft
# model ''` -- the same lesson as the target model, one flag later.
DSPARK = ["-md", cached(BONSAI, "Ternary-Bonsai-27B-dspark-Q4_1.gguf") or "",
          "--spec-type", "draft-dspark", "--spec-draft-n-max", "4",
          "-ngld", "999"]

# Arms carry their own repo. A single REPO constant would have resolved the
# Bonsai quant tag against the Qwen repo and fetched the wrong artifact.
# Every arm resolves to a PATH. `-hf repo:tag` was used until 2026-08-19, when
# it stalled an unattended queue for eleven minutes: llama.cpp performs an
# ONLINE etag check on each launch even for a fully cached file, and with a
# large download saturating the link it logged
# `common_pull_file: download failed ... retrying after 2 seconds` in a loop.
# A boot that needs no network should not be able to fail because of one.
# Pre-V3 byte counts, pinned so these arms can never silently resolve to the
# Dynamic V3 file of the same name.
_PREV3 = {
    "UD-Q4_K_XL": 17923394624,
    "UD-Q3_K_XL": 13441059904,
    "UD-Q2_K_XL": 10676423744,
    "UD-IQ2_XXS":  9010048064,
    "UD-IQ2_M":   10319907904,
    "UD-IQ3_XXS": 11913559104,
}
_Q = lambda fn: cached(QWEN, "Qwen3.8-27B-%s.gguf" % fn, _PREV3.get(fn))

ARMS = {
    "q4-tuned":      (None, _Q("UD-Q4_K_XL"), MTP),  # production control
    "q2kxl-nomtp":   (None, _Q("UD-Q2_K_XL"), []),   # research: test MTP off first
    "q2kxl-mtp2":    (None, _Q("UD-Q2_K_XL"), MTP),
    "iq2xxs-nomtp":  (None, _Q("UD-IQ2_XXS"), []),
    "iq2xxs-mtp2":   (None, _Q("UD-IQ2_XXS"), MTP),
    # Q2_g64, NOT Q2_0. Prism ships two ternary packs: Q2_0 is g128 and needs
    # their llama.cpp fork; Q2_g64 is the pack they describe as "matching the
    # 64-value-group Q2_0 packing in llama.cpp", which is what our build has.
    # Whether mainline actually serves it is the first thing these arms test.
    # Fetched by exact filename, never by quant tag: `-hf repo:Q2_0` matches
    # PQ2_0 by substring, and the two files are byte-for-byte the same size.
    "bonsai-nospec": (None, cached(BONSAI, "Ternary-Bonsai-27B-Q2_g64.gguf"), []),
    "bonsai-dspark": (None, cached(BONSAI, "Ternary-Bonsai-27B-Q2_g64.gguf"), DSPARK),
    # Artifacts fetched with `hf download` by exact filename are addressed by
    # PATH (repo=None -> -m). A `-hf repo:tag` reference would risk a second
    # full download of a file already on disk, and tag matching is by substring
    # -- which is how `:Q2_0` started fetching `PQ2_0.gguf` earlier today.
    "iq1m-nomtp":    (None, 'C:\\Users\\xenod\\.cache\\huggingface\\hub\\models--AtomicChat--Qwen3.8-27B-GGUF\\snapshots\\ca10ebceb1887be9d33b838770a36b39d75a8a4c\\Qwen3.8-27B-AD-IQ1_M.gguf', []),
    "ornith9b-nomtp":(None, 'C:\\Users\\xenod\\.cache\\huggingface\\hub\\models--ornith-ai--Ornith-1.0-9B-GGUF\\snapshots\\3296bc7a404871a72ac3f1903f561459c09b5c17\\ornith-1.0-9b-Q6_K.gguf', []),
    "bonsai-g64":    (None, 'C:\\Users\\xenod\\.cache\\huggingface\\hub\\models--prism-ml--Ternary-Bonsai-27B-gguf\\snapshots\\abbae723028d71be674e71e1a71201a6f43fab22\\Ternary-Bonsai-27B-Q2_g64.gguf', []),

    # --- resolved from the cache at import time; None until downloaded, which
    # the arena reports as FAILED TO START instead of crashing the sweep ---

    # Same size class as Unsloth's UD-IQ2_XXS (8,561 vs 8,592 MiB), different
    # quantizer. The research asked for exactly this comparison and assumed it
    # needed a second vendor's 9.9 GB file; it does not.
    "adiq2xxs":      (None, cached("AtomicChat/Qwen3.8-27B-GGUF",
                                   "Qwen3.8-27B-AD-IQ2_XXS.gguf"), []),
    "iq2m-nomtp":    (None, _Q("UD-IQ2_M"), []),
    "iq3xxs-nomtp":  (None, _Q("UD-IQ3_XXS"), []),

    # MoE. 35B total, ~3B active per token. The research assumed these needed
    # CPU expert offload at ~20 GiB; Unsloth's low-bit builds are 9.4-10.7 GiB,
    # so the first question is simply whether they are RESIDENT -- 35B of
    # capacity doing 3B of work per token with no PCIe traffic is a regime
    # nothing else measured here can imitate. `-ncmoe` arms test the fallback.
    "qwen36moe":     (None, cached("unsloth/Qwen3.6-35B-A3B-GGUF",
                                   "Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf"), []),
    "qwen36moe-iq1": (None, cached("unsloth/Qwen3.6-35B-A3B-GGUF",
                                   "Qwen3.6-35B-A3B-UD-IQ1_M.gguf"), []),
    "qwen36moe-cpu": (None, cached("unsloth/Qwen3.6-35B-A3B-GGUF",
                                   "Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf"),
                      ["--n-cpu-moe", "34"]),
    "ornith35moe":   (None, cached("unsloth/Ornith-1.0-35B-GGUF",
                                   "Ornith-1.0-35B-UD-IQ2_XXS.gguf"), []),
    "ornith35-iq1s": (None, cached("unsloth/Ornith-1.0-35B-GGUF",
                                   "Ornith-1.0-35B-UD-IQ1_S.gguf"), []),
    "ornith35-cpu":  (None, cached("unsloth/Ornith-1.0-35B-GGUF",
                                   "Ornith-1.0-35B-UD-IQ2_XXS.gguf"),
                      ["--n-cpu-moe", "34"]),

    "ornith9b-q8":   (None, cached("ornith-ai/Ornith-1.0-9B-GGUF",
                                   "ornith-1.0-9b-Q8_0.gguf"), []),
    # Prism's phone-class operating point: ternary at 1.125 bpw, 3.54 GiB.
    "bonsai-1bit":   (None, cached("prism-ml/Bonsai-27B-gguf",
                                   "Bonsai-27B-Q1_0.gguf"), []),
    "gptoss20b":     (None, cached("unsloth/gpt-oss-20b-GGUF",
                                   "gpt-oss-20b-Q4_K_M.gguf"), []),

    # --- Unsloth Dynamic V3, published 2026-08-19T16:39:23Z, mid-session ---
    # Every arm above this line is the PRE-V3 build of the same repo. The files
    # were replaced in place: same names, different sizes and OIDs. Keeping both
    # generations as separate arms is the only way the comparison stays honest,
    # and it is why nothing here is addressed by `-hf repo:tag`.
    #
    #     UD-IQ2_XXS   9,010,048,064 -> 7,266,070,528
    #     UD-Q2_K_XL  10,676,423,744 -> 9,828,981,664
    #     UD-Q4_K_XL  17,923,394,624 -> 17,559,178,144
    #     UD-IQ2_M     deleted; UD-IQ2_S added
    #
    # V3 also ships the 1-bit builds this project spent a day hunting for. At
    # 5.77 GiB, IQ1_S leaves ~4.6 GB of headroom against the 1.18 GB the current
    # production artifact leaves -- and four CPU-resident layers were measured to
    # cost half the decode throughput on this card.
    "v3-iq1s":   (None, cached(QWEN, "Qwen3.8-27B-UD-IQ1_S.gguf",  6192222208), []),
    "v3-iq1m":   (None, cached(QWEN, "Qwen3.8-27B-UD-IQ1_M.gguf",  6729166848), []),
    "v3-iq2xxs": (None, cached(QWEN, "Qwen3.8-27B-UD-IQ2_XXS.gguf", 7266070528), []),
    "v3-iq2s":   (None, cached(QWEN, "Qwen3.8-27B-UD-IQ2_S.gguf",  8371970048), []),
    "v3-q2kxl":  (None, cached(QWEN, "Qwen3.8-27B-UD-Q2_K_XL.gguf", 9828981664), []),
    "v3-q4kxl":  (None, cached(QWEN, "Qwen3.8-27B-UD-Q4_K_XL.gguf", 17559178144), MTP),
}


CODE_PROMPT = """Here is a Python class:

class LRUCache:
    def __init__(self, cap):
        self.cap = cap
        self.data = {}
        self.order = []
    def get(self, k):
        if k not in self.data:
            return None
        self.order.remove(k)
        self.order.append(k)
        return self.data[k]
    def put(self, k, v):
        if k in self.data:
            self.order.remove(k)
        elif len(self.data) >= self.cap:
            victim = self.order.pop()
            del self.data[victim]
        self.data[k] = v
        self.order.append(k)

Rewrite this class exactly as given, but rename the attribute "order" to "usage" everywhere. Output the full class."""


def post(path, payload, timeout=1800):
    req = urllib.request.Request(BASE_URL + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def kill_server():
    cmd = ("$c=Get-NetTCPConnection -LocalPort %d -State Listen "
           "-ErrorAction SilentlyContinue; if($c){Stop-Process -Id "
           "$c.OwningProcess -Force}" % PORT)
    subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True)
    time.sleep(6)


def vram():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return [int(x) for x in out.split(",")]


def start(tag, repo, quant, extra, round_no):
    kill_server()
    free_before = vram()[1]
    log = ROOT / "logs" / ("arena-r%d-%s.log" % (round_no, tag))
    if not repo and quant is None:
        # cached() found nothing: the artifact is still downloading. Report it
        # as a non-start rather than passing the literal string "None" to -m,
        # which llama.cpp would treat as a filename and fail on obscurely.
        fh = log.open("w", encoding="utf-8")
        fh.write("artifact not present in the Hugging Face cache yet\n")
        fh.close()
        return None, None, log, free_before
    ref = ["-hf", repo + ":" + quant] if repo else ["-m", quant]
    args = ([EXE] + ref + ["--alias", tag] + COMMON + extra +
            ["--host", "127.0.0.1", "--port", str(PORT)])
    fh = log.open("w", encoding="utf-8", errors="replace")
    p = subprocess.Popen(args, stdout=fh, stderr=subprocess.STDOUT)
    for _ in range(90):
        time.sleep(4)
        try:
            urllib.request.urlopen(BASE_URL + "/health", timeout=3).read()
            break
        except Exception:
            if p.poll() is not None:
                fh.close()
                return None, None, log, free_before
    else:
        fh.close()
        return None, None, log, free_before
    time.sleep(2)
    fh.flush()
    return p, fh, log, free_before


def measure(reps):
    """Decode on the code-rewrite prompt -- the probe every tuning decision in
    this project was read from. An 11-token prompt stayed inside 9.86-11.90
    tok/s across EVERY configuration tested and would resolve nothing (04 s3)."""
    tg, pp, dn, da = [], [], 0, 0
    for _ in range(reps):
        r = post("/completion", {"prompt": CODE_PROMPT, "n_predict": 160,
                                 "temperature": 0.7, "cache_prompt": False})
        t = r["timings"]
        if t["predicted_per_second"] <= 0:
            raise ValueError("zero decode rate -- a dead sample, not a slow one")
        tg.append(t["predicted_per_second"])
        pp.append(t["prompt_per_second"])
        dn += int(t.get("draft_n", 0))
        da += int(t.get("draft_n_accepted", 0))
    g = post("/completion", {"prompt": CODE_PROMPT, "n_predict": 120,
                             "temperature": 0.0, "top_k": 1, "seed": 42,
                             "cache_prompt": False})
    h = hashlib.sha256(g["content"].encode()).hexdigest()[:16].upper()
    return tg, pp, dn, da, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--arms", default="q4-tuned,q2kxl-nomtp,q2kxl-mtp2")
    ap.add_argument("--out", default="arena-quant.jsonl")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",")]
    for a in arms:
        if a not in ARMS:
            sys.exit("unknown arm %r; known: %s" % (a, ", ".join(ARMS)))

    out = ROOT / "results" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    per_round = dict((a, []) for a in arms)

    for rnd in range(1, args.rounds + 1):
        order = arms if rnd % 2 else list(reversed(arms))
        print("\n===== round %d : %s =====" % (rnd, " -> ".join(order)), flush=True)
        for tag in order:
            repo, quant, extra = ARMS[tag]
            p, fh, log, free_before = start(tag, repo, quant, extra, rnd)
            if p is None:
                print("  %-14s FAILED TO START -- see %s" % (tag, log.name), flush=True)
                continue
            used, free = vram()
            try:
                gpu, cpu = parse_layer_split(log.read_text(encoding="utf-8", errors="replace"))
            except ValueError as e:
                gpu = cpu = None
                print("    layer split unavailable: %s" % e, flush=True)
            try:
                tg, pp, dn, da, h = measure(args.reps)
            except Exception as e:
                print("  %-14s MEASURE FAILED: %s" % (tag, e), flush=True)
                p.kill()
                fh.close()
                continue

            row = dict(round=rnd, arm=tag, repo=repo, quant=quant, mtp=bool(extra),
                       vram_free_before=free_before, vram_used=used, vram_free=free,
                       gpu_layers=gpu, cpu_layers=cpu,
                       code_tok_s=[round(v, 2) for v in sorted(tg)],
                       code_median=round(median(tg), 2),
                       pp_median=round(median(pp), 1),
                       draft_n=dn, draft_accepted=da,
                       accept_pct=round(100.0 * da / dn, 1) if dn else None,
                       greedy_hash=h)
            per_round[tag].append(row["code_median"])
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print("  %-14s split %s+%s  free_before %s  free %s  code %s %s  "
                  "pp %s  acc %s  %s"
                  % (tag, gpu, cpu, free_before, free, row["code_median"],
                     row["code_tok_s"], row["pp_median"], row["accept_pct"], h),
                  flush=True)
            p.kill()
            fh.close()

    kill_server()
    print("\n===== PAIRED RESULT (baseline = %s) =====" % arms[0])
    base = per_round[arms[0]]
    for tag in arms[1:]:
        if not base or len(per_round[tag]) != len(base):
            print("  %-14s unpaired (%d vs %d rounds) -- not comparable"
                  % (tag, len(base), len(per_round[tag])))
            continue
        d = paired_deltas(base, per_round[tag])
        verdict = "RESOLVED" if d["resolved"] else "under drift floor / inconsistent sign"
        print("  %-14s per-round %s  mean %+.2f%%  range %+.2f..%+.2f%%  -> %s"
              % (tag, d["per_round_pct"], d["mean_pct"], d["min_pct"],
                 d["max_pct"], verdict))
        rec = dict(kind="PAIRED", baseline=arms[0], arm=tag,
                   baseline_rounds=base, candidate_rounds=per_round[tag])
        rec.update(d)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
