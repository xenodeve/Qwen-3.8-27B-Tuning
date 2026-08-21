"""DFlash2 against the incumbent, paired within a round, on ONE binary.

THE QUESTION. `ngram-mod` is this project's speculation champion because it
costs 0 MiB. DFlash2 costs 1.1 GB of a 12 GB card. Public numbers are all from
bigger hardware -- atomic.chat's 47.4 -> 140.6 tok/s is an RTX 6000 -- so the
only way to know what it does here is to run it here.

WHY EVERY ARM RUNS ON BUILD 10499, INCLUDING THE ONES ALREADY MEASURED.
DFlash2 needs the PR build; `ngram-mod`'s recorded numbers are from 10472.
Comparing across those two would confound the decoder with the build, and
nothing in the result could separate them. So the incumbent is re-measured
here, on the same binary, in the same rounds.

WHY ARMS ALTERNATE. Free VRAM at boot moves 9,326-10,732 MiB and `--fit`
follows it; raw decode is not comparable across boots. Arms are run
round-robin and rotated each round, and the verdict comes from
`harness.paired_deltas`, which reports a range and refuses to call an effect
resolved unless it clears 13.6 % AND keeps its sign.

WHY n_predict IS NOT 160. Every decoder verdict this project holds was decided
on 160-token generations, and whether that understates speculation is an OPEN
question in the ledger (CORRECTIONS 8). Measuring a drafter with the budget
that is under suspicion for hiding drafters would be walking into a trap we
wrote down ourselves.

WHY THE LAYER SPLIT IS PARSED WITH expect_layers. A drafter adds its own
assignment passes and it is assigned LAST, so the default read of this log
returns the drafter's (6, 0) -- a healthy-looking split for the wrong model, in
which a spill of the target's 65 layers cannot appear. Issue #17.

WHAT THIS CANNOT TELL YOU. Throughput, not task success. This project's metric
is verified accepted coding tasks per hour, and a decoder whose output is
byte-identical to no-speculation (ngram) and one whose output is not are not
interchangeable on that metric just because tok/s says so.
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness import (median, parse_layer_split, draft_acceptance,
                     paired_deltas, vram_settled, VRAM_MIN_RISE_MIB)

ROOT = Path(r"C:\AI\qwen38-tuning")
EXE = r"C:\AI\llama.cpp-dflash2\llama-server.exe"
TARGET = (r"C:\Users\xenod\.cache\huggingface\hub\models--unsloth--Qwen3.8-27B-GGUF"
          r"\snapshots\27af057ecb382ddfea5d12837360a8980560e3ed"
          r"\Qwen3.8-27B-UD-IQ2_XXS.gguf")
DRAFTER = (r"C:\Users\xenod\.cache\huggingface\hub"
           r"\models--z-lab--Qwen3.8-27B-DFlash2-GGUF"
           r"\snapshots\57ab3265056d4024870b0621cfc2c127537020ed"
           r"\Qwen3.8-27B-DFlash2-Q4_K_M.gguf")
BASE = "http://127.0.0.1:8080"
TARGET_LAYERS = 65          # Qwen3.8-27B: 64 blocks plus the MTP head
N_PREDICT = 512
N_GEN = 3                   # timed generations per arm per round, after a warm turn

# ngram-mod's tuned window, copied from worker-iq2xxs-deep.ps1 so the incumbent
# is measured as it is actually served rather than at defaults.
NGRAM = ["--spec-ngram-mod-n-match", "12",
         "--spec-ngram-mod-n-min", "16", "--spec-ngram-mod-n-max", "32"]
DFLASH = ["-md", DRAFTER, "--spec-draft-n-max", "4", "-ngld", "99"]

ARMS = [
    ("none", []),
    ("ngram-mod", ["--spec-type", "ngram-mod"] + NGRAM),
    ("dflash2", ["--spec-type", "draft-dflash"] + DFLASH),
    # Verified supported by reading common/arg.cpp:4155 -- --spec-type is a
    # comma-separated list -- not by trusting a forum post.
    ("dflash2+ngram", ["--spec-type", "draft-dflash,ngram-mod"] + DFLASH + NGRAM),
]


def vram():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip()
    return [int(x) for x in o.split(",")]


def port_owner():
    """PID listening on the arena port, or None. Reads only, never stops it."""
    r = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "$c = Get-NetTCPConnection -LocalPort 8080 -State Listen "
         "-ErrorAction SilentlyContinue; if ($c) { $c.OwningProcess }"],
        capture_output=True, text=True)
    out = (r.stdout or "").strip().splitlines()
    return int(out[0]) if out and out[0].strip().isdigit() else None


def require_exclusive_port():
    """Refuse to run while another orchestrator holds the port.

    CLAUDE.md: "Two orchestrators cannot share port 8080. An armed queue once
    killed a running corpus and the summary still printed a plausible number."
    On 2026-08-22 this arena reproduced that -- a second run was launched while
    the first was finishing, and the older teardown killed the younger server.
    The younger log ends mid-load with no error, because it did not fail.

    It failed loudly only because the kill landed during a load. Landing
    between generations it would have produced a short arm with a believable
    rate, which is the failure this project exists to refuse.
    """
    pid = port_owner()
    if pid is not None:
        raise RuntimeError(
            "port 8080 is already held by pid %d. Another orchestrator is "
            "running; stop it before starting a measurement, or this run and "
            "that one will kill each other's servers." % pid)


def kill():
    """Stop any llama-server. Returns True if one was actually running.

    The caller needs to know: a floor for the settle check is only meaningful
    when something WAS resident. Applying it when nothing was running demands a
    rise that cannot happen, and the wait times out every time -- which is what
    the first version of this file did.
    """
    # Exit code, not the message: taskkill returns 0 when it killed something
    # and 128 when the image was not found, and that holds whatever language
    # the machine reports errors in. Matching on "SUCCESS" would return False
    # on a localised Windows and silently skip every settle wait -- the same
    # fault, quieter.
    r = subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                       capture_output=True, text=True)
    return r.returncode == 0


def wait_for_vram_release(floor_mib=None, limit_s=90, poll_s=3):
    """WDDM frees a 12 GB allocation in stages.

    Starting the next arm too early passes /health against memory the driver
    still holds, then dies on the first /completion -- instrument fault 7.

    Delegates to harness.vram_settled rather than comparing readings here. The
    first version of this function did compare them, and lost `floor_mib` in
    the process: "stopped moving" alone cannot tell *release finished* from
    *release has not begun*, because two polls taken before the driver does
    anything agree perfectly. The floor is the free reading taken WHILE the
    model was still resident, plus the minimum rise a real teardown must
    produce -- which a release that never started cannot reach.
    """
    readings = []
    for _ in range(int(limit_s / poll_s)):
        time.sleep(poll_s)
        readings.append(vram()[1])
        if vram_settled(readings, floor_mib=floor_mib):
            return readings
    print("    VRAM still moving after %ds: %s" % (limit_s, readings[-4:]),
          flush=True)
    return readings


def post(path, payload, timeout):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


_BLOCK = ("// section {i}\n"
          "static int helper_{i}(int a, int b) {{\n"
          "    int acc = a;\n"
          "    for (int k = 0; k < b; ++k) acc += (k ^ a) % 7;\n"
          "    return acc;\n"
          "}}\n\n")


# Real source from this repo. A coding worker reads real files, and real files
# barely repeat: measured with harness.line_repetition_pct, harness.py is 4.8 %
# duplicate lines and opencode_corpus.py is 0.6 %, against 66.2 % for the
# synthetic blocks above.
REAL_SOURCES = ["harness.py", "depth_sweep.py", "model_arena.py",
                "opencode_corpus.py", "kv_sweep.py"]


def filler(n_tokens, regime="synthetic"):
    """Roughly n_tokens of prompt, identical for every arm.

    A drafter's acceptance depends on how predictable the text is, so varying
    the prompt between arms would measure the text instead of the decoder.

    TWO REGIMES, BECAUSE THE ARMS DO DIFFERENT JOBS. `ngram-mod` drafts by
    matching text it has already seen in the context; it is cheap and strong
    exactly where the answer is already on screen, and has nothing to offer
    where the model is writing something new. DFlash2 is a trained drafter and
    does not need to have seen the text before. Measuring only one regime
    measures one of them at its best and the other away from it.

      synthetic  66.2 % duplicate lines -- ngram-mod's best case, and the trap
                 depth_sweep.py already names in its own header: "the sweep
                 prompt is 84.5 % duplicate lines, so treat the smaller number
                 as the real one".
      real-code  ~4 % duplicate lines -- this repo's own source, which is what
                 the worker actually reads.

    A verdict from one regime does not carry to the other, the same way a
    verdict at one depth does not carry to another depth.
    """
    if regime == "real-code":
        parts, budget = [], n_tokens * 3
        for name in REAL_SOURCES:
            path = Path(__file__).parent / name
            if path.is_file():
                parts.append(path.read_text(encoding="utf-8", errors="replace"))
            if sum(len(x) for x in parts) >= budget:
                break
        text = "".join(parts)[:budget]
        return text + ("\n# Explain what vram_settled guards against, "
                       "then write a test for it.\n")

    out, i = [], 0
    while sum(len(s) for s in out) < n_tokens * 3:
        out.append(_BLOCK.format(i=i))
        i += 1
    return "".join(out) + "\n// Explain what helper_3 computes, then rewrite it.\n"


def stop_server():
    """Kill the server AND wait for the driver to hand the memory back.

    One function, because separating them is how the wait died: run_arm's
    `finally` called kill(), so start()'s kill() then found nothing to kill,
    returned False, and skipped the wait for every arm in the run. The guard
    was present, called, and inert.

    Reading free VRAM before the kill is what makes the floor mean anything --
    it is how much has to come back.
    """
    resident_free = vram()[1]
    if not kill():
        return                      # nothing was running; nothing to wait for
    wait_for_vram_release(floor_mib=resident_free + VRAM_MIN_RISE_MIB)


def start(ctx, extra, tag, boot_s=240):
    stop_server()
    free_before = vram()[1]
    log = ROOT / "logs" / ("dflash2-" + tag + ".log")
    args = [EXE, "-m", TARGET, "--alias", "qwen38", "-c", str(ctx),
            "-ngl", "auto", "--fit", "on", "--fit-target", "768", "-fa", "on",
            "-np", "1", "-t", "18", "-b", "2048", "-ub", "256",
            "-ctk", "q4_0", "-ctv", "q4_0",
            "--no-mmproj-auto", "-lv", "5",
            "--host", "127.0.0.1", "--port", "8080"] + extra
    fh = log.open("w", encoding="utf-8", errors="replace")
    p = subprocess.Popen(args, stdout=fh, stderr=subprocess.STDOUT)
    for _ in range(boot_s // 3):
        time.sleep(3)
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


def rate(t):
    """A rate of zero is missing data whatever the token count says."""
    r = t.get("predicted_per_second")
    return r if t.get("predicted_n") and r and r > 0 else None


def run_arm(ctx, label, extra, rnd, regime="synthetic"):
    tag = (label.replace("+", "-") + "-" + regime
           + "-c" + str(ctx) + "-r" + str(rnd))
    p, fh, log, free_before = start(ctx, extra, tag)
    row = dict(ctx=ctx, arm=label, round=rnd, regime=regime,
               args=" ".join(extra),
               n_predict=N_PREDICT, free_before=free_before,
               loaded=p is not None)
    if p is None:
        row["note"] = "server failed to start"
        print("    %-15s FAILED TO LOAD" % label, flush=True)
        return row

    try:
        prompt = filler(int(ctx * 0.5), regime)
        samp = {"temperature": 0.0, "top_k": 1, "seed": 42}
        # Warm turn: pays the cold prefill once so the timed generations measure
        # decode rather than prefill. Its timings are discarded deliberately.
        #
        # FULL LENGTH, not a token or two. A 16-token warm turn paid the prefill
        # but left the n-gram table nearly empty, and the first TIMED generation
        # of every ngram arm then came in 35-40 % low -- 69.8 against 113.4 and
        # 114.2 in the same boot, while dflash2 and none showed no such step
        # because neither builds a table from the text it just emitted. Median
        # of three mostly absorbed it, which is precisely why it could have gone
        # unnoticed: a systematic bias hidden inside a summary statistic.
        post("/completion",
             dict({"prompt": prompt, "n_predict": N_PREDICT, "cache_prompt": True},
                  **samp),
             timeout=900)
        timings, rates = [], []
        for _ in range(N_GEN):
            r = post("/completion",
                     dict({"prompt": prompt, "n_predict": N_PREDICT,
                           "cache_prompt": True}, **samp),
                     timeout=900)
            timings.append(r["timings"])
            rates.append(rate(r["timings"]))
        good = [x for x in rates if x]
        row.update(tg_samples=rates,
                   tg_med=median(good) if good else None,
                   acceptance=draft_acceptance(timings))
        fh.flush()
        text = log.read_text(encoding="utf-8", errors="replace")
        row["split"] = "%d+%d" % parse_layer_split(text, expect_layers=TARGET_LAYERS)
        row["free_after"] = vram()[1]
        print("    %-15s %6.2f tok/s   split %-6s acc %s"
              % (label, row["tg_med"] or 0.0, row["split"], row["acceptance"]),
              flush=True)
    except Exception as exc:               # a failed arm is a row, not a crash
        row["note"] = "%s: %s" % (type(exc).__name__, exc)
        print("    %-15s ERROR %s" % (label, exc), flush=True)
    finally:
        stop_server()
        fh.close()
    return row


def report(rows):
    by_ctx = {}
    for r in rows:
        by_ctx.setdefault(r["ctx"], []).append(r)
    for ctx, rs in sorted(by_ctx.items()):
        print("\nctx=%d" % ctx)
        series = {}
        for label, _ in ARMS:
            vals = [r["tg_med"] for r in rs
                    if r["arm"] == label and r.get("tg_med")]
            if vals:
                series[label] = vals
        if "ngram-mod" not in series:
            print("  no ngram-mod rounds -- nothing to pair against")
            continue
        base = series["ngram-mod"]
        for label, vals in series.items():
            shown = [round(v, 1) for v in vals]
            if label == "ngram-mod":
                print("  %-15s %s  (baseline)" % (label, shown))
                continue
            if len(vals) != len(base):
                print("  %-15s %s  NOT PAIRED (%d vs %d rounds) -- no verdict"
                      % (label, shown, len(vals), len(base)))
                continue
            d = paired_deltas(base, vals)
            verdict = "RESOLVED" if d["resolved"] else "within noise / inconsistent"
            print("  %-15s %s  %+.1f%% [%+.1f, %+.1f]  %s"
                  % (label, shown, d["mean_pct"], d["min_pct"], d["max_pct"],
                     verdict))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, nargs="+", default=[16384])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--regime", choices=["synthetic", "real-code"],
                    nargs="+", default=["synthetic"])
    ap.add_argument("--out",
                    default=str(ROOT / "results" / "dflash2-arena.jsonl"))
    a = ap.parse_args()

    require_exclusive_port()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with out.open("a", encoding="utf-8") as fh:
        for ctx in a.ctx:
          for regime in a.regime:
            for rnd in range(1, a.rounds + 1):
                # Rotate so no arm always runs first: within a boot the earlier
                # arm sees a cleaner GPU, and a fixed order hands that advantage
                # to the same arm every round.
                k = (rnd - 1) % len(ARMS)
                order = ARMS[k:] + ARMS[:k]
                print("  ctx=%d %s round %d: %s"
                      % (ctx, regime, rnd,
                         " -> ".join(l for l, _ in order)), flush=True)
                for label, extra in order:
                    row = run_arm(ctx, label, extra, rnd, regime)
                    rows.append(row)
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()

    print("\nwrote %d rows to %s" % (len(rows), out))
    report(rows)


if __name__ == "__main__":
    sys.exit(main())
