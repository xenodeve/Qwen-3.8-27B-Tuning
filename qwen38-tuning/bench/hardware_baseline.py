"""One boot, everything a new card changes, before any sweep is paid for.

WHY THIS EXISTS. On 2026-08-23 the RTX 4070 SUPER 12 GB was replaced with an
RTX 5060 Ti 16 GB. Every number this project holds -- 96.92 tok/s at ctx 98,304,
the 45-376 MiB band where DFlash2 becomes unreliable, the 13.6 % noise floor,
`11,069 MiB free` reported identically in all 552 logs -- was measured on Ada
with 12 GB. `CLAUDE.md` forbids comparing raw decode across BOOTS; across
hardware it is not a comparison at all.

Three things could each invalidate a sweep before it starts, and one boot
answers all three:

  1. The binary is built with CMAKE_CUDA_ARCHITECTURES=89 (Ada). The card is
     compute capability 12.0 (Blackwell). If there is no SASS and no usable
     PTX, it does not run; if it JIT-compiles, the first boot is slow and the
     kernel path may differ. Boot time and a real decode answer this.
  2. Free VRAM as llama.cpp sees it is NOT nvidia-smi's number -- CORRECTIONS 27,
     confirmed again on this card with a game running: nvidia-smi said 7,682 MiB
     free while llama.cpp reported 15,172. Only the log line counts.
  3. `--fit` acted on 2 of 150 boots on the old card. With 4 GB more headroom it
     should act even less, but "should" is not a measurement.

Prints one block per depth and exits non-zero if anything is off. It does not
sweep and it does not compare -- it establishes what the new machine reports.
"""
import argparse
import json
import re
import sys
import time
import urllib.request

import dflash2_arena as A
from harness import generation_is_measurable

ENDPOINT = "http://127.0.0.1:8080/completion"

# Regexes as module constants: an f-string expression may not contain a
# backslash before Python 3.12 and this box runs 3.11.
RX = {
    "device":   r"CUDA0\s+:\s+(.+?)\s+\((\d+) MiB, (\d+) MiB free\)",
    "n_ctx":    r"llama_context: n_ctx +searchable",
    "ctx":      r"llama_context: n_ctx += (\d+)",
    "ubatch":   r"llama_context: n_ubatch += (\d+)",
    "rs_seq":   r"llama_context: n_rs_seq += (\d+)",
    "model":    r"CUDA0 model buffer size = +([\d.]+) MiB",
    "cpu_map":  r"CPU_Mapped model buffer size = +([\d.]+) MiB",
    "kv":       r"CUDA0 KV buffer size = +([\d.]+) MiB",
    "rs":       r"CUDA0 RS buffer size = +([\d.]+) MiB",
    "compute":  r"CUDA0 compute buffer size = +([\d.]+) MiB",
    "fit":      r"common_params_fit_impl: (will leave \d+ >= \d+ MiB[^\n]*|cannot meet[^\n]*)",
    "cram":     r"prompt cache is (enabled, size limit: \d+ MiB|disabled)",
}


def grab(text, key, last=False):
    ms = re.findall(RX[key], text)
    if not ms:
        return None
    return ms[-1] if last else ms[0]


def gen(prompt, n_predict):
    body = json.dumps({
        "prompt": prompt, "n_predict": n_predict, "temperature": 0.0,
        "top_k": 1, "seed": 42, "cache_prompt": True, "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(ENDPOINT, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read().decode())["timings"], time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, nargs="+", default=[98304])
    ap.add_argument("--chars", type=int, default=150000,
                    help="prompt size in CHARACTERS, not tokens")
    ap.add_argument("--n-predict", type=int, default=512)
    args = ap.parse_args()

    text = (A.CORPUS_DIR / A.CORPUS_FILES["real-code-deep"]).read_text(
        encoding="utf-8", errors="replace")[:args.chars]
    ok = True

    for ctx in args.ctx:
        t_boot = time.time()
        p, fh, log, free_smi = A.start(
            ctx, ["--spec-type", "ngram-mod"] + A.NGRAM, f"hwbase-{ctx}",
            boot_s=900)
        boot_s = time.time() - t_boot
        if p is None:
            print(f"ctx {ctx:,}: BOOT FAILED after {boot_s:.1f}s -- {log}")
            ok = False
            continue

        timings, wall = gen(text, args.n_predict)
        measurable = generation_is_measurable([timings], n_predict=args.n_predict)
        free_after_smi = A.vram()[1]
        A.stop_server()

        raw = log.read_text(encoding="utf-8", errors="replace")
        dev = re.search(RX["device"], raw)
        split = A.parse_layer_split(raw, expect_layers=A.TARGET_LAYERS)
        depth = timings["prompt_n"] + timings["cache_n"]
        tg = timings.get("predicted_per_second")

        print(f"\n===== ctx {ctx:,} =====")
        if dev:
            print(f"  device            {dev.group(1)}")
            print(f"  VRAM as llama.cpp sees it   total {int(dev.group(2)):,} MiB, "
                  f"free {int(dev.group(3)):,} MiB")
        print(f"  VRAM as nvidia-smi saw it   free {free_smi:,} -> {free_after_smi:,} MiB")
        print(f"  boot wall         {boot_s:.1f} s   (JIT would be minutes, not seconds)")
        print(f"  split             {split[0]}+{split[1]}")
        print(f"  n_ctx/n_ubatch/n_rs_seq   {grab(raw,'ctx')} / {grab(raw,'ubatch')} / {grab(raw,'rs_seq')}")
        print(f"  model / CPU_Mapped        {grab(raw,'model',last=True)} / {grab(raw,'cpu_map')} MiB")
        print(f"  KV / RS / compute         {grab(raw,'kv',last=True)} / "
              f"{grab(raw,'rs',last=True)} / {grab(raw,'compute',last=True)} MiB")
        print(f"  fit               {grab(raw,'fit',last=True)}")
        print(f"  prompt cache      {grab(raw,'cram')}")
        print(f"  prompt depth      {depth:,} tokens, {timings['prompt_n']:,} evaluated")
        print(f"  prefill           {timings['prompt_ms']:,.1f} ms")
        print(f"  decode            {tg} tok/s   predicted_n={timings.get('predicted_n')}   "
              f"measurable={measurable}")
        if split[1] != 0:
            print(f"  *** {split[1]} layers on CPU -- not a clean baseline ***")
            ok = False
        if not measurable:
            print("  *** generation too short to measure ***")
            ok = False

    print("\nBASELINE READ OK" if ok else "\nBASELINE HAS PROBLEMS -- read above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
