"""Does pinning `-ngl` and disabling `--fit` even boot at the served window?

THE PRIZE. `CLAUDE.md` forbids comparing raw decode across boots because free
VRAM at boot moves 9,326-10,732 MiB and `--fit` follows it. The RTX 3090 scan
lists the counter-move among its never-set flags and rates it *"highest value on
this list for measurement integrity"*: give `-c` and `-ngl` explicit numbers and
`--fit` has nothing left to adjust (`common/fit.cpp` only touches arguments the
user did not set).

2026-08-23 produced direct evidence that this matters. Three `-ub 128` boots
logged byte-identical allocation -- same `n_ubatch`, same 428.27 MiB compute
buffer, same `projected to use 8827 MiB vs 10919`, same `will leave 2091 >= 768`
-- and yet `free_after`, sampled while the server was running, read 759, 757 and
**1,214 MiB**. The round with 457 MiB more spare ran 6 % faster. Nothing in the
experiment caused that; something else on the machine released memory.

WHY THIS IS A PREFLIGHT AND NOT THE SWEEP. Pinning removes llama.cpp's ability
to back off, so if `--fit` was quietly reducing anything, `-ngl 65 --fit off`
turns a silent adjustment into an OOM. That is the better failure -- but it is
worth one boot to find out before spending ten on a sweep that cannot start.

Boots each configuration once, reads the layer split, the buffer sizes and the
fitted context back out of the log, and stops.
"""
import re
import sys

import dflash2_arena as A

CTX = 98304
NGRAM = ["--spec-type", "ngram-mod"] + A.NGRAM

ARMS = [
    ("fit-auto-base", NGRAM),
    ("pinned",        NGRAM + ["-ngl", "65", "--fit", "off"]),
]


def grab(text, pattern, cast=str):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else None


ok = True
for name, extra in ARMS:
    p, fh, log, free_before = A.start(CTX, extra, f"pinpre-{name}", boot_s=600)
    if p is None:
        tail = log.read_text(encoding="utf-8", errors="replace")[-600:]
        print(f"{name:<15} BOOT FAILED\n  log: {log}\n  tail: {tail}", flush=True)
        ok = False
        continue
    free_after = A.vram()[1]
    A.stop_server()
    text = log.read_text(encoding="utf-8", errors="replace")

    # Regexes held in locals: an f-string expression may not contain a
    # backslash before Python 3.12, and this box runs 3.11.
    n_ctx = grab(text, r"llama_context: n_ctx += (\d+)")
    model_buf = grab(text, r"CUDA0 model buffer size = +([\d.]+) MiB")
    kv_buf = grab(text, r"CUDA0 KV buffer size = +([\d.]+) MiB")
    compute = grab(text, r"CUDA0 compute buffer size = +([\d.]+) MiB")
    fit_says = grab(text, r"(will leave \d+ >= \d+ MiB of free device memory)")
    split = A.parse_layer_split(text, expect_layers=A.TARGET_LAYERS)

    print(f"{name:<15} split={split[0]}+{split[1]}  n_ctx={n_ctx}", flush=True)
    print(f"{'':<15} model_buf={model_buf}  kv_buf={kv_buf}  compute={compute}",
          flush=True)
    print(f"{'':<15} free_before={free_before}  free_after={free_after}  "
          f"fit_says={fit_says}", flush=True)
    print(flush=True)

print("PREFLIGHT PASS -- both configurations boot; the sweep can proceed" if ok else
      "PREFLIGHT FAIL -- see above before spending a sweep")
sys.exit(0 if ok else 1)
