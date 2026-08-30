"""Prove `-ub` in `extra` actually reaches the context before sweeping it.

server_argv() hardcodes `-ub 256` and appends the arm's extra after it, so every
arm in the `ubatch` set carries the flag twice. `tests/test_ubatch_arm_set.py`
pins the argv ordering; it cannot pin llama.cpp's behaviour. Only a boot can.

If the parser kept the FIRST occurrence, all three arms would run at 256, the
sweep would come back flat, and flat would be written up as "-ub has no effect"
-- the `--spec-ngram-mod-n-min` failure repeated: twelve boots, a plausible
spread, no effect present to find.

Boots once per value, reads `llama_context: n_ubatch = N` and the compute buffer
line out of the log, and stops. Exits non-zero the moment a boot reports a value
it was not asked for.
"""
import re
import sys

import dflash2_arena as A

CTX = 98304
WANT = [256, 128, 64]
BASE = ["--spec-type", "ngram-mod"] + A.NGRAM   # no sidecar: this is about -ub

ok = True
for ub in WANT:
    p, fh, log, free_before = A.start(CTX, BASE + ["-ub", str(ub)],
                                      f"ubpre-{ub}", boot_s=600)
    if p is None:
        print(f"ub={ub}: BOOT FAILED -- {log}")
        ok = False
        continue
    free_after = A.vram()[1]
    A.stop_server()
    text = log.read_text(encoding="utf-8", errors="replace")

    got = re.search(r"llama_context:\s+n_ubatch\s+=\s+(\d+)", text)
    got = int(got.group(1)) if got else None
    bufs = re.findall(r"sched_reserve:\s+CUDA0 compute buffer size =\s+([\d.]+) MiB", text)
    buf = max(float(b) for b in bufs) if bufs else None

    verdict = "OK" if got == ub else f"*** MISMATCH: asked {ub}, context used {got} ***"
    if got != ub:
        ok = False
    print(f"ub={ub:<4} n_ubatch={got}  compute_buffer_max={buf} MiB  "
          f"free_before={free_before}  free_after={free_after}  {verdict}",
          flush=True)

print()
print("PREFLIGHT PASS -- the last -ub wins, sweep is meaningful" if ok else
      "PREFLIGHT FAIL -- do not run the sweep; it would measure one value three times")
sys.exit(0 if ok else 1)
