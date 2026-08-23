"""Boot one server for the prefix-cache depth probe, and hold it.

Deliberately a separate step from the probe: both depths must be measured in
ONE boot, because free VRAM at boot moves 9,326-10,732 MiB and `--fit` follows
it, so a shallow round and a deep round from different boots are not
comparable. The probe erases the slot between depths instead.

Flags chosen to match what the workers actually serve -- `--spec-type ngram-mod`,
ctx 98,304 -- plus `--slot-save-path`, which is what the erase endpoint needs
(`server-context.cpp:4549`) and which changes no allocation.
"""
import sys
import time

import dflash2_arena as A

CTX = 98304
EXTRA = ["--spec-type", "ngram-mod",
         "--spec-ngram-mod-n-match", "12",
         "--spec-ngram-mod-n-max", "32",
         "--spec-ngram-mod-n-min", "16",
         "--slot-save-path", r"C:\AI\qwen38-tuning\logs\slots"]

p, fh, log, free_before = A.start(CTX, EXTRA, "prefix-depth", boot_s=600)
if p is None:
    print(f"BOOT FAILED -- see {log}")
    sys.exit(1)

print(f"UP  ctx={CTX}  free_before={free_before} MiB  log={log}")
print(f"free_now={A.vram()[1]} MiB")
sys.stdout.flush()

# hold the process open; the probe runs against it from another shell
while p.poll() is None:
    time.sleep(5)
print("server exited")
