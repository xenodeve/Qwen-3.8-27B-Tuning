"""Boot once per `-cram` value and run the A/B/A/B/A swap against each.

Two boots, not one: `-cram` is a server flag, so the arms cannot share a
process. Both run back to back on the same machine state, which is the closest
this gets to a paired comparison — and the effect being looked for is a full
cold prefill appearing or not appearing, which is not a 13.6 %-floor question.
"""
import subprocess
import sys
from pathlib import Path

import dflash2_arena as A

CTX = 98304
CHARS = 150000
BASE = ["--spec-type", "ngram-mod"] + A.NGRAM
HERE = Path(__file__).parent

ok = True
for cram in ("8192", "0"):
    p, fh, log, free_before = A.start(CTX, BASE + ["-cram", cram],
                                      f"cram-{cram}", boot_s=600)
    if p is None:
        print(f"-cram {cram}: BOOT FAILED -- {log}")
        ok = False
        continue
    print(f"\n########## -cram {cram}  (free_before={free_before})", flush=True)
    r = subprocess.run([sys.executable, str(HERE / "prompt_cache_swap.py"),
                        "--chars", str(CHARS), "--tag", cram],
                       cwd=str(HERE))
    if r.returncode != 0:
        ok = False
    A.stop_server()

print("\nDONE" if ok else "\nDONE WITH FAILURES -- read above")
sys.exit(0 if ok else 1)
