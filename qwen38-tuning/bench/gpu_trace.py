"""Sample GPU state on a fixed interval while something else is being measured.

WHY THIS EXISTS, and it was found the hard way. The arena records `split` per
arm, and `65+0` reads as "fully resident, everything is fine". At ctx 98,304 the
served profile loads `65/65 layers to GPU` and prefill still collapses from
~924 tok/s to 64-98 -- a 10x loss that NO existing column shows, because the
column that would have shown it does not exist.

What did show it, once someone looked by hand: **208 MiB free**, 100 %
utilisation, and **79 W on a ~220 W card**. High utilisation at low power is
the signature of a memory-bound stall, not of compute. The layers were resident
and the working set was thrashing anyway.

So residency is necessary and not sufficient, and a benchmark that records only
the layer split will call a thrashing configuration healthy.

DELIBERATELY SEPARATE FROM THE ARENA. It attaches to a run that is already
going, writes its own file, and never touches the server or the port. A sampler
that can perturb the measurement it samples is worse than no sampler, and this
project has already had a unit test kill a live server mid-round.

Usage:
    python gpu_trace.py --out ../results/gpu-trace.jsonl --interval 5
Stops on Ctrl-C or --duration.
"""
import argparse
import json
import subprocess
import sys
import time

import gpu_device

FIELDS = ["memory.used", "memory.free", "utilization.gpu", "utilization.memory",
          "temperature.gpu", "clocks.current.sm", "power.draw",
          # PCIe link state. Both fields downtrain when the card is idle, so a
          # reading taken between runs says nothing about the slot -- they are
          # here so the trace carries them WHILE the GPU is working, which is
          # the only form issue #51 stage 4 can use.
          "pcie.link.gen.current", "pcie.link.width.current"]


def sample():
    """One reading, or None if nvidia-smi is momentarily unavailable.

    Returns None rather than raising: a sampler that dies at second 900 of a
    two-hour run loses the rest of the trace, and a dropped sample is a gap
    the reader can see in the timestamps.
    """
    # Routed through `gpu_device` so the sample names one card. The previous
    # form took `splitlines()[0]`, which on a two-card machine is whichever the
    # driver listed first -- a full trace of the wrong GPU, with correct
    # timestamps and nothing to say so (issue #50).
    try:
        parts = gpu_device.query(FIELDS)
    except Exception:
        return None
    if len(parts) != len(FIELDS):
        return None
    out = {}
    for name, raw in zip(FIELDS, parts):
        key = name.replace(".", "_")
        try:
            out[key] = float(raw) if "." in raw else int(raw)
        except ValueError:
            out[key] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--duration", type=float, default=None,
                    help="seconds; omit to run until interrupted")
    a = ap.parse_args()

    t0 = time.time()
    n = dropped = 0
    with open(a.out, "a", encoding="utf-8") as fh:
        try:
            while a.duration is None or time.time() - t0 < a.duration:
                s = sample()
                if s is None:
                    dropped += 1
                else:
                    s["t"] = round(time.time() - t0, 2)
                    s["wall"] = time.strftime("%H:%M:%S")
                    fh.write(json.dumps(s) + "\n")
                    fh.flush()          # a trace that dies with the process is no trace
                    n += 1
                time.sleep(a.interval)
        except KeyboardInterrupt:
            pass
    print("wrote %d samples (%d dropped) to %s" % (n, dropped, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
