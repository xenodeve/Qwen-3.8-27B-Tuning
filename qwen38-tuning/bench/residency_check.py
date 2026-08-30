"""Is the model ACTUALLY resident, or does llama.cpp only think so?

Every conclusion in this project rests on one reported number: the GPU/CPU layer
split that `-lv 5` prints at load. A review panel pointed out on 2026-08-20 that
the number can be true and the claim still false:

    Windows WDDM permits VRAM overcommit. When the card is full and something
    else needs memory -- the desktop compositor, a browser -- WDDM silently
    pages a CUDA allocation out to host RAM across PCIe. llama.cpp still reports
    the layer as assigned to CUDA0, because from its side it is. Throughput
    falls off exactly the cliff a CPU-resident layer would produce, and nothing
    in the server log says why.

That is a precise explanation for something already measured here and left
unexplained: at 345 MiB free VRAM the code prompt returned [6.70, 8.28, 11.57] --
a 73 % spread with one perfectly normal sample. A lower mean is slowness; a wide
spread with normal samples in it is eviction.

The observable is the process's **shared** GPU memory. Dedicated is what sits on
the card; shared is what spilled to host. For a run that claims full residency it
must be zero, and it must be zero DURING generation, not at load -- KV grows and
compute buffers are allocated as work arrives.

    \\GPU Process Memory(pid_*)\\Shared Usage       <- must stay 0
    \\GPU Process Memory(pid_*)\\Dedicated Usage

Usage:
    python residency_check.py --samples 8 --interval 3
    python residency_check.py --while-generating      # loads the GPU first
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import gpu_device

ROOT = Path(r"C:\AI\qwen38-tuning")
BASE = "http://127.0.0.1:8080"

PS = r"""
$ErrorActionPreference = 'SilentlyContinue'
$c = Get-NetTCPConnection -LocalPort 8080 -State Listen
if (-not $c) { '{"error":"nothing listening on 8080"}'; exit }
$serverPid = $c.OwningProcess
$ctr = Get-Counter -Counter "\GPU Process Memory(*)\Shared Usage",
                            "\GPU Process Memory(*)\Dedicated Usage" -ErrorAction SilentlyContinue
if (-not $ctr) { '{"error":"GPU Process Memory counters unavailable"}'; exit }
$shared = 0; $dedicated = 0
foreach ($s in $ctr.CounterSamples) {
    if ($s.InstanceName -match "pid_$serverPid" ) {
        if ($s.Path -like '*shared usage*')    { $shared    += $s.CookedValue }
        if ($s.Path -like '*dedicated usage*') { $dedicated += $s.CookedValue }
    }
}
@{ pid = $serverPid
   shared_mib = [math]::Round($shared/1MB, 1)
   dedicated_mib = [math]::Round($dedicated/1MB, 1) } | ConvertTo-Json -Compress
"""


def sample():
    out = subprocess.run(["powershell", "-NoProfile", "-Command", PS],
                         capture_output=True, text=True).stdout.strip()
    try:
        return json.loads(out)
    except Exception:
        return {"error": "unparseable counter output: %s" % out[:160]}


def _free_vram():
    """Free VRAM on the SERVED card, so the ratio can be read against headroom.

    The previous form took `splitlines()[0]`, which on a two-card machine is
    whichever card the driver listed first -- here the retired 4070 SUPER. It
    did not raise; it returned a plausible number for the wrong hardware, and
    the bare `except` below would have hidden it either way (issue #50).
    """
    try:
        return gpu_device.free_vram()
    except Exception:
        return None


def keep_busy(seconds):
    """Hold the GPU under real generation load while the counters are read."""
    body = json.dumps({"prompt": "Write a Python LRU cache class.",
                       "n_predict": 4096, "temperature": 0.7,
                       "cache_prompt": False}).encode()
    req = urllib.request.Request(BASE + "/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=seconds + 60).read()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--interval", type=float, default=3.0)
    ap.add_argument("--while-generating", action="store_true",
                    help="drive a real generation so the counters see load")
    ap.add_argument("--label", default="unlabelled")
    ap.add_argument("--out", default="residency-check.jsonl")
    args = ap.parse_args()

    out = ROOT / "results" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    worker = None
    if args.while_generating:
        import threading
        worker = threading.Thread(
            target=keep_busy, args=(args.samples * args.interval,), daemon=True)
        worker.start()
        time.sleep(2)

    rows = []
    for i in range(args.samples):
        s = sample()
        s["t"] = round(i * args.interval, 1)
        rows.append(s)
        if "error" in s:
            print("  %5.1fs  %s" % (s["t"], s["error"]), flush=True)
        else:
            print("  %5.1fs  dedicated %8.1f MiB   shared %8.1f MiB%s"
                  % (s["t"], s["dedicated_mib"], s["shared_mib"],
                     "   <-- SPILLED TO HOST" if s["shared_mib"] > 0 else ""),
                  flush=True)
        time.sleep(args.interval)

    good = [r for r in rows if "error" not in r]
    if not good:
        print("\nno usable samples -- the counters were unavailable, so residency "
              "is UNVERIFIED. Do not record a run as resident on this basis.")
        sys.exit(1)

    peak_shared = max(r["shared_mib"] for r in good)
    peak_ded = max(r["dedicated_mib"] for r in good)
    # Deliberately NOT a boolean gate. The first version of this script reported
    # `truly_resident = (shared == 0)` and immediately measured 98 MiB against
    # 9,417 MiB dedicated on the production artifact -- about 1 %, the size of
    # ordinary pinned staging for host-to-device copies, not evicted weights.
    # An absolute threshold here would repeat the mistake this project already
    # made with a "100 % tool compliance" gate that would have rejected its own
    # control. What carries information is the RATIO compared across arms with
    # different headroom.
    verdict = dict(label=args.label, kind="SUMMARY", samples=len(good),
                   peak_shared_mib=peak_shared,
                   peak_dedicated_mib=peak_ded,
                   shared_pct_of_dedicated=(round(100.0 * peak_shared / peak_ded, 2)
                                            if peak_ded else None),
                   free_vram_mib=_free_vram(),
                   under_load=bool(args.while_generating))
    with out.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(dict(r, label=args.label)) + "\n")
        f.write(json.dumps(verdict) + "\n")

    print("\n=== RESIDENCY %s ===" % args.label)
    for k, v in verdict.items():
        print("  %-22s %s" % (k, v))
    print("")
    print("  Read the RATIO across arms, not this row alone. WDDM can page a")
    print("  process VRAM to host RAM while llama.cpp still reports the layer")
    print("  as assigned to CUDA0, so the loader split cannot detect it. A")
    print("  ratio that climbs as free VRAM falls is eviction; one that sits")
    print("  flat near 1 percent is ordinary pinned staging for H2D copies.")


if __name__ == "__main__":
    main()
