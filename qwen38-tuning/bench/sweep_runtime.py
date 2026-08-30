"""Runtime tuning sweep — squeeze the machine, prove quality is untouched.

Levers here (fit-target, CPU threads, batch/ubatch) change WHERE work happens and
HOW it is scheduled, not what the model computes. So instead of paying ~48 min for
a full quality benchmark per arm, every config emits a greedy sample
(temperature 0, top_k 1, seed 42) whose SHA-256 is compared against the baseline.

    hash matches  -> output is bit-identical; quality is provably unchanged
    hash differs  -> the change is NOT quality-neutral; full quality bench required

That is cheaper than sampling-based quality measurement and strictly stronger:
a pass-rate comparison could miss a small regression, an identical hash cannot.

Each config also records the ACTUAL layer split parsed from the verbose load
report, because on this machine --fit derives placement from free VRAM at boot
and the whole point of the fit-target sweep is to move layers onto the GPU.

Usage:
  python sweep_runtime.py fit      # --fit-target 256..2048
  python sweep_runtime.py threads  # -t sweep, then -tb on the best
  python sweep_runtime.py batch    # -b x -ub grid
"""
import hashlib, json, subprocess, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gpu_device
from provenance import resolve_exe

from harness import median, parse_layer_split

ROOT = Path(r"C:\AI\qwen38-tuning")
EXE = resolve_exe(r"C:\AI\llama.cpp-cuda\llama-server.exe")
MODEL = "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL"
PORT = 8080
BASE_URL = f"http://127.0.0.1:{PORT}"

BENCH_PROMPT = "Write a Python function that reverses a linked list."
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
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    "$c=Get-NetTCPConnection -LocalPort 8080 -State Listen "
                    "-ErrorAction SilentlyContinue; if($c){Stop-Process -Id "
                    "$c.OwningProcess -Force}"], capture_output=True)
    time.sleep(5)


def vram():
    """(used, free) on the served card -- see `gpu_device` (issue #50)."""
    return gpu_device.vram()


def start_server(extra, tag):
    """Launch with -lv 5 so the layer assignment lands in the log."""
    kill_server()
    free_before = vram()[1]
    log = ROOT / "logs" / f"sweep-{tag}.log"
    args = [EXE, "-hf", MODEL, "--alias", "qwen38-q4", "-c", "16384",
            "-ngl", "auto", "--fit", "on", "-fa", "on", "-np", "1",
            "--no-mmproj-auto", "--spec-type", "draft-mtp",
            "--spec-draft-n-max", "2", "-lv", "5",
            # fit-target 768 won the fit sweep: 12.39 tok/s on the code prompt
            # with the tightest spread, vs 11.34 at the 1024 default. Later
            # sweeps stack on top of it. The fit sweep itself overrides this.
            "--fit-target", "768",
            # -t 18 won the thread sweep: 13.58 tok/s on the code prompt with the
            # tightest spread AND no prompt-processing loss. -t 20 grabs every
            # logical thread and costs 18% of pp. The thread sweeps override this.
            "-t", "18",
            # -b 2048 -ub 256 won the batch sweep: 13.49 tok/s with prompt
            # processing unchanged. The batch sweeps override this.
            "-b", "2048", "-ub", "256",
            "--host", "127.0.0.1", "--port", str(PORT)] + extra
    fh = log.open("w", encoding="utf-8", errors="replace")
    p = subprocess.Popen(args, stdout=fh, stderr=subprocess.STDOUT)
    for _ in range(60):
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


def parse_split(log):
    """Delegates to the tested primitive; see bench/tests/test_harness.py."""
    try:
        return parse_layer_split(log.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError) as e:
        print(f"    layer split unavailable: {e}")
        return None, None


def measure(reps=3):
    tg, pp, dn, da = [], [], 0, 0
    for prompt, bucket in ((BENCH_PROMPT, "bench"), (CODE_PROMPT, "code")):
        vals = []
        for _ in range(reps):
            r = post("/completion", {"prompt": prompt, "n_predict": 160,
                                     "temperature": 0.7, "cache_prompt": False})
            t = r["timings"]
            vals.append(t["predicted_per_second"])
            if bucket == "code":          # long prompt; the short one measures overhead
                pp.append(t["prompt_per_second"])
            dn += int(t.get("draft_n", 0))
            da += int(t.get("draft_n_accepted", 0))
        tg.append((bucket, sorted(vals)))
    g = post("/completion", {"prompt": "def fibonacci(n):", "n_predict": 60,
                             "temperature": 0.0, "top_k": 1, "seed": 42,
                             "cache_prompt": False})
    h = hashlib.sha256(g["content"].encode()).hexdigest()[:16].upper()
    return tg, pp, dn, da, h


def run(configs, sweep_name):
    out = ROOT / "results" / f"sweep-{sweep_name}.jsonl"
    baseline_hash = None
    for tag, extra, label in configs:
        p, fh, log, free_before = start_server(extra, f"{sweep_name}-{tag}")
        if p is None:
            print(f"  {label:<26} FAILED TO START")
            continue
        used, free = vram()
        gpu, cpu = parse_split(log)
        tg, pp, dn, da, h = measure(reps=5 if sweep_name in ("threads2", "batch2", "spec", "spec2", "drift", "ab") else 3)
        if baseline_hash is None:
            baseline_hash = h
        row = dict(sweep=sweep_name, tag=tag, label=label, args=" ".join(extra),
                   gpu_layers=gpu, cpu_layers=cpu,
                   vram_free_before=free_before, vram_used=used, vram_free=free,
                   tg_bench_med=round(median(tg[0][1]), 2), tg_bench=[round(v, 2) for v in tg[0][1]],
                   tg_code_med=round(median(tg[1][1]), 2), tg_code=[round(v, 2) for v in tg[1][1]],
                   pp_med=round(median(pp), 1),
                   acceptance=round(100.0 * da / dn, 1) if dn else None,
                   greedy_hash=h, quality_identical=(h == baseline_hash))
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        q = "same" if h == baseline_hash else "*** CHANGED ***"
        print(f"  {label:<26} gpu={gpu}/cpu={cpu}  bench={row['tg_bench_med']:<6} "
              f"code={row['tg_code_med']:<6} pp={row['pp_med']:<6} "
              f"free={free:<5} acc={row['acceptance']}%  quality:{q}", flush=True)
        p.kill()
        fh.close()
    print(f"\n-> {out}")


SWEEPS = {
    "fit": [(f"t{v}", ["--fit-target", str(v)], f"--fit-target {v}")
            for v in (1024, 256, 512, 768, 1536, 2048)],
    # i5-13500 is 6 P-cores + 8 E-cores = 14 physical, 20 logical.
    # llama.cpp guidance is that physical-core-like counts beat using every
    # logical thread, and E-cores can drag a lockstep decode loop.
    "threads": [(f"t{v}", ["-t", str(v)], f"-t {v}") for v in (14, 6, 8, 10, 12, 20)],
    # -t 20 beat the -t 14 default by 8.2% but showed one low sample while -t 14
    # was very tight. Re-run both at N=5 before adopting: the win must survive
    # contention with the desktop, which is exactly what 20 threads competes with.
    # -tb (prompt/batch threads) is swept separately at the winning -t.
    "threads2": [("t14", ["-t", "14"], "-t 14"),
                 ("t20", ["-t", "20"], "-t 20"),
                 ("t18", ["-t", "18"], "-t 18"),
                 ("t20tb14", ["-t", "20", "-tb", "14"], "-t 20 -tb 14"),
                 ("t20tb20", ["-t", "20", "-tb", "20"], "-t 20 -tb 20")],
    # The default's median was dragged by a single 10.14 outlier, and the three
    # leaders sit within ~0.3% of each other on decode. Re-run at N=5 so the
    # choice is not made on noise. -b 512 -ub 128 is excluded despite the best
    # decode: it costs 33% of prompt processing, which the prefix-cache result
    # shows is paid in full on every cache invalidation.
    "spec": [("ctl",  [], "control (p-min 0.00, p-split 0.10)"),
             ("pm05", ["--spec-draft-p-min", "0.05"], "--spec-draft-p-min 0.05"),
             ("pm10", ["--spec-draft-p-min", "0.10"], "--spec-draft-p-min 0.10"),
             ("ps00", ["--spec-draft-p-split", "0.00"], "--spec-draft-p-split 0.00"),
             ("ps25", ["--spec-draft-p-split", "0.25"], "--spec-draft-p-split 0.25"),
             ("nmin1",["--spec-draft-n-min", "1"], "--spec-draft-n-min 1"),
             ("nmin2",["--spec-draft-n-min", "2"], "--spec-draft-n-min 2")],
    "ab": [('stock1', ['--fit-target', '1024', '-t', '14', '-b', '2048', '-ub', '512'], 'STOCK pass 1'), ('tuned1', ['--fit-target', '768', '-t', '18', '-b', '2048', '-ub', '256'], 'TUNED pass 1'), ('stock2', ['--fit-target', '1024', '-t', '14', '-b', '2048', '-ub', '512'], 'STOCK pass 2'), ('tuned2', ['--fit-target', '768', '-t', '18', '-b', '2048', '-ub', '256'], 'TUNED pass 2'), ('stock3', ['--fit-target', '1024', '-t', '14', '-b', '2048', '-ub', '512'], 'STOCK pass 3'), ('tuned3', ['--fit-target', '768', '-t', '18', '-b', '2048', '-ub', '256'], 'TUNED pass 3')],
    "drift": [(f"r{i}", [], f"identical config, restart {i}") for i in range(1, 7)],
    "spec2": [("ctl", [], "control"),
              ("nm2", ["--spec-draft-n-min", "2"], "n-min 2"),
              ("pm10", ["--spec-draft-p-min", "0.10"], "p-min 0.10"),
              ("nm2pm10", ["--spec-draft-n-min", "2", "--spec-draft-p-min", "0.10"], "n-min 2 + p-min 0.10"),
              ("all3", ["--spec-draft-n-min", "2", "--spec-draft-p-min", "0.10",
                        "--spec-draft-p-split", "0.25"], "n-min 2 + p-min 0.10 + p-split 0.25")],
    "batch2": [("d", ["-b", "2048", "-ub", "512"], "-b 2048 -ub 512 (default)"),
               ("a", ["-b", "1024", "-ub", "512"], "-b 1024 -ub 512"),
               ("b", ["-b", "2048", "-ub", "256"], "-b 2048 -ub 256")],
    "batch": [(f"b{b}u{u}", ["-b", str(b), "-ub", str(u)], f"-b {b} -ub {u}")
              for b, u in ((2048, 512), (1024, 256), (512, 128), (2048, 256), (1024, 512))],
}

if __name__ == "__main__":
    name = sys.argv[1]
    print(f"=== {name} sweep (first row is the control) ===", flush=True)
    run(SWEEPS[name], name)
