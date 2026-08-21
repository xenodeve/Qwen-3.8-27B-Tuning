"""Run the deep-context corpus against a running server.

Answers the one question left open by the 16K corpus: does Q8_0 KV damage task
success at the depth where it actually buys throughput?

The whole corpus shares one ~45K-token repository prefix, so `cache_prompt` pays
that prefill once and every task after it evaluates only its own question. That
makes 18 samples at 64K affordable, and it matches how an agent works — one cold
turn over the repo, then many cheap turns.

Usage:
    python run_deep_bench.py --label q4-64k-f16 --attempts 3
"""
import argparse, json, re, subprocess, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from deep_tasks import DEEP_TASKS, build_repo
from deep_tasks_v2 import DEEP_TASKS_V2, build_repo as build_repo_v2

ROOT = Path(r"C:\AI\qwen38-tuning")
OUT = ROOT / "results" / "deep-quality.jsonl"
WORK = ROOT / "bench" / "_deepwork"
CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def post(url, payload, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def extract_code(text):
    blocks = CODE_RE.findall(text or "")
    return max(blocks, key=len) if blocks else (text or "")


def verify(code, test_src, tag):
    WORK.mkdir(parents=True, exist_ok=True)
    f = WORK / f"{tag}.py"
    f.write_text(code + "\n\n# ---- verification ----\n" + test_src, encoding="utf-8")
    try:
        p = subprocess.run([sys.executable, str(f)], capture_output=True,
                           text=True, timeout=20)
        return p.returncode == 0, (p.stderr or "").strip().splitlines()[-1:] or [""]
    except subprocess.TimeoutExpired:
        return False, ["TIMEOUT"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--v2", action="store_true",
                    help="harder corpus with decoys, multi-hop and aggregation")
    ap.add_argument("--blocks", type=int, default=None,
                    help="repo size in blocks; 680 ~= 44K tokens (64K window), "
                         "1550 ~= 100K tokens (128K window). Planted shards are "
                         "placed by percentage, so depth scales with it.")
    args = ap.parse_args()

    tasks = DEEP_TASKS_V2 if args.v2 else DEEP_TASKS
    if args.v2:
        repo = build_repo_v2(args.blocks) if args.blocks else build_repo_v2()
    else:
        repo = build_repo()
    print(f"repository prefix: {len(repo)} chars (~{len(repo)/3.6:.0f} tokens)", flush=True)

    rows, t_start = [], time.time()
    for task in tasks:
        for attempt in range(1, args.attempts + 1):
            payload = {
                "messages": [
                    {"role": "developer", "content":
                     "You are a precise Python engineer working in the repository "
                     "shown by the user. Answer using ONLY values found in that "
                     "repository. Reply with one fenced ```python block and nothing else."},
                    {"role": "user", "content": repo + "\n\n" + task["prompt"]},
                ],
                "temperature": args.temperature, "top_p": 0.95,
                "top_k": 20, "min_p": 0.0,
                "max_tokens": args.max_tokens,
                "chat_template_kwargs": {"reasoning_effort": "medium"},
                # True on purpose: the shared repo prefix is the point.
                "cache_prompt": True,
            }
            t0 = time.time()
            try:
                r = post(args.endpoint, payload)
            except Exception as e:
                print(f"  {task['id']:<22} #{attempt} REQUEST FAILED {e}", flush=True)
                rows.append(dict(label=args.label, task=task["id"], attempt=attempt,
                                 passed=False, error=f"REQUEST: {e}"))
                continue
            wall = time.time() - t0

            msg = r["choices"][0]["message"]
            usage = r.get("usage", {})
            ok, err = verify(extract_code(msg.get("content", "")), task["test"],
                             f"{args.label}_{task['id']}_{attempt}")
            row = dict(label=args.label, task=task["id"], depth=task["depth"],
                       attempt=attempt, passed=ok, wall_s=round(wall, 1),
                       completion_tokens=usage.get("completion_tokens"),
                       prompt_tokens=usage.get("prompt_tokens"),
                       error=(err[0][:180] if not ok else None))
            rows.append(row)
            with OUT.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"  {task['id']:<22} #{attempt} {'PASS' if ok else 'FAIL'} "
                  f"{row['wall_s']}s", flush=True)

    total = time.time() - t_start
    passed = sum(1 for r in rows if r["passed"])
    summary = dict(label=args.label, kind="SUMMARY", total_attempts=len(rows),
                   passed=passed,
                   pass_rate=round(100 * passed / len(rows), 1) if rows else 0,
                   total_wall_s=round(total, 1),
                   verified_tasks_per_hour=round(3600 * passed / total, 1) if total else 0)
    with OUT.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary) + "\n")
    print(f"\n=== {args.label}: {summary['pass_rate']}% ({passed}/{len(rows)}), "
          f"{summary['verified_tasks_per_hour']}/hr, wall {summary['total_wall_s']}s")


if __name__ == "__main__":
    main()
