"""Run the coding-task corpus against a llama-server and verify by execution.

Primary output is VERIFIED TASKS PER HOUR: pass rate x throughput, which is the
metric the optimization plan names. Raw tok/s is reported too, but a config that
decodes fast and fails tests is worse than a slow one that passes.

Verification runs the model's own code in a subprocess with a timeout, so a
model that emits an infinite loop scores a failure rather than hanging the run.

Usage:
    python run_bench.py --label q4-mtp2 --attempts 3
    python run_bench.py --label q4-t06 --temperature 0.6
"""
import argparse, json, re, subprocess, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tasks import TASKS

ROOT = Path(r"C:\AI\qwen38-tuning")
RESULTS = ROOT / "results"
WORK = ROOT / "bench" / "_work"


def post(url, payload, timeout=1800):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text):
    """Take the largest fenced block; fall back to the whole reply.

    Largest rather than first: models often emit a short usage example after the
    real implementation, and taking the first block would test the example.
    """
    blocks = CODE_RE.findall(text or "")
    if blocks:
        return max(blocks, key=len)
    return text or ""


def verify(code, test_src, task_id, attempt):
    WORK.mkdir(parents=True, exist_ok=True)
    f = WORK / f"{task_id}_{attempt}.py"
    f.write_text(code + "\n\n# ---- verification ----\n" + test_src, encoding="utf-8")
    try:
        p = subprocess.run(
            [sys.executable, str(f)], capture_output=True, text=True, timeout=20
        )
        return p.returncode == 0, (p.stderr or "").strip().splitlines()[-1:] or [""]
    except subprocess.TimeoutExpired:
        return False, ["TIMEOUT: code did not terminate in 20s"]
    except Exception as e:  # pragma: no cover
        return False, [f"HARNESS: {e}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--reasoning-effort", default="medium")
    ap.add_argument("--max-tokens", type=int, default=3072)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "quality-bench.jsonl"

    rows, t_start = [], time.time()
    for task in TASKS:
        for attempt in range(1, args.attempts + 1):
            payload = {
                "messages": [
                    {"role": "developer", "content":
                     "You are a precise Python engineer. Reply with one fenced "
                     "```python block containing only the requested code. No "
                     "explanation, no usage examples, no tests."},
                    {"role": "user", "content": task["prompt"]},
                ],
                "temperature": args.temperature, "top_p": args.top_p,
                "top_k": args.top_k, "min_p": args.min_p,
                "max_tokens": args.max_tokens,
                "chat_template_kwargs": {"reasoning_effort": args.reasoning_effort},
                "cache_prompt": False,
            }
            t0 = time.time()
            try:
                r = post(args.endpoint, payload)
            except Exception as e:
                rows.append(dict(label=args.label, task=task["id"], attempt=attempt,
                                 passed=False, error=f"REQUEST: {e}", wall_s=None))
                print(f"  {task['id']} #{attempt}: REQUEST FAILED {e}", flush=True)
                continue
            wall = time.time() - t0

            msg = r["choices"][0]["message"]
            usage = r.get("usage", {})
            code = extract_code(msg.get("content", ""))
            ok, err = verify(code, task["test"], task["id"], attempt)

            row = dict(
                label=args.label, task=task["id"], difficulty=task["difficulty"],
                attempt=attempt, passed=ok, wall_s=round(wall, 1),
                completion_tokens=usage.get("completion_tokens"),
                reasoning_chars=len(msg.get("reasoning_content") or ""),
                tok_s=round((usage.get("completion_tokens") or 0) / wall, 2) if wall else None,
                error=(err[0][:200] if not ok else None),
                temperature=args.temperature, reasoning_effort=args.reasoning_effort,
            )
            rows.append(row)
            with out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"  {task['id']:<16} #{attempt} {'PASS' if ok else 'FAIL'} "
                  f"{row['wall_s']}s {row['tok_s']} tok/s", flush=True)

    total_wall = time.time() - t_start
    done = [r for r in rows if r.get("wall_s") is not None]
    passed = sum(1 for r in rows if r["passed"])
    toks = [r["tok_s"] for r in done if r["tok_s"]]

    summary = dict(
        label=args.label, kind="SUMMARY",
        total_attempts=len(rows), passed=passed,
        pass_rate=round(100 * passed / len(rows), 1) if rows else 0,
        total_wall_s=round(total_wall, 1),
        verified_tasks_per_hour=round(3600 * passed / total_wall, 1) if total_wall else 0,
        median_tok_s=round(sorted(toks)[len(toks) // 2], 2) if toks else None,
        temperature=args.temperature, reasoning_effort=args.reasoning_effort,
    )
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary) + "\n")

    print("\n=== SUMMARY " + args.label + " ===")
    for k, v in summary.items():
        print(f"  {k:<26} {v}")

    by_diff = {}
    for r in rows:
        d = r.get("difficulty", "?")
        by_diff.setdefault(d, [0, 0])
        by_diff[d][1] += 1
        by_diff[d][0] += 1 if r["passed"] else 0
    print("  by difficulty:")
    for d, (p, n) in sorted(by_diff.items()):
        print(f"    {d:<8} {p}/{n}")


if __name__ == "__main__":
    main()
