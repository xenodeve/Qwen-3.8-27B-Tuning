"""Run the coding corpus through OpenCode against the local server.

**This measures the worker that ships.** `run_retry_bench.py` sends a 35-token
developer message to `/v1/chat/completions` and grades the reply for a fenced
code block. Nothing in production does that: the real worker is OpenCode with a
tool loop, and it does not answer with a code fence -- it **writes a file**.

So the two harnesses do not merely differ in prompt size. They grade different
shapes of work, and the failure this project has chased for two days -- 58.3 %
"output contract pass", the model looping in its reasoning and never emitting a
fence -- is a property of a task shape that OpenCode never asks for.

Each task gets a **fresh empty directory**, so the worker cannot see another
task's answer and a leftover file cannot be graded as a pass. The prompt names
the file to write and nothing else; the grade comes from executing that file
against the same hidden assertions `run_retry_bench.py` uses, so the two are
comparable on the only axis that matters.

    python bench/opencode_corpus.py --label lean-iq2xxs --tasks 10

Assumes a server already listening on 8080 and an OpenCode config directory
prepared with a provider pointing at it. Neither is started here: this measures
a harness, and starting the thing under test from inside the measurement is how
you end up measuring the starter.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tasks import TASKS
from run_bench import verify

ROOT = Path(r"C:\AI\qwen38-tuning")
OPENCODE = r"C:\Users\xenod\.bun\bin\opencode.exe"

# The lean profile, measured 2026-08-21: 99,073 -> ~4,650 tokens of prefix.
# Every one of these was found by reading the binary's own flag table; none is
# documented anywhere we could find.
LEAN_ENV = {
    "OPENCODE_DISABLE_CLAUDE_CODE": "1",        # broad: prompt AND skills
    "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
    "OPENCODE_DISABLE_DEFAULT_PLUGINS": "1",
}


def kill_server():
    """OpenCode keeps a per-project server alive between invocations.

    `run` attaches to whatever is already listening, and that server carries the
    project root it was first started with -- so a second task launched from a
    different directory has its files written into the FIRST task's project.
    Observed 2026-08-21: every answer landed in C:\AI\qwen38-tuning while the
    harness looked in the task directory and recorded "no file written" on work
    the model had done correctly.

    `opencode run` takes no project argument -- the positional is the message --
    so the only way to change the root is to make it start a fresh server.
    """
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    "Get-Process opencode -ErrorAction SilentlyContinue | "
                    "Stop-Process -Force"], capture_output=True)
    time.sleep(2)


def run_task(task, workdir, cfgdir, model, timeout_s):
    """One task in a directory emptied first, one OpenCode invocation.

    The directory is SHARED across tasks and emptied between them, rather than
    one directory per task. That is forced by OpenCode, not chosen: its server
    persists between invocations and carries the project root it first started
    with, `run` has no project argument, and killing the server per task costs a
    cold boot that pushed a 37 s task past 150 s. Emptying is what keeps a
    previous task's answer from being graded as this one's.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    for f in workdir.iterdir():
        if f.is_file():
            f.unlink()
        else:
            shutil.rmtree(f, ignore_errors=True)

    target = f"{task['id']}.py"
    prompt = (
        f"{task['prompt']}\n\n"
        f"Write the code to a file named exactly `{target}` in the current "
        f"directory. Do not create any other file. Do not write tests. "
        f"When the file is written, stop and report only the filename."
    )

    env = dict(os.environ)
    env.update(LEAN_ENV)
    env["OPENCODE_CONFIG_DIR"] = str(cfgdir)

    # Stream the transcript to disk as it happens rather than collecting it and
    # writing at the end. The first version of this function used
    # capture_output=True, and when the run was killed mid-hang there was
    # nothing on disk at all -- the same "record only on the happy path" fault
    # that cost this project a queue step at 02:35 today.
    o_path, e_path = workdir / "_stdout.txt", workdir / "_stderr.txt"
    t0 = time.time()
    with o_path.open("w", encoding="utf-8", errors="replace") as of,          e_path.open("w", encoding="utf-8", errors="replace") as ef:
        proc = subprocess.Popen([OPENCODE, "run", "-m", model, prompt],
                                cwd=str(workdir), env=env, stdout=of, stderr=ef,
                                text=True, encoding="utf-8", errors="replace")
        try:
            rc = proc.wait(timeout=timeout_s)
            note = ""
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
            rc = None
            note = f"timeout after {timeout_s}s"
    wall = time.time() - t0
    out = o_path.read_text(encoding="utf-8", errors="replace")
    err = e_path.read_text(encoding="utf-8", errors="replace")

    produced = sorted(f.name for f in workdir.iterdir()
                      if f.is_file() and not f.name.startswith("_"))
    code_path = workdir / target
    row = dict(task=task["id"], difficulty=task["difficulty"], wall_s=round(wall, 1),
               rc=rc, files=produced, note=note,
               # The corpus contract was "one fenced block, nothing else".
               # OpenCode's is "one file, nothing else". Same intent, and this
               # is the column that says whether it was honoured.
               wrote_target=code_path.exists(),
               extra_files=[f for f in produced if f != target])

    if not code_path.exists():
        tail = " / ".join(l.strip() for l in out.strip().splitlines()[-4:])
        row.update(accepted=False,
                   error="target file not written" + (f" | said: {tail[:200]}" if tail else " | no output at all"))
        return row

    code = code_path.read_text(encoding="utf-8", errors="replace")
    ok, err_lines = verify(code, task["test"], task["id"], "oc")
    row.update(accepted=bool(ok), code_chars=len(code),
               error=None if ok else " | ".join(err_lines)[:300])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", default="local/qwen38")
    ap.add_argument("--cfgdir", default=r"C:\AI\ocworker\cfg-local")
    ap.add_argument("--work", default=r"C:\AI\ocworker\corpus")
    ap.add_argument("--tasks", type=int, default=len(TASKS))
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", default="results/opencode-corpus.jsonl")
    args = ap.parse_args()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(args.work)
    cfgdir = Path(args.cfgdir)
    if not (cfgdir / "opencode.json").exists():
        raise SystemExit(f"no opencode.json in {cfgdir}")

    # Once, not per task: this is what fixes the project root for every task
    # that follows, and it is the only lever `run` gives us over it.
    kill_server()

    rows = []
    print(f"label={args.label}  model={args.model}  tasks={args.tasks}", flush=True)
    for t in TASKS[:args.tasks]:
        r = run_task(t, work, cfgdir, args.model, args.timeout)
        r["label"] = args.label
        rows.append(r)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\n")
        mark = "PASS" if r["accepted"] else "fail"
        extra = f"  +{r['extra_files']}" if r["extra_files"] else ""
        print("  %-16s %-7s %-5s %6.1fs%s  %s"
              % (t["id"], t["difficulty"], mark, r["wall_s"], extra,
                 (r.get("error") or "")[:70]), flush=True)

    n = sum(1 for r in rows if r["accepted"])
    wrote = sum(1 for r in rows if r["wrote_target"])
    clean = sum(1 for r in rows if r["wrote_target"] and not r["extra_files"])
    wall = sum(r["wall_s"] for r in rows)
    summary = dict(kind="SUMMARY", label=args.label, model=args.model,
                   tasks=len(rows), accepted=n,
                   wrote_target=wrote, single_file=clean,
                   wall_s=round(wall, 1),
                   tasks_per_hour=round(3600.0 * n / wall, 1) if wall else None)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
    print("\n%s: accepted %d/%d  wrote target %d/%d  single file %d/%d  "
          "%.0f s total  %s accepted tasks/hour"
          % (args.label, n, len(rows), wrote, len(rows), clean, len(rows),
             wall, summary["tasks_per_hour"]), flush=True)


if __name__ == "__main__":
    main()
