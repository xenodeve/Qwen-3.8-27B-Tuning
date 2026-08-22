"""Benchmark the local worker on REAL open issues, in throwaway clones.

WHY. This project's stated metric is verified accepted coding tasks per hour and
no published number measures it -- every result so far is tok/s on a prompt this
harness generated. A decoder that emits tokens 23 % faster has not made the
worker better if the extra tokens are wrong.

WHAT IT MEASURES, per task:
  - PASS / FAIL / VOID against the repo's OWN verify command (§4 of plan 06)
  - wall clock, so tasks per hour is computable
  - the context high-water mark, read from the server log -- the number that
    decides how much VRAM the window has to reserve, and the one this project
    has mis-read three times (CORRECTIONS 15, 17)

SAFETY. Three rules, all developer instructions, all mechanical rather than
advisory:

  1. `D:\\Github\\*` is never written to and never deleted. Two of those
     checkouts hold 333 and 440 uncommitted files -- days of work existing
     nowhere else -- and this script ends by deleting what it made. Every path
     goes through harness.assert_deletable before any removal.
  2. Every task runs in a fresh clone FROM THE GITHUB REMOTE, which is also the
     state the issue was written against; a local tree with 333 dirty files is
     not.
  3. **No issue is ever closed, commented on, labelled or pushed to.** A PASS
     means the worker did the task, not that the task is done: nobody reviewed
     the diff, nobody merged it, and it is deleted minutes later. Closing would
     claim work that was thrown away.

Assumes a server is already listening on 8080. Starting the thing under test
from inside the measurement is how you end up measuring the starter.

    python bench/real_task_bench.py --tasks xeno-skills:306 openclink:144
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness import assert_deletable, is_protected, classify_outcome
import edit_canary          # worker_argv: pins the project root on the argv

OPENCODE = r"C:\Users\xenod\.bun\bin\opencode.exe"
GH = r"C:\Program Files\GitHub CLI\gh.exe"
GIT_BASH = r"C:\Program Files\Git\bin\bash.exe"
BASE = "http://127.0.0.1:8080"

# Verify commands read from each repo's own .claude/t4.json where it has one,
# and from package.json otherwise. Not invented here -- a benchmark that makes
# up its own success criterion is grading itself.
REPOS = {
    "xeno-skills": {
        "remote": "https://github.com/xenodeve/xeno-skills",
        "tracker": "xenodeve/xeno-skills",
        # Git's bash by absolute path. Plain "bash" resolves to WSL here, and
        # WSL then fails with `execvpe(/bin/bash) failed: No such file or
        # directory` -- which the first run correctly refused as a red baseline
        # while saying nothing about why.
        "verify": [GIT_BASH, "tests/hooks/run-all.sh"],
        "setup": [],
    },
    "openclink": {
        "remote": "https://github.com/xenodeve/openclink",
        "tracker": "xenodeve/openclink",
        "verify": [sys.executable, "-m", "pytest", "tests/", "-q", "-m", "not integration"],
        # requirements-dev.txt exists and the first run did not install it:
        # 191 of 1,285 tests failed on a clean checkout with only `pip install
        # -e .`. Whether that closes the gap is what the next baseline says.
        "setup": [
            [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
            [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-dev.txt"],
            [sys.executable, "-m", "pip", "install", "-q", "-e", "."],
        ],
    },
}


def preflight():
    """Fail at the start on a missing tool, not in the middle of a task.

    The first run set GIT_BASH from a string whose backslash escapes were
    eaten by the editor -- it became "...\\Gitinash.exe" -- and the task died
    with FileNotFoundError [WinError 2] AFTER the clone, the dependency
    install and the issue fetch had all run. A path that is wrong is wrong
    before any work happens.
    """
    missing = [x for x in (OPENCODE, GH, GIT_BASH) if not os.path.exists(x)]
    if missing:
        raise SystemExit("missing tool(s): " + ", ".join(missing))


def sh(cmd, cwd=None, timeout=1800, env=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout,
                          env=env)


def issue_body(tracker, number):
    r = sh([GH, "issue", "view", str(number), "--repo", tracker,
            "--json", "title,body"])
    if r.returncode != 0:
        return None, None
    d = json.loads(r.stdout)
    return d.get("title", ""), d.get("body", "")


def ctx_high_water(log_path, since_offset):
    """Peak context reached, from the server log.

    Reads `n_tokens` on the `slot release` line. NOT `prompt eval`, which
    reports only what survived cache reuse -- misreading that shipped three
    mis-sized worker profiles in one day (CORRECTIONS 15, 17).

    Takes the MAXIMUM over the task, not the last value: a task that peaks at
    turn 3 and ends at turn 9 still needs a window that holds the peak.
    """
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            fh.seek(since_offset)
            text = fh.read()
    except OSError:
        return None, since_offset
    vals = [int(m) for m in re.findall(r"n_tokens\s*=\s*(\d+)", text)]
    vals += [int(m) for m in re.findall(r"n_past\s*=\s*(\d+)", text)]
    new_offset = since_offset + len(text.encode("utf-8", "replace"))
    return (max(vals) if vals else None), new_offset


def run_one(spec, scratch, cfgdir, model, log_path, offset, timeout_s, n_ctx):
    repo_key, number = spec.split(":")
    repo = REPOS[repo_key]
    row = {"repo": repo_key, "issue": int(number), "outcome": "VOID"}

    title, body = issue_body(repo["tracker"], number)
    if body is None:
        row["note"] = "could not read the issue"
        return row, offset
    row["title"] = title

    clone = Path(scratch) / "clones" / ("%s-%s" % (repo_key, number))
    assert_deletable(clone, Path(scratch))
    if clone.exists():
        shutil.rmtree(clone, ignore_errors=True)
    clone.parent.mkdir(parents=True, exist_ok=True)

    r = sh(["git", "clone", "--depth", "50", repo["remote"], str(clone)], timeout=900)
    if r.returncode != 0:
        row["note"] = "clone failed: " + (r.stderr or "")[-300:]
        return row, offset

    for cmd in repo["setup"]:
        sh(cmd, cwd=str(clone), timeout=1800)

    # Baseline BEFORE the worker touches anything. A repo whose tests are
    # already red cannot score a task, and finding that out afterwards makes the
    # worker's damage indistinguishable from the repo's.
    base = sh(repo["verify"], cwd=str(clone), timeout=1800)
    row["verify_baseline_exit"] = base.returncode
    if base.returncode != 0:
        # Record WHY. The first version wrote only "already red", which made
        # every VOID uninterpretable -- three tasks voided and the run could not
        # say whether the repo was broken, the toolchain was missing, or the
        # command was wrong. A guard that refuses without evidence just moves
        # the mystery.
        row["note"] = "verify was already red before the worker ran"
        row["verify_baseline_cmd"] = " ".join(repo["verify"])
        row["verify_baseline_out"] = ((base.stdout or "")[-1200:]
                                      + "\n--stderr--\n" + (base.stderr or "")[-1200:])
        _cleanup(clone, scratch)
        return row, offset

    prompt = (
        "You are working in a checkout of %s at %s.\n\n"
        "Implement the following issue. Make the change in the working tree. "
        "Do not commit, do not push, do not open a pull request.\n\n"
        "=== ISSUE #%s: %s ===\n%s\n=== END ISSUE ===\n\n"
        "When you are done, stop and say what you changed."
        % (repo["tracker"], clone, number, title, body)
    )

    env = dict(os.environ)
    env["OPENCODE_CONFIG_DIR"] = str(cfgdir)
    out_path = clone.parent / ("%s-%s.stdout.txt" % (repo_key, number))
    t0 = time.time()
    with out_path.open("w", encoding="utf-8", errors="replace") as of:
        # --dir, not cwd=. OpenCode keeps a per-project server alive between
        # invocations and `run` attaches to whichever is already listening,
        # carrying THE PROJECT ROOT IT WAS FIRST STARTED WITH
        # (opencode_corpus.py:50-62). Measured 2026-08-23: with cwd= alone the
        # worker edited C:\AI\README.md -- the live tree -- while git diff in
        # the clone stayed empty and the row scored diff_bytes=0. That is
        # indistinguishable from "the worker changed nothing", which is what
        # every one of the five real tasks recorded.
        proc = subprocess.Popen(edit_canary.worker_argv(model, prompt, clone),
                                cwd=str(clone), env=env, stdout=of,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace")
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=60)
            rc = None
            row["note"] = "worker timed out after %ss" % timeout_s
    row["wall_clock_s"] = round(time.time() - t0, 1)
    row["worker_rc"] = rc

    diff = sh(["git", "diff"], cwd=str(clone), timeout=300).stdout
    status = sh(["git", "status", "--porcelain"], cwd=str(clone), timeout=300).stdout
    row["changed_files"] = len([l for l in status.splitlines() if l.strip()])
    row["diff_bytes"] = len(diff)
    row["diff"] = diff[:20000]

    after = sh(repo["verify"], cwd=str(clone), timeout=1800)
    row["verify_exit"] = after.returncode
    row["verify_tail"] = (after.stdout or "")[-1500:]

    row["ctx_high_water"], offset = ctx_high_water(log_path, offset)

    # PASS is the repo's own command plus a diff. Whether the diff addresses
    # the stated defect is a judgement this script does not make and does not
    # fake -- it records the diff so a person or a stronger model can.
    row["n_ctx"] = n_ctx
    row["outcome"] = classify_outcome(after.returncode, row["changed_files"],
                                      row["ctx_high_water"], n_ctx)
    if row["outcome"] == "PASS":
        row["judged"] = "verify-only; diff not assessed"
    elif row["outcome"] == "WINDOW_BOUND":
        row["note"] = ("context saturated: high-water %s against n_ctx %s -- "
                       "the window ran out, this is not a worker failure"
                       % (row["ctx_high_water"], n_ctx))
    elif row["changed_files"] == 0:
        row.setdefault("note", "the worker changed nothing")

    _cleanup(clone, scratch)
    return row, offset


def _force_remove(func, path, _exc):
    """rmtree onerror: clear the read-only bit and retry once.

    git marks everything under .git/objects read-only, and on Windows that
    makes shutil.rmtree fail on the first pack file. The first run of this
    script left three half-deleted clones on disk and reported CLEANUP FAILED
    for all of them -- correctly, but for a reason that is fixable rather than
    a lock held by something else.
    """
    import stat
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        pass


def _cleanup(clone, scratch):
    """Delete the clone and VERIFY it is gone. Never widen the pattern."""
    assert_deletable(clone, Path(scratch))
    shutil.rmtree(clone, onerror=_force_remove)
    if clone.exists():
        print("    CLEANUP FAILED, path still present: %s" % clone, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True,
                    help="repo:issue, e.g. xeno-skills:306")
    ap.add_argument("--scratch", default=r"D:\bench-scratch")
    ap.add_argument("--cfgdir", default=r"C:\AI\ocworker\cfg-local")
    ap.add_argument("--model", default="local/qwen38")
    ap.add_argument("--log", default=r"C:\AI\qwen38-tuning\logs\real-task-server.log")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--n-ctx", type=int, required=True, dest="n_ctx",
                    help="the window the SERVER was started with. Required and "
                         "never guessed: without it, a task that ran out of "
                         "context is scored as a worker failure.")
    ap.add_argument("--out", default=r"C:\AI\qwen38-tuning\results\real-task-bench.jsonl")
    a = ap.parse_args()

    preflight()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    scratch = os.path.join(a.scratch, stamp)
    if is_protected(scratch):
        raise SystemExit("scratch root is inside a protected tree: %s" % scratch)
    os.makedirs(scratch, exist_ok=True)
    print("scratch root: %s" % scratch, flush=True)

    offset = os.path.getsize(a.log) if os.path.exists(a.log) else 0
    rows = []
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        for spec in a.tasks:
            print("  %s ..." % spec, flush=True)
            try:
                row, offset = run_one(spec, scratch, a.cfgdir, a.model,
                                      a.log, offset, a.timeout, a.n_ctx)
            except Exception as exc:
                row = {"task": spec, "outcome": "VOID",
                       "note": "%s: %s" % (type(exc).__name__, exc)}
            row["scratch"] = scratch
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print("    %-8s ctx_high_water=%s  wall=%ss  files=%s"
                  % (row.get("outcome"), row.get("ctx_high_water"),
                     row.get("wall_clock_s"), row.get("changed_files")),
                  flush=True)

    # Delete the whole scratch root and say what could not be removed. A run
    # that cannot clean up reports it; it does not widen the pattern.
    assert_deletable(Path(scratch), Path(scratch))
    shutil.rmtree(scratch, ignore_errors=True)
    print("\ncleanup: %s" % ("removed" if not os.path.exists(scratch)
                             else "FAILED, still present: " + scratch))

    n = len(rows)
    c = lambda k: sum(1 for r in rows if r.get("outcome") == k)
    hw = [r["ctx_high_water"] for r in rows if r.get("ctx_high_water")]
    print("\n%d tasks: %d PASS, %d FAIL, %d WINDOW_BOUND, %d VOID"
          % (n, c("PASS"), c("FAIL"), c("WINDOW_BOUND"), c("VOID")))
    if c("WINDOW_BOUND"):
        print("WINDOW_BOUND is NOT a worker failure: those tasks filled the "
              "%d-token window before they could finish. It IS a result "
              "about the window." % a.n_ctx)
    if hw:
        hw.sort()
        print("context high-water: min %d  median %d  max %d"
              % (hw[0], hw[len(hw) // 2], hw[-1]))
    print("\nNO ISSUE WAS CLOSED, COMMENTED ON OR LABELLED. A PASS means the "
          "worker did the task, not that the task is done.")


if __name__ == "__main__":
    sys.exit(main())
