"""Can the worker EDIT AN EXISTING TRACKED FILE in a fresh clone? Nothing else.

THE QUESTION THIS ANSWERS, AND THE ONE IT DOES NOT. Five real GitHub issues ran
1,427-2,400 s each and changed no files; three exited rc=0 having decided they
were done. Before any of that is attributed to the model, the tool path has to
be cleared.

The broad version -- "can this worker write files at all" -- is already answered
YES: `results/opencode-corpus.jsonl` records `wrote_target: True` on 7 of 11
rows with real filenames. But those tasks CREATED A NEW FILE in an empty scratch
directory. A real task must `read -> edit -> save` a file that already exists in
a git checkout. Different tool, different trust situation, and untested.

So this probes exactly that and reports what it observed. It does not measure
throughput, it does not judge the model, and it cannot tell you why a task was
abandoned -- only whether abandoning it was even a choice the worker had.

WHY THE OUTCOMES ARE SHAPED THIS WAY. Each sends the investigation somewhere
different, so collapsing them would waste the run:

    EDITED             a diff exists. The edit path works; look elsewhere.
    TOOL_DENIED        the tool was refused. Permission gating is real.
    NO_EDIT_ATTEMPTED  the tool was never called. A model or harness behaviour,
                       not a permission one.
    EDIT_NO_DIFF       an edit was claimed and the tree is unchanged. A write
                       that landed somewhere else, or was reverted.

A DETECTOR THAT CANNOT REPORT AN UNOBSERVED SUCCESS. `EDITED` is returned only
against a real diff -- never against the worker's own claim, and never against
an exit code. The five failing tasks all exited 0, so exit status is precisely
the signal that has already been shown to lie here.

AND THE TRANSCRIPT SURVIVES. Last time it was written inside the clone and
deleted with the scratch root, which is why five 40-minute failures have no
mechanism attached. Here it is written beside the clone and saved on every path,
including the failing ones -- especially those.

NEEDS THE SERVER. The worker talks to llama-server on :8080, so this cannot run
beside a benchmark that owns the port.
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
from harness import assert_deletable

OPENCODE = r"C:\Users\xenod\.bun\bin\opencode.exe"
GIT = "git"

OUTCOMES = ("EDITED", "TOOL_DENIED", "NO_EDIT_ATTEMPTED", "EDIT_NO_DIFF")

# Word-boundary, so "credited" or "editorial" cannot pass for a tool call.
_EDIT_WORD = re.compile(r"\bedit(s|ed|ing)?\b", re.I)
_DENIED = re.compile(
    r"permission denied|not permitted|denied|refus(?:ed|ing)|"
    r"requires? approval|not allowed", re.I)


def classify(rc, diff_bytes, transcript):
    """What the run actually demonstrated. The diff outranks every claim.

    `rc` is accepted and deliberately not used to decide EDITED in either
    direction: a worker that exits 0 having done nothing is the case under
    investigation, and a worker that crashes after a successful edit still
    edited. The artifact is the evidence.
    """
    if diff_bytes > 0:
        return "EDITED"
    if _DENIED.search(transcript or ""):
        return "TOOL_DENIED"
    if _EDIT_WORD.search(transcript or ""):
        return "EDIT_NO_DIFF"
    return "NO_EDIT_ATTEMPTED"


def transcript_path(clone, scratch):
    """Beside the clone, never inside it, so per-task cleanup cannot take it."""
    clone, scratch = Path(clone), Path(scratch)
    return scratch / "transcripts" / (clone.name + ".stdout.txt")


def save_transcript(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="replace")
    return path


def assert_tracked(repo, rel, tracked):
    """Refuse a target git does not already track.

    An untracked target silently converts this into the create-a-new-file case,
    which is already known to work -- so the canary would come back green
    having tested nothing.
    """
    if rel not in tracked:
        raise ValueError(
            "%s is not tracked by git in %s; the canary must edit an EXISTING "
            "tracked file or it degrades into the create-a-file case that is "
            "already known to pass" % (rel, repo))


def worker_argv(model, prompt, workdir):
    r"""The command line, with the working directory pinned EXPLICITLY.

    `cwd=` on the subprocess is not enough and was not enough here. OpenCode
    keeps a per-project server alive between invocations; `run` attaches to
    whichever is already listening, and that server carries **the project root
    it was first started with** (`opencode_corpus.py:50-62`, written 2026-08-21).

    Measured 2026-08-23: launched with `cwd=<clone>` and told to edit
    `README.md`, the worker edited `C:\AI\README.md` -- the live repository --
    while `git diff` in the clone stayed empty and the row scored
    `diff_bytes=0`. That is indistinguishable from the worker choosing to do
    nothing, and it is what five real tasks recorded.

    So the directory goes on the argv. An empty one raises rather than
    defaulting, because defaulting is precisely how the live tree got edited.
    """
    if not workdir:
        raise ValueError(
            "workdir is required: OpenCode resolves paths against the project "
            "root of whatever server it attaches to, so an unset directory "
            "silently targets the wrong tree")
    return [OPENCODE, "run", "--dir", str(Path(workdir).resolve()),
            "-m", model, prompt]


def sh(args, cwd=None, timeout=600, env=None):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout, encoding="utf-8", errors="replace",
                          env=env)


def run(remote, rel, scratch, cfgdir, model, prompt, timeout_s):
    """One canary. Returns a row; writes the transcript whatever happens."""
    scratch = Path(scratch)
    clone = scratch / "clones" / ("canary-" + str(int(time.time())))
    assert_deletable(clone, scratch)
    clone.parent.mkdir(parents=True, exist_ok=True)

    row = {"remote": remote, "target": rel, "outcome": "VOID"}

    r = sh([GIT, "clone", "--depth", "5", remote, str(clone)], timeout=900)
    if r.returncode != 0:
        row["note"] = "clone failed: " + (r.stderr or "")[-300:]
        return row

    tracked = sh([GIT, "ls-files"], cwd=str(clone)).stdout.split()
    assert_tracked(clone, rel, tracked)

    env = dict(os.environ)
    env["OPENCODE_CONFIG_DIR"] = str(cfgdir)

    t0 = time.time()
    p = sh(worker_argv(model, prompt, clone), cwd=str(clone),
           timeout=timeout_s, env=env)
    row["wall_s"] = round(time.time() - t0, 1)
    row["rc"] = p.returncode

    transcript = (p.stdout or "") + "\n--- stderr ---\n" + (p.stderr or "")
    row["transcript"] = str(save_transcript(transcript_path(clone, scratch),
                                            transcript))

    diff = sh([GIT, "diff"], cwd=str(clone)).stdout
    status = sh([GIT, "status", "--porcelain"], cwd=str(clone)).stdout
    row["diff_bytes"] = len(diff)
    row["changed_files"] = len([l for l in status.splitlines() if l.strip()])
    row["outcome"] = classify(p.returncode, row["diff_bytes"], transcript)
    row["diff_head"] = diff[:800]

    shutil.rmtree(clone, ignore_errors=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default="https://github.com/xenodeve/openclink")
    ap.add_argument("--target", default="README.md")
    ap.add_argument("--scratch", default=r"D:\bench-scratch")
    ap.add_argument("--cfgdir", default=r"C:\AI\ocworker\cfg-local")
    ap.add_argument("--model", default="local/qwen38")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", default=r"C:\AI\qwen38-tuning\results\edit-canary.jsonl")
    ap.add_argument("--round", choices=["trivial", "reasoned", "both"],
                    default="both")
    a = ap.parse_args()

    # Two rounds, because they fail differently. The first needs no
    # comprehension at all, so a failure there is purely the tool path. The
    # second requires read -> reason -> edit, which is the shape a real task has.
    prompts = {
        "trivial": (
            "Open the existing file %s. Change exactly one line by appending "
            "the word CANARY to the end of the first non-empty line. Save the "
            "file. Then run `git diff` and show me its output. Do NOT create "
            "any new file." % a.target),
        "reasoned": (
            "Read the existing file %s. Find the first heading in it and add "
            "one short sentence directly beneath that heading summarising what "
            "the heading is about. Save the file, then run `git diff` and show "
            "the output. Do NOT create any new file." % a.target),
    }
    todo = list(prompts) if a.round == "both" else [a.round]

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fh:
        for name in todo:
            print("=== canary: %s ===" % name, flush=True)
            row = run(a.remote, a.target, a.scratch, a.cfgdir, a.model,
                      prompts[name], a.timeout)
            row["round"] = name
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print("  outcome=%s  rc=%s  diff_bytes=%s  wall=%ss"
                  % (row["outcome"], row.get("rc"), row.get("diff_bytes"),
                     row.get("wall_s")), flush=True)
            print("  transcript: %s" % row.get("transcript"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
