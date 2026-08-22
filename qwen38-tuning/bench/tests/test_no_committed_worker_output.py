"""Model-emitted code from a benchmark run is never committed.

THE STANDING INSTRUCTION. The developer's rule for benchmark runs is
"ลบ code ที่พ่นตอน benchmark ให้หมดด้วย" -- delete all the code the benchmark
spat out -- because a benchmark's output is a measurement artifact and not
something this project ships. `real_task_bench.py` obeys it for the clones it
makes under the scratch root, enforced by `harness.assert_deletable` and
`test_scratch_safety.py`. The runners that generate code IN TREE had no such
guard.

WHAT WAS ACTUALLY WRONG. `run_bench.py` writes model output to `bench/_work/`
and `.gitignore` covers it. `run_deep_bench.py` writes to `bench/_deepwork/`
and nothing covered it, so **314 generated files, 557 KB, sat committed in the
repository**. Two runners, the same class of artifact, one line of .gitignore
between them -- and the one that was missed is the one nobody looked at again.

WHY THIS TEST IS THE SHAPE IT IS. It does not name `_deepwork`, and it does not
read any runner's source to discover where that runner writes. It asserts the
invariant directly against git: nothing under a `bench/_*` scratch directory is
tracked. A future runner that invents `_widework/` is caught without anyone
remembering to update a list -- which is exactly the kind of remembering that
failed here.
"""

import os
import subprocess

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(BENCH))


def tracked_under(pathspec):
    r = subprocess.run(["git", "ls-files", "--", pathspec],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return [line for line in r.stdout.splitlines() if line.strip()]


def test_no_generated_scratch_directory_under_bench_is_tracked():
    """`bench/_*` is the naming convention for benchmark scratch. None of it ships."""
    tracked = tracked_under("qwen38-tuning/bench/_*")
    assert tracked == [], (
        "%d generated file(s) are committed under bench/_*; the first few are %s. "
        "Benchmark output is deleted, not shipped -- remove them and add the "
        "directory to .gitignore beside qwen38-tuning/bench/_work/."
        % (len(tracked), tracked[:5])
    )
