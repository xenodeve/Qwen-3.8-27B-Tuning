r"""A real-task row must name the model and the build, read from the server's own log.

THE GAP THIS CLOSES, and it is the third instance of one mistake.

  2026-08-24, morning: two llama-server builds print identical version strings
  and differ 2.2x in prefill. Rows did not record the binary -> provenance.py,
  test_exe_provenance.py.

  2026-08-24, afternoon: two model files, one 6,929 MiB and one 9,373 MiB, run
  the identical arm. Arena rows did not record the model -> resolve_target,
  test_target_provenance.py.

  2026-08-24, evening: the two REAL-TASK rows that actually compared those two
  models came back with `target` absent from both:

      real-task-dflash2ngram.jsonl   FAIL          hw=69401  target=n/a
      real-task-q2kxl.jsonl          WINDOW_BOUND  hw=98303  target=n/a

`real_task_bench.py` deliberately does not start the server -- "starting the
thing under test from inside the measurement is how you end up measuring the
starter" -- so it never sees `-m`. The two rows comparing the artifacts at the
centre of the decision are, in the files, indistinguishable by model. The only
reason anyone knows which is which is that a human set an environment variable
and remembers doing it, which is not evidence.

WHY THE LOG AND NOT THE HTTP API

The harness already opens the server log to read `ctx_high_water`, so this costs
one more scan of a file it has in hand, and it works after the fact on a log from
a run that has already finished. An HTTP call only answers while the server is
still up.

llama.cpp writes both facts at startup, before any request:

    srv    load_model: loading model 'C:\...\Qwen3.8-27B-UD-Q2_K_XL.gguf'
    common_param: system_info: ... CUDA : ARCHS = 890,1200 | USE_GRAPHS = 1 |
                  BLACKWELL_NATIVE_FP4 = 1 | CPU : SSE3 = 1 | ...

The second line is worth as much as the first: it carries the compiled
architecture list AND the feature flags, so a row can say the build had
`ARCHS = 890,1200` rather than requiring cuobjdump on a dll that may since have
been replaced.

BOTH READS START AT BYTE ZERO, not at the harness's rolling offset. The boot
happens before the first task, so an offset-relative read would find neither.

WHAT THIS CANNOT DO is notice that the log belongs to a different server than
the one answering on 8080. `log_fault()` covers the case where the log is stale
by checking that it grew; nothing here re-checks identity per task.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import real_task_bench as rtb

BOOT = (
    "0.00.092.319 I cmn  common_param: system_info: n_threads = 18 "
    "(n_threads_batch = 18) / 20 | CUDA : ARCHS = 890,1200 | USE_GRAPHS = 1 | "
    "BLACKWELL_NATIVE_FP4 = 1 | CPU : SSE3 = 1 | AVX2 = 1 | \n"
    "0.00.110.494 I srv    load_model: loading model "
    "'C:\\cache\\Qwen3.8-27B-UD-Q2_K_XL.gguf'\n"
    "0.00.110.498 I srv    load_model: local path "
    "'C:\\cache\\Qwen3.8-27B-UD-Q2_K_XL.gguf'\n"
    "0.00.151.462 D llama_model_loader: loaded meta data with 47 key-value pairs "
    "and 81 tensors from C:\\cache\\Qwen3.8-27B-DFlash2-Q4_K_M.gguf\n"
)


def write_log(tmp_path, text):
    p = tmp_path / "server.log"
    p.write_text(text, encoding="utf-8")
    return str(p)


# ------------------------------------------------------------------ the model

def test_the_target_model_is_read_from_the_boot_line(tmp_path):
    assert rtb.server_model(write_log(tmp_path, BOOT)) == \
        r"C:\cache\Qwen3.8-27B-UD-Q2_K_XL.gguf"


def test_the_drafter_is_not_mistaken_for_the_target(tmp_path):
    """The drafter appears in the log too, via `loaded meta data ... from`, and
    on this machine it is loaded FIRST. Matching that line instead would label
    every dflash2 row with the 1.05 GiB drafter as its model."""
    got = rtb.server_model(write_log(tmp_path, BOOT))
    assert "DFlash2" not in got


def test_a_log_without_the_boot_line_reports_absence(tmp_path):
    """None, not a guess. A row that names the wrong model is worse than a row
    that names none -- the first is believed."""
    assert rtb.server_model(write_log(tmp_path, "nothing useful here\n")) is None


def test_a_missing_log_reports_absence(tmp_path):
    assert rtb.server_model(str(tmp_path / "nope.log")) is None


# ------------------------------------------------------------------ the build

def test_the_build_info_is_read_and_carries_the_arch_list(tmp_path):
    info = rtb.server_build_info(write_log(tmp_path, BOOT))
    assert "ARCHS = 890,1200" in info
    assert "BLACKWELL_NATIVE_FP4 = 1" in info


def test_the_build_info_is_absent_rather_than_empty(tmp_path):
    assert rtb.server_build_info(write_log(tmp_path, "no system_info line\n")) is None


# ---------------------------------------------------------------- reading rule

def test_both_reads_start_at_byte_zero_not_at_an_offset(tmp_path):
    """The boot precedes every task, so a read relative to the harness's rolling
    offset would find neither line. Padding the front must not hide them."""
    padded = ("filler\n" * 5000) + BOOT
    path = write_log(tmp_path, padded)
    assert rtb.server_model(path) is not None
    assert rtb.server_build_info(path) is not None


# ------------------------------------------------------------------ the wiring

SRC = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "real_task_bench.py"), encoding="utf-8").read()


def test_the_row_records_both():
    assert 'row["target"] = server_model(' in SRC, \
        "server_model is defined but the row still does not name the model"
    assert 'row["build"] = server_build_info(' in SRC, \
        "the build line is read and dropped"
