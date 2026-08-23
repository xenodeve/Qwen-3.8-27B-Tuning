r"""A row must say which MODEL FILE produced it, not only which binary.

WHY THIS EXISTS, and it is the same lesson twice.

`test_exe_provenance.py` was written on 2026-08-24 because two llama-server
builds on this machine print identical version strings and differ 2.2x in
prefill, so a row that did not record `exe` could not be attributed. That fix
recorded the binary. **It did not record the model**, and the very next question
asked of this harness is a quant comparison:

    UD-IQ2_XXS   6,929 MiB   2.16 bpw   the artifact we serve
    UD-Q2_K_XL   9,373 MiB              the artifact an external ladder says
                                        sits above a 10-point quality cliff

Two runs of the same task on those two files produce rows that are identical in
every recorded field -- same arm, same argv, same ctx, same corpus hash, same
exe, same cuda_archs. `dflash2_arena.TARGET` is a module constant and nothing
writes it down.

`docs/results/01-artifacts.md` opens with the reason this matters here:

    Every model file this project has loaded. `bpw` is the real bits per weight
    from the loader's tensor histogram, **not** the filename.

and the register lists TWO different files both named
`Qwen3.8-27B-UD-Q2_K_XL.gguf`, 9,373 MiB and 10,181 MiB, in different snapshot
directories. A filename cannot identify an artifact here. A path and a size can.

WHAT THIS FILE PINS

The override, so both artifacts stay measurable without editing the module; and
the recording, so a JSONL of mixed quants can be separated afterwards. The size
is recorded beside the path because the two Q2_K_XL files differ by 808 MiB and
a moved cache would otherwise make the path a lie.

WHAT IT CANNOT DO is verify the file's contents match its name -- that needs the
loader's own tensor histogram, which only appears in the server log. The register
is where bpw lives; this only guarantees you can tell which log to open.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dflash2_arena as arena
import provenance

V3_IQ2XXS = (r"C:\Users\xenod\.cache\huggingface\hub\models--unsloth--Qwen3.8-27B-GGUF"
             r"\snapshots\27af057ecb382ddfea5d12837360a8980560e3ed"
             r"\Qwen3.8-27B-UD-IQ2_XXS.gguf")
V3_Q2KXL = (r"C:\Users\xenod\.cache\huggingface\hub\models--unsloth--Qwen3.8-27B-GGUF"
            r"\snapshots\27af057ecb382ddfea5d12837360a8980560e3ed"
            r"\Qwen3.8-27B-UD-Q2_K_XL.gguf")


# ---------------------------------------------------------------- the override

def test_resolve_target_falls_through_to_the_default(monkeypatch):
    monkeypatch.delenv(provenance.TARGET_ENV_VAR, raising=False)
    assert provenance.resolve_target(V3_IQ2XXS) == V3_IQ2XXS


def test_resolve_target_honours_the_environment(monkeypatch):
    monkeypatch.setenv(provenance.TARGET_ENV_VAR, V3_Q2KXL)
    assert provenance.resolve_target(V3_IQ2XXS) == V3_Q2KXL


def test_an_empty_variable_does_not_produce_an_empty_model_path(monkeypatch):
    monkeypatch.setenv(provenance.TARGET_ENV_VAR, "")
    assert provenance.resolve_target(V3_IQ2XXS) == V3_IQ2XXS


def test_the_two_env_vars_are_distinct():
    """One variable for both would make pointing at a different model silently
    also point at a different binary."""
    assert provenance.ENV_VAR != provenance.TARGET_ENV_VAR


# --------------------------------------------------------------- the recording

def test_model_size_reads_the_file(tmp_path):
    p = tmp_path / "fake.gguf"
    p.write_bytes(b"x" * 1234)
    assert provenance.model_size_mib(str(p)) == pytest.approx(1234 / 1048576)


def test_model_size_reports_absence_rather_than_zero():
    """0.0 MiB would read as an empty model rather than a missing one, and a
    missing model is the case where the path in the row is wrong."""
    assert provenance.model_size_mib(r"C:\AI\nope\missing.gguf") is None


@pytest.mark.skipif(not (os.path.isfile(V3_IQ2XXS) and os.path.isfile(V3_Q2KXL)),
                    reason="both artifacts needed to prove they are distinguishable")
def test_the_two_artifacts_are_distinguishable_by_size():
    """The whole point. Same vendor, same snapshot, same naming scheme -- and
    9,373 against 6,929 MiB is what tells two rows apart afterwards."""
    a = provenance.model_size_mib(V3_IQ2XXS)
    b = provenance.model_size_mib(V3_Q2KXL)
    assert round(a) != round(b)
    # 7,266,070,528 and 9,828,981,664 bytes exactly. The decimal-GB forms --
    # 7.27 and 9.83 -- are what the external ladder calls "7.3 GB" and "9.8 GB",
    # which is how those two artifacts were identified as ours.
    assert round(a) == 6929 and round(b) == 9374


# ------------------------------------------------------------------ the wiring

SRC = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "dflash2_arena.py"), encoding="utf-8").read()


def test_the_arena_resolves_its_target_through_provenance():
    assert "TARGET = resolve_target(" in SRC, \
        "TARGET is still a bare constant; the override cannot reach it"


def test_server_argv_launches_the_module_target():
    argv = arena.server_argv(16384, [])
    assert argv[argv.index("-m") + 1] == arena.TARGET


def test_the_row_records_the_target_and_its_size():
    assert 'target=TARGET' in SRC, "the row does not record which model ran"
    assert "model_size_mib(TARGET)" in SRC, \
        "the path is recorded without the size, so a moved cache makes it a lie"
