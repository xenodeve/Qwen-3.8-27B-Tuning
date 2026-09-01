r"""The `cpumoe` arm set, run the way the +63.1 % row was run.

This family was measured twice at ctx 16,384 with a short synthetic prompt. The
first pass reported `GGML_OP_OFFLOAD_MIN_BATCH=8` at **+8.01 %**; a second pass
with the arms rotated through every position brought the same arm to **+0.38 %**
and showed VRAM identical across all four to within 14 MiB. Neither run is
comparable with `nvfp4-final-147456.jsonl`, and the shallow instrument has since
been shown to miss a 24 % effect that the deep one resolves at 0.3 % spread
(`allreduce-147456.jsonl`).

So the arms move to the arena, on the configuration the +63.1 % figure was
measured on, and these tests hold what makes the run readable:

  * the control IS `nvfp4-final`'s winning arm, not a retyped copy of it
  * every other arm is that control plus exactly ONE thing
  * `minbatch8` differs only in the environment, so its argv must match the
    control byte for byte

`--cpu-moe` is kept as the maximum-effect arm: it sends every MoE weight to the
host. If the family does anything at all on this artifact, that is where it
shows.
"""
import os
import sys

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BENCH)

import dflash2_arena as arena  # noqa: E402


def _arms():
    return dict((a[0], (a[1], a[2])) for a in arena.ARM_SETS["cpumoe"])


def test_the_arm_set_exists_with_the_four_arms_that_were_measured_shallow():
    assert sorted(_arms()) == ["cmoe-all", "minbatch8", "ncmoe8", "off"]


def test_the_control_is_the_arm_the_63_percent_figure_was_measured_on():
    ours = _arms()["off"][0]
    theirs = dict((a[0], a[1]) for a in arena.ARM_SETS["nvfp4-final"])["nvfp4-mtp+nm24"]
    assert ours == theirs


def test_each_flag_arm_is_the_control_plus_exactly_one_thing():
    base = _arms()["off"][0]
    assert _arms()["ncmoe8"][0] == base + ["--n-cpu-moe", "8"]
    assert _arms()["cmoe-all"][0] == base + ["--cpu-moe"]


def test_the_env_arm_does_not_also_move_argv():
    """`GGML_OP_OFFLOAD_MIN_BATCH` is an environment variable. If its arm also
    changed the command line, the row would have two causes."""
    base, base_env = _arms()["off"]
    argv, env = _arms()["minbatch8"]
    assert argv == base
    assert env["GGML_OP_OFFLOAD_MIN_BATCH"] == "8"
    assert {k: v for k, v in env.items() if k != "GGML_OP_OFFLOAD_MIN_BATCH"} == dict(base_env)


def test_no_arm_quietly_changes_the_split_or_the_target():
    """Residency before arithmetic: an arm that moved `-ts` or the model file
    would produce a delta this project has already had to retract once."""
    base = " ".join(_arms()["off"][0])
    for name, (argv, _) in _arms().items():
        joined = " ".join(argv)
        assert "-sm tensor -ts 7819,15490 -ub 1024" in joined, name
        assert arena.NVFP4_VERY_LOW in argv, name
        assert joined.startswith(base) or joined == base, name
