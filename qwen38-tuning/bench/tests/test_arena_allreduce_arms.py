r"""The `allreduce` arm set, and the one-variable property it has to keep.

`GGML_CUDA_ALLREDUCE` is task #48 and has never been run. Under `-sm tensor`
every layer pays an all-reduce across the two cards on every token, so it is the
one environment variable that sits directly on this machine's decode path.
`ggml-cuda.cu:1231-1240` accepts `nccl | internal | none`; NCCL is not compiled
into our binary (the code warns and falls back), and Windows defaults to
`internal` -- so the A/B that exists here is the default against `none`.

It was first measured at ctx 16,384 with a short synthetic prompt, and the
developer rejected that: the figure this project quotes for the served pairing
(+63.1 %) came from `nvfp4-final-147456.jsonl` -- ctx 147,456, regime
`real-code-vendor`, `-ts 7819,15490`, `tg_med` over samples, with
`free_before`/`free_after` recorded per row. A number from a different
instrument cannot be compared with it.

So the arm set below runs inside the arena, on the winning configuration, and
these tests hold the two properties that make its output comparable:

  * both arms carry BYTE-IDENTICAL argv, so the only difference is the variable
  * the argv is the +63.1 % pairing, not a retyped approximation of it
"""
import os
import sys

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BENCH)

import dflash2_arena as arena  # noqa: E402


def test_the_arm_set_exists():
    assert "allreduce" in arena.ARM_SETS


def test_both_arms_carry_byte_identical_argv():
    """The whole point is one variable. CORRECTIONS 26 and 28 are what a delta
    with two causes looks like afterwards."""
    arms = arena.ARM_SETS["allreduce"]
    assert len(arms) == 2
    argvs = [a[1] for a in arms]
    assert argvs[0] == argvs[1], "the arms differ in argv, so the env is not the only variable"


def test_only_the_env_differs_and_it_is_the_variable_under_test():
    arms = dict((a[0], a[2]) for a in arena.ARM_SETS["allreduce"])
    names = sorted(arms)
    assert names == ["allreduce-none", "internal-default"], names
    base = arms["internal-default"]
    none = arms["allreduce-none"]
    assert "GGML_CUDA_ALLREDUCE" not in base, "the control must not set the variable"
    assert none["GGML_CUDA_ALLREDUCE"] == "none"
    # everything else in the environment has to match, or the env is two variables
    assert {k: v for k, v in none.items() if k != "GGML_CUDA_ALLREDUCE"} == dict(base)


def test_the_argv_is_the_configuration_the_63_percent_figure_was_measured_on():
    """Not an approximation of it: the served NVFP4 + MTP + n-match 24 pairing,
    on the same `-ts` the 147,456 run used."""
    argv = arena.ARM_SETS["allreduce"][0][1]
    joined = " ".join(argv)
    assert "-sm tensor -ts 7819,15490 -ub 1024" in joined
    assert "draft-mtp,ngram-mod" in joined
    assert "--spec-ngram-mod-n-match 24" in joined
    assert "--spec-draft-n-max 3" in joined
    assert arena.NVFP4_VERY_LOW in argv


def test_the_control_arm_matches_the_nvfp4_final_arm_it_must_reproduce():
    """If the control drifts from `nvfp4-final`'s winning arm, the run measures
    something the +63.1 % figure never described."""
    ours = arena.ARM_SETS["allreduce"][0][1]
    theirs = dict((a[0], a[1]) for a in arena.ARM_SETS["nvfp4-final"])["nvfp4-mtp+nm24"]
    assert ours == theirs
