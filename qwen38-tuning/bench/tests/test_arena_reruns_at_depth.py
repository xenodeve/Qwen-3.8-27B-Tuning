r"""The six shallow-screen questions, re-asked where an effect can be seen.

`GGML_CUDA_ALLREDUCE` was screened at ctx 16,384 and looked as flat as everything
else in that sweep; at 147,456 on the real corpus it is worth **24 %**, resolved
at a 0.3 % per-arm spread. So "no effect at 16,384" was never a verdict about the
served depth -- it was a statement about an instrument that could not see one.
Every arm set here re-asks one of those questions, plus `--kv-unified`, which had
never been tested at any depth.

The invariant these tests hold is the one that makes the answers comparable with
everything else measured in this campaign: **the control of each set is
`nvfp4-final`'s winning arm object itself**, and each other arm differs from it by
exactly one thing -- one argv addition, one argv substitution, or one environment
entry. Nothing here checks a number.
"""
import os
import sys

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BENCH)

import dflash2_arena as arena  # noqa: E402
import pytest  # noqa: E402

SETS = ["spec-order", "mtp-nmax", "mtp-pmin", "backend-sampling",
        "launch-queues", "kv-unified", "builds-nvfp4"]


def _winning_arm():
    return dict((a[0], a[1]) for a in arena.ARM_SETS["nvfp4-final"])["nvfp4-mtp+nm24"]


@pytest.mark.parametrize("name", SETS)
def test_every_control_is_the_winning_arm_itself(name):
    assert arena.ARM_SETS[name][0][1] == _winning_arm()


@pytest.mark.parametrize("name", SETS)
def test_every_arm_differs_from_its_control_by_at_most_one_thing(name):
    arms = arena.ARM_SETS[name]
    base_argv, base_env = arms[0][1], arms[0][2]
    for arm_name, argv, env in arms[1:]:
        added = [x for x in argv if x not in base_argv]
        replaced = len(argv) == len(base_argv) and argv != base_argv
        env_delta = {k: v for k, v in env.items() if base_env.get(k) != v}
        moved = (1 if added or replaced else 0) + (1 if env_delta else 0)
        assert moved == 1, (name, arm_name, added, env_delta)


def test_the_build_set_pins_a_binary_on_every_arm_including_the_control():
    """`arm_exe` falls back to the module EXE. CORRECTIONS 41 is what a build
    comparison looks like when one arm inherits it."""
    for arm_name, _, env in arena.ARM_SETS["builds-nvfp4"]:
        assert arena.ENV_VAR in env, arm_name
    pinned = [env[arena.ENV_VAR] for _, _, env in arena.ARM_SETS["builds-nvfp4"]]
    assert len(set(pinned)) == len(pinned), pinned


def test_the_order_set_really_reverses_the_decoder_order():
    got = [" ".join(a[1]).split("--spec-type ")[1].split(" ")[0]
           for a in arena.ARM_SETS["spec-order"]]
    assert got == ["draft-mtp,ngram-mod", "ngram-mod,draft-mtp"], got
