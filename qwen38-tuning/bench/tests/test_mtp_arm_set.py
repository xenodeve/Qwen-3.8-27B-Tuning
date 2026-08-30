r"""The MTP arms must use the head baked into the target, and never a sidecar.

WHY THIS SET EXISTS AT ALL, and why it is not a re-run of something measured.

`docs/results/02-decoders.md` carries `draft-mtp` at **+81 % at 16K and -71 % at
131,072**, and the same page records why that measurement could not have used a
baked-in head:

    Can `draft-mtp` run on `UD-IQ2_S` alone?  **No.** "model doesn't contain MTP
    layers" -- the weights are a separate 1.3 GB file passed with `-md`

Every prior figure therefore paid **564 MiB** for a sidecar head on an artifact
that had none. `UD-Q2_K_XL` is different: it reports `n_layer_all = 65`, offloads
`66/66`, and its boot log shows the block loading out of the main file --

    create_tensor: loading tensor blk.64.nextn.eh_proj.weight
    llama_model_loader: - kv 28: qwen35.nextn_predict_layers u32 = 1
    common_speculative_init_result: creating MTP draft context against the
        TARGET model '...Qwen3.8-27B-UD-Q2_K_XL.gguf'

-- so `--spec-type draft-mtp` with **no `-md`** is a configuration this project
has never run.

THE ONE THING THAT WOULD SILENTLY INVALIDATE IT

Passing `-md`. Then the head comes from a file rather than from the target, the
sidecar's weights are added to `fit_params_target` (`server-context.cpp:1074`,
gated only on "was -md given"), and the arm measures the thing that was already
measured while being labelled the thing that was not. A test asserts its absence
rather than trusting the author of the arm.

WHAT THE PROBE ALREADY ESTABLISHED, so nobody re-derives it

Booted at ctx 98,304 on `UD-Q2_K_XL`, `draft-mtp` with no `-md`:

    model 8965.31 | KV 1728.00 | MTP KV 384.00 | RS 598.50 (n_max=3)
    compute 472.27 + 82.01 | total 12,230 MiB, leaving 2,942

against `dflash2+ngram`'s 12,973 leaving 2,199. **743 MiB back, not the 1,394 a
first estimate suggested** -- the model buffer itself grows 334.74 MiB when the
head is used, and the MTP context costs 466 MiB that `--fit` adds to its own
target (768 -> 1234).

WHAT THIS FILE CANNOT DO is check the arms are run against a model that HAS an
MTP head. Nothing in an arm names a model. On an artifact without one the server
refuses at boot with "model doesn't contain MTP layers", which is the correct
loud failure and is why no guard is attempted here.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dflash2_arena as arena

ARMS = arena.ARM_SETS["mtp"]
BY_LABEL = {arena.arm_parts(a)[0]: arena.arm_parts(a) for a in ARMS}


def spec_types(extra):
    """The value following `--spec-type`, split on commas."""
    i = extra.index("--spec-type")
    return extra[i + 1].split(",")


def test_the_set_has_both_arms():
    assert set(BY_LABEL) == {"draft-mtp", "draft-mtp+ngram"}, sorted(BY_LABEL)


def test_neither_arm_passes_a_sidecar_drafter():
    """-md would make the head come from a file instead of the target, and the
    row would carry the label of a configuration it did not run."""
    for label, (_l, extra, _env) in BY_LABEL.items():
        assert "-md" not in extra, f"{label} passes -md; the head must come from the target"
        assert "-ngld" not in extra, f"{label} passes -ngld, which only means anything with -md"


def test_both_arms_request_draft_mtp():
    for label, (_l, extra, _env) in BY_LABEL.items():
        assert "draft-mtp" in spec_types(extra), label


def test_only_one_arm_adds_ngram():
    plain = spec_types(BY_LABEL["draft-mtp"][1])
    paired = spec_types(BY_LABEL["draft-mtp+ngram"][1])
    assert plain == ["draft-mtp"]
    assert "ngram-mod" in paired and "draft-mtp" in paired


def test_the_ngram_arm_carries_the_window_the_profiles_serve():
    """If it used defaults it would be measuring a window nothing ships, and the
    comparison against the incumbent would carry two changes instead of one."""
    extra = BY_LABEL["draft-mtp+ngram"][1]
    for flag, value in (("--spec-ngram-mod-n-match", "12"),
                        ("--spec-ngram-mod-n-min", "16"),
                        ("--spec-ngram-mod-n-max", "32")):
        assert flag in extra, flag
        assert extra[extra.index(flag) + 1] == value, flag


def test_the_two_arms_differ_only_in_the_ngram_half():
    """A delta with two causes cannot be attributed -- the shape of
    CORRECTIONS 26 and 28."""
    plain = [x for x in BY_LABEL["draft-mtp"][1] if x not in ("--spec-type", "draft-mtp")]
    paired = BY_LABEL["draft-mtp+ngram"][1]
    extra_only = [x for x in paired if x not in ("--spec-type", "draft-mtp,ngram-mod")]
    assert plain == [], f"the plain arm carries flags beyond --spec-type: {plain}"
    assert extra_only == arena.NGRAM, extra_only


def test_neither_arm_sets_an_environment_variable():
    for label, (_l, _e, env) in BY_LABEL.items():
        assert env == {}, f"{label} varies the environment as well: {env}"
