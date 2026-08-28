r"""The split mode has never been tried ON NVFP4, and that is not a small gap.

WHAT IS ALREADY KNOWN, AND WHAT IT IS KNOWN ABOUT. `-sm tensor` beat `-sm layer`
by +65.4 % [+64.2, +67.3] at ctx 147,456 and +59.5 % at 16,384, and `-sm row`
cannot load at all on this pair (`device CUDA0 does not support split buffers`).
Every one of those numbers was taken on **UD-Q4_K_XL**, on 2026-08-26, **with
speculation OFF on both sides**.

WHY THAT DOES NOT SETTLE IT. This project holds that a verdict at one depth does
not transfer to another, and this session added: it does not transfer across
ARTIFACTS either -- `n-match 24` lost on UD-Q4_K_XL and wins by +27.1 % here,
`map-k` declined 100 % of its drafts there and is +15.4 % here, and MTP's
prompt-copying turned out to belong to the artifact rather than to MTP. The
split verdict is the last big one still being quoted across an artifact change.

AND THERE IS A MECHANISM, NOT ONLY A CAUTION. Two things this configuration
wants are blocked BY the tensor split specifically:

  1. `set_sampler: backend sampling not supported with SPLIT_MODE_TENSOR; using
     CPU` appears in every boot, and `draft-mtp` announces `backend_sampling=1`
     immediately before it is disabled. This arm set is a speculative pairing;
     the earlier layer-vs-tensor measurement had speculation off on both sides
     and so could not have seen this at all.
  2. no second model loads under `-sm tensor` -- `draft-dflash` aborts in
     `ggml-backend-meta.cpp`, and the vision projector is expected to hit the
     same wall. The register already shows `-sm layer` + `draft-dflash` loading
     and running at 52.11 tok/s, so layer demonstrably CAN host one.

So the honest question is not "is tensor faster" but "is tensor still faster
once the thing being split is NVFP4 and the decoder is the one we would serve".

ONE VARIABLE. Same file, same decoder, same n-gram, same depth, same corpus.
Only `-sm` and the `-ts` that belongs to it move: `-ts` is meaningless under
layer, where llama.cpp already splits by free VRAM and `-ts 1,1` measured +1.8 %,
inside noise.
"""
import os
import sys

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BENCH)

import dflash2_arena as A  # noqa: E402

SET = "nvfp4-split"


def arms():
    return A.ARM_SETS[SET]


def test_the_arm_set_exists():
    assert SET in A.ARM_SETS


def test_it_is_a_pair_and_the_baseline_is_what_we_would_serve():
    labels = [a[0] for a in arms()]
    assert len(labels) == 2, labels
    assert any(l.endswith("-base") for l in labels), labels
    assert labels[0].endswith("-base"), (
        "the baseline must be first so the report names it: %s" % labels)


def test_the_baseline_is_the_tensor_split():
    label, extra, _ = arms()[0]
    assert "tensor" in extra[extra.index("-sm") + 1], extra
    assert "-ts" in extra, "the tensor split without a computed ratio is 0.38 tok/s"


def test_the_arm_is_the_layer_split():
    label, extra, _ = arms()[1]
    assert extra[extra.index("-sm") + 1] == "layer", extra


def test_the_layer_arm_carries_no_ts():
    """Under layer llama.cpp already splits by free VRAM; `-ts 1,1` there
    measured +1.8 %, inside any floor. Passing one would vary two things."""
    _, extra, _ = arms()[1]
    assert "-ts" not in extra, extra


def test_only_the_split_differs():
    """Same model, same decoder, same n-gram, same micro-batch."""
    def without_split(extra):
        out, skip = [], 0
        for i, tok in enumerate(extra):
            if skip:
                skip -= 1
                continue
            if tok in ("-sm", "-ts"):
                skip = 1
                continue
            out.append(tok)
        return out
    a, b = without_split(arms()[0][1]), without_split(arms()[1][1])
    assert a == b, "the arms differ by more than the split:\n  %s\n  %s" % (a, b)


def test_both_arms_run_the_nvfp4_file_with_its_baked_in_head():
    for label, extra, _ in arms():
        assert A.NVFP4_VERY_LOW in extra, label
        assert "draft-mtp,ngram-mod" in extra, label
        assert "-md" not in extra, "%s: the head is IN the file" % label


def test_both_arms_carry_the_retuned_ngram():
    """n-match 24 is the value measured on THIS artifact. Reverting to 12 here
    would compare the split against a differently tuned decoder."""
    for label, extra, _ in arms():
        i = extra.index("--spec-ngram-mod-n-match")
        assert extra[i + 1] == "24", (label, extra[i + 1])


def test_both_arms_ask_for_both_cards():
    for label, extra, env in arms():
        assert env.get("CUDA_VISIBLE_DEVICES") == A.BOTH_CARDS, label
