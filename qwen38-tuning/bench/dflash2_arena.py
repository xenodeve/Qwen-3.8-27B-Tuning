"""DFlash2 against the incumbent, paired within a round, on ONE binary.

THE QUESTION. `ngram-mod` is this project's speculation champion because it
costs 0 MiB. DFlash2 costs 1.1 GB of a 12 GB card. Public numbers are all from
bigger hardware -- atomic.chat's 47.4 -> 140.6 tok/s is an RTX 6000 -- so the
only way to know what it does here is to run it here.

WHY EVERY ARM RUNS ON BUILD 10499, INCLUDING THE ONES ALREADY MEASURED.
DFlash2 needs the PR build; `ngram-mod`'s recorded numbers are from 10472.
Comparing across those two would confound the decoder with the build, and
nothing in the result could separate them. So the incumbent is re-measured
here, on the same binary, in the same rounds.

WHY ARMS ALTERNATE. Free VRAM at boot moves 9,326-10,732 MiB and `--fit`
follows it; raw decode is not comparable across boots. Arms are run
round-robin and rotated each round, and the verdict comes from
`harness.paired_deltas`, which reports a range and refuses to call an effect
resolved unless it clears 13.6 % AND keeps its sign.

WHY n_predict IS NOT 160. Every decoder verdict this project holds was decided
on 160-token generations, and whether that understates speculation is an OPEN
question in the ledger (CORRECTIONS 8). Measuring a drafter with the budget
that is under suspicion for hiding drafters would be walking into a trap we
wrote down ourselves.

WHY THE LAYER SPLIT IS PARSED WITH expect_layers. A drafter adds its own
assignment passes and it is assigned LAST, so the default read of this log
returns the drafter's (6, 0) -- a healthy-looking split for the wrong model, in
which a spill of the target's 65 layers cannot appear. Issue #17.

WHAT THIS CANNOT TELL YOU. Throughput, not task success. This project's metric
is verified accepted coding tasks per hour, and a decoder whose output is
byte-identical to no-speculation (ngram) and one whose output is not are not
interchangeable on that metric just because tok/s says so.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gpu_device
from harness import (observed_spread_pct, classify_against_floors,
                     residency_note, archs_missing_for_gpus,
                     NOISE_FLOOR_PCT,
                     median, parse_layer_split, target_layer_count,
                     generation_is_original, copied_window_fraction,
                     draft_acceptance,
                     paired_deltas, vram_settled, VRAM_MIN_RISE_MIB,
                     parse_spec_impl_stats, generation_is_measurable)
from provenance import (resolve_exe, resolve_target, resolve_effort,
                        cuda_archs, model_size_mib, ENV_VAR)

ROOT = Path(r"C:\AI\qwen38-tuning")
# THE SERVED BINARY, so the arena measures what we serve unless told otherwise.
# This defaulted to C:\AI\llama.cpp-dflash2 -- CMAKE_CUDA_ARCHITECTURES=89, 141
# sm_89 cubins, no sm_120a, no PTX -- long after a compute-capability-12.0 card
# was installed. That produced fifteen published rows on the wrong machine
# (2026-08-27) and the boot-time guard has stopped four launches since, the last
# on 2026-08-29 while starting the split-mode sweep. A guard firing that often
# is a guard working around a wrong default.
#
# QWEN38_LLAMA_EXE still overrides it -- that is how the patched mirror build
# gets measured -- and the boot-time architecture check still runs, because a
# default cannot know what card comes next.
DEFAULT_EXE = r"C:\AI\llama.cpp-blackwell\llama-server.exe"
EXE = resolve_exe(DEFAULT_EXE)
TARGET = resolve_target(
    r"C:\Users\xenod\.cache\huggingface\hub\models--unsloth--Qwen3.8-27B-GGUF"
    r"\snapshots\27af057ecb382ddfea5d12837360a8980560e3ed"
    r"\Qwen3.8-27B-UD-IQ2_XXS.gguf")
DRAFTER = (r"C:\Users\xenod\.cache\huggingface\hub"
           r"\models--z-lab--Qwen3.8-27B-DFlash2-GGUF"
           r"\snapshots\57ab3265056d4024870b0621cfc2c127537020ed"
           r"\Qwen3.8-27B-DFlash2-Q4_K_M.gguf")
# NVFP4 with the MTP head baked in -- no -md sidecar, no mirror patch, and it
# runs on the SERVED binary. Verified from the GGUF header: 448 NVFP4 tensors.
NVFP4_VERY_LOW = (r"C:\Users\xenod\.cache\huggingface\hub"
                  r"\models--esatapedico--Qwen3.8-27B-NVFP4-MTP-GGUF"
                  r"\snapshots\bcd7a7d3e251d4ec0fd15c72584b5eb9e0981383"
                  r"\Qwen3.8-27B-NVFP4-MTP-VERY-LOW.gguf")

# The 535 MiB DFlash2 drafter, not the 1,090 MiB one this project started with.
# Measured 2026-08-27: its Meta buffer is 538.42 MiB against 786.35, it reaches
# 163,840 where Q4_K_M does not, and its author's own table puts throughput
# within a few percent of the larger file at every n_max.
DFLASH_SMALL = (r"C:\Users\xenod\.cache\huggingface\hub"
                r"\models--HermiHg--Qwen3.8-27B-DFlash2-Q2_K_S-MIX-GGUF"
                r"\snapshots\3a802866ab98104e56d2c0b33442004b5b39ab08"
                r"\Qwen3.8-27B-DFlash2-Q2_K_S-MIX.gguf")

BASE = "http://127.0.0.1:8080"

# Set explicitly from 2026-08-24. Everything before that date ran at the chat
# template's `xhigh` with an unlimited budget -- never chosen, just never set --
# so NOTHING measured after this line is comparable to an earlier figure without
# saying so. The row records it for that reason.
EFFORT = resolve_effort()
N_PREDICT = 512
N_GEN = 3                   # timed generations per arm per round, after a warm turn

# ngram-mod's tuned window, copied from worker-iq2xxs-deep.ps1 so the incumbent
# is measured as it is actually served rather than at defaults.
NGRAM = ["--spec-ngram-mod-n-match", "12",
         "--spec-ngram-mod-n-min", "16", "--spec-ngram-mod-n-max", "32"]
DFLASH = ["-md", DRAFTER, "--spec-draft-n-max", "4", "-ngld", "99"]

# The decoder every worker-*.ps1 actually serves. Named once so an arm set that
# means "the incumbent, with one thing changed" cannot drift from the incumbent
# by being retyped. `arm_parts` copies it, so sharing the object is safe.
SERVED_NGRAM = ["--spec-type", "ngram-mod"] + NGRAM

# The two-card serving shape, held constant wherever the arm varies
# something else. `-sm tensor` WITHOUT a ratio is the even split, and on
# this pair that is the 0.38 tok/s configuration (CORRECTIONS 33) -- so the
# ratio travels with the split mode, never separately.
DUAL_TENSOR = ["-sm", "tensor", "-ts", "7819,15490", "-ub", "1024"]

# The other split mode, as it is MEANT to be used: no `-ts`. Under `-sm layer`
# llama.cpp already divides by free VRAM, and a ratio there measured +1.8 %
# [+0.6, +4.1] -- inside any floor. Passing one would vary two things at once.
# `-ub` is held at the tensor arm's value so the micro-batch is not a second
# variable either.
DUAL_LAYER = ["-sm", "layer", "-ub", "1024"]

# The two cards, by UUID. Indexes are a position in an enumeration the driver
# can reorder; after a reorder an index keeps working and means a different
# card (issue #50). SUPER_4070 is the RETIRED 12 GB card -- it is named here
# only so issue #51 can measure what adding it does, never as a default.
TI_5060 = gpu_device.SERVED_GPU_UUID
SUPER_4070 = "GPU-fba37e4b-ea9e-66e9-c3fd-a16b2e833bc4"
BOTH_CARDS = SUPER_4070 + "," + TI_5060

ARMS = [
    ("none", []),
    ("ngram-mod", SERVED_NGRAM),
    ("dflash2", ["--spec-type", "draft-dflash"] + DFLASH),
    # Verified supported by reading common/arg.cpp:4155 -- --spec-type is a
    # comma-separated list -- not by trusting a forum post.
    ("dflash2+ngram", ["--spec-type", "draft-dflash,ngram-mod"] + DFLASH + NGRAM),
]


def arm_parts(arm):
    """Normalise an arm to `(label, extra_argv, env)`.

    Arms are `(label, extra)` or `(label, extra, env)`. Most of what llama.cpp's
    CUDA backend can be told is a flag, but twelve knobs are environment
    variables (`grep getenv ggml/src/ggml-cuda/`), and at least one of them --
    `GGML_CUDA_GRAPH_OPT` -- is an optimisation that is off unless asked for.
    Without this, testing one meant exporting it and re-running the whole sweep,
    which is a comparison ACROSS BOOTS and forbidden here.

    A malformed arm raises rather than defaulting: silently reading it as
    `(label, [])` would run the control config under the arm's name and publish
    the result as the arm's.
    """
    if len(arm) == 2:
        label, extra = arm
        env = {}
    elif len(arm) == 3:
        label, extra, env = arm
    else:
        raise ValueError("arm must be (label, extra) or (label, extra, env), "
                         "got %r" % (arm,))
    return label, list(extra), dict(env)


def launch_env(env):
    """The process environment with the GPU pin, then `env`, layered on top.

    Layered, not replaced: llama-server needs PATH and CUDA_PATH to start, and a
    bare dict would fail in a way that looks like a bad flag.

    The pin sits BELOW `env` so an arm can lift it -- issue #51 measures both
    cards on purpose, and an arm set is the right place to say so. It sits above
    nothing else: without it, `--main-gpu` defaults to 0, which on this machine
    is the retired 4070 SUPER, and llama.cpp spreads layers over both cards
    while every row keeps claiming one (#50).

    `pin_env()` raises if the card is absent. That is deliberate -- an absent
    UUID leaves llama-server with no devices and it runs on CPU, which produces
    a row rather than an error.
    """
    # `env or {}` and not `env`: `run_arm`'s own signature defaults it to None,
    # so the control arm reaches here as None the moment anything calls this
    # without going through an arm set. It raised `'NoneType' object is not a
    # mapping` -- loud, which is why it survived unnoticed; the fix is to make
    # "no arm environment" mean the empty one it already means everywhere else.
    return {**os.environ, **gpu_device.pin_env(), **(env or {})}


def _ngram(n_min, n_match=12, n_max=32):
    return ["--spec-ngram-mod-n-match", str(n_match),
            "--spec-ngram-mod-n-min", str(n_min),
            "--spec-ngram-mod-n-max", str(n_max)]


def _nvfp4_mtp(ngram="ngram-mod"):
    """The NVFP4 target with its baked-in MTP head, and one n-gram beside it.

    n-max 3 is draft-mtp's own default (common.h:325) and what the +41.2 %
    measurement used. The head is in the file, so no -md.
    """
    return ["-m", NVFP4_VERY_LOW,
            "--spec-type", "draft-mtp," + ngram,
            "--spec-draft-n-max", "3"]


def _pair(extra_ngram=None, n_draft=4, extra=()):
    return (["--spec-type", "draft-dflash,ngram-mod",
             "-md", DRAFTER, "--spec-draft-n-max", str(n_draft), "-ngld", "99"]
            + (extra_ngram if extra_ngram is not None else NGRAM) + list(extra))


# Named arm sets. The default set answers "which decoder"; the others answer
# "which setting of the decoder we already chose", which is where the measured
# levers are.
# ---- the build A/B ----------------------------------------------------------
# ONE binary against another, alternating inside one session, which is the only
# admissible way to ask this. Icon B against icon 7 gave +26 % at matched depth,
# one reading per side, in different boots -- and CORRECTIONS 23 measured the
# same arm drifting 48.9 % across boots at depth. That number cannot carry a
# verdict; this arm set can.
#
# IT IS STILL UNANSWERED, and one attempt has already failed silently.
# `layer-pairings` looked like it settled this on 2026-08-30 and did not: every
# arm launched the module default while every row recorded the pin, so the two
# "builds" were one binary and the null it produced was an artefact
# (CORRECTIONS 40 and 41). `start()` is fixed and this set is the instrument.
# Until it is RUN, +26 % is CONTESTED -- not confirmed, not refuted.
#
# BOTH ARMS CARRY THE SAME `extra`. The only difference is the binary, because
# an arm set that also moved a flag produces a delta with two causes -- the
# shape of CORRECTIONS 26 and 28.
#
# THE PATH ON ONE SIDE ONLY IS DELIBERATE. Studio's binary finds NO CUDA device
# with a bare PATH and serves from the CPU without saying so; CUDA 13 keeps
# cudart64_13.dll in %CUDA_PATH%ind, not in. Ours ships its own cublas
# beside it and needs nothing, so adding the directory to both sides would be
# adding a variable to the arm that did not need it.
STUDIO_EXE = os.path.join(os.path.expanduser("~"), ".unsloth", "llama.cpp", "build", "bin", "Release", "llama-server.exe")
_CUDA = os.environ.get("CUDA_PATH", "")
STUDIO_ENV = {
    ENV_VAR: STUDIO_EXE,
    "PATH": ";".join([os.path.dirname(STUDIO_EXE),
                      os.path.join(_CUDA, "bin"),
                      os.path.join(_CUDA, "bin", "x64"),
                      os.environ.get("PATH", "")]),
}

BUILD_AB = [
    ("build-10499-ours", SERVED_NGRAM),
    ("build-10679-unsloth", SERVED_NGRAM, STUDIO_ENV),
]

# OUR side of a build A/B, PINNED rather than inherited. `arm_exe` falls back to
# the module `EXE`, which `resolve_exe` reads from QWEN38_LLAMA_EXE at import --
# and exporting that variable is exactly how the patched mirror gets measured.
# A developer with it set would turn a build A/B into B/B or A/C without
# changing a single label, and every row would still carry a plausible number.
SERVED_ENV = {ENV_VAR: DEFAULT_EXE}

# The artifact with the nextn head in the main file. `draft-mtp` has no sidecar,
# so on an artifact without one it does not merely lose -- it cannot run. Named
# here rather than reached through QWEN38_TARGET: the three-way at 65,536 was
# run by exporting that variable, which makes the arm set an incomplete
# description of its own experiment.
Q4_K_XL = (r"C:\Users\xenod\.cache\huggingface\hub"
           r"\models--unsloth--Qwen3.8-27B-GGUF"
           r"\snapshots\f1bfb127c64f7072bdd2cad55f258b9c8b2910fe"
           r"\Qwen3.8-27B-UD-Q4_K_XL.gguf")


# The PATCHED build. `draft-dflash` under `-sm tensor` aborts without it, and
# the existing tensor dflash arm sets reach it by requiring QWEN38_LLAMA_EXE to
# be exported -- which makes an arm set an incomplete description of its own
# experiment, and turns a forgotten export into either a dead sweep or, worse,
# a live one on whatever the variable did hold.
MIRROR_EXE = r"C:\AI\llama.cpp-mirror\build-mirror\bin\llama-server.exe"
MIRROR_ENV = {ENV_VAR: MIRROR_EXE, "CUDA_VISIBLE_DEVICES": BOTH_CARDS}


def _layer_pairing(decoder, extra):
    """One decoder's argv under the layer split, identical on both builds.

    Built once and shared by the two arms so the pair CANNOT drift: if the two
    builds ran different flags for one decoder, "their build is faster" and
    "their arm had a different flag" would be the same number. `arm_parts`
    copies the list, so sharing the object is safe.
    """
    return (DUAL_LAYER + ["-m", Q4_K_XL, "--spec-type", decoder]
            + list(extra) + NGRAM)


# ---- 2026-08-30: DFlash2 against MTP on BOTH builds, issue #56 --------------
#
# THE SPLIT MODE IS FORCED, NOT CHOSEN. Under `-sm tensor` DFlash2 cannot load
# on build 10679 at all: it aborts at ggml-backend-meta.cpp:543,
# GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0), the same assertion
# our own unpatched binary dies on (#52, 5f87e12). Probed at ctx 65,536 on
# UD-Q4_K_XL with both cards, all four combinations load under `-sm layer` and
# only there. Our SERVED binary runs DFlash2 there with no mirror patch, which
# is what makes a same-flags comparison possible at all.
#
# SO NOTHING HERE MAY BE COMPARED TO `dual-pairings`. That table is `-sm
# tensor` on the patched mirror; this is `-sm layer` on two unpatched builds.
# Two variables move between them. The comparison is valid INSIDE this run.
#
# WHY OUR BUILD RUNS IN THE SAME ROUNDS. `+26 % from the newer build` rests on
# two single readings from two different boots, against a measured 48.9 %
# same-arm drift at this depth (CORRECTIONS 23). Quoting the existing tables
# for our side would repeat that error while looking like a control.
#
# WHY THE INCUMBENT IS IN IT. Without `ngram-mod` on each build the two drafter
# figures have nothing to be a percentage of, and the run answers a comparison
# nobody asked for.
#
# THE DRAFT DEPTHS ARE DELIBERATE AND DIFFERENT. 4 for DFlash2 and 3 for MTP
# are the values every figure in 02-decoders.md was taken at; matching them to
# each other would make this incomparable with the whole register to remove a
# difference neither drafter shares a meaning for -- MTP's ceiling is unread,
# DFlash2's is block_size - 1 = 7.
LAYER_PAIRINGS = [
    (label % build, extra, {**env, "CUDA_VISIBLE_DEVICES": BOTH_CARDS})
    for build, env in (("b10499", SERVED_ENV), ("b10679", STUDIO_ENV))
    for label, extra in (
        ("ngram-mod %s", _layer_pairing("ngram-mod", [])),
        # NO -md. The head is inside UD-Q4_K_XL; a sidecar would add about
        # 1.4 GB for nothing and quietly make this a different experiment.
        ("mtp+ngram %s", _layer_pairing("draft-mtp,ngram-mod",
                                        ["--spec-draft-n-max", "3"])),
        ("dflash+ngram %s", _layer_pairing("draft-dflash,ngram-mod",
                                           ["-md", DRAFTER, "-ngld", "99",
                                            "--spec-draft-n-max", "4"])),
    )
]


def _tensor_arm(decoder, extra):
    """One arm of the draft-depth set, on the tensor split with its ratio.

    `-ts` travels with `-sm tensor` and is never optional: the even split is the
    0.38 tok/s configuration on this pair (CORRECTIONS 33).
    """
    return (DUAL_TENSOR + ["-m", Q4_K_XL, "--spec-type", decoder]
            + list(extra) + NGRAM)


# ---- 2026-08-30: --spec-draft-n-max 4 against 7 where DFlash2 is fastest -----
#
# `draft-dflash,ngram-mod` under `-sm tensor` on the patched mirror is the
# fastest paired figure this project holds: 65.1 / 64.3 / 63.8 tok/s at ctx
# 65,536, +123.8 % over the incumbent (results/dual-pairings-65536.jsonl).
# EVERY ONE OF THOSE ROUNDS RAN AT n_max 4 -- a value the ledger records as
# "chosen without knowing either number". common.h:325 defaults it to 3;
# speculative.cpp:989 clamps at block_size - 1, and this drafter's block_size is
# 8, so the ceiling is 7. 7 was measured once, in another configuration, and
# took 25 % off DFlash2's wall clock.
#
# THE HEADROOM IS THE RISK, AND IT IS PROBED RATHER THAN ASSUMED. The recurrent
# state is 149.62 x (1 + n_max), so this costs about 449 MiB on the arm that
# already finishes with the least in the register. Probed 2026-08-30 at exactly
# this configuration, both load AND answer a real request -- n_max 4 leaves
# [1043, 770] MiB and n_max 7 leaves [870, 462]. But 462 on CUDA1 sits between
# the two numbers the profile measured (336 died on the first request, 488
# survived 135,233 tokens), so the arm may still die on the real corpus at
# depth. The arena records that; it is not a reason to leave the arm out.
#
# MTP IS HELD AT 3, DELIBERATELY. qwen35.nextn_predict_layers = 1 -- the head
# predicts ONE token ahead -- and 7 measured -56 % on it with acceptance falling
# from 0.48-0.61 to 0.38-0.44. Matching the two drafters would be matching a
# number that does not mean the same thing twice. It is here so the ordering
# found under `-sm layer` on 2026-08-30 can be re-checked at each drafter's own
# best-known depth, in one rotation.
#
# THE INCUMBENT IS HERE FOR THE SAME REASON IT IS IN `layer-pairings`: without
# it the two dflash figures can only be compared to a table from another boot
# series, which is the thing this bench exists to stop.
# ---- 2026-08-30: DFlash2 on NVFP4, given everything it is known to want -----
#
# results/nvfp4-dflash-147456.jsonl put draft-dflash,ngram-mod at +0.2 % WITH
# THE SIGN FLIPPING against draft-mtp,ngram-mod, and that is the number the
# ledger cites for "DFlash2 has no case on this artifact". It is also what
# stands between the developer and a head-less NVFP4 file
# (esatapedico/...-BUDGET is 134 MiB smaller, ...-STARVED 257 MiB): stripping
# the MTP head only pays if something else drafts better.
#
# THAT RUN GAVE DFLASH2 NONE OF WHAT IT IS NOW KNOWN TO WANT. ctx 147,456 --
# above its 131,072 ceiling on the other artifact and more than twice its
# measured best of 65,536 -- at --spec-draft-n-max 3, with the n-gram at
# n-match 12, a window this register records COLLAPSING on NVFP4 (acceptance
# 55.4 -> 22.1) while 24 wins and is what +63.1 % was measured with.
#
# This set gives it all three at once: 65,536, n_max 4, n-match 24. THREE
# VARIABLES MOVE TOGETHER, DELIBERATELY -- the question is binary, does DFlash2
# have a case here at all, and a handicapped run cannot answer it. If it wins,
# attributing the win to depth, draft size or window is a SEPARATE experiment
# and the write-up must say so.
#
# NO MTP ARM: excluded by the developer, who has measured it repeatedly. The
# consequence travels with the result -- this set CANNOT compare DFlash2 to MTP,
# because MTP's NVFP4 figures are from other boots and this depth carries a
# measured 48.9 % same-arm drift (CORRECTIONS 23). The incumbent stays so the
# DFlash2 figure has something in its OWN rounds to be a percentage of.
NVFP4_DFLASH_65536 = [
    ("nvfp4-ngram nm24",
     DUAL_TENSOR + ["-m", NVFP4_VERY_LOW, "--spec-type", "ngram-mod"]
     + _ngram(16, n_match=24),
     {ENV_VAR: MIRROR_EXE, "CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
    ("nvfp4-dflash+ngram nm24 n4",
     DUAL_TENSOR + ["-m", NVFP4_VERY_LOW,
                    "--spec-type", "draft-dflash,ngram-mod",
                    "-md", DFLASH_SMALL, "-ngld", "99",
                    "--spec-draft-n-max", "4"] + _ngram(16, n_match=24),
     {ENV_VAR: MIRROR_EXE, "CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
]


# ---- 2026-08-30: the SAME arm at MTP's depth, issue #50 ---------------------
#
# `nvfp4-dflash-65536` settled that DFlash2 works on NVFP4: +67.9 %
# [+65.8, +71.5] RESOLVED over the incumbent, acceptance 50.0 against the
# discredited run's 22.1. What it could NOT do is compare DFlash2 to MTP, whose
# NVFP4 figures are at 147,456 -- a different depth and a different boot.
#
# ONE ARM, WHICH THIS BENCH NORMALLY REFUSES. The three-arm rotation was
# declined: MTP has been measured repeatedly at this exact setting. Checked
# rather than accepted, and the check is what makes this admissible:
#
#   nvfp4-final-147456.jsonl         nvfp4-mtp+nm24   39.43 / 42.61 / 42.55
#   nvfp4-ngram-retune-147456.jsonl  mtp+nm24         43.10 / 42.99 / 42.93
#
# Six rounds, TWO INDEPENDENT BOOT SERIES, same artifact, depth, window and
# n_max. Five of six fall in 42.5-43.1 and the full range spans 9.3 %.
# CORRECTIONS 23 measures up to 48.9 % same-arm drift at depth and that is how
# +26 % happened -- but this comparator does not have that spread. Comparing
# across boots to a six-round, two-series, 9.3 % comparator is a different act
# from comparing to one reading, and the write-up must say which it is.
#
# EVERYTHING MTP'S ROWS HELD IS HELD HERE. The only differences are the decoder
# and --spec-draft-n-max: 3 for MTP, 4 for DFlash2, each its own measured best.
# nextn_predict_layers = 1 and block_size = 8 do not mean the same thing, so
# matching the two would match a number rather than a setting.
#
# THE RISK, BEFORE THE RUN. At 65,536 this arm finished with 2,828 MiB free.
# 147,456 adds about 1,440 MiB of KV at 18.00 KiB/token and n_max 2 -> 4 adds
# 299 more, leaving roughly 1,100. The earlier 147,456 DFlash2 run used n_max 2
# and loaded. If this fails to load, THAT is the result and the fallback is 2.
NVFP4_DFLASH_147456 = [
    ("nvfp4-dflash+ngram nm24 n4",
     DUAL_TENSOR + ["-m", NVFP4_VERY_LOW,
                    "--spec-type", "draft-dflash,ngram-mod",
                    "-md", DFLASH_SMALL, "-ngld", "99",
                    "--spec-draft-n-max", "4"] + _ngram(16, n_match=24),
     {ENV_VAR: MIRROR_EXE, "CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
]


TENSOR_DRAFT_DEPTH = [
    ("ngram-mod", _tensor_arm("ngram-mod", []), MIRROR_ENV),
    # NO -md: the nextn head is inside UD-Q4_K_XL.
    ("mtp+ngram n3", _tensor_arm("draft-mtp,ngram-mod",
                                 ["--spec-draft-n-max", "3"]), MIRROR_ENV),
    ("dflash+ngram n4", _tensor_arm("draft-dflash,ngram-mod",
                                    ["-md", DRAFTER, "-ngld", "99",
                                     "--spec-draft-n-max", "4"]), MIRROR_ENV),
    ("dflash+ngram n7", _tensor_arm("draft-dflash,ngram-mod",
                                    ["-md", DRAFTER, "-ngld", "99",
                                     "--spec-draft-n-max", "7"]), MIRROR_ENV),
]


ARM_SETS = {
    "decoders": ARMS,
    "build-ab": BUILD_AB,
    "layer-pairings": LAYER_PAIRINGS,
    "tensor-draft-depth": TENSOR_DRAFT_DEPTH,
    "nvfp4-dflash-65536": NVFP4_DFLASH_65536,
    "nvfp4-dflash-147456": NVFP4_DFLASH_147456,

    # `GGML_CUDA_GRAPH_OPT` -- NEVER RUN HERE. An optimisation that is off
    # unless asked for, and nothing in this project has ever asked.
    #
    #     static bool enable_graph_optimization = [] {
    #         const char * env = getenv("GGML_CUDA_GRAPH_OPT");
    #         return env != nullptr && atoi(env) == 1;    // ggml-cuda.cu:4330
    #     }();
    #
    # It additionally requires CUDA graphs to be in use and EXACTLY ONE device
    # (ggml-cuda.cu:4342), both true here: GGML_CUDA_GRAPHS was ON in the build,
    # and there is one card. Decode at batch 1 is a long sequence of small
    # kernels, which is the case graph optimisation exists for, so this is the
    # runtime knob most likely to move the number that matters.
    #
    # Both arms carry the SAME argv -- the incumbent `ngram-mod` window every
    # worker profile serves -- so the only difference between them is the
    # variable. An arm set that also changed a flag would produce a delta with
    # two causes, which is the shape of CORRECTIONS 26 and 28.
    #
    # WHAT THIS CANNOT SHOW: nothing in argv or the boot banner echoes the
    # variable back, so there is no independent confirmation that llama.cpp read
    # it. A null result here means "no effect OR not applied" and must be
    # written up that way.
    # `--spec-draft-n-max 7` -- THE CEILING, NEVER SET.
    #
    # common.h:325 defaults n_max to 3. speculative.cpp:989 takes the ceiling
    # from the drafter's own metadata -- block_size - 1 -- and our boot log
    # prints `block_size=8` for DFlash2, so 7. Every DFlash2 figure this repo
    # holds, report 29 included, was measured at 4: a value the ledger records as
    # "chosen without knowing either number", with two independent reviews
    # calling it the largest unclaimed lever on the list.
    #
    # Cost is known and flat: the Gated DeltaNet recurrent state is
    # 149.62 x (1 + n_max) and does not scale with context, so 4 -> 7 is
    # +448.84 MiB. On UD-Q2_K_XL that is 12,973 -> 13,422 of 15,172.
    #
    # ONE NUMBER CHANGED. Both arms are byte-identical to arms already measured
    # apart from `--spec-draft-n-max`; a delta with two causes is unattributable
    # and is what CORRECTIONS 26 and 28 both are.
    #
    # THE MTP ARM'S CEILING IS UNREAD. draft-mtp prints no block_size line, and
    # the server CLAMPS WITH A WARNING rather than an error -- so a run can
    # silently draft 3 while the label says 7. Read `n_max=` back out of the boot
    # log before believing either row.
    "n-max-7": [
        ("dflash2+ngram n7", _pair(n_draft=7), {}),
        ("draft-mtp+ngram n7",
         ["--spec-type", "draft-mtp,ngram-mod"] + NGRAM + ["--spec-draft-n-max", "7"], {}),
    ],

    # `draft-mtp` USING THE HEAD BAKED INTO THE TARGET -- never run here before.
    #
    # 02-decoders.md carries draft-mtp at +81 % @16K and -71 % @131,072, and the
    # same page records why those could not have used a baked-in head:
    #
    #     Can `draft-mtp` run on `UD-IQ2_S` alone?  No. "model doesn't contain
    #     MTP layers" -- the weights are a separate 1.3 GB file passed with -md
    #
    # So every prior figure paid 564 MiB for a sidecar on an artifact that had
    # none. `UD-Q2_K_XL` reports n_layer_all = 65, offloads 66/66, and its boot
    # log shows blk.64.nextn.* loading out of the main file, with
    # `creating MTP draft context against the TARGET model`.
    #
    # NO -md ON PURPOSE. Passing one moves the head into a file, adds its weights
    # to fit_params_target (server-context.cpp:1074, gated only on "was -md
    # given"), and makes the arm measure the configuration that was already
    # measured while carrying the label of the one that was not.
    #
    # Probed at ctx 98,304: model 8965.31, KV 1728.00, MTP KV 384.00, RS 598.50
    # at the n_max=3 default, compute 472.27 + 82.01 -- 12,230 MiB leaving 2,942,
    # against dflash2+ngram's 12,973 leaving 2,199. 743 MiB back, not the 1,394 a
    # first estimate suggested: the model buffer grows 334.74 MiB when the head
    # is used, and --fit raises its own target 768 -> 1234 for the MTP context.
    #
    # REQUIRES A MODEL WITH AN MTP HEAD. Nothing in an arm names a model; on an
    # artifact without one the server refuses at boot, which is the right loud
    # failure.
    "mtp": [
        ("draft-mtp", ["--spec-type", "draft-mtp"], {}),
        ("draft-mtp+ngram", ["--spec-type", "draft-mtp,ngram-mod"] + NGRAM, {}),
    ],

    # THE SERVED ARM AGAINST ITS OWN ABLATIONS -- issue #44.
    #
    # `worker-q2kxl-mtp.ps1` serves `draft-mtp,ngram-mod` at ctx 147,456, and
    # that choice rests on ONE UNPAIRED SESSION PER ARM on a single real task
    # (report 35). Two things say that is not enough:
    #
    #   02-decoders.md carries draft-mtp at +81 % @16K and -71 % @131,072. We
    #   serve DEEPER than the depth where the sign flipped. That figure used a
    #   sidecar head on an artifact with none, so it does not transfer -- which
    #   is the point: the figure that would transfer was never taken.
    #
    #   An operator on the same RTX 5060 Ti published the paired curve we lack
    #   (researchs/hf-discussion-5060ti-mtp): 2.08x at 2,500 decaying to 1.72x
    #   at 25,400, and his measurement stops there. Ours runs six times deeper.
    #
    # THREE ARMS. Dropping MTP alone leaves `ngram-mod` -- the decoder every
    # other worker profile serves, and the real alternative. Dropping both is
    # the only honest floor. Two arms would have made the answer "MTP or
    # nothing", which is not the choice in front of us.
    #
    # ONE VARIABLE BETWEEN NEIGHBOURS. The ngram window is byte-identical in the
    # two arms that have one; `tests/test_served_ablation_arm_set.py` asserts it
    # rather than trusting this comment. A delta with two causes is what
    # CORRECTIONS 26 and 28 both are.
    #
    # NO -md, for the reason the "mtp" set above gives at length.
    "served-ablation": [
        ("draft-mtp+ngram", ["--spec-type", "draft-mtp,ngram-mod"] + NGRAM, {}),
        ("ngram-mod", SERVED_NGRAM, {}),
        ("none", [], {}),
    ],

    # ---- issue #51 stage 2: what does the second card buy? ------------------
    #
    # A second GPU was connected on 2026-08-26: RTX 5060 Ti 16 GB (sm_120)
    # beside the retired RTX 4070 SUPER 12 GB (sm_89). 28 GB total, PXB
    # topology, no NVLink. Nothing in the register describes a two-card run.
    #
    # WHY `ngram-mod` AND NOT THE SERVED DECODER. draft-mtp's head is weights,
    # and weights get PLACED -- on a split model llama.cpp decides which card
    # holds them. That makes the drafter's location a second variable moving
    # with the first, and a delta with two causes is what CORRECTIONS 26 and 28
    # both are. `ngram-mod` costs 0 MiB and has nothing to place, so the only
    # thing differing between these arms is where the target model lives.
    #
    # WHY `solo` IS FIRST AND ALSO THE FLOOR. Its round-to-round spread IS the
    # noise floor for this machine, which has to be re-derived rather than
    # inherited: the second card draws power and shares the bus, so this is a
    # new configuration, and CORRECTIONS 23 already showed the floor moving
    # from 13.6 % at 16,384 to 48.9 % at 65,536 on one card alone.
    #
    # `-sm row` is included because it is the mode that trades PCIe traffic for
    # parallelism, and the link here is the open question -- the 5060 Ti reads
    # x4 of a possible x16 at idle and nobody has yet looked under load.
    "dual-gpu": [
        # "-base" is not decoration: report() picks the baseline by that suffix,
        # and without it the first arm ALPHABETICALLY becomes the thing every
        # delta is measured from -- here that would be `both-layer`, silently
        # inverting the question.
        ("solo-5060ti-base", SERVED_NGRAM, {"CUDA_VISIBLE_DEVICES": TI_5060}),
        ("both-layer", SERVED_NGRAM, {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("both-row", SERVED_NGRAM + ["-sm", "row"],
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
    ],

    # ---- issue #51 stage 2, the clean half ----------------------------------
    #
    # The `dual-gpu` set above measured -78.3 % decode with a spread of 0.8 %
    # per arm -- and the two arms DECODED DIFFERENT TEXT. ngram-mod accepted
    # 93.3 % on one card and 58.5 % on two. That is not sampling noise to be
    # averaged out: SAMPLER is already greedy, and the text differs because
    # splitting the model changes the order of the reductions and therefore the
    # logits. On a split model you cannot decode the same tokens as on one card,
    # ever, so a speculative decode rate can never be a clean hardware
    # comparison between these two configurations.
    #
    # With speculation OFF every token costs exactly one forward pass whatever
    # the token is. The rate stops depending on the text -- the same property
    # that already makes prefill comparable, and prefill on the identical 6,621
    # token prompt says two cards are 57 % FASTER, the opposite sign.
    "dual-gpu-nospec": [
        ("solo-nospec-base", [], {"CUDA_VISIBLE_DEVICES": TI_5060}),
        ("both-nospec", [], {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
    ],

    # ---- issue #52 stage 1: how should the model be divided? ----------------
    #
    # #51 ran llama.cpp's DEFAULTS and stopped there. The default splits by free
    # VRAM -- measured 41:59 -- and these two cards are asymmetric in SPEED as
    # well as size: the 4070 SUPER does 0.798 ms per prefill token against the
    # 5060 Ti's 1.517 (results/09-hardware.md). A capacity-proportional split
    # therefore hands MORE work to the SLOWER card, which is not obviously right
    # and has never been tested.
    #
    # `-ts 1,1` is the smallest honest alternative: an even split moves ~8.35
    # GiB of a 16.69 GiB artifact onto each card, which the 4070's 11,069 MiB
    # free accommodates, and it shifts work toward the faster chip.
    #
    # `-sm tensor` is a mode this project has never run at all. The help calls
    # it "split weights and KV across GPUs (parallelized, EXPERIMENTAL)" -- a
    # different trade from `layer`'s pipeline. `-sm row` is NOT here: it cannot
    # load on this pair ("device CUDA0 does not support split buffers").
    #
    # Speculation is off in all three for the reason CORRECTIONS 32 gives at
    # length: the split changes the reduction order, which changes the logits,
    # which changes the text -- and a speculative rate is partly a measure of
    # how predictable the text is. Prefill is read separately from the log,
    # where the prompt is identical across arms and the confound cannot reach.
    "dual-split": [
        ("layer-default-base", [], {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("split-tensor", ["-sm", "tensor"], {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("ts-even", ["-ts", "1,1"], {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
    ],

    # ---- issue #52 stage 2: the micro-batch, on the split that won ----------
    #
    # `-ub 256` was chosen against a SINGLE card (results/05-runtime-flags.md).
    # Two cards change the arithmetic twice over: `-sm tensor` moves activations
    # between the cards inside every layer rather than once per boundary, and
    # the link carrying that traffic is gen4 x4 on the 5060 Ti -- a quarter of
    # what the other card has (CORRECTIONS 31). A wider micro-batch amortises a
    # transfer over more tokens, which is exactly the shape of a narrow link.
    #
    # Every arm carries `-sm tensor` because that is what the profile now
    # serves: sweeping a second knob on a configuration nobody runs measures
    # a machine that does not exist.
    #
    # -b stays at 2048 throughout. -ub above -b is silently clamped, so moving
    # both at once would make some arms identical to their neighbours without
    # anything in the row saying which.
    "dual-ubatch": [
        ("ub-256-base", ["-sm", "tensor"], {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("ub-128", ["-sm", "tensor", "-ub", "128"], {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("ub-512", ["-sm", "tensor", "-ub", "512"], {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("ub-1024", ["-sm", "tensor", "-ub", "1024"], {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
    ],

    # ---- issue #52 stage 3: the KV type, on the split that won --------------
    #
    # `q4_0` was never a preference. It bought residency on 12 GB and then on
    # 16 GB, and this build compiles a flash-attention kernel for only four
    # types -- f16, bf16, q4_0, q8_0 (issue #43). With 28 GB and 4,670 MiB still
    # free at 229,376, q8_0 is affordable for the first time in this project.
    #
    # Residency is the gate, not throughput: q8_0 doubles the KV bytes per
    # token, and an arm that spills is not a faster arm, it is a different one.
    # Both arms carry `-ub 1024` because that is what stage 2 won and what the
    # profile now serves. Sweeping the KV type on a micro-batch nobody runs
    # would measure a machine that does not exist.
    "dual-kv": [
        ("kv-q4-0-base", ["-sm", "tensor", "-ub", "1024"],
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("kv-q8-0", ["-sm", "tensor", "-ub", "1024", "-ctk", "q8_0", "-ctv", "q8_0"],
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
    ],

    # ---- issue #52 stage 4: does any of it survive the depth we serve? ------
    #
    # Everything in stages 1-3 is ctx 16,384. CORRECTIONS 23 says a verdict at
    # one depth does not transfer, and this project has watched `draft-mtp` go
    # from +81 % at 16K to -71 % at 131,072 on one artifact. So -sm tensor's
    # +59.5 % is a hypothesis at 147,456 until it is measured there.
    #
    # Both arms carry -ub 1024, the stage 2 winner, so the split is the only
    # thing moving. The BASELINE is the default split -- if the profile's own
    # configuration were the baseline, a null would read as "the change is
    # safe" when the question is whether the change was right.
    #
    # This also re-derives the noise floor AT DEPTH, which is the acceptance
    # criterion stages 1-3 cannot satisfy: per-arm spread here, not the under
    # 0.8 % measured at 16,384.
    "dual-depth": [
        ("layer-1024-base", ["-ub", "1024"], {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("tensor-1024", ["-sm", "tensor", "-ub", "1024"],
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
    ],

    # ---- issue #52, the last lever: the decoder on the tuned dual config ----
    #
    # Measured at the SERVED depth on the configuration the profile actually
    # runs -- -sm tensor, -ub 1024, both cards -- because a decoder verdict
    # taken on a different split is a verdict about that split.
    #
    # CORRECTIONS 32 applies but is weaker here than in the dual-gpu set: all
    # three arms share one hardware configuration, so the text differs by
    # DECODER rather than by device placement. That is issue #44's confound,
    # not #51's, and it is the reason `none` is included -- it is the only arm
    # whose rate cannot be moved by what the model chose to write.
    #
    # draft-mtp carries NO -md: UD-Q4_K_XL reports the nextn head in the main
    # file, and adding -md brings back a 1,393.90 MiB sidecar for nothing
    # (worker-q2kxl-mtp.ps1 says why at length).
    "dual-decoder": [
        ("ngram-mod-base", ["-sm", "tensor", "-ub", "1024"] + SERVED_NGRAM,
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("draft-mtp+ngram", ["-sm", "tensor", "-ub", "1024",
                             "--spec-type", "draft-mtp,ngram-mod"] + NGRAM,
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("none", ["-sm", "tensor", "-ub", "1024"],
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
    ],

    # ---- issue #52: the one comparison the tensor split leaves open ---------
    #
    # No external drafter loads under -sm tensor -- draft-mtp and draft-dflash
    # both abort at ggml-backend-meta.cpp:1522, because the Meta backend cannot
    # host a second model. So the choice is not "which decoder" but a PAIR:
    #
    #     -sm tensor + ngram-mod   (fast split, only the weightless decoder)
    #     -sm layer  + dflash2     (slower split, every decoder available)
    #
    # Bare, layer was 17.4 tok/s against tensor's 28.7 -- but dflash2+ngram was
    # the FASTEST arm on one card at 98,304 (ledger, issue #40), so the drafter
    # could in principle close a 65 % gap. Nothing here says whether it does.
    #
    # -ts on the tensor arm is the ratio the profile computes on this machine,
    # so the arm is the served configuration rather than an idealised one.
    "dual-drafter": [
        ("tensor-ngram-base",
         ["-sm", "tensor", "-ts", "7819,15490", "-ub", "1024"] + SERVED_NGRAM,
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("layer-dflash-ngram",
         ["-ub", "1024", "--spec-type", "draft-dflash,ngram-mod",
          "-md", DRAFTER, "--spec-draft-n-max", "4", "-ngld", "99"] + NGRAM,
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("layer-ngram",
         ["-ub", "1024"] + SERVED_NGRAM,
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
    ],

    # ---- issue #52: MTP is available on the tensor split after all ----------
    #
    # The earlier verdict "no external drafter loads under -sm tensor" was too
    # strong, and the assertions say why. At 147,456 on the EVEN split MTP died
    # at ggml-backend-meta.cpp:1522, GGML_ASSERT(bufs.back() != nullptr) -- a
    # buffer allocation returning null, which is what an out-of-memory looks
    # like there. With the computed -ts freeing 2.9 GB on the display card it
    # loads. So that failure was the same root cause as the 0.38 tok/s
    # incident, wearing a different error message.
    #
    # DFlash2 is genuinely different: it dies at ggml-backend-meta.cpp:543,
    # GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0) -- a graph split
    # axis, not a buffer -- at ctx 16,384 with -ub 128, where memory pressure
    # is as low as this configuration goes. Pinning the drafter with -devd and
    # disabling backend sampling both change nothing.
    #
    # NO -md on the MTP arm: UD-Q4_K_XL carries the nextn head in the main
    # file, and -md would add a 1.4 GB sidecar for nothing.
    "dual-mtp": [
        ("ngram-mod-base",
         ["-sm", "tensor", "-ts", "7819,15490", "-ub", "1024"] + SERVED_NGRAM,
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("draft-mtp+ngram",
         ["-sm", "tensor", "-ts", "7819,15490", "-ub", "1024",
          "--spec-type", "draft-mtp,ngram-mod", "--spec-draft-n-max", "3"] + NGRAM,
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("none",
         ["-sm", "tensor", "-ts", "7819,15490", "-ub", "1024"],
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
    ],

    # ---- 2026-08-27: the n-gram family, on two cards, at the served depth ---
    #
    # WHY THIS SET EXISTS. `--spec-type` takes eleven values and FIVE of them
    # are weightless n-gram variants. None needs the Meta backend to host a
    # second model, which is what kills `draft-dflash` and `draft-mtp` under
    # `-sm tensor`, so all of them are available here and only one has ever
    # been run on this machine.
    #
    # The family WAS swept -- on the old single 12 GB card. `ngram-map-k` led
    # at 16,384 (+135.89 % against ngram-mod's +112.55 %) and lost at 131,072
    # (+120.54 % against +200.22 %). Those magnitudes are UPPER BOUNDS: the
    # prompt was 84.5 % duplicate lines (instrument fault 8), and every
    # elimination was decided on 160-token generations (CORRECTIONS 8). Run
    # this on a real-code regime, not the synthetic one.
    #
    # `n-match` rides along because it is the same question at a finer grain
    # and shares the baseline: 24 wins at 16,384, 16 wins at 65,536, and we
    # ship 12, which is the second-worst arm at the deeper of the two. It moves
    # no allocation, so the only cost is the boot it shares with the variants.
    #
    # `ngram-cache` IS EXCLUDED. Its greedy hash 3EFE93950A8A980E differs from
    # the same-depth baseline 04E5CAB1D14525C0 -- it changes the answer, so it
    # is not draft-and-verify, whatever rate it posts.
    #
    # EVERY ARM CARRIES `-ts`. `dual-decoder` does not, so its 147,456 rows ran
    # the EVEN split -- the configuration that decoded at 0.38 tok/s, which
    # report 36 section 4 records and tells the reader not to quote. The value
    # here is the same one `dual-drafter` and `dual-mtp` use, held constant
    # across arms so the split is not a variable.
    "dual-ngram-family": [
        ("ngram-mod-base", DUAL_TENSOR + SERVED_NGRAM,
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("nm-16", DUAL_TENSOR + ["--spec-type", "ngram-mod"] + _ngram(16, 16),
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("nm-24", DUAL_TENSOR + ["--spec-type", "ngram-mod"] + _ngram(16, 24),
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        # The variants run at THEIR OWN defaults. Tuning a loser is wasted
        # boots, and each carries a different parameter family
        # (--spec-ngram-map-k-size-n/-m/-min-hits) that only matters if it wins.
        ("map-k", DUAL_TENSOR + ["--spec-type", "ngram-map-k"],
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("map-k4v", DUAL_TENSOR + ["--spec-type", "ngram-map-k4v"],
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
    ],

    # ---- 2026-08-27: DFlash2 on the tensor split, PATCHED BINARY ONLY -------
    #
    # A local patch mirrors the target's output projection so `result_output` is
    # not axis 0, which is what TOP_K could not take. draft-dflash loads under
    # -sm tensor for the first time. This set measures what it buys against the
    # decoder the profile actually serves.
    #
    # REQUIRES QWEN38_LLAMA_EXE pointing at C:\AI\llama.cpp-mirror. On the
    # served binary every drafter arm aborts at ggml-backend-meta.cpp:543, which
    # is a loud failure and therefore an acceptable way to find out.
    #
    # NOT COMPARABLE TO ANY EXISTING ROW. The patch changes the target's split,
    # so both arms here are on a machine no other row was taken on. Read it as
    # "how much does DFlash2 buy on the tensor split", never as a rate beside
    # docs/results/.
    #
    # DEPTH IS THE WHOLE QUESTION. 147,456 and 98,304 OOM with the drafter
    # resident; 65,536 loads and answers a 34,278-token request. Run this at
    # 65,536 or below.
    "dual-dflash-tensor": [
        ("ngram-mod-base", DUAL_TENSOR + SERVED_NGRAM,
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("dflash+ngram", DUAL_TENSOR + ["--spec-type", "draft-dflash,ngram-mod",
                                        "-md", DRAFTER, "-ngld", "99",
                                        "--spec-draft-n-max", "4"] + NGRAM,
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        # The solo drafter, because the pair and the drafter alone measured
        # +48.5 % and +34.7 % at 16,384 -- different arms, and on the tensor
        # split nobody knows which one leads.
        ("dflash", DUAL_TENSOR + ["--spec-type", "draft-dflash",
                                  "-md", DRAFTER, "-ngld", "99",
                                  "--spec-draft-n-max", "4"],
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
    ],

    # ---- 2026-08-27: WHICH PAIRING DO WE SERVE ------------------------------
    #
    # The deployment question, which `dual-dflash-tensor` did not ask. That set
    # compared the drafter against nothing and against the incumbent, and told
    # us the gain is the PAIRING (+113.1 %) rather than the drafter (+19.4 %).
    # It never contained the obvious rival.
    #
    # draft-mtp carries no second file -- the head is inside UD-Q4_K_XL -- and
    # it LOADS at ctx 147,456 on the SERVED, UNPATCHED binary: 66+0, CUDA0 with
    # 1,571 MiB free and CUDA1 with 861, costing about 2,750 MiB. draft-dflash
    # cannot reach that depth at all; the ladder put its ceiling at 65,536. So
    # if MTP is anywhere near on rate it wins outright, because it costs neither
    # the patch nor three quarters of the window.
    #
    # ITS RATE IS UNKNOWN, NOT BAD. Three paired rounds at 147,456 were VOIDED:
    # copied_window_fraction [0.519, 0.0, 0.23], identical across rounds and so
    # deterministic. The middle round is 0.0 -- one round did not copy -- which
    # is why this is worth running again at another depth rather than closed.
    # The unpaired 44.5 / 54.3 / 92.7 tok/s read before the guard ran are
    # exactly what CORRECTIONS 32 says not to trust.
    #
    # Run at 65,536, where all three can load, so the comparison is a decoder
    # comparison and not a depth comparison. Requires the patched binary for the
    # dflash arm; on the served one that arm aborts loudly.
    "dual-pairings": [
        ("ngram-mod-base", DUAL_TENSOR + SERVED_NGRAM,
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        # NO -md. The head is in the main file; a sidecar would add 1.4 GB for
        # nothing and quietly make this a different experiment.
        ("mtp+ngram", DUAL_TENSOR + ["--spec-type", "draft-mtp,ngram-mod",
                                     "--spec-draft-n-max", "3"] + NGRAM,
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("dflash+ngram", DUAL_TENSOR + ["--spec-type", "draft-dflash,ngram-mod",
                                        "-md", DRAFTER, "-ngld", "99",
                                        "--spec-draft-n-max", "4"] + NGRAM,
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
    ],

    # ---- 2026-08-29: NVFP4 against the artifact we serve ---------------------
    #
    # NVFP4 is the ONLY weight format this build has a Blackwell fast path for:
    # mmq-config-blackwell.cuh covers exactly GGML_TYPE_MXFP4 and
    # GGML_TYPE_NVFP4 and nothing else, and under the computed -ts the 5060 Ti
    # carries about 70 % of the weights. It is also not Blackwell-only --
    # mmq.cuh:129 defines a "Generic NVFP4" SRAM layout, so the 4070 runs it too.
    #
    # THE FILE, read from its own header: arch qwen35, 1202 tensors, 448 NVFP4,
    # 744 F32, 9 Q2_K and 1 Q3_K. The 448-tensor NVFP4 backbone is byte-identical
    # across every tier of that repo; the tiers differ in about ten tensors, which
    # is where the 14,173-to-31,599 MiB spread comes from. The FORMAT is not the
    # variable -- which tensors stay high is.
    #
    # THE MODEL IS IN THE ARM. server_argv hardcodes -m TARGET, so an arm that
    # varies the artifact must append its own; llama.cpp takes the last, the same
    # plain-setter behaviour the -ub set relies on.
    #
    # WHY nvfp4-ngram IS HERE. One unpaired run of NVFP4 + draft-mtp reported
    # draft acceptance 0.21053, 12 accepted of 57 generated, mean len 1.63 --
    # against 0.488-0.554 and mean 16-18 for the ngram-mod we serve. Without the
    # no-MTP arm the sweep could not tell "NVFP4 is slower" from "MTP is slower".
    #
    # 147,456 because both artifacts hold it: NVFP4 VERY-LOW loads at 229,376,
    # UD-Q4_K_XL reaches about 250,000, and comparing at a depth only one can
    # reach would put depth in the comparison.
    "nvfp4-vs-q4": [
        ("q4-ngram-base", DUAL_TENSOR + SERVED_NGRAM,
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("nvfp4-ngram", DUAL_TENSOR + ["-m", NVFP4_VERY_LOW] + SERVED_NGRAM,
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("nvfp4-mtp+ngram", DUAL_TENSOR + ["-m", NVFP4_VERY_LOW,
                                           "--spec-type", "draft-mtp,ngram-mod",
                                           "--spec-draft-n-max", "3"] + NGRAM,
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
    ],

    # ---- 2026-08-29: DFlash2 beside NVFP4, against the head NVFP4 ships with -
    #
    # WHERE THIS STARTS. At 147,456, three paired rounds on real vendor code:
    # q4-ngram-base 24.4/25.6/25.7, nvfp4-ngram -22.4 % RESOLVED, and
    # nvfp4-mtp+ngram +41.2 % [+39.9, +43.0] RESOLVED. So the MTP head inside
    # the NVFP4 file is worth more than the artifact change itself, and
    # ngram-mod ALONE on NVFP4 is a loss -- its acceptance falls 55.4 -> 22.1
    # because that artifact writes text the n-gram cannot predict.
    #
    # WHICH IS EXACTLY DFLASH2'S CASE. It drafts from the model, not from
    # repetition. On UD-Q4_K_XL at 65,536 the draft-dflash,ngram-mod pairing was
    # +123.8 % against +38.9 % for draft-mtp,ngram-mod. If that ordering carries
    # onto NVFP4 at the served depth it beats the current champion.
    #
    # REQUIRES THE PATCHED BINARY. DFlash2's selector runs a TopK over the
    # TARGET's LM head; under -sm tensor those logits are axis 0 and llama.cpp
    # aborts at ggml-backend-meta.cpp:543. vLLM refuses the same component from
    # the other side -- "DFlash2 requires an unquantized target LM head for
    # candidate TopK". The mirror costs 1,080 MiB, measured.
    #
    # MEMORY IS THE RISK: nvfp4-ngram finished with 3,797 MiB free, and the
    # drafter buffer plus the mirror are about 1,618 of it.
    "nvfp4-dflash": [
        ("nvfp4-mtp+ngram", DUAL_TENSOR + ["-m", NVFP4_VERY_LOW,
                                           "--spec-type", "draft-mtp,ngram-mod",
                                           "--spec-draft-n-max", "3"] + NGRAM,
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("nvfp4-dflash+ngram", DUAL_TENSOR + ["-m", NVFP4_VERY_LOW,
                                              "--spec-type", "draft-dflash,ngram-mod",
                                              "-md", DFLASH_SMALL, "-ngld", "99",
                                              "--spec-draft-n-max", "3"] + NGRAM,
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
    ],

    # ---- 2026-08-29: retune the n-gram FOR NVFP4 ----------------------------
    #
    # n-match 12 is what every profile serves and it was chosen on UD-Q4_K_XL:
    # at 147,456 it beat 16 and 24, and both map-k variants declined 100 % of
    # their drafts. On NVFP4 the same setting collapses -- acceptance 55.4 ->
    # 22.1, and beside MTP it reports `ngram-mod decline 97.2 %`. It is barely
    # firing.
    #
    # That is not a decoder fault. ngram-mod drafts from repetition in the text
    # the model is producing, so a different artifact writing differently is a
    # different problem for it. This project's rule that a verdict at one DEPTH
    # does not transfer applies to ARTIFACTS too, and nothing had tested it.
    #
    # THE DRAFTER IS HELD AT draft-mtp because that is what would be served:
    # NVFP4 + draft-mtp + ngram-mod is +41.2 % [+39.9, +43.0] over the served
    # configuration, and DFlash2 beside it added +0.2 % with the sign flipping
    # while costing 650 MiB of headroom and a patched binary.
    #
    # THAT +0.2 % IS RETRACTED (CORRECTIONS 42) and this comment is kept only
    # as the reason the set was BUILT that way. The arm it describes ran at
    # n-match 12 -- the very window this set was measuring -- so it never
    # tested DFlash2 at the 24 that won. See NVFP4_DFLASH_65536.
    #
    # Costs only boots -- none of these settings moves an allocation.
    "nvfp4-ngram-retune": [
        ("mtp+nm12-base", DUAL_TENSOR + _nvfp4_mtp() + _ngram(16, 12),
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("mtp+nm16", DUAL_TENSOR + _nvfp4_mtp() + _ngram(16, 16),
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("mtp+nm24", DUAL_TENSOR + _nvfp4_mtp() + _ngram(16, 24),
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        # The variants run at their own defaults: tuning a loser is wasted boots,
        # and each carries a different parameter family.
        ("mtp+map-k", DUAL_TENSOR + _nvfp4_mtp("ngram-map-k"),
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("mtp+map-k4v", DUAL_TENSOR + _nvfp4_mtp("ngram-map-k4v"),
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
    ],

    # ---- 2026-08-29: the proposal against the incumbent, in ONE run ----------
    #
    # Two verdicts exist and they were taken in different runs: NVFP4+MTP at
    # n-match 12 was +41.2 % over the served config, and n-match 24 was +27.1 %
    # over n-match 12 on NVFP4. MULTIPLYING THEM WOULD BE A CROSS-RUN
    # COMPARISON, which this project forbids -- the spread across boots is
    # measured and its cause is unknown. This set is the only figure that may be
    # quoted for the decision.
    #
    # 24 is the value that LOST on UD-Q4_K_XL at this same depth, where 12 beat
    # both 16 and 24 and map-k declined 100 % of its drafts. On NVFP4 map-k
    # recovers to +15.4 % RESOLVED and 24 wins at spread 0.4 %. n-gram tuning
    # does not survive an artifact change; nothing had tested that.
    #
    # NEITHER ARM NEEDS THE PATCH. The MTP head is in the file, so no -md and no
    # mirrored output projection: both run the SERVED binary.
    # ---- 2026-08-29: the split mode, ON NVFP4 --------------------------------
    #
    # `-sm tensor` beat `-sm layer` by +65.4 % at this depth -- measured on
    # UD-Q4_K_XL, on 2026-08-26, with SPECULATION OFF ON BOTH SIDES. This
    # session established twice over that a verdict does not survive an artifact
    # change, and the split verdict is the last big one still being quoted
    # across one.
    #
    # There is also a mechanism, not only a caution. Every boot prints
    # `set_sampler: backend sampling not supported with SPLIT_MODE_TENSOR; using
    # CPU`, and draft-mtp announces `backend_sampling=1` immediately before it
    # is disabled. A comparison with speculation off could not have seen that.
    # `-sm layer` + draft-dflash is already on record loading and running at
    # 52.11 tok/s, so layer can host what tensor cannot.
    #
    # The residency guard does not block this pair: it reads CPU spill from the
    # load report ("66+0" vs "55+11"), and both modes report 66+0 when resident.
    # If one of them spills, the report SHOULD refuse the delta -- that is the
    # guard working, not a problem with the arm set.
    "nvfp4-split": [
        ("tensor-base", DUAL_TENSOR + _nvfp4_mtp() + _ngram(16, 24),
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("layer", DUAL_LAYER + _nvfp4_mtp() + _ngram(16, 24),
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
    ],

    # ---- 2026-08-29: is the n-gram beside MTP earning its place? -------------
    #
    # draft-mtp ALONE has never been run on this artifact. Every NVFP4 arm here
    # carries an n-gram, and the one exception is the n-gram alone.
    #
    # Two pieces of evidence point opposite ways. FOR the n-gram: n-match 24 is
    # +27.1 % RESOLVED over 12 on this corpus with MTP fixed. AGAINST it: on the
    # developer's real agent traffic ngram-mod generated 5 drafts in 4,653
    # calls, and Unsloth Studio's single runs on this same file put MTP alone
    # ahead of MTP+ngram.
    #
    # THIS RUNS ON real-code-vendor, WHICH IS THE N-GRAM'S BEST CASE -- repeated
    # vendor source. So the n-gram should win here. If it loses even here that is
    # decisive; if it wins, the agent-traffic question stays open and needs a
    # regime this project does not have. Do not generalise the win.
    #
    # n-max is swept in the same boots because it is free: 2 is llama.cpp's
    # documented default for MTP on GPU, 3 is our deviation, and real-use
    # acceptance per position (0.690, 0.448, 0.284) says 3 should hold.
    # ---- 2026-08-29: what does CPU draft-sampling cost? ----------------------
    #
    # The developer pushed back on "the CPU sampler is not the bottleneck", and
    # was right: the layer-vs-tensor pair changed the split AND the offload at
    # once, so -31 % bounds the offload's benefit only from above.
    #
    #   -bs, --backend-sampling        enable backend sampling   default DISABLED
    #   --spec-draft-backend-sampling  offload DRAFT sampling    default ENABLED
    #
    # The MAIN sampler is on the CPU under both splits -- nothing here passes
    # -bs. What tensor loses is the DRAFT offload, and the logs say it exactly:
    # tensor prints `set_sampler: backend sampling not supported with
    # SPLIT_MODE_TENSOR; using CPU`, layer does not.
    #
    # layer is the ONLY split where the offload works, so it is the only place
    # the offload can be varied alone. The delta is its worth, X -- which makes
    # tensor's true advantage about 31 % + X, and tells us the size of a tax
    # this configuration pays and cannot avoid.
    #
    # MEASURED ON layer, WHICH IS NOT WHAT WE SERVE. The number is about the
    # offload, not about a servable configuration.
    # ---- 2026-08-29: the -Beta bundle, borrowed whole from Unsloth Studio ----
    #
    # Nine settings Studio uses and we do not, adopted together so one paired
    # sweep can say whether the bundle is worth bisecting. If it wins, bisect.
    # If it loses, the eleven-flag diff stops being interesting and the RAM
    # question becomes a pure trade rather than a hoped-for free win.
    #
    # The RAM is the reason it exists: a real session held 20.4 GB working set
    # and 34.4 GB private, and --ctx-checkpoints 32 at ~350 MiB each is where it
    # went. The arena cannot see host RAM, so THIS SWEEP ONLY ANSWERS THE SPEED
    # HALF -- read the process RAM separately.
    "beta-bundle": [
        ("default-base", DUAL_TENSOR + _nvfp4_mtp() + _ngram(16, 24),
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("beta", DUAL_TENSOR + ["-m", NVFP4_VERY_LOW,
                        "--spec-type", "draft-mtp,ngram-mod",
                        "--spec-draft-n-max", "2"]
               + _ngram(16, 24)
               + ["--cache-ram", "0", "--ctx-checkpoints", "0",
                  "--load-mode", "none", "--kv-unified", "-t", "2"],
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
    ],

    "draft-sampling-cost": [
        ("layer-bs-on", DUAL_LAYER + _nvfp4_mtp() + _ngram(16, 24),
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("layer-bs-off", DUAL_LAYER + _nvfp4_mtp() + _ngram(16, 24)
                       + ["--no-spec-draft-backend-sampling"],
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
    ],

    "nvfp4-mtp-solo": [
        ("mtp+nm24-base", DUAL_TENSOR + _nvfp4_mtp() + _ngram(16, 24),
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("mtp-solo", DUAL_TENSOR + ["-m", NVFP4_VERY_LOW,
                                    "--spec-type", "draft-mtp",
                                    "--spec-draft-n-max", "3"],
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("mtp-solo-nmax2", DUAL_TENSOR + ["-m", NVFP4_VERY_LOW,
                                          "--spec-type", "draft-mtp",
                                          "--spec-draft-n-max", "2"],
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
    ],

    "nvfp4-final": [
        ("q4-ngram-base", DUAL_TENSOR + SERVED_NGRAM,
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("nvfp4-mtp+nm24", DUAL_TENSOR + _nvfp4_mtp() + _ngram(16, 24),
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
    ],

    "graph-opt": [
        ("graph-opt-off", SERVED_NGRAM, {}),
        ("graph-opt-on", SERVED_NGRAM, {"GGML_CUDA_GRAPH_OPT": "1"}),
    ],

    # `GGML_CUDA_ALLREDUCE` -- task #48, never run. Under `-sm tensor` every
    # layer pays an all-reduce across the two cards on every token, so this is
    # the one environment variable sitting directly on our decode path.
    #
    #     const char * env = getenv("GGML_CUDA_ALLREDUCE");   // ggml-cuda.cu:1222
    #     ... "nccl" | "internal" | "none"                    // :1231-1240
    #
    # NCCL is not compiled into our binary -- the code warns and falls back --
    # and Windows defaults to `internal`, so the A/B that exists here is the
    # default against `none`, and `nccl` would only re-measure the default.
    #
    # The argv is `nvfp4-final`'s winning arm, reused rather than retyped: this
    # has to be comparable with the +63.1 % row in nvfp4-final-147456.jsonl, and
    # a first attempt measured at ctx 16,384 with a short synthetic prompt was
    # rejected for exactly that reason. Run it the way that row was run:
    #
    #     python dflash2_arena.py --arms allreduce --ctx 147456 \
    #         --regime real-code-vendor --rounds 3
    #
    # WHAT THIS CANNOT SHOW: nothing in argv or the boot banner echoes the
    # variable back, so the row records the env the launcher set, not a value
    # read from the process.
    "allreduce": [
        ("internal-default", DUAL_TENSOR + _nvfp4_mtp() + _ngram(16, 24),
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("allreduce-none", DUAL_TENSOR + _nvfp4_mtp() + _ngram(16, 24),
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS, "GGML_CUDA_ALLREDUCE": "none"}),
    ],

    # The tensor-split RATIO on an NVFP4 artifact -- a different question from
    # the one `-ts 1,1` answered.
    #
    # That row (results 09, "+1.8 %, noise", and the page carries a red
    # retraction on the sentence after it) was measured under `-sm layer` on
    # `UD-Q4_K_XL`, where BOTH cards run the same kernel and a ratio has nothing
    # to buy. The same page closes native FP4 as "unreachable for us" and says
    # exactly why: *"Native FP4 needs MXFP4 or NVFP4 weights. That is an artifact
    # swap, not a flag."*
    #
    # The artifact was swapped. On NVFP4, `mmq.cu:131`
    #
    #     const bool use_native_fp4 = blackwell_mma_available(cc) &&
    #         (src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_NVFP4);
    #
    # is true on the 5060 Ti and false on the 4070 SUPER, where
    # blackwell_mma_available() is false by construction. The cards run DIFFERENT
    # kernels over the same tensors, and `-sm tensor` splits every layer across
    # both -- so no layer runs the fast path alone. Tilting the budget toward the
    # Blackwell card is the only knob that changes that balance.
    #
    # Three points with the total held constant, so the proportion is the single
    # variable and the claim -- that the line slopes -- can fail. Headroom is the
    # binding limit: at runtime the 4070 holds ~11.2 GB of 12.0 and the 5060 Ti
    # ~14.5 GB of 16.0, so `tilt-5060` is the largest push those numbers allow.
    # If it OOMs, that is the answer to how far this can go.
    "ts-ratio": [
        ("control", ["-sm", "tensor", "-ts", "7819,15490", "-ub", "1024"]
         + _nvfp4_mtp() + _ngram(16, 24), {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("tilt-5060", ["-sm", "tensor", "-ts", "7309,16000", "-ub", "1024"]
         + _nvfp4_mtp() + _ngram(16, 24), {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("tilt-4070", ["-sm", "tensor", "-ts", "9009,14300", "-ub", "1024"]
         + _nvfp4_mtp() + _ngram(16, 24), {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
    ],

    # `--threads` -- lever rank 5, never swept once. Everything is GPU-resident,
    # so 18 may only be contention; Unsloth Studio serves the same artifact with
    # 2. `arm_parts` already puts `-t 18` in the base argv, which is what the
    # worker serves, so the control adds nothing and the arms override.
    "threads": [
        ("t18", DUAL_TENSOR + _nvfp4_mtp() + _ngram(16, 24),
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("t8", DUAL_TENSOR + _nvfp4_mtp() + _ngram(16, 24) + ["-t", "8"],
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("t2", DUAL_TENSOR + _nvfp4_mtp() + _ngram(16, 24) + ["-t", "2"],
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
    ],

    # The `ngram-mod` window -- lever rank 2, and `--spec-ngram-mod-n-max` has
    # NEVER been swept here at any depth. Every arena run on this artifact
    # reports the drafter declining 97-98 % of the calls it receives, so the
    # window is worth asking about. Two steps rather than one jump: n-max alone,
    # then Studio's 48/64 pair, so a result can be attributed to a half.
    # CORRECTIONS 38 is why Studio's numbers are a candidate and not a
    # recommendation -- 48 and 64 are its defaults, not its choices.
    "ngram-window": [
        ("ours-16-32", DUAL_TENSOR + _nvfp4_mtp() + _ngram(16, 24),
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("nmax-64", DUAL_TENSOR + _nvfp4_mtp() + _ngram(16, 24, 64),
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("studio-48-64", DUAL_TENSOR + _nvfp4_mtp() + _ngram(48, 24, 64),
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
    ],

    # Confirm the +15.63 %, and find whether 64 is the peak.
    #
    # `ngram-window-147456.jsonl`: n-max 64 measured 52.76 against the served
    # 32's 45.63, per-arm spreads 0.8 % and 1.2 %, every row 66+0 with free_after
    # inside 26 MiB. The harness labelled it "clears this run's spread, not the
    # applied floor" -- that floor is 13.6 %, measured at ctx 16,384 on Ada, and
    # CLAUDE.md says it must be re-derived at depth. Unconfirmed until repeated.
    #
    # 64 IS llama.cpp's own default. `--help`: "maximum number of ngram tokens
    # ... (default: 64)". Our 32 is a deviation BELOW it, and
    # worker-q4-dual.ps1:1252-1264 says why -- 16/32 were "held constant rather
    # than chosen", and a 48/64 attempt was "REVERTED WITHOUT A VERDICT" because
    # on agent traffic the drafter recorded `#gen drafts = 0` and the change was
    # inert either way.
    #
    # THAT CAVEAT TRAVELS WITH ANY RESULT HERE. This corpus makes the drafter
    # fire; the served workload may not. The ladder measures the corpus.
    #
    # n-min stays at 16: studio-48-64 was -10.58 % in the same run, so carrying
    # 48 up the ladder would fold a measured loss into every rung.
    "ngram-nmax-ladder": [
        ("nmax-32-served", DUAL_TENSOR + _nvfp4_mtp() + _ngram(16, 24, 32),
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("nmax-64-default", DUAL_TENSOR + _nvfp4_mtp() + _ngram(16, 24, 64),
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("nmax-96", DUAL_TENSOR + _nvfp4_mtp() + _ngram(16, 24, 96),
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("nmax-128", DUAL_TENSOR + _nvfp4_mtp() + _ngram(16, 24, 128),
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
    ],

    # KV cache type at the served depth -- issue #46, and the asymmetric arm was
    # IMPOSSIBLE until 2026-09-01. `-ctk q8_0 -ctv q4_0` exits during load on
    # every binary this project had, because `fattn.cu:442` drops each K != V
    # pair unless GGML_CUDA_FA_ALL_QUANTS was compiled in; a build with it ON now
    # exists at F:\llama-build\faq and runs the pair.
    #
    # ALL THREE arms pin that build. Pinning only the asymmetric one would put a
    # second variable -- the binary -- inside a KV comparison.
    #
    # `arm_parts` already passes `-ctk q4_0 -ctv q4_0`, so the control adds
    # nothing and the others override by last-wins.
    "kv-type": [
        ("q4-q4", DUAL_TENSOR + _nvfp4_mtp() + _ngram(16, 24),
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS,
          ENV_VAR: r"F:\llama-build\faq\build\bin\llama-server.exe"}),
        ("q8-q4", DUAL_TENSOR + _nvfp4_mtp() + _ngram(16, 24) + ["-ctk", "q8_0", "-ctv", "q4_0"],
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS,
          ENV_VAR: r"F:\llama-build\faq\build\bin\llama-server.exe"}),
        ("q8-q8", DUAL_TENSOR + _nvfp4_mtp() + _ngram(16, 24) + ["-ctk", "q8_0", "-ctv", "q8_0"],
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS,
          ENV_VAR: r"F:\llama-build\faq\build\bin\llama-server.exe"}),
    ],

    # The step between the served ratio and the one that broke.
    #
    # `ts-ratio` found the slope and its edge in one run: tilt-4070 at 61.3 % is
    # -18.2 % [-20.6, -16.5] RESOLVED, and tilt-5060 at 68.6 % was VOIDED in all
    # three rounds -- not for memory (it loaded 66+0 with 2,286 MiB free) but by
    # the prompt-copy guard, copied_frac [0, 0, 0.539] reproducing to the digit.
    # Acceptance peaks at the control too: 44.2 / 58.8 / 50.9 across the three.
    #
    # `push` is the voided ratio carried forward unchanged. The 5060 Ti was
    # emptied to 14 MiB of 16,311 before this run, which should NOT matter --
    # the arm was rejected for output, not for OOM. If it scores now, that
    # reading was wrong and the void was memory pressure after all.
    "ts-ratio-fine": [
        ("control", ["-sm", "tensor", "-ts", "7819,15490", "-ub", "1024"]
         + _nvfp4_mtp() + _ngram(16, 24), {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("mid", ["-sm", "tensor", "-ts", "7573,15736", "-ub", "1024"]
         + _nvfp4_mtp() + _ngram(16, 24), {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("push", ["-sm", "tensor", "-ts", "7309,16000", "-ub", "1024"]
         + _nvfp4_mtp() + _ngram(16, 24), {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
    ],

    # The MoE-offload family, at the depth that can see it.
    #
    # Measured twice at ctx 16,384 on a short synthetic prompt: the first pass
    # put `GGML_OP_OFFLOAD_MIN_BATCH=8` at +8.01 %, and a second pass with the
    # arms rotated through every position brought the same arm to +0.38 % with
    # VRAM identical across all four to within 14 MiB. Neither is comparable
    # with nvfp4-final-147456.jsonl, and that shallow instrument has since been
    # shown to miss a 24 % effect the arena resolves at 0.3 % spread
    # (allreduce-147456.jsonl). So it is re-run here, properly.
    #
    #   --n-cpu-moe N   common/arg.cpp:2728 -- pushes an ffn_*_exps buffer-type
    #                   override for each of the first N blocks
    #   --cpu-moe       common/arg.cpp:2721 -- the same for every block. Kept as
    #                   the MAXIMUM-effect arm: if the family does anything at
    #                   all here, this is where it shows
    #   GGML_OP_OFFLOAD_MIN_BATCH   ggml-cuda.cu:5501, default 32. It gates
    #                   ggml_backend_cuda_device_offload_op, which the scheduler
    #                   consults ONLY for weights already in a host buffer
    #                   (ggml-backend.cpp:959) -- so without one of the two
    #                   flags above it has nothing to act on
    #
    # WHAT THE ARTIFACT SAYS, and what the run has to confirm or refute: reading
    # the served GGUF's header gives 1,202 tensors and **zero** whose name
    # contains `exps` -- 48 `ssm_*` blocks and 17 attention blocks, dense FFN
    # throughout, no `expert_count` key. If that is right the overrides match
    # nothing and all four arms are one configuration. The measurement decides.
    "cpumoe": [
        ("off", DUAL_TENSOR + _nvfp4_mtp() + _ngram(16, 24),
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("ncmoe8", DUAL_TENSOR + _nvfp4_mtp() + _ngram(16, 24) + ["--n-cpu-moe", "8"],
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("cmoe-all", DUAL_TENSOR + _nvfp4_mtp() + _ngram(16, 24) + ["--cpu-moe"],
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS}),
        ("minbatch8", DUAL_TENSOR + _nvfp4_mtp() + _ngram(16, 24),
         {"CUDA_VISIBLE_DEVICES": BOTH_CARDS, "GGML_OP_OFFLOAD_MIN_BATCH": "8"}),
    ],

    # `--spec-ngram-mod-n-min` -- MEASURED, NO EFFECT. Kept so nobody re-runs it.
    #
    # The hypothesis was that it gates how often ngram-mod fires: at n_min 16 on
    # real code it declines 93.7 % of the calls it receives, and when it does
    # fire it is worth 16.7 tokens against draft-dflash's 2.9, so letting short
    # drafts through looked like a large free win.
    #
    # It is not, and the reason is a misreading of common/speculative.cpp:1993.
    # In draft_one, `i` counts DRAFT TOKENS ALREADY PRODUCED, not matched
    # context. So n_min is a minimum draft LENGTH, and the declines happen at
    # i = 0 -- the table misses on the very first successor -- where no value of
    # n_min can help.
    #
    # 16 / 8 / 4 / 2 measured 79.7 / 79.7 / 79.7 / 79.8 tok/s over three rounds,
    # a spread of 0.15 %, on the frozen corpus.
    "ngram-nmin": [
        ("nmin-16-base", _pair(_ngram(16))),
        ("nmin-8",       _pair(_ngram(8))),
        ("nmin-4",       _pair(_ngram(4))),
        ("nmin-2",       _pair(_ngram(2))),
    ],

    # `--spec-draft-n-max` is a VRAM knob, priced at 149.62 MiB per unit:
    # need_n_rs_seq() returns draft.n_max (common/common.h:390) and the
    # recurrent state is allocated once per draft position. Default is 3
    # (common.h:325); the DFlash clamp is block_size-1 = 7
    # (speculative.cpp:989). Every arm here records free_after, because the
    # throughput number is meaningless if the deeper arm spilled a layer.
    "draft-n": [
        ("n-3-default", _pair(n_draft=3)),
        ("n-4-base",    _pair(n_draft=4)),
        ("n-7-clamp",   _pair(n_draft=7)),
    ],

    # `--spec-ngram-mod-n-match` -- the width of the context window the n-gram
    # table is keyed on. Default 24 (common.h:352); we run 12. The external
    # RTX 3090 stack measured a SHORTER, more recent match predicting better on
    # quote-and-explain work: 3.21 against 2.69 tokens per step at 12 vs 32
    # (docs/researchs/syv-rtx3090). This is the knob that governs the 94-97 %
    # decline rate measured here -- `n_min` does not, and was swept to prove it.
    #
    # THE TRAP, from the source read in researchs/llamacpp-flag-semantics:
    # ngram-mod is registered ABOVE draft-dflash (speculative.cpp:2545 vs 2551)
    # and the cascade stops at the first non-empty draft (2725-2726). Lowering
    # n_match raises ngram's fire rate BY SUPPRESSING dflash calls. An arm can
    # fire twice as often and decode SLOWER. Read the per-impl counters in the
    # `impl` column, not just tok/s.
    #
    # Not below 6: the key collapses, acceptance falls under the reset
    # threshold, and the table wipe at speculative.cpp:2044-2054 fires
    # repeatedly -- that measures the reset loop, not the flag.
    "ngram-nmatch": [
        ("nmatch-24-default", _pair(_ngram(16, n_match=24))),
        ("nmatch-16",         _pair(_ngram(16, n_match=16))),
        ("nmatch-12-base",    _pair(_ngram(16, n_match=12))),
        ("nmatch-8",          _pair(_ngram(16, n_match=8))),
    ],

    # The two winners of 2026-08-22, crossed. --spec-draft-n-max 7 measured
    # +23.4 % RESOLVED at n_match 12; --spec-ngram-mod-n-match 24 measured
    # +34.6 % RESOLVED at n_max 4. NEITHER RESULT LICENSES THEIR SUM, and both
    # act on the same drafter through the same cascade, so an interaction is
    # the expected case rather than the surprising one: n_max sets how many
    # tokens draft-dflash produces per call, n_match sets how often ngram-mod
    # pre-empts it entirely (speculative.cpp:2545 above 2551).
    #
    # A 2x2 rather than one "both" arm, because "both" against a single
    # baseline cannot tell a real interaction from a replication failure of
    # either single effect -- and both singles were measured on a different day
    # with a different boot-VRAM roll. Here all four ride the same rounds.
    #
    # WATCH free_after. n_max 7 costs 149.625 MiB per unit of recurrent state
    # (common.h:390) and fitted 65+0 with 443 MiB free at this depth; n_match
    # moves no allocation at all (ngram-mod.cpp:60-62). If the 7 arms spill a
    # layer, the throughput column is measuring residency, not the flags.
    "draft-n-x-nmatch": [
        ("combo-base-n4-m12", _pair(_ngram(16, n_match=12), n_draft=4)),
        ("combo-n7-m12",      _pair(_ngram(16, n_match=12), n_draft=7)),
        ("combo-n4-m24",      _pair(_ngram(16, n_match=24), n_draft=4)),
        ("combo-n7-m24-both", _pair(_ngram(16, n_match=24), n_draft=7)),
    ],

    # `-ub` / `--ubatch-size` -- one of the 48 flags from the RTX 3090 scan that
    # exist here and have never been set. The scan's claim, unverified when
    # written: n_ubatch is the single knob that sizes the worst-case compute
    # buffer, because the reserve pass builds the prompt-processing graph at
    # n_tokens = min(n_ctx, n_ubatch).
    #
    # THE VRAM HALF IS ALREADY ANSWERED, AND THE ANSWER IS "TOO SMALL".
    # `ubatch_preflight.py`, 2026-08-23, one boot per value at ctx 98,304:
    #
    #   -ub 256 -> n_ubatch 256, compute buffer 472.27 MiB, 825 MiB free
    #   -ub 128 -> n_ubatch 128, compute buffer 428.27 MiB, 869 MiB free
    #   -ub  64 -> n_ubatch  64, compute buffer 406.27 MiB, 891 MiB free
    #
    # A 4x cut in ubatch returns **66 MiB**. The arms that need it -- the ones
    # loading DFlash2 -- finish with 45-376 MiB free and are unreliable there
    # (CORRECTIONS.md 26); 66 MiB moves them to 111-442, the same band. So this
    # set does NOT use the drafter: pairing it with dflash would spend hours on
    # timeouts to re-answer a question the preflight closed.
    #
    # WHAT IS LEFT TO MEASURE is the other direction. A smaller ubatch means
    # more, smaller prompt-processing steps, and nothing here has ever measured
    # what that costs. The baseline is `ngram-mod`, which is what all four
    # worker profiles serve and which lands within 4 % over six boots -- a
    # stable enough floor that a real -ub effect cannot hide in it.
    #
    # THE ARGV CARRIES -ub TWICE. server_argv() hardcodes 256 and appends the
    # arm's extra after it, so these arms rely on llama.cpp keeping the LAST
    # occurrence. tests/test_ubatch_arm_set.py pins the ordering; the preflight
    # above proved the parser honours it, reading `n_ubatch` back out of the
    # boot log. Without that check a set that silently ran three arms at 256
    # would report a flat result -- which is how --spec-ngram-mod-n-min wasted
    # twelve boots.
    "ubatch": [
        ("ub-256-base", ["--spec-type", "ngram-mod"] + NGRAM + ["-ub", "256"]),
        ("ub-128",      ["--spec-type", "ngram-mod"] + NGRAM + ["-ub", "128"]),
        ("ub-64",       ["--spec-type", "ngram-mod"] + NGRAM + ["-ub", "64"]),
    ],

    # Pinned allocation -- the RTX 3090 scan's *"highest value on this list for
    # measurement integrity"*, and it is aimed at a constraint this project
    # imposes on itself rather than at tok/s.
    #
    # `CLAUDE.md` forbids comparing raw decode across boots because free VRAM at
    # boot moves 9,326-10,732 MiB and `--fit` follows it. `common/fit.cpp` only
    # adjusts arguments the user did NOT set, so giving `-ngl` a number and
    # turning `--fit` off leaves it nothing to do. If that lowers the
    # boot-to-boot spread, the standing constraint becomes negotiable.
    #
    # THE EVIDENCE THAT MADE THIS WORTH A SWEEP, 2026-08-23. Three `-ub 128`
    # boots logged byte-identical allocation -- same n_ubatch, same 428.27 MiB
    # compute buffer, same `projected to use 8827 MiB vs 10919`, same
    # `will leave 2091 >= 768` -- and `free_after`, sampled while the server
    # ran, read 759, 757 and **1,214 MiB**. The round with 457 MiB more spare
    # ran 6 % faster. Nothing in the experiment caused that.
    #
    # NEVER SWEPT, AND IT SHOULD NOT BE. The preflight closed the question more
    # cheaply than ten boots could: `pinned` and `fit-auto-base` agree on every
    # observable at ctx 98,304 -- 65+0, n_ctx 98304, model 6,521.13 MiB, KV
    # 1,728.00, compute 472.27, free_after 1,427 -- because `--fit` had nothing
    # to pin. Reading every log this project has kept says why: llama.cpp has
    # reported **11,069 MiB free in all 552 of them**, and 148 of the 150 boots
    # on our own artifact end in "no changes needed". `--fit` cannot follow a
    # number it never sees change. CORRECTIONS.md 27, which retracts the stated
    # cause of the no-cross-boot rule while leaving the rule itself standing.
    #
    # KEPT ANYWAY, for two reasons. The arms are the control if the boot picture
    # ever changes -- another machine, another artifact, a depth where `--fit`
    # does act, as it did twice for `n-7-clamp` at 65,536. And the test beside
    # them pins the double-flag override that any future arm set will need.
    #
    # READ THE SPREAD, NOT THE MEDIAN if it is ever run. `paired_deltas` answers
    # "which arm is faster", which is not the question. The question is whether
    # `pinned` varies less across rounds, so the useful columns are the per-arm
    # range and `free_after`.
    #
    # THE BASELINE PASSES NO OVERRIDE ON PURPOSE. server_argv() already
    # hardcodes `-ngl auto --fit on`, which is the configuration every
    # measurement in this project has used; restating it in the arm would let
    # the baseline drift away from the prefix without the test noticing.
    # tests/test_pinned_alloc_arm_set.py pins both halves, and
    # `pinned_alloc_preflight.py` proves the pinned form boots before any sweep
    # spends ten of them -- pinning removes llama.cpp's ability to back off, so
    # anything `--fit` was quietly reducing becomes an OOM instead.
    "pinned-alloc": [
        ("fit-auto-base", ["--spec-type", "ngram-mod"] + NGRAM),
        ("pinned",        ["--spec-type", "ngram-mod"] + NGRAM
                          + ["-ngl", "65", "--fit", "off"]),
    ],

    # `--spec-draft-p-min` -- MEASURED, NULL at these values. Kept so nobody
    # re-runs it, and because the counters carry a lesson the rate does not.
    #
    # 0.00 / 0.10 / 0.25 measured 70.2,76.0,76.4 / 74.5,76.5,76.2 /
    # 75.2,74.8,75.6 over three paired rounds: +2.2 % and +1.5 %, both inside
    # the 13.6 % floor with the sign flipping.
    #
    # THE COUNTERS ARE THE RESULT. At 0.10 every per-impl counter is
    # byte-identical to the baseline -- the early-stop NEVER FIRED. At 0.25 it
    # fired on 2.2 % of draft calls and moved dflash efficiency 47.7 -> 49.8 %
    # for no throughput. The algebraic bound below (1/sum >= 1/16) is correct
    # and was still too generous: on real code the selector's confidence sits
    # above 0.10 essentially always. Starting the arms above a proven worst-case
    # bound was NECESSARY AND NOT SUFFICIENT -- the bound says nothing about
    # where the distribution actually lives.
    #
    # Above 0.25 is untested. No measured reason to expect a win: dflash
    # already keeps only 2.91 of 5 drafted tokens, so a value aggressive enough
    # to bite often starts discarding tokens that would have been accepted.
    #
    # `--spec-draft-p-min` defaults to 0.00 (common.h:329), i.e. the DFlash2
    # confidence early-stop is off. Trimming low-confidence tail positions
    # narrows the verify batch, which also moves the flash-attention kernel
    # choice. No VRAM cost.
    # Every value here is ABOVE 1/16 = 0.0625 on purpose. The greedy check at
    # speculative.cpp:1264-1268 compares 1/sum where sum = SUM exp(scores[k] -
    # scores[argmax]) over the 16 selector candidates; the argmax term is
    # exactly 1 and every other term is <= 1, so 1/sum is in [0.0625, 1.0].
    # ANY p_min <= 0.0625 is mathematically identical to 0.00 -- a ladder
    # starting at 0.05 would repeat the n_min error exactly.
    "p-min": [
        ("pmin-0-base", _pair()),
        ("pmin-0.10",   _pair(extra=["--spec-draft-p-min", "0.10"])),
        ("pmin-0.25",   _pair(extra=["--spec-draft-p-min", "0.25"])),
    ],
}


def vram():
    """[used, free] on the served card -- see `gpu_device` (issue #50)."""
    used, free = gpu_device.vram()
    return [used, free]


def free_for_env(env):
    """Free MiB across the cards THIS ARM will actually use.

    `vram()` answers for the served card. Writing that into a two-card row
    produced `free_before: 15983` on an arm spread over 28 GB -- a believable
    number describing half the hardware. It is not a throughput field, which is
    why it survived a whole sweep before anyone looked at it.

    The sum is a CEILING, not a promise: a layer cannot straddle two cards, so
    free memory does not really add up. Residency is still read from the layer
    split in llama.cpp's own log.
    """
    uuids = [u.strip() for u in
             launch_env(env)["CUDA_VISIBLE_DEVICES"].split(",") if u.strip()]
    return gpu_device.total_vram(uuids)[1]



def caps_for_env(env):
    """Compute capabilities of the cards THIS ARM will actually use.

    Mirrors `free_for_env`, and for the same reason: the ambient
    CUDA_VISIBLE_DEVICES is not the arm's, so asking the module-level helper
    would answer for the wrong set of cards.
    """
    uuids = [u.strip() for u in
             launch_env(env)["CUDA_VISIBLE_DEVICES"].split(",") if u.strip()]
    return [gpu_device.query(["compute_cap"], u)[0] for u in uuids]

def port_owner():
    """PID listening on the arena port, or None. Reads only, never stops it."""
    r = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "$c = Get-NetTCPConnection -LocalPort 8080 -State Listen "
         "-ErrorAction SilentlyContinue; if ($c) { $c.OwningProcess }"],
        capture_output=True, text=True)
    out = (r.stdout or "").strip().splitlines()
    return int(out[0]) if out and out[0].strip().isdigit() else None


def require_exclusive_port():
    """Refuse to run while another orchestrator holds the port.

    CLAUDE.md: "Two orchestrators cannot share port 8080. An armed queue once
    killed a running corpus and the summary still printed a plausible number."
    On 2026-08-22 this arena reproduced that -- a second run was launched while
    the first was finishing, and the older teardown killed the younger server.
    The younger log ends mid-load with no error, because it did not fail.

    It failed loudly only because the kill landed during a load. Landing
    between generations it would have produced a short arm with a believable
    rate, which is the failure this project exists to refuse.
    """
    pid = port_owner()
    if pid is not None:
        raise RuntimeError(
            "port 8080 is already held by pid %d. Another orchestrator is "
            "running; stop it before starting a measurement, or this run and "
            "that one will kill each other's servers." % pid)


def kill():
    """Stop any llama-server. Returns True if one was actually running.

    The caller needs to know: a floor for the settle check is only meaningful
    when something WAS resident. Applying it when nothing was running demands a
    rise that cannot happen, and the wait times out every time -- which is what
    the first version of this file did.
    """
    # Exit code, not the message: taskkill returns 0 when it killed something
    # and 128 when the image was not found, and that holds whatever language
    # the machine reports errors in. Matching on "SUCCESS" would return False
    # on a localised Windows and silently skip every settle wait -- the same
    # fault, quieter.
    r = subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                       capture_output=True, text=True)
    return r.returncode == 0


def wait_for_vram_release(floor_mib=None, limit_s=90, poll_s=3):
    """WDDM frees a 12 GB allocation in stages.

    Starting the next arm too early passes /health against memory the driver
    still holds, then dies on the first /completion -- instrument fault 7.

    Delegates to harness.vram_settled rather than comparing readings here. The
    first version of this function did compare them, and lost `floor_mib` in
    the process: "stopped moving" alone cannot tell *release finished* from
    *release has not begun*, because two polls taken before the driver does
    anything agree perfectly. The floor is the free reading taken WHILE the
    model was still resident, plus the minimum rise a real teardown must
    produce -- which a release that never started cannot reach.
    """
    readings = []
    for _ in range(int(limit_s / poll_s)):
        time.sleep(poll_s)
        readings.append(vram()[1])
        if vram_settled(readings, floor_mib=floor_mib):
            return readings
    print("    VRAM still moving after %ds: %s" % (limit_s, readings[-4:]),
          flush=True)
    return readings


def post(path, payload, timeout):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


_BLOCK = ("// section {i}\n"
          "static int helper_{i}(int a, int b) {{\n"
          "    int acc = a;\n"
          "    for (int k = 0; k < b; ++k) acc += (k ^ a) % 7;\n"
          "    return acc;\n"
          "}}\n\n")


# The real-code prompt comes from a FROZEN FILE, not from live source.
#
# INSTRUMENT FAULT, 2026-08-22. This used to read this directory's own
# harness.py, depth_sweep.py, model_arena.py, opencode_corpus.py and
# kv_sweep.py and slice the first n*3 characters. Appending 3,045 bytes to
# harness.py between two runs (24,306 -> 27,351) moved the 24,576-character
# window from "harness.py plus 270 characters of depth_sweep.py" to
# "harness.py alone", and the same arm with byte-identical arguments then
# measured 78.9 tok/s in one run and 105.4 in the other. Nothing was
# throttling: 49 C, no power cap, zero throttle counters. The workload had
# changed underneath the measurement, and the operator was the one changing it.
#
# corpora/real-code.txt is that source concatenated at commit 674ea4b, the
# tree report 29 was measured on, so those numbers stay interpretable.
CORPUS_DIR = Path(__file__).parent / "corpora"


# Frozen corpora, by regime. Each is committed evidence: its hash is stamped
# into every row measured against it, so a file that changes changes a visible
# number instead of changing nothing anybody can see.
#
# real-code.txt is NEVER modified or replaced. Rows going back to report 29 name
# its hash, and rewriting it would silently reinterpret all of them.
# real-code-deep.txt was added because the shallow one tops out around ctx
# 30,600 and the served window is 98,304 -- see build-deep-corpus.py for what
# went into it and what was deliberately left out.
CORPUS_FILES = {
    "real-code": "real-code.txt",
    "real-code-deep": "real-code-deep.txt",
    # A third text, from llama.cpp's gguf-py -- written by people who have
    # never seen this repo. It exists to separate "any prompt this long
    # collapses" from "this text at this slice does" (issue #44). Rows are
    # compared WITHIN a corpus, never across: it carries its own hash.
    "real-code-vendor": "real-code-vendor.txt",
}


def corpus_hash(regime):
    """Short hash of the frozen corpus, or None for a generated regime."""
    name = CORPUS_FILES.get(regime)
    if name is None:
        return None
    import hashlib
    return hashlib.sha256((CORPUS_DIR / name).read_bytes()).hexdigest()[:16]


def filler(n_tokens, regime="synthetic"):
    """Roughly n_tokens of prompt, identical for every arm AND every run.

    A drafter's acceptance depends on how predictable the text is, so varying
    the prompt between arms would measure the text instead of the decoder --
    and varying it between RUNS makes two runs incomparable, which is the
    fault described above.

    TWO REGIMES, BECAUSE THE ARMS DO DIFFERENT JOBS. `ngram-mod` drafts by
    matching text it has already seen in the context; it is strong exactly
    where the answer is already on screen and has nothing to offer where the
    model is writing something new. DFlash2 is a trained drafter and does not
    need to have seen the text before.

      synthetic       66.2 % duplicate lines -- ngram-mod's best case, and the
                      trap depth_sweep.py names in its own header.
      real-code       frozen real source, which is what the worker actually
                      reads. 91,868 chars: good to about ctx 30,600.
      real-code-deep  the same idea at 406,146 chars, good to about ctx
                      135,000, for the windows we actually serve.

    A verdict from one regime does not carry to the other, the same way a
    verdict at one depth does not carry to another depth. `real-code` and
    `real-code-deep` are DIFFERENT REGIMES for this reason and not one corpus
    that grew: they carry different hashes and rows are compared within a
    corpus, never across.

    A CORPUS TOO SMALL FOR THE WINDOW RAISES. This used to return
    `text[:n_tokens * 3]`, so any request above what the file holds silently
    produced a shorter prompt -- a run at ctx 65,536 that actually measured
    ~30,600 and reported a perfectly plausible rate for a window it never
    filled. Nothing in the row said so. Every verdict taken before this guard
    existed was at ctx 16,384, which `real-code.txt` covers with room to spare,
    so none of them is affected; the next one up would have been.
    """
    name = CORPUS_FILES.get(regime)
    if name is not None:
        text = (CORPUS_DIR / name).read_text(encoding="utf-8", errors="replace")
        want = n_tokens * 3
        if len(text) < want:
            raise ValueError(
                "%s holds %d chars but ctx %d needs %d. Truncating would "
                "measure a shorter window than the one reported. Use "
                "--regime real-code-deep, or extend the corpus with new source "
                "(never by tiling this one -- see CORRECTIONS.md 20)."
                % (name, len(text), n_tokens, want)
            )
        return text[:want] + ("\n# Explain what vram_settled guards against, "
                              "then write a test for it.\n")

    out, i = [], 0
    while sum(len(s) for s in out) < n_tokens * 3:
        out.append(_BLOCK.format(i=i))
        i += 1
    return "".join(out) + "\n// Explain what helper_3 computes, then rewrite it.\n"


def stop_server():
    """Kill the server AND wait for the driver to hand the memory back.

    One function, because separating them is how the wait died: run_arm's
    `finally` called kill(), so start()'s kill() then found nothing to kill,
    returned False, and skipped the wait for every arm in the run. The guard
    was present, called, and inert.

    Reading free VRAM before the kill is what makes the floor mean anything --
    it is how much has to come back.
    """
    resident_free = vram()[1]
    if not kill():
        return                      # nothing was running; nothing to wait for
    wait_for_vram_release(floor_mib=resident_free + VRAM_MIN_RISE_MIB)


def arm_exe(env):
    """The binary THIS ARM runs: its own `QWEN38_LLAMA_EXE`, else the module's.

    `EXE` is resolved once at import, which was right while every arm in a run
    shared a binary. It stopped being right the moment the question became
    "does build 10679 decode faster than 10499", because that is a comparison
    the harness could only make ACROSS boots -- and this bench exists because
    across-boot comparisons are not admissible here.
    """
    return (env or {}).get(ENV_VAR) or EXE


def server_argv(ctx, extra, env=None, verify=False):
    """The exact command line `start()` launches, without launching it.

    Extracted so a test can assert on what reaches llama-server rather than on
    what an arm set intended. The two differ whenever an arm overrides a flag
    this function hardcodes: `extra` is appended, so the argv carries the flag
    twice and only llama.cpp's last-wins parsing (`common/arg.cpp`, plain
    setters) decides which value is used. An arm set that got that backwards
    would run every arm at the hardcoded value and report a flat sweep.
    """
    exe = arm_exe(env)
    if verify and not os.path.isfile(exe):
        # REFUSE. Falling back to the module default would run both arms of a
        # build comparison on ONE build and report a flat sweep as a result.
        raise FileNotFoundError(
            "arm asked for %s and it is not a file. Not falling back to %s: "
            "a build A/B whose arms silently share a binary is worse than no "
            "A/B, because it looks complete." % (exe, EXE))
    return [exe, "-m", TARGET, "--alias", "Qwen3.8-27B-arena", "-c", str(ctx),
            "-ngl", "auto", "--fit", "on", "--fit-target", "768", "-fa", "on",
            "-np", "1", "-t", "18", "-b", "2048", "-ub", "256",
            "-ctk", "q4_0", "-ctv", "q4_0",
            "--no-mmproj-auto", "-lv", "5",
            "--reasoning-effort", EFFORT,
            "--host", "127.0.0.1", "--port", "8080"] + list(extra)


def start(ctx, extra, tag, boot_s=240, env=None):
    stop_server()
    free_before = free_for_env(env or {})
    log = ROOT / "logs" / ("dflash2-" + tag + ".log")
    # `env` IS NOT OPTIONAL HERE. Without it `arm_exe` falls back to the module
    # `EXE` and the process runs the default while `new_row` records the binary
    # the arm pinned -- CORRECTIONS 34's shape, one seam below the one its test
    # covers. It voided a six-arm build A/B on 2026-08-30: every row named
    # `llama.cpp-mirror` and every process was `llama.cpp-blackwell`, and the
    # only visible symptom was draft counters that matched to the digit across
    # the two "builds", which reads as greedy determinism until you check.
    # `verify=True` so a pin at a path that is not there stops the run instead
    # of silently becoming the default one boot later.
    args = server_argv(ctx, extra, env=env, verify=True)
    fh = log.open("w", encoding="utf-8", errors="replace")
    p = subprocess.Popen(args, stdout=fh, stderr=subprocess.STDOUT,
                         env=launch_env(env or {}))
    for _ in range(boot_s // 3):
        time.sleep(3)
        if p.poll() is not None:
            fh.close()
            return None, None, log, free_before
        try:
            urllib.request.urlopen(BASE + "/health", timeout=3).read()
            time.sleep(2)
            fh.flush()
            # The binary must carry kernels for every card this arm can see.
            # Checked HERE, on the first boot, because the alternative is
            # fifteen plausible rows: on 2026-08-27 a sweep ran at ctx 147,456
            # with 66+0 residency on a build whose CMAKE_CUDA_ARCHITECTURES was
            # 89, while a capability 12.0 card was visible and in use. Read from
            # the run's own `system_info` line rather than predicted from the
            # DLL -- see harness.archs_missing_for_gpus.
            missing = archs_missing_for_gpus(
                log.read_text(encoding="utf-8", errors="replace"),
                caps_for_env(env or {}))
            if missing:
                stop_server()
                fh.close()
                raise SystemExit(
                    "FATAL: %s has no CUDA kernels for compute capability %s, "
                    "which is visible to this arm.\n"
                    "  Every row from this binary would be a plausible number "
                    "measured on the wrong machine.\n"
                    "  Set QWEN38_LLAMA_EXE to a build whose "
                    "CMAKE_CUDA_ARCHITECTURES covers every installed card."
                    % (EXE, ", ".join(missing)))
            return p, fh, log, free_before
        except Exception:
            pass
    fh.close()
    return None, None, log, free_before


def rate(t):
    """A rate of zero is missing data whatever the token count says."""
    r = t.get("predicted_per_second")
    return r if t.get("predicted_n") and r and r > 0 else None


def record_fault(row, exc):
    """Append a fault to `row["note"]` without erasing what is already there.

    `run_arm` writes WHY a row is unmeasurable, then keeps working, and the
    later work can raise. Assigning to `note` in the handler destroyed the
    diagnosis: on 2026-08-24 eighteen rows at ctx 147,456 reported

        ValueError: no assignment pass has 65 layers; passes seen: [66, 66, 66]

    while the real problem -- every generation producing 9 tokens against a
    512-token budget -- had already been written to that field and was gone
    (issue #44). A harness that deletes its own evidence cannot be debugged.
    """
    fault = "%s: %s" % (type(exc).__name__, exc)
    row["note"] = (row["note"] + " | " + fault) if row.get("note") else fault


SAMPLER = {"temperature": 0.0, "top_k": 1, "seed": 42}


def completion_payload(prompt, ignore_eos=False):
    """The body of every /completion this arena sends.

    One function so a forced row and a natural one can be diffed: the ONLY key
    that may differ between them is `ignore_eos`, and a test asserts it.

    Absent rather than false by default. A row measured before this option
    existed carries neither the key nor the column, and writing `false` into
    the request would make the two look different in a packet capture while
    being identical in behaviour.
    """
    body = dict({"prompt": prompt, "n_predict": N_PREDICT,
                 "cache_prompt": True}, **SAMPLER)
    if ignore_eos:
        body["ignore_eos"] = True
    return body


def arm_target(ctx, extra):
    """The model an arm actually loads, read off the argv that launches it.

    `server_argv` hardcodes `-m TARGET` and appends the arm's flags, so an arm
    that overrides `-m` puts it twice on the command line and llama.cpp's
    last-wins parsing decides. Reading the LAST `-m` of the resolved argv is
    therefore the same answer the server gives itself -- and it cannot drift
    from `server_argv`, which a second scan of `extra` alone could.

    `-md` names the DRAFTER and is a different token: a speculative arm has two
    models and only one of them is the target.
    """
    argv = server_argv(ctx, extra)
    hits = [i for i, tok in enumerate(argv) if tok == "-m" and i + 1 < len(argv)]
    return argv[hits[-1] + 1]


def new_row(ctx, arm, rnd, regime, extra, env, free_before, ignore_eos=False,
            loaded=True):
    """The columns every row carries, and why each one is there.

    Four of these -- exe/cuda_archs, env, target/target_mib, effort -- were each
    added on 2026-08-24 AFTER a comparison had already been made without them.
    `ignore_eos` is the fifth and is being added before rather than after.
    """
    return dict(
        ctx=ctx, arm=arm, round=rnd, regime=regime,
        args=" ".join(extra),
        n_predict=N_PREDICT, free_before=free_before,
        corpus=corpus_hash(regime),
        # Which binary produced this number. Two builds on this machine report
        # the same version string and differ 2x in prefill; without these two
        # fields the JSONL cannot tell them apart afterwards.
        # THE ARM'S binary, not the module default. CORRECTIONS 34 is this
        # column's neighbour making exactly this mistake: `target` recorded the
        # module default for every row, so every NVFP4 arm was written down as
        # having run the Q4 control's file, and the test guarding it stayed
        # green throughout. A build A/B whose rows all say 10499 is worse than
        # no A/B -- it looks complete.
        exe=arm_exe(env), cuda_archs=cuda_archs(arm_exe(env)),
        # WHICH CARDS. Resolved from the environment a launch would actually
        # get, not from what the arm asked for -- the control arm asks for
        # nothing, and a silent column reads like "the usual card" right up
        # until the usual card changes, which here it did when a second GPU was
        # installed on 2026-08-26 (issue #50).
        devices=launch_env(env)["CUDA_VISIBLE_DEVICES"],
        # Which MODEL, with its size: two files on this machine share the name
        # UD-Q2_K_XL and differ by 808 MiB, so the path alone is not an identity
        # if the cache ever moves. Resolved from the arm's own `-m` when it has
        # one -- on 2026-08-29 an NVFP4 arm and the Q4 control both recorded the
        # Q4 path, because this column read the module default and an arm can
        # change its target by overriding `-m`.
        target=arm_target(ctx, extra),
        target_mib=model_size_mib(arm_target(ctx, extra)),
        # Which reasoning effort. Everything before 2026-08-24 ran at the
        # template's xhigh; a row without this field is one of those.
        effort=EFFORT,
        # Whether the generation budget was FORCED. Past the point the model
        # would have stopped it is decoding text it did not choose to write, and
        # draft acceptance is a property of how predictable that text is -- so a
        # forced row's acceptance is not comparable with a natural one's. Arm
        # against arm within one forced run is unaffected: every arm decodes
        # under the same rule.
        ignore_eos=bool(ignore_eos),
        # Always present, even when empty: absent and {} must not be the same
        # value, or "this arm set nothing" reads the same as "this row predates
        # the feature".
        # `env or {}`, so "this arm set nothing" is the empty dict this comment
        # already promises rather than a TypeError. `run_arm` defaults env to
        # None, so the control arm reaches here as None the moment a caller
        # skips the arm-set path.
        env=dict(env or {}),
        loaded=loaded)


def mark_arm_dead(label, dead, reason):
    """Record that an arm cannot load, and why.

    `dead` is a plain dict the sweep owns, not module state: two sweeps in one
    process must not inherit each other's failures.
    """
    dead[label] = reason


def should_skip_arm(label, dead):
    return label in dead


def first_line_of_failure(log_path):
    """The llama.cpp line that explains a refusal, or None.

    An arm that will not load says why -- "dflash requires ctx_other to be
    set", "device CUDA0 does not support split buffers", a GGML_ASSERT. Reading
    it here puts the reason in the JSONL instead of leaving it in a log file
    nobody opens.
    """
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return None
    wanted = ("GGML_ASSERT", "error loading", "failed to initialize",
              "does not support", "requires ", "out of memory",
              "not implemented")
    for line in text.splitlines():
        if any(w in line for w in wanted):
            return line.strip()[-200:]
    return None


def skipped_row(ctx, label, rnd, regime, extra, env, dead, ignore_eos=False):
    """A row for a round an arm was NOT tried in.

    Recorded rather than omitted. A missing row makes an impossible arm look
    unpaired -- report() prints "NOT PAIRED (1 vs 3 rounds)" and the reader
    concludes the sweep was interrupted, when in fact the arm cannot exist.
    """
    row = new_row(ctx, label, rnd, regime, extra, env, 0,
                  ignore_eos=ignore_eos, loaded=False)
    row["measurable"] = False
    row["note"] = ("not retried -- failed to load in an earlier round: %s"
                   % dead.get(label, "reason not captured"))
    return row


def run_arm(ctx, label, extra, rnd, regime="synthetic", env=None,
            ignore_eos=False):
    env = env or {}
    tag = (label.replace("+", "-") + "-" + regime
           + "-c" + str(ctx) + "-r" + str(rnd))
    p, fh, log, free_before = start(ctx, extra, tag, env=env)
    row = new_row(ctx, label, rnd, regime, extra, env, free_before,
                  ignore_eos=ignore_eos, loaded=p is not None)
    if p is None:
        why = first_line_of_failure(log)
        row["note"] = ("server failed to start"
                       + (" -- " + why if why else ""))
        print("    %-15s FAILED TO LOAD%s"
              % (label, ("  " + why) if why else ""), flush=True)
        return row

    try:
        prompt = filler(int(ctx * 0.5), regime)
        body = completion_payload(prompt, ignore_eos=ignore_eos)
        # Warm turn: pays the cold prefill once so the timed generations measure
        # decode rather than prefill. Its timings are discarded deliberately.
        #
        # FULL LENGTH, not a token or two. A 16-token warm turn paid the prefill
        # but left the n-gram table nearly empty, and the first TIMED generation
        # of every ngram arm then came in 35-40 % low -- 69.8 against 113.4 and
        # 114.2 in the same boot, while dflash2 and none showed no such step
        # because neither builds a table from the text it just emitted. Median
        # of three mostly absorbed it, which is precisely why it could have gone
        # unnoticed: a systematic bias hidden inside a summary statistic.
        post("/completion", body, timeout=900)
        timings, rates, contents = [], [], []
        for _ in range(N_GEN):
            r = post("/completion", body, timeout=900)
            timings.append(r["timings"])
            rates.append(rate(r["timings"]))
            contents.append(r.get("content"))
        # A row whose generations produced almost nothing is not a measurement.
        # At ctx 65,536 the corpus ran out, the model answered in 2-4 tokens
        # against a 512-token budget, and the arena reported a tight RESOLVED
        # -56.5 % computed over that. Refuse the row instead of ranking it.
        long_enough = generation_is_measurable(timings, N_PREDICT)
        # A 512-token verbatim copy of the prompt passes the length gate and is
        # not a decode measurement: the first row taken on real-code-vendor read
        # 195.13 tok/s with ngram-mod accepting 1,911 of 1,912 drafted tokens in
        # runs of 32.85, because the model was continuing the corpus rather than
        # answering the instruction appended to it (issue #44).
        original = generation_is_original(contents, prompt)
        measurable = long_enough and original
        good = [x for x in rates if x]
        row.update(tg_samples=rates,
                   predicted_n=[t.get("predicted_n") for t in timings],
                   measurable=measurable,
                   tg_med=(median(good) if (good and measurable) else None),
                   acceptance=draft_acceptance(timings))
        row["copied_frac"] = [round(copied_window_fraction(c, prompt), 3)
                              for c in contents]
        if not long_enough:
            row["note"] = ("generations too short to measure: predicted_n=%s "
                           "against n_predict=%d" %
                           ([x.get("predicted_n") for x in timings], N_PREDICT))
        elif not original:
            row["note"] = ("generations copy the prompt rather than answer it: "
                           "12-word windows found verbatim in the prompt = %s"
                           % row["copied_frac"])
        fh.flush()
        text = log.read_text(encoding="utf-8", errors="replace")
        row["split"] = "%d+%d" % parse_layer_split(
            text, expect_layers=target_layer_count(text))
        # Per-implementation counters. The pooled acceptance line cannot say
        # which speculator served which fraction, and with a chained
        # --spec-type that is the whole question.
        row["impl"] = parse_spec_impl_stats(text)
        row["free_after"] = vram()[1]
        impl = row.get("impl") or {}
        decl = "  ".join("%s decline %s%% len %s" %
                         (k, d["decline_pct"], d["mean_acc_len"])
                         for k, d in sorted(impl.items()))
        if not measurable:
            print("    %-15s NOT MEASURABLE  %s" % (label, row["note"]), flush=True)
        else:
            print("    %-15s %6.2f tok/s  split %-6s acc %-6s free %5s  %s"
                  % (label, row["tg_med"], row["split"],
                     row["acceptance"], row.get("free_after"), decl),
                  flush=True)
    except Exception as exc:               # a failed arm is a row, not a crash
        record_fault(row, exc)
        print("    %-15s ERROR %s" % (label, exc), flush=True)
    finally:
        stop_server()
        fh.close()
    return row


def report(rows):
    # Grouped by regime as well as depth. Pooling them averages a decoder's best
    # case with its worst and reports the result as its performance: on
    # 2026-08-22 this printed ngram-mod as [119.7, 119.4, 119.3, 53.0, 52.5,
    # 49.3] -- one baseline spanning both prompts -- and every delta computed
    # against it was meaningless. It did not crash and it did not look wrong.
    groups = {}
    for r in rows:
        groups.setdefault((r["ctx"], r.get("regime", "synthetic")), []).append(r)
    for (ctx, regime), rs in sorted(groups.items()):
        print("\nctx=%d  regime=%s" % (ctx, regime))
        # Arms come from the rows, not from a module constant: a named arm set
        # has different arm names, and reading ARMS here reported an empty
        # series for every sweep that was not the decoder comparison.
        series = {}
        # The residency each arm actually ran at, collected in the same pass.
        # `run_arm` has recorded this since the two-card work and the report
        # never read it -- see harness.residency_note.
        splits = {}
        for r in rs:
            if r.get("tg_med"):
                series.setdefault(r["arm"], []).append(r["tg_med"])
                splits.setdefault(r["arm"], []).append(r.get("split"))

        # The baseline is the arm the sweep varies FROM. Name it "*-base" or
        # call it ngram-mod; otherwise the first arm seen is used, and an
        # unlabelled baseline is a silent choice, so the header prints it.
        base_name = (next((k for k in series if k.endswith("-base")), None)
                     or ("ngram-mod" if "ngram-mod" in series else None)
                     or (sorted(series)[0] if series else None))
        if base_name is None:
            print("  no rows with a rate -- nothing to pair against")
            continue
        base = series[base_name]
        print("  baseline: %s" % base_name)

        # The floor a verdict was reached against, printed rather than assumed.
        # NOISE_FLOOR_PCT is 13.6 -- Ada, at ctx 16,384 -- and CLAUDE.md says it
        # does not transfer. At 147,456 on two cards this run's own arms spread
        # under 2 %, and the report called a tight, sign-consistent -13.3 %
        # "within noise" purely because of the imported constant.
        base_spread = observed_spread_pct(base)
        print("  floor applied: %.1f %% (NOISE_FLOOR_PCT, Ada @ ctx 16,384)"
              % NOISE_FLOOR_PCT)
        print("  this run's baseline spread: %s"
              % ("%.1f %% over %d rounds" % (base_spread, len(base))
                 if base_spread is not None else "unknown (one round)"))

        for label, vals in series.items():
            shown = [round(v, 1) for v in vals]
            spread = observed_spread_pct(vals)
            spread_s = ("%.1f %%" % spread) if spread is not None else "n/a"
            if label == base_name:
                print("  %-15s %s  spread %s  (baseline)"
                      % (label, shown, spread_s))
                continue
            if len(vals) != len(base):
                print("  %-15s %s  NOT PAIRED (%d vs %d rounds) -- no verdict"
                      % (label, shown, len(vals), len(base)))
                continue
            # Residency before arithmetic. A spilled arm and a resident one are
            # different machines, and a delta between them describes the spill.
            note = residency_note(splits[base_name], splits[label])
            if note:
                print("  %-15s %s  spread %s  NOT COMPARABLE (%s) -- no verdict"
                      % (label, shown, spread_s, note))
                continue
            d = paired_deltas(base, vals)
            if not d["resolved"] and d["min_pct"] * d["max_pct"] <= 0:
                # sign changes across rounds -- no amount of floor helps
                verdict = "inconsistent in sign"
            else:
                verdict = classify_against_floors(
                    d["mean_pct"], max(spread or 0.0, base_spread or 0.0))
                if verdict == "resolved":
                    verdict = "RESOLVED"
            print("  %-15s %s  spread %s  %+.1f%% [%+.1f, %+.1f]  %s"
                  % (label, shown, spread_s, d["mean_pct"],
                     d["min_pct"], d["max_pct"], verdict))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, nargs="+", default=[16384])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--regime",
                    choices=["synthetic", "real-code", "real-code-deep",
                             "real-code-vendor"],
                    nargs="+", default=["synthetic"])
    ap.add_argument("--arms", choices=sorted(ARM_SETS), default="decoders")
    # OFF by default. Every decoder figure this project holds was taken without
    # it, and turning it on globally would make new rows quietly incomparable
    # with old ones -- the shape of four fixes made on 2026-08-24, each added
    # after a comparison had already been made without the field. The row
    # carries the flag either way (issue #44).
    ap.add_argument("--ignore-eos", action="store_true",
                    help="force the full n_predict budget. Needed where the "
                         "model stops on EOS after a few tokens and the row is "
                         "voided; NOT comparable with a natural row's draft "
                         "acceptance")
    ap.add_argument("--out",
                    default=str(ROOT / "results" / "dflash2-arena.jsonl"))
    a = ap.parse_args()

    require_exclusive_port()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with out.open("a", encoding="utf-8") as fh:
        for ctx in a.ctx:
          for regime in a.regime:
            # Per ctx and regime, not global: an arm that cannot load at
            # 262,144 may load fine at 16,384, and inheriting the verdict
            # across depths would skip a measurement that was available.
            dead = {}
            for rnd in range(1, a.rounds + 1):
                # Rotate so no arm always runs first: within a boot the earlier
                # arm sees a cleaner GPU, and a fixed order hands that advantage
                # to the same arm every round.
                arms = ARM_SETS[a.arms]
                k = (rnd - 1) % len(arms)
                order = arms[k:] + arms[:k]
                print("  ctx=%d %s round %d: %s"
                      % (ctx, regime, rnd,
                         " -> ".join(arm_parts(a)[0] for a in order)), flush=True)
                for arm in order:
                    label, extra, env = arm_parts(arm)
                    # An arm that could not load will not load. The argv is
                    # byte-identical between rounds and the failure is a
                    # capability, not a resource -- `dflash requires ctx_other
                    # to be set` does not become true on the second try. Each
                    # retry cost a boot plus a full VRAM-release wait for a
                    # result already known (developer, 2026-08-27).
                    if should_skip_arm(label, dead):
                        row = skipped_row(ctx, label, rnd, regime, extra, env,
                                          dead, ignore_eos=a.ignore_eos)
                        print("    %-15s SKIPPED -- %s"
                              % (label, dead[label]), flush=True)
                    else:
                        row = run_arm(ctx, label, extra, rnd, regime, env,
                                      ignore_eos=a.ignore_eos)
                        if not row.get("loaded", True):
                            mark_arm_dead(label, dead, row.get("note", "failed to load"))
                    rows.append(row)
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()

    print("\nwrote %d rows to %s" % (len(rows), out))
    report(rows)


if __name__ == "__main__":
    sys.exit(main())
