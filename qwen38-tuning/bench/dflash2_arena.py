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
                     NOISE_FLOOR_PCT,
                     median, parse_layer_split, target_layer_count,
                     generation_is_original, copied_window_fraction,
                     draft_acceptance,
                     paired_deltas, vram_settled, VRAM_MIN_RISE_MIB,
                     parse_spec_impl_stats, generation_is_measurable)
from provenance import (resolve_exe, resolve_target, resolve_effort,
                        cuda_archs, model_size_mib)

ROOT = Path(r"C:\AI\qwen38-tuning")
EXE = resolve_exe(r"C:\AI\llama.cpp-dflash2\llama-server.exe")
TARGET = resolve_target(
    r"C:\Users\xenod\.cache\huggingface\hub\models--unsloth--Qwen3.8-27B-GGUF"
    r"\snapshots\27af057ecb382ddfea5d12837360a8980560e3ed"
    r"\Qwen3.8-27B-UD-IQ2_XXS.gguf")
DRAFTER = (r"C:\Users\xenod\.cache\huggingface\hub"
           r"\models--z-lab--Qwen3.8-27B-DFlash2-GGUF"
           r"\snapshots\57ab3265056d4024870b0621cfc2c127537020ed"
           r"\Qwen3.8-27B-DFlash2-Q4_K_M.gguf")
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
    return {**os.environ, **gpu_device.pin_env(), **env}


def _ngram(n_min, n_match=12, n_max=32):
    return ["--spec-ngram-mod-n-match", str(n_match),
            "--spec-ngram-mod-n-min", str(n_min),
            "--spec-ngram-mod-n-max", str(n_max)]


def _pair(extra_ngram=None, n_draft=4, extra=()):
    return (["--spec-type", "draft-dflash,ngram-mod",
             "-md", DRAFTER, "--spec-draft-n-max", str(n_draft), "-ngld", "99"]
            + (extra_ngram if extra_ngram is not None else NGRAM) + list(extra))


# Named arm sets. The default set answers "which decoder"; the others answer
# "which setting of the decoder we already chose", which is where the measured
# levers are.
ARM_SETS = {
    "decoders": ARMS,

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

    "graph-opt": [
        ("graph-opt-off", SERVED_NGRAM, {}),
        ("graph-opt-on", SERVED_NGRAM, {"GGML_CUDA_GRAPH_OPT": "1"}),
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


def server_argv(ctx, extra):
    """The exact command line `start()` launches, without launching it.

    Extracted so a test can assert on what reaches llama-server rather than on
    what an arm set intended. The two differ whenever an arm overrides a flag
    this function hardcodes: `extra` is appended, so the argv carries the flag
    twice and only llama.cpp's last-wins parsing (`common/arg.cpp`, plain
    setters) decides which value is used. An arm set that got that backwards
    would run every arm at the hardcoded value and report a flat sweep.
    """
    return [EXE, "-m", TARGET, "--alias", "qwen38", "-c", str(ctx),
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
    args = server_argv(ctx, extra)
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
        exe=EXE, cuda_archs=cuda_archs(EXE),
        # WHICH CARDS. Resolved from the environment a launch would actually
        # get, not from what the arm asked for -- the control arm asks for
        # nothing, and a silent column reads like "the usual card" right up
        # until the usual card changes, which here it did when a second GPU was
        # installed on 2026-08-26 (issue #50).
        devices=launch_env(env)["CUDA_VISIBLE_DEVICES"],
        # Which MODEL, with its size: two files on this machine share the name
        # UD-Q2_K_XL and differ by 808 MiB, so the path alone is not an identity
        # if the cache ever moves.
        target=TARGET, target_mib=model_size_mib(TARGET),
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
        env=dict(env),
        loaded=loaded)


def run_arm(ctx, label, extra, rnd, regime="synthetic", env=None,
            ignore_eos=False):
    env = env or {}
    tag = (label.replace("+", "-") + "-" + regime
           + "-c" + str(ctx) + "-r" + str(rnd))
    p, fh, log, free_before = start(ctx, extra, tag, env=env)
    row = new_row(ctx, label, rnd, regime, extra, env, free_before,
                  ignore_eos=ignore_eos, loaded=p is not None)
    if p is None:
        row["note"] = "server failed to start"
        print("    %-15s FAILED TO LOAD" % label, flush=True)
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
        for r in rs:
            if r.get("tg_med"):
                series.setdefault(r["arm"], []).append(r["tg_med"])

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
                    row = run_arm(ctx, label, extra, rnd, regime, env,
                                  ignore_eos=a.ignore_eos)
                    rows.append(row)
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()

    print("\nwrote %d rows to %s" % (len(rows), out))
    report(rows)


if __name__ == "__main__":
    sys.exit(main())
