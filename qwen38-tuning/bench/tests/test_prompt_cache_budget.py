r"""The prompt cache is smaller than one of our conversations, and that is the
single largest cost this project has measured.

MEASURED 2026-09-02, `logs/serve-20260902-034815.log`, a live two-agent Claude
Code session on icon 2 (NVFP4-MTP-VERY-LOW, build 10729, ctx 200,704). Four
hours, 303 completed requests, and in the LAST THIRTY MINUTES of it:

    wall     1,801 s
    prefill  1,478 s over 1,033,213 tokens
    decode     239 s ->      7,024 tokens @ 29.4 tok/s
    FORCED re-prefill 1,229 s = 68.2 % OF WALL, all ten events after an eviction

The server prefilled 147 tokens for every token it emitted. Over the whole
session the same figure is 30.5 % of wall -- the average is diluted by the first
hour, when the conversation was still small, and it climbs as the window fills.

WHY, AND IT IS NOT THE CHECKPOINTS. The block above `$betaArg` in the profile
blames the hybrid: "the recurrent half cannot be rewound to a shared prefix". For
`--ctx-checkpoints 0` that was measured and stands. For THIS it is wrong, and the
log says so in the line before every re-prefill:

    srv   prompt_save: - saving prompt with length 45619, total state size = 1131.811 MiB
    srv         alloc: - making room for prompt cache entry, removing oldest entry (size = 7028.285 MiB)
    slot operator (): id 0 | task 70315 | new prompt, task.n_tokens = 160447
    slot operator (): id 0 | task 70315 | checking checkpoint with [45590, 45590] against 3...
    slot operator (): id 0 | task 70315 | forcing full prompt re-processing

`against 3` is `pos_min_thold`: the incoming prompt shares THREE tokens with what
the slot holds. They are different conversations -- the main agent at 160k and a
sub-agent at 45k, alternating on one slot -- so discarding every checkpoint is
correct. The conversation that could have been reused was in the prompt cache,
and it had been evicted one line earlier to make room.

`--cache-ram` defaults to 8192 MiB (`common/common.h:632`) and the profile never
passed it, so that default was in force. At ctx 200,704 on this artifact a single
conversation's state outgrows it -- three times in that session llama.cpp refused
to cache one at all:

    srv alloc: - prompt state size 9801.444 MiB exceeds cache size limit 8192.000 MiB, skipping

31 `making room` evictions discarded 122,276 MiB. The mechanism itself is
healthy: 37 prompt-cache restores succeeded in the same session, one recovering a
35,733-token prefix whole (`found better prompt with f_keep = 1.000`), and
`--cache-idle-slots` is already on by default. **The budget was the bug.**

Full evidence and the source reading:
https://github.com/xenodeve/Qwen-3.8-27B-Tuning/issues/70#issuecomment-5502598376

WHY 16384 AND NOT MORE. The largest entry seen is 9,801 MiB, the sub-agent's is
about 2,029 MiB, so 16 GiB holds both today with room to grow. It is not a
comfortable margin and it is not meant to be: the host has 47.7 GB and this
server already commits 34.35 GB of it, so a bigger cap trades a re-prefill for
paging, which is the one way this change can lose. **The log reads out its own
verdict** -- if `exceeds cache size limit` or `making room` come back, 16384 was
not enough and the next value is an experiment, not a guess.

ONE FLAG, NOT TWO. `--ctx-checkpoints` was the other candidate: entries would
shrink if fewer were kept. It is deliberately left alone. Checkpoints earn their
place in the same log -- 220 successful `restored context checkpoint` -- and the
cap is never the limit, `created context checkpoint 11 of 32` being the highest
ever reached. Changing both would leave the next session unable to say which one
moved.
"""
import functools
import os
import re

import pytest

import _invocation

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(BENCH))
PROFILE = os.path.join(ROOT, "qwen38-tuning", "scripts", "worker-q4-dual.ps1")

SERVED = "16384"

# Every profile this change reaches. Shared so two tests cannot drift into
# asking the same question of different profiles.
SERVING = [
    ("-Nvfp4", "-Deep", "-Vision"),   # icon 2, where the 68.2 % was measured
    ("-Nvfp4", "-Deep"),
    ("-Nvfp4",),                      # icon 1, same artifact, 147,456
    (),                               # the Q4 dual profile
]


@functools.lru_cache(maxsize=None)
def _argv(args):
    """Only the previewed command line, and one dry run per distinct argument
    set. A value quoted in the profile's prose is not a value the server
    receives -- traps.md 16, which is why this slices at the marker rather than
    searching the whole output. `_invocation.resolved` already fails if the
    profile did not preview at all."""
    out = _invocation.resolved(PROFILE, *args)
    return out[out.index("WhatIf: would run"):]


# ------------------------------------------------- the profiles that serve it

@pytest.mark.parametrize("args", SERVING)
def test_the_prompt_cache_is_bigger_than_one_conversation(args):
    """8192 MiB is llama.cpp's default and it was never chosen here -- the same
    shape as `--spec-ngram-mod-n-max 32`, which cost 15 % for the same reason."""
    out = _argv(args)
    assert re.search(r"--cache-ram\s+" + SERVED + r"\b", out), out


@pytest.mark.parametrize("args", SERVING)
def test_the_flag_is_passed_exactly_once(args):
    """`--cache-ram` appearing twice would leave which value wins to llama.cpp's
    argument parser, and a reader of the command line unable to say."""
    out = _argv(args)
    assert len(re.findall(r"--cache-ram\b", out)) == 1, out


# ---------------------------------------------------- and the one that must not

def test_the_beta_bundle_still_disables_the_prompt_cache():
    """`-Beta` is Unsloth Studio's bundle and Studio sets `--cache-ram 0`. It is
    a measurement of THEIR configuration; raising it there would change what the
    arm is testing while the name stayed the same."""
    out = _argv(("-Nvfp4", "-Deep", "-Beta"))
    assert re.search(r"--cache-ram\s+0\b", out), out
    assert SERVED not in out, out
    assert len(re.findall(r"--cache-ram\b", out)) == 1, out


def test_the_checkpoint_budget_was_not_moved_at_the_same_time():
    """One flag changes, so the next real session can attribute the difference.
    220 checkpoint restores succeeded in the measured session and the cap was
    never reached -- there is nothing here to fix."""
    out = _argv(("-Nvfp4", "-Deep", "-Vision"))
    assert "--ctx-checkpoints" not in out, out
    assert "-ctxcp" not in out, out
