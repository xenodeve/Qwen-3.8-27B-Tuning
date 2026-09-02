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

WHY 24576, AND WHY 16384 WAS SERVED FIRST FOR A WRONG REASON. Replaying the
log's own recorded entry sizes through the same `alloc()`/`update()` arithmetic,
counting the 52 forced re-prefills that would have found their prefix:

     8192 (llama.cpp's default)   0 / 52
    16384                        35 / 52
    24576                        43 / 52
       -1  == no size limit      13 / 52

A SIMULATION, labelled as one -- the prefix test is a heuristic, not the server's
`f_keep`/`f_sim` rule.

`-1` IS A TRAP. `server-task.h:613` maps a negative to `limit_size = 0`, and
`update()` gates its dynamic token raise on `limit_size > 0`, so `-1` pins the
cap at its constructor value, `n_ctx` = 200,704 tokens, against two live
conversations of 167k + 46k = 213k.

16384 was picked over 24576 on "the host commits 34.35 GB of 47.7, so a bigger
cap trades a re-prefill for paging". **Two errors.** `--cache-ram` is a CAP, NOT
A RESERVATION -- `alloc()` only resizes to the state actually stored, so raising
it costs the difference the cache really holds, about 4-6 GB against 16.5 GB of
free commit. And the commit limit is not fixed: `AutomaticManagedPagefile` is
True on a 932 GB WD_BLACK SN850X, measured at 1,809 MB/s write and 5,332 MB/s
read, so a 7 GiB entry faulted back costs ~1.3 s against the 200-250 s
re-prefill it replaces. The server already runs mostly paged -- 34.5 GB private
against a 4.5 GB working set.

**The log still reads out its own verdict** -- if `exceeds cache size limit` or
`making room` come back, 24576 was not enough. If decode falls while the
hard-fault rate climbs, the paging trade went the wrong way and this is one
constant to revert.

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

SERVED = "24576"

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


def test_the_checkpoint_budget_is_four_because_the_log_never_used_a_fourth():
    """32 is llama.cpp's default and 28 of those slots were never touched.

    This test used to assert the flag was ABSENT, on the reasoning that
    "checkpoints earn their place -- 220 restores -- and the cap is never the
    limit". The second half was true and irrelevant: reaching the cap is not the
    question, USING what is held is. Measured over the same session, counting
    how deep each successful restore had to search a list that
    `create_checkpoint` keeps newest-last and `std::find_if` walks in reverse
    (`server-context.cpp:3324-3336`):

        240 restores
          newest checkpoint      185    77.1 %
          second newest           52    98.8 %
          third newest             3   100.0 %
          deeper                   0

    752 checkpoints were created at 151-834 MiB each, median 320. The highest
    slot ever reached is `created context checkpoint 11 of 32`, and the deepest
    ever RESTORED FROM is the third.

    Four keeps one slot of margin over the deepest observed use. It is safe
    because the cap evicts the OLDEST and always admits the new one
    (`server-context.cpp:2317-2324`), so a smaller cap keeps the newest K --
    exactly the ones the restores reach for. A cap that refused new checkpoints
    instead would freeze the set at its oldest members and this would be a
    regression rather than a cleanup.

    WHAT IT BUYS. `alloc()` counts checkpoints into `state_size_new`
    (`server-task.cpp:1723-1728`), so they are why an entry overflows
    `--cache-ram`. One cached entry in the log holds 116,241 tokens with
    **11 checkpoints at 7,755 MiB total**, of which roughly 5,150 MiB is
    checkpoints. At four that entry is about 4,500 MiB.

    WHAT IT COSTS, stated because the next session must not misread the log:
    this moves a SECOND flag before `--cache-ram 24576` has been measured, so if
    `making room for prompt cache entry` disappears, the two cannot be told
    apart. It is shipped anyway because 28 unused slots are waste rather than an
    experimental arm.
    """
    out = _argv(("-Nvfp4", "-Deep", "-Vision"))
    assert re.search(r"--ctx-checkpoints\s+4\b", out), out
    # `arg.cpp:1695` gives this one option THREE names. A second spelling on the
    # same command line would silently override the first, and the reader of the
    # argv could not say which value won.
    assert len(re.findall(r"(?:--ctx-checkpoints|-ctxcp|--swa-checkpoints)\b", out)) == 1, out


def test_beta_still_carries_no_checkpoint_flag_of_its_own():
    """`--ctx-checkpoints 0` LEFT the `-Beta` bundle on 2026-08-29, measured:
    on a hybrid model it makes every turn re-prefill from token 0, 51.6 s at the
    served depth (CORRECTIONS 39). It must not come back through this door --
    `-Beta` inherits the same 4, not a zero."""
    out = _argv(("-Nvfp4", "-Deep", "-Beta"))
    assert not re.search(r"--ctx-checkpoints\s+0\b", out), out
    assert re.search(r"--ctx-checkpoints\s+4\b", out), out
