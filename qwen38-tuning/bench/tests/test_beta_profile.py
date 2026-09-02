r"""A profile that adopts what Unsloth Studio does, as ONE testable bundle.

Called `-Beta` from 2026-08-29 -- it was `-Lean` while the bundle was only
about memory, and the name stopped fitting once it also carried decoder
settings the developer wanted tried.

Studio runs the same model file on the same two cards and differs from us on
eleven flags (`docs/researchs/unsloth-studio-config-2026-08-29.md`). Adopting
them one at a time is eleven sweeps; adopting them all silently is a profile
nobody can reason about. `-Beta` is the middle: **one switch, one bundle, one
paired measurement**, and if the bundle wins it gets bisected.

WHAT IS IN THE BUNDLE, AND WHY EACH ONE

  --cache-ram 0        MEASURED HERE: a real session held 20.4 GB working set
                       and 34.4 GB private. Studio disables it. This is the HOST
                       store for evicted prompts -- it carries a conversation
                       across a slot change, not across a turn. Still open.
  (--ctx-checkpoints 0 was in this bundle and was MEASURED OUT on 2026-08-29:
   on a hybrid model it makes every turn re-prefill from token 0, 51.6 s at the
   served depth. It is a different mechanism from --cache-ram and was only ever
   grouped with it because Studio sets both.)
  --load-mode none     VENDOR: Studio's auto "picks None when it can prove the
                       model fits without paging, since a mapped read is slower".
  --kv-unified         Studio sets it; may be inert at -np 1.
  --threads 2          Studio uses 2 against our 18. Everything is GPU-resident,
                       and the draft sampler falls back to the CPU under this
                       split, so the right number is not obvious either way.
  --metrics            free Prometheus endpoint. No throughput claim.

WHAT IS DELIBERATELY *NOT* IN IT

  the n-gram parameters and the --spec-type order -- they have their own arm
  sets and belong in a sweep, not in a profile;
  --parallel 4 -- Studio shares one window across four slots; we serve one
  conversation and want the whole window for it;
  -c ~41,000 -- their `auto` picks shallow "while prioritizing GPU speed"
  (VENDOR). Depth is the point of this machine;
  -ub 512 -- their default. Ours is 1024 and MEASURED at +10.1 % prefill;
  --spec-draft-n-max 2 -- their documented default. Ours is 3 and our own
  acceptance per position (0.690, 0.448, 0.284) supports it;
  the sampler -- a QUALITY lever, and quality is unmeasured on every artifact
  here.

SEPARATE FROM THE BUNDLE, AND NOT OPTIONAL: `--fit on --fit-target 768` is
MEASURED INERT under `-sm tensor` -- `llama_params_fit is not implemented for
SPLIT_MODE_TENSOR` on every boot. It is removed from the profile outright, for
every arm, because a flag that does nothing is a flag a reader believes did
something. Studio passes `--fit off` for the same reason.
"""
import os
import re
import subprocess

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(BENCH))
PROFILE = os.path.join(ROOT, "qwen38-tuning", "scripts", "worker-q4-dual.ps1")
SERVE = os.path.join(ROOT, "serve.ps1")


def _whatif(script, *args):
    r = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", script, "-WhatIf"] + list(args),
        capture_output=True, text=True, timeout=180)
    return r.stdout + r.stderr


# ---------------------------------------------- the inert flag, removed for good

@pytest.mark.parametrize("args", [(), ("-Nvfp4",), ("-Nvfp4", "-Beta")])
def test_fit_is_turned_OFF_not_merely_unstated(args):
    """`--fit` DEFAULTS TO ON, so removing the flag changes nothing.

    This started as "delete the inert flag", and booting it showed the fitting
    warning still there -- `--fit [on|off] ... default: 'on'`. Silence needs
    `--fit off`, which is what Unsloth Studio passes. Deleting the flag was a
    no-op dressed as a cleanup, and only running it said so.

    `--fit-target` goes with it: it is a margin for a fitting step that never
    runs under this split.
    """
    out = _whatif(PROFILE, *args)
    assert re.search(r"--fit\s+off", out), out
    assert "--fit-target" not in out, out


@pytest.mark.parametrize("args", [(), ("-Nvfp4",), ("-Nvfp4", "-Beta")])
def test_the_split_is_still_tensor(args):
    """Removing --fit must not have touched what makes it inert."""
    out = _whatif(PROFILE, *args)
    assert re.search(r"-sm\s+tensor", out), out


# ------------------------------------------------------------------ the bundle

BUNDLE = ["--cache-ram", "0",
          "--load-mode", "none", "--kv-unified", "--metrics",
          "-t", "2"]


def test_beta_applies_the_whole_bundle():
    """`--ctx-checkpoints 0` was here until 2026-08-29 and was MEASURED OUT --
    see test_beta_does_not_disable_context_checkpoints for the 51.6 s."""
    out = _whatif(PROFILE, "-Nvfp4", "-Beta")
    for flag, value in (("--cache-ram", "0"), ("--load-mode", "none")):
        assert re.search(re.escape(flag) + r"\s+" + value, out), (flag, out)
    assert "--kv-unified" in out, out
    assert "--metrics" in out, out
    assert re.search(r"-t\s+2\b", out), out


def test_without_beta_none_of_it_appears():
    """Opt-in. The bundle is unmeasured and must not leak into the default.

    `--cache-ram` LEFT THIS LIST 2026-09-02, and it is not a leak. The default
    profile now names the flag at its own MEASURED value, `24576`, for a reason
    that has nothing to do with Studio: llama.cpp's 8192 MiB default is smaller
    than one of our conversations at ctx 200,704, and a live session spent
    68.2 % of its last half hour re-prefilling what the prompt cache had just
    evicted (issue #70, `bench/tests/test_prompt_cache_budget.py`).

    `--ctx-checkpoints` LEFT IT TOO, 2026-09-02, for the same reason and with
    the sharper version of the same evidence. Of 240 successful restores in that
    session, 185 used the newest checkpoint, 52 the second, 3 the third and none
    went deeper, while the default holds 32. The profile now names it at **4**.
    Studio's value is **0**, which CORRECTIONS 39 measured as a fault on this
    hybrid -- every turn re-prefills from token 0, 51.6 s at the served depth.

    What the guard has to say is therefore narrower and stronger than "absent":
    the default must not carry Studio's VALUES. `--cache-ram 0` disables the
    store and 24576 raises it; `--ctx-checkpoints 0` disables checkpoints and 4
    trims them. Same flags, opposite decisions.
    """
    out = _whatif(PROFILE, "-Nvfp4")
    for flag in ("--load-mode", "--kv-unified", "--metrics"):
        assert flag not in out, (flag, out)
    assert not re.search(r"--cache-ram\s+0\b", out), out
    assert re.search(r"--cache-ram\s+24576\b", out), out
    assert not re.search(r"--ctx-checkpoints\s+0\b", out), out
    assert re.search(r"--ctx-checkpoints\s+4\b", out), out
    assert re.search(r"-t\s+18\b", out), out


def test_beta_keeps_everything_we_measured():
    """The bundle borrows what Studio does DIFFERENTLY. It must not quietly
    revert a value this project measured and won on.

    The n-gram left this list on 2026-08-29 and that is not a quiet revert: it
    is the one change here made ON PURPOSE against a number of ours, because
    that number (+27.1 % for n-match 24) was measured on a corpus where the
    n-gram fires and the developer's workload is one where it does not -- two
    sessions at `#gen drafts = 0`. `test_beta_runs_MTP_ALONE` carries the
    reasoning; this list keeps what is NOT in dispute.
    """
    out = _whatif(PROFILE, "-Nvfp4", "-Beta")
    assert re.search(r"-ub\s+1024", out), "-ub 1024 is +10.1 % prefill, MEASURED"
    assert re.search(r"--spec-draft-n-max\s+3", out), "2 was measured slower here"
    assert re.search(r"-np\s+1", out), "one conversation gets the whole window"


def test_beta_reaches_the_profile_through_serve():
    out = _whatif(SERVE, "-Dual", "-Nvfp4", "-Beta")
    assert re.search(r"-Beta\s+True", out), out


def test_the_banner_says_the_bundle_is_unmeasured():
    out = _whatif(SERVE, "-Dual", "-Nvfp4", "-Beta")
    assert "UNMEASURED" in out.upper(), out


# ---------------------------------------------------- the launcher, at the developer's request

BETA = os.path.join(ROOT, "launchers", "serve-dual-nvfp4-beta.bat")
BETA_LAN = os.path.join(ROOT, "launchers", "serve-dual-nvfp4-beta-lan.bat")
BOTH_BETA = [BETA, BETA_LAN]


def read_bat(path):
    with open(path, "rb") as fh:
        return fh.read().decode("ascii")


@pytest.mark.parametrize("path", BOTH_BETA)
def test_the_beta_launcher_exists(path):
    """Shipped for an UNMEASURED bundle, which is normally forbidden here --
    the exception is that the developer asked for it in order to be the
    measurement. The file must say that, which the next test checks."""
    assert os.path.exists(path), path


@pytest.mark.parametrize("path", BOTH_BETA)
def test_it_admits_the_bundle_is_unmeasured(path):
    """A launcher that presents a hypothesis as a result is trap 17 again."""
    assert "UNMEASURED" in read_bat(path).upper(), path


@pytest.mark.parametrize("path", BOTH_BETA)
def test_it_asks_for_the_bundle_and_the_same_everything_else(path):
    """AT 200,704, because that is the depth actually served.

    This shipped at 147,456 first, on the reasoning that it made a clean A/B
    against the shallow pair. The developer runs 200k. An A/B against a
    configuration nobody uses answers a question nobody asked, so the pair moved
    and its partner is now serve-dual-nvfp4-deep.bat.
    """
    t = read_bat(path)
    for flag in ("-Dual", "-Nvfp4", "-Vision", "-Beta", "-Deep"):
        assert flag in t, (path, flag)


@pytest.mark.parametrize("path", BOTH_BETA)
def test_it_is_readable_by_cmd(path):
    raw = open(path, "rb").read()
    raw.decode("ascii")
    assert not raw.startswith(b"\xef\xbb\xbf"), "a BOM makes cmd choke"
    assert b"\r\n" in raw


def test_only_the_lan_beta_one_exposes():
    assert "-Lan" not in read_bat(BETA)
    assert "-Lan" in read_bat(BETA_LAN)


def test_it_names_the_number_to_compare_against():
    """The point of the icon is an A/B the developer runs. A launcher that does
    not say what to compare it with leaves them to remember.

    Asserted as "two working-set figures at this depth", not as two literals:
    the first version hardcoded 15.28 and 2.03, which were measured at ctx
    65,536, and moving the pair to 200,704 made them the wrong numbers for the
    file they were in.
    """
    t = read_bat(BETA)
    assert t.count("GB working set") >= 2, (
        "the .bat must state BOTH sides of the memory pair it claims")
    assert "200,704" in t, "it must say which depth those figures came from"


# ------------------------------------------------- thinking, the Unsloth mechanism

def test_beta_uses_the_models_own_template_with_kwargs():
    """Studio does not pass a template file at all.

    It uses the one inside the GGUF and steers it with
    `--chat-template-kwargs {"enable_thinking": true, "preserve_thinking": true}`.
    We pass `--chat-template-file qwen38-late-system.jinja` plus
    `--reasoning-effort medium`, and NEITHER the template's reason for existing
    nor the choice of `medium` is written down anywhere in this repository.

    `-Beta` borrows their mechanism whole, which is the only way to find out
    whether ours is doing anything. Note what `preserve_thinking` maps to on our
    side: `--reasoning-preserve`, a flag we do not set and which our own boot log
    suggests -- "chat template supports preserving reasoning, consider enabling
    it via --reasoning-preserve".

    THEIR MECHANISM, ADAPTED TO OUR BINARY, BECAUSE BOOTING IT SAID SO. Copying
    `--chat-template-kwargs {"enable_thinking": true, "preserve_thinking": true}`
    verbatim starts and thinks, but the log answers back twice:

        W Setting 'enable_thinking' via --chat-template-kwargs is deprecated.
          Use --reasoning on / --reasoning off instead.
        I chat template supports preserving reasoning, consider enabling it via
          --reasoning-preserve

    So one kwarg is deprecated and the other DOES NOTHING -- the server still
    asks for `--reasoning-preserve` after being handed `preserve_thinking`.
    Copying a command line from a different build is copying its bugs.

    WHAT THIS TEST ITSELF GOT WRONG, 2026-08-29. It asserted `--reasoning-effort`
    was ABSENT -- borrowing their mechanism was read as borrowing the whole line,
    effort included. But Studio does not drop the effort; it sends `medium` per
    REQUEST (`reasoningEffort: "medium"`, both n-max threads in `studio.db`).
    Dropping the flag with no client to supply it hands the choice to the chat
    template, and the served boot log then read

        Reasoning effort is set to xhigh. Please think carefully through ...

    -- the exact default `test_reasoning_effort_default.py` exists to end,
    restored by a switch, with decode healthy the whole time. Only the template
    FILE is Studio's to omit. See CORRECTIONS 36.

    AND THEN IT GOT THE TEMPLATE WRONG TOO, 2026-08-31, issue #58. "Only the
    template FILE is Studio's to omit" is true about STUDIO and false about
    what we can serve. Studio omits it safely because Studio's client never
    sends a system message after the user turn; Claude Code sends one every
    session, and without the patched file Qwen3.8's own template RAISES on it.
    Five hub icons answered HTTP 500 to every request -- fifteen in a row in
    `logs/serve-20260831-023636.log` -- because this assertion was read as "the
    omission belongs to -Beta" rather than "the omission belongs to whoever
    asks for it".

    So the omission moved to `-StockTemplate` and this test now asks for it by
    name. `-Beta` alone carries the template, which is what the two assertions
    below the docstring check. The sweep that keeps EVERY icon honest lives in
    `test_chat_template_travels.py`; two branches is one fewer than the number
    that can break.
    """
    out = _whatif(PROFILE, "-Nvfp4", "-Deep", "-Beta", "-StockTemplate")
    assert "--chat-template-file" not in out, out
    assert re.search(r"--reasoning-effort\s+medium", out), out
    assert "--chat-template-kwargs" not in out, "deprecated on this build"
    assert re.search(r"--reasoning\s+on", out), out
    assert "--reasoning-preserve" in out, out

    # -Beta WITHOUT the switch keeps the template. This is the line whose
    # absence cost five icons every Claude Code request (issue #58).
    plain = _whatif(PROFILE, "-Nvfp4", "-Deep", "-Beta")
    assert "qwen38-late-system.jinja" in plain, plain


def test_without_beta_the_template_file_is_still_used():
    """The default keeps what it has always had. This is an experiment, not a
    migration."""
    out = _whatif(PROFILE, "-Nvfp4", "-Deep")
    assert "qwen38-late-system.jinja" in out, out
    assert re.search(r"--reasoning-effort\s+medium", out), out
    assert "--chat-template-kwargs" not in out, out


def test_no_json_blob_has_to_cross_two_shells():
    """The first version passed a JSON object as one argv entry, which has to
    survive PowerShell and then cmd intact. Using the flags llama.cpp actually
    wants removes the problem rather than solving it."""
    out = _whatif(PROFILE, "-Nvfp4", "-Deep", "-Beta")
    assert "{" not in out.split("llama-server.exe")[-1], out


# ------------------------------------- the two decoder values added on request

def test_beta_keeps_our_draft_depth_because_2_was_MEASURED_SLOWER():
    """REVERTED 2026-08-29, hours after it was added, on the developer's own use.

    -Beta carried Studio's `--spec-draft-n-max 2` for one afternoon. The
    developer said it felt slower and the server's own counters say why -- not
    the rate, which is a cross-session comparison, but the MECHANISM:

        n-max 3    297 drafts -> 891 tokens   = 3 per draft, mean acc len 2.80
        n-max 2    887 drafts -> 1774 tokens  = 2 per draft, mean acc len 2.12

    The acceptance RATE barely moved (0.60 -> 0.54). The accepted LENGTH fell
    24 %, which is what a shorter draft buys you: every verify step advances
    less far. Decode read 43-45 tok/s before and 25-33 after.

    That is exactly what the per-position acceptance predicted --
    (0.690, 0.448, 0.284), so position three lands 28 % of the time and cutting
    it costs. The prediction is now measured on this machine, on this workload.

    Studio's 2 is documented for MTP on GPU; it is not right HERE. A default
    from another product is still a verdict from another configuration.
    """
    out = _whatif(PROFILE, "-Nvfp4", "-Deep", "-Beta")
    assert re.search(r"--spec-draft-n-max\s+3\b", out), out


def _retired_test_beta_takes_unsloths_draft_depth():
    """2, not our 3.

    3 is llama.cpp's own default (`--spec-draft-n-max N (default: 3)`) and we
    were getting it by not setting anything. Studio sets 2 deliberately -- its
    UI documents 2 for MTP on GPU and 3 for CPU/Mac, so 2 is THEIR choice for a
    GPU run, not a standard.

    Our real-use counters argue for 3: acceptance per position
    (0.690, 0.448, 0.284), so position three still lands 28 % of the time.
    Putting 2 in the bundle is how that argument gets tested rather than
    repeated.
    """
    out = _whatif(PROFILE, "-Nvfp4", "-Deep", "-Beta")
    assert re.search(r"--spec-draft-n-max\s+2\b", out), out


def test_beta_runs_MTP_ALONE():
    """No n-gram at all, at the developer's request, 2026-08-29.

    Two independent observations point the same way and neither is ours alone:

      Studio's fastest of eight runs on this same model file was `draft-mtp`
      by itself -- 54.95 tok/s against 52.28 and 49.72 for MTP+ngram.

      On this machine, on real agent traffic, `ngram-mod` DOES NOT FIRE. Two
      sessions logged `#gen drafts = 0`, and an earlier eighteen-minute session
      logged 5 drafts in 4,653 calls. A decoder that never produces a draft
      cannot be earning the call.

    WHAT THIS IS NOT. Our own corpus measures `ngram-mod` at n-match 24 as
    +27.1 % over 12 -- on `real-code-vendor`, which is repeated vendor source
    and exactly the text an n-gram is good at. That number is real and it is
    about a different workload. **The default keeps the n-gram; -Beta drops it.**
    Which is right depends on what you are doing, and this repository has
    measured only one of the two.
    """
    out = _whatif(PROFILE, "-Nvfp4", "-Deep", "-Beta")
    assert re.search(r"--spec-type\s+draft-mtp(?!,)", out), out


def test_beta_carries_no_ngram_parameter_at_all():
    """A parameter for a decoder that is not loaded is a flag that does nothing
    and a reader who thinks it did -- the same fault as the inert `--fit on`
    this profile carried for weeks."""
    out = _whatif(PROFILE, "-Nvfp4", "-Deep", "-Beta")
    assert "--spec-ngram" not in out, out


def test_the_default_still_pairs_mtp_with_the_ngram():
    """+63.1 % was measured with the pairing, on the corpus this project uses.
    Dropping it there would throw away the only decoder result it has."""
    out = _whatif(PROFILE, "-Nvfp4", "-Deep")
    assert re.search(r"--spec-type\s+draft-mtp,ngram-mod", out), out
    assert re.search(r"--spec-ngram-mod-n-match\s+24", out), out


def _retired_test_beta_keeps_our_ngram_bounds():
    """REVERTED 2026-08-29, the same afternoon, and for a different reason than
    the draft depth.

    `n-min 48 / n-max 64` are llama.cpp's defaults, which Studio never sets;
    ours are 16 / 32. They rode into -Beta alongside `--spec-draft-n-max 2` and
    the two sessions that followed recorded, on BOTH sides:

        ngram-mod: #gen drafts = 0

    The n-gram never fired once on agent traffic, so the change could not have
    been measured -- it was inert, not better and not worse. Keeping an inert
    deviation inside a bundle makes the bundle harder to reason about for
    nothing, so it goes back.

    This is a different verdict from the draft depth: that one LOST, this one
    was NEVER EXERCISED. The register says both, because "reverted" alone would
    read as if 48/64 had been tried and beaten.

    What would exercise it is a workload where an n-gram fires at all, and this
    project does not have one. See the guide's tier-2 entry on dropping
    `ngram-mod` for the same reason.
    """
    out = _whatif(PROFILE, "-Nvfp4", "-Deep", "-Beta")
    assert re.search(r"--spec-ngram-mod-n-min\s+16", out), out
    assert re.search(r"--spec-ngram-mod-n-max\s+32", out), out
    assert re.search(r"--spec-ngram-mod-n-match\s+24", out), out


def _retired_test_beta_takes_unsloths_ngram_bounds():
    """48 and 64 -- which are llama.cpp's DEFAULTS, not Studio's tuning.

    `--spec-ngram-mod-n-min` defaults to 48 and `n-max` to 64. Studio simply
    does not set them. WE are the ones deviating, to 16 and 32, and this
    project's own register shows those two carried through from an older sweep
    where they were "held constant" rather than chosen.

    `n-match` stays 24 on both sides -- also the default, and separately
    measured here at +27.1 % over 12.
    """
    out = _whatif(PROFILE, "-Nvfp4", "-Deep", "-Beta")
    assert re.search(r"--spec-ngram-mod-n-min\s+48\b", out), out
    assert re.search(r"--spec-ngram-mod-n-max\s+64\b", out), out
    assert re.search(r"--spec-ngram-mod-n-match\s+24\b", out), out


def test_the_default_serves_n_max_64_and_leaves_the_draft_depth_at_3():
    r"""`n-max` 32 -> 64 on 2026-09-02, and the draft depth deliberately NOT moved.

    64 is llama.cpp's own default; our 32 was carried through an older sweep
    where it was "held constant" rather than chosen. Measured at the served
    147,456 on the real-code corpus, three independent boot series:
    **+15.63 %, +14.85 %, +14.52 %** (`results/ngram-window-147456.jsonl`,
    `results/ngram-nmax-ladder-147456.jsonl`, `results/won-levers-combo-147456.jsonl`).
    96 and 128 fall back to about +2 %, so 64 is a peak and not a direction.

    **The draft depth stays at 3 and that is the whole reason this test names
    both.** `--spec-draft-n-max 4` is +5.76 % on its own at this depth -- itself a
    reversal of the 16,384 screen, where 3 beat 4 -- but the two together measure
    **-4.61 %**, worse than changing nothing, at 0.1-0.5 % spreads. Naive
    multiplication would have predicted +21 %. A future reader who adopts the
    second winner because the first one worked will make the profile slower, so
    the assertion below is a guard and not bookkeeping.

    `n-min` stays 16: 48/64 measured -10.58 % in the same run, so Studio's pair
    must not be copied whole (CORRECTIONS 38).
    """
    out = _whatif(PROFILE, "-Nvfp4", "-Deep")
    assert re.search(r"--spec-draft-n-max\s+3\b", out), out
    assert re.search(r"--spec-ngram-mod-n-min\s+16\b", out), out
    assert re.search(r"--spec-ngram-mod-n-max\s+64\b", out), out
    assert not re.search(r"--spec-draft-n-max\s+4\b", out), (
        "the draft depth was raised to 4 alongside n-max 64; measured together "
        "they are -4.61 %, worse than changing neither")


# --------------------------------------------- context checkpoints, and the 51 s

def test_beta_does_not_disable_context_checkpoints():
    r"""`--ctx-checkpoints 0` is what made every turn re-prefill from zero.

    This artifact is HYBRID -- Gated DeltaNet recurrent state beside attention
    KV -- and the recurrent half cannot be rewound to a shared prefix. Without a
    checkpoint to restore from, llama.cpp gives up on the whole prompt and says
    so, once per request:

        forcing full prompt re-processing due to lack of cache data
        (likely due to SWA or hybrid/recurrent memory, see PR 13194)

    In serve-20260829-125227.log, the `-Beta` boot, that line appears on ALL
    THREE requests it served: 17,881 tokens, then 46,998, then 46,997 -- the
    last two the same conversation, re-read from the first token, 51.6 s each
    at ~911 tok/s before a character came back.

    The same binary, same artifact, same day, with checkpoints left at their
    default (serve-20260829-073741.log):

        context checkpoints enabled, max = 32, min spacing = 8192
        restored context checkpoint (pos_min = 321, n_past = 322, size = 150.890 MiB)

    forced full re-processing ONCE in the whole session, and its turns prefilled
    13, 29, 285, 829, 1,358 tokens. Checkpoints are the mechanism for exactly
    the memory this model has, and they work on it.

    `0` was copied from Unsloth Studio along with `--cache-ram 0`. Studio serves
    at half our window and its own logs show it reusing a 39,616-token prefix
    anyway, so the setting costs them less than it costs us -- and WHY it costs
    them less is not settled here. `--kv-unified` is the candidate: we set it,
    they do not.

    Cost of the default: 150.89 MiB per checkpoint, at most 32, no closer
    together than 8,192 tokens -- so about six of them at the depth we serve.
    Host RAM, not VRAM.
    """
    out = _whatif(PROFILE, "-Nvfp4", "-Deep", "-Beta")
    assert not re.search(r"--ctx-checkpoints\s+0(?!\d)", out), (
        "-Beta disables context checkpoints, which on this hybrid model means "
        "every request re-prefills from token 0 -- 51.6 s at the served depth")
