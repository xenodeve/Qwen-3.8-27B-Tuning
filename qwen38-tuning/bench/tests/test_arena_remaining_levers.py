r"""The three levers that had never been swept once, run inside the arena.

`--threads`, the `ngram-mod` window, and the KV cache type are lever ranks 5, 2
and (via issue #46) an open question in `39-OPTIMISATION-GUIDE.md`. None had a
measurement at the served depth, and the ALLREDUCE result showed why that
matters: at ctx 16,384 every arm of every sweep landed inside about a percent,
while the same variable at 147,456 was worth 24 % resolved at a 0.3 % spread.

The arena's base argv already carries `-t 18` and `-ctk q4_0 -ctv q4_0`
(`arm_parts`), and arm extras are appended after, so llama.cpp's last-wins
parsing makes an override the single moving part. That is why every control here
can be `nvfp4-final`'s winning arm unchanged rather than a near-copy.

`kv-type` pins the FA_ALL_QUANTS build. `-ctk q8_0 -ctv q4_0` does not merely run
slowly on the default binary -- it exits during load, because `fattn.cu:442`
drops every K != V pair unless the flag was compiled in. Pinning it for the
control too keeps the binary constant across the arm set.
"""
import os
import sys

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BENCH)

import dflash2_arena as arena  # noqa: E402

FAQ_EXE = r"F:\llama-build\faq\build\bin\llama-server.exe"


def _set(name):
    return dict((a[0], (a[1], a[2] if len(a) > 2 else {})) for a in arena.ARM_SETS[name])


def _control_argv():
    return dict((a[0], a[1]) for a in arena.ARM_SETS["nvfp4-final"])["nvfp4-mtp+nm24"]


# --- threads ---------------------------------------------------------------

def test_threads_control_is_the_winning_arm_untouched():
    """The base argv already says `-t 18`, which is what the worker serves, so
    the control must add nothing at all."""
    assert _set("threads")["t18"][0] == _control_argv()


def test_threads_arms_override_only_the_thread_count():
    base = _set("threads")["t18"][0]
    assert _set("threads")["t8"][0] == base + ["-t", "8"]
    assert _set("threads")["t2"][0] == base + ["-t", "2"]


# --- ngram window ----------------------------------------------------------

def test_ngram_window_control_is_the_winning_arm_untouched():
    assert _set("ngram-window")["ours-16-32"][0] == _control_argv()


def test_ngram_window_moves_n_max_and_then_n_min_on_top():
    """Two steps, not one jump: n-max alone, then Studio's pair. A single arm at
    48/64 could not say which half did anything."""
    def val(argv, flag):
        return argv[argv.index(flag) + 1]
    a = _set("ngram-window")
    assert val(a["ours-16-32"][0], "--spec-ngram-mod-n-max") == "32"
    assert val(a["nmax-64"][0], "--spec-ngram-mod-n-max") == "64"
    assert val(a["nmax-64"][0], "--spec-ngram-mod-n-min") == "16"
    assert val(a["studio-48-64"][0], "--spec-ngram-mod-n-max") == "64"
    assert val(a["studio-48-64"][0], "--spec-ngram-mod-n-min") == "48"
    for name in a:
        assert val(a[name][0], "--spec-ngram-mod-n-match") == "24"


# --- KV type ---------------------------------------------------------------

def test_every_kv_arm_pins_the_fa_all_quants_build():
    """`-ctk q8_0 -ctv q4_0` EXITS DURING LOAD on the default binary. If only the
    asymmetric arm were pinned, the set would compare two builds as well as two
    KV types."""
    for name, (_, env) in _set("kv-type").items():
        assert env.get(arena.ENV_VAR) == FAQ_EXE, name


def test_kv_control_adds_nothing_because_the_base_is_already_q4_0():
    assert _set("kv-type")["q4-q4"][0] == _control_argv()


def test_kv_arms_override_only_the_cache_types():
    base = _set("kv-type")["q4-q4"][0]
    assert _set("kv-type")["q8-q4"][0] == base + ["-ctk", "q8_0", "-ctv", "q4_0"]
    assert _set("kv-type")["q8-q8"][0] == base + ["-ctk", "q8_0", "-ctv", "q8_0"]


def test_the_asymmetric_arm_is_the_one_that_needed_the_rebuild():
    """Recorded so a reader knows why this set exists at all: the pair was
    unusable on every binary this project had until 2026-09-01."""
    assert "-ctk" in _set("kv-type")["q8-q4"][0]
    argv = _set("kv-type")["q8-q4"][0]
    i = argv.index("-ctk")
    assert (argv[i + 1], argv[i + 3]) == ("q8_0", "q4_0")
