r"""Forcing the generation budget is allowed, and a row that did it must say so.

WHY THE OPTION EXISTS (issue #44). At ctx 147,456 the arena could not produce a
measurable row on `real-code-deep`: the model stopped on EOS after 9 tokens
against a 512-token budget, and `generation_is_measurable` correctly refused all
eighteen. Sweeping the prompt length from cold showed it is **not** a length
threshold --

    43,162 -> 512   46,909 -> 1   51,038 -> 1   54,310 -> 512
    57,780 -> 512   60,831 -> 512   64,210 -> 9

-- the failures are not monotonic, so what decides it is WHERE `filler` happens
to cut the corpus, not how much it took. With `ignore_eos` the same 64,210-token
prompt runs the full 512 and the text is a real answer, opening `<think>` and
reasoning about `vram_settled`, which is the function the instruction asks about.
So the model can answer; greedy decoding just reaches EOS first.

WHY IT IS OFF BY DEFAULT AND STAMPED ON THE ROW.

Every decoder figure this project holds was taken without it. Turning it on
globally would make new rows quietly incomparable with old ones, which is the
shape of four separate fixes made on 2026-08-24 -- `exe`, `env`, `target`,
`effort` -- each added *after* a comparison had already been made without it.
The default is therefore off, and the row carries the flag either way.

WHAT IT COSTS, stated because it is not free. Past the point where the model
would have stopped, it is decoding continuation text it did not choose to write.
Draft acceptance is a property of how predictable that text is, so an
`ignore_eos` row's acceptance is not comparable with a natural row's even at the
same depth and corpus. Throughput comparisons BETWEEN ARMS within one
`ignore_eos` run are unaffected, because every arm decodes under the same rule.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dflash2_arena as arena


def test_the_default_is_off_so_nothing_changes_silently():
    assert arena.completion_payload("p")["prompt"] == "p"
    assert "ignore_eos" not in arena.completion_payload("p"), (
        "the key must be absent by default, not present and false -- a row "
        "recorded before this option existed carries neither")


def test_the_flag_puts_it_in_the_request():
    assert arena.completion_payload("p", ignore_eos=True)["ignore_eos"] is True


def test_the_budget_and_the_cache_setting_are_unchanged_by_it():
    """Only one thing may differ between a forced row and a natural one."""
    base = arena.completion_payload("p")
    forced = arena.completion_payload("p", ignore_eos=True)
    assert base["n_predict"] == forced["n_predict"] == arena.N_PREDICT
    assert base["cache_prompt"] == forced["cache_prompt"]
    assert set(forced) - set(base) == {"ignore_eos"}


def test_the_sampler_is_the_same_greedy_one():
    """temp 0 / top_k 1 / seed 42 is what every prior decoder row used."""
    p = arena.completion_payload("p")
    assert p["temperature"] == 0.0 and p["top_k"] == 1 and p["seed"] == 42


def test_the_command_line_offers_it():
    import subprocess
    out = subprocess.run([sys.executable, str(Path(arena.__file__)), "--help"],
                         capture_output=True, text=True, timeout=120)
    assert "--ignore-eos" in (out.stdout + out.stderr)


def test_a_row_records_whether_it_was_forced():
    """Without this a forced row and a natural row are indistinguishable in the
    results file, and the acceptance columns are not comparable."""
    row = arena.new_row(ctx=147456, arm="none", rnd=1, regime="real-code-deep",
                        extra=[], env={}, free_before=1234, ignore_eos=True)
    assert row["ignore_eos"] is True
    natural = arena.new_row(ctx=147456, arm="none", rnd=1,
                            regime="real-code-deep", extra=[], env={},
                            free_before=1234)
    assert natural["ignore_eos"] is False
