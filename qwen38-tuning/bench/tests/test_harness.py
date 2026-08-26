"""Tests for the measurement primitives.

Each test below exists because the untested version of that function failed
SILENTLY during this project and corrupted a published table. They are
regression tests against real incidents, not coverage for its own sake.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from harness import median


def test_median_of_odd_returns_middle_not_max():
    # The PowerShell original indexed [int](3/2) -> 2 (banker's rounding), so it
    # returned the MAX and every sweep table was mislabelled "median".
    assert median([7.71, 8.24, 9.99]) == 8.24


def test_median_of_even_averages_the_two_middles():
    assert median([1.0, 2.0, 3.0, 4.0]) == 2.5


def test_median_does_not_require_sorted_input():
    assert median([9.99, 7.71, 8.24]) == 8.24


def test_median_of_single_sample():
    assert median([5.5]) == 5.5


def test_median_of_empty_raises():
    # Silently returning 0.0 or None would put a plausible number in a report.
    try:
        median([])
    except ValueError:
        return
    raise AssertionError("median([]) must raise, not invent a value")


# ── load_jsonl ────────────────────────────────────────────────────────────────
from harness import load_jsonl


def test_load_jsonl_reads_plain_rows(tmp_path):
    f = tmp_path / "r.jsonl"
    f.write_text('{"a":1}\n{"a":2}\n', encoding="utf-8")
    assert [r["a"] for r in load_jsonl(f)] == [1, 2]


def test_load_jsonl_survives_a_bom_on_the_first_line(tmp_path):
    # PowerShell 5.1's Add-Content -Encoding utf8 writes a BOM on first write.
    # Parsing as plain utf-8 raised JSONDecodeError on line 1 and the original
    # `except: pass` dropped it -- silently deleting the BASELINE row from
    # every published sweep table.
    f = tmp_path / "r.jsonl"
    f.write_bytes(b'\xef\xbb\xbf{"a":1}\n{"a":2}\n')
    assert [r["a"] for r in load_jsonl(f)] == [1, 2]


def test_load_jsonl_ignores_blank_lines(tmp_path):
    f = tmp_path / "r.jsonl"
    f.write_text('{"a":1}\n\n  \n{"a":2}\n', encoding="utf-8")
    assert len(load_jsonl(f)) == 2


def test_load_jsonl_raises_on_a_corrupt_line(tmp_path):
    # A truncated row must not vanish: that is exactly how the BOM bug hid.
    f = tmp_path / "r.jsonl"
    f.write_text('{"a":1}\n{"a":\n', encoding="utf-8")
    try:
        load_jsonl(f)
    except ValueError as e:
        assert "2" in str(e)   # names the offending line
        return
    raise AssertionError("a corrupt line must raise, not be skipped")


def test_load_jsonl_missing_file_raises(tmp_path):
    try:
        load_jsonl(tmp_path / "nope.jsonl")
    except FileNotFoundError:
        return
    raise AssertionError("missing results file must raise")


# ── parse_layer_split ─────────────────────────────────────────────────────────
from harness import parse_layer_split

LOG = """\
0.01.283.294 D load_tensors: layer   0 assigned to device CUDA0, is_swa = 0
0.01.283.295 D load_tensors: layer   1 assigned to device CPU, is_swa = 0
0.01.283.296 D load_tensors: layer   2 assigned to device CUDA0, is_swa = 0
"""


def test_parse_layer_split_counts_cpu_despite_the_trailing_comma():
    # The device token is "CUDA0," / "CPU," -- an exact == "CPU" test matched
    # nothing, so a real 32/33 split was published as gpu=32 cpu=0.
    assert parse_layer_split(LOG, total=3) == (2, 1)


def test_parse_layer_split_uses_the_last_pass_only():
    # llama.cpp prints a reserve pass first; counting both doubles everything.
    doubled = LOG + LOG.replace("CUDA0", "CPU")
    assert parse_layer_split(doubled, total=3) == (0, 3)


def test_parse_layer_split_raises_when_counts_do_not_add_up():
    # 32+0 != 65 should have been loud the first time.
    bad = "load_tensors: layer 0 assigned to device WEIRD, is_swa = 0\n"
    try:
        parse_layer_split(bad, total=1)
    except ValueError:
        return
    raise AssertionError("an unaccounted device must raise")


def test_parse_layer_split_raises_when_no_lines_match():
    try:
        parse_layer_split("nothing here", total=65)
    except ValueError:
        return
    raise AssertionError("no assignment lines must raise, not return (0,0)")


# ── project_prefill ───────────────────────────────────────────────────────────
from harness import project_prefill_seconds


def test_project_prefill_matches_the_published_numbers():
    # 518.8 tok/s at 16384 -> ~32 s, the figure quoted in the reports.
    assert round(project_prefill_seconds(518.8, 16384)) == 32
    assert round(project_prefill_seconds(518.8, 262144)) == 505


def test_project_prefill_rejects_nonsense_rate():
    for bad in (0, -1):
        try:
            project_prefill_seconds(bad, 16384)
        except ValueError:
            continue
        raise AssertionError(f"rate {bad} must raise")


# ── deep-context corpus ───────────────────────────────────────────────────────
# The corpus is only a valid instrument if the planted facts are actually
# present, actually unique, and the tests actually encode the planted values.
# A silently-wrong corpus would produce a confident quality verdict from noise.
from deep_tasks import build_repo, PLANTED, DEEP_TASKS


def test_repo_contains_every_planted_shard_exactly_once():
    repo = build_repo(n_blocks=120)
    for f in PLANTED:
        assert repo.count(f"class Handler{f['shard']:04d}:") == 1, f["shard"]


def test_planted_constants_are_unique_in_the_repo():
    # If a planted TIMEOUT_MS also appears on a routine block, a model could be
    # right by accident and the task would not test retrieval at all.
    repo = build_repo(n_blocks=120)
    for f in PLANTED:
        assert repo.count(f"TIMEOUT_MS = {f['timeout']}") == 1, f["timeout"]
        assert repo.count(f'CHECKSUM_FIELD = "{f["field"]}"') == 1, f["field"]


def test_routine_blocks_share_one_default_so_the_contrast_task_is_answerable():
    repo = build_repo(n_blocks=120)
    assert repo.count("MAX_RETRIES = 2") == 120 - len(PLANTED)


def test_repo_fills_a_deep_cache_without_overflowing_a_64k_window():
    # Measured on this tokenizer: 415889 chars -> 112319 tokens = 3.70 chars/token.
    # The first version of this test checked only the lower bound, so a corpus
    # of 112K tokens passed and then failed every request with HTTP 400 against
    # a 65536-token window. Both bounds now.
    tokens = len(build_repo()) / 3.70
    assert tokens > 35000, f"too shallow to stress the cache: ~{tokens:.0f} tokens"
    assert tokens < 52000, f"will not fit 64K with room to answer: ~{tokens:.0f} tokens"


def test_every_task_test_matches_its_planted_values():
    by_shard = {f["shard"]: f for f in PLANTED}
    checks = {
        "deep_retries_17": str(by_shard[17]["retries"]),
        "deep_timeout_94": str(by_shard[94]["timeout"]),
        "deep_field_203": by_shard[203]["field"],
        "deep_combine_310": by_shard[310]["field"],
    }
    for task in DEEP_TASKS:
        if task["id"] in checks:
            assert checks[task["id"]] in task["test"], task["id"]


# ── deep corpus v2 ────────────────────────────────────────────────────────────
# v2 exists to break v1's 100% ceiling, so its own instrument must be sound:
# decoys must actually be present and distinguishable, the dependency chain must
# be walkable, and every asserted value must match the planted data.
from deep_tasks_v2 import build_repo as build_v2, PLANTED as PLANTED_V2, DEEP_TASKS_V2


def test_v2_each_authoritative_shard_appears_once():
    repo = build_v2()
    for f in PLANTED_V2:
        assert repo.count(f"class Handler{f['shard']:04d}:") == 1, f["shard"]


def test_v2_has_decoys_and_they_are_marked_deprecated():
    repo = build_v2()
    assert repo.count("DEPRECATED = True") >= 8, "decoys missing"
    assert repo.count("DEPRECATED = False") == len(PLANTED_V2)


def test_v2_decoy_constants_never_collide_with_authoritative_ones():
    # A decoy sharing a planted value would make a wrong retrieval score as right.
    repo = build_v2()
    for f in PLANTED_V2:
        assert repo.count(f"TIMEOUT_MS = {f['timeout']}") == 1, f["timeout"]
        assert repo.count(f'CHECKSUM_FIELD = "{f["field"]}"') == 1, f["field"]


def test_v2_dependency_chain_is_walkable():
    by = {f["shard"]: f for f in PLANTED_V2}
    for f in PLANTED_V2:
        if f["depends"] is not None:
            assert f["depends"] in by, f"{f['shard']} points at a missing shard"


def test_v2_asserted_values_match_the_planted_data():
    by = {f["shard"]: f for f in PLANTED_V2}
    t = {x["id"]: x["test"] for x in DEEP_TASKS_V2}
    assert str(by[203]["retries"]) in t["v2_authoritative_203"]
    assert str(by[417]["timeout"]) in t["v2_authoritative_417"]
    assert by[203]["field"] in t["v2_hop_417_to_203"]
    assert str(by[417]["timeout"]) in t["v2_hop_1508_to_417"]
    assert str(by[1508]["retries"]) in t["v2_chain_2941_to_1508"]
    assert str(sum(f["retries"] for f in PLANTED_V2)) in t["v2_sum_all_retries"]


def test_v2_slowest_shard_assertion_is_actually_the_slowest():
    slowest = max(PLANTED_V2, key=lambda f: f["timeout"])["shard"]
    assert f"slowest() == {slowest}" in dict(
        (x["id"], x["test"]) for x in DEEP_TASKS_V2)["v2_max_timeout_shard"]


def test_v2_repo_fits_a_64k_window_with_room_to_answer():
    tokens = len(build_v2()) / 3.70
    assert 30000 < tokens < 52000, f"~{tokens:.0f} tokens"


def test_v2_deep_variant_fits_a_128k_window_with_room_to_answer():
    # 1550 blocks is the size used for the 128K runs. Same both-bounds rule as
    # the 64K variant: too small and it never exercises the window, too large
    # and every request returns HTTP 400 instead of a result.
    tokens = len(build_v2(1550)) / 3.70
    assert 85000 < tokens < 115000, f"~{tokens:.0f} tokens"


def test_v2_deep_variant_still_plants_each_shard_once():
    # Planted shards are placed by percentage, so the count must survive scaling.
    repo = build_v2(1550)
    for f in PLANTED_V2:
        assert repo.count(f"class Handler{f['shard']:04d}:") == 1, f["shard"]


# ---------------------------------------------------------------------------
# paired_deltas — added 2026-08-19 for the cross-model arena.
#
# Comparing two MODELS cannot be interleaved inside one boot: the weights differ,
# so each arm needs its own server. The only defence left against the 13.6 %
# restart drift documented in report 04 is to alternate boots (A/B/A/B) and pair
# by round. This function is that pairing, and it is exactly the arithmetic that
# went wrong twice before -- a "median" that held the max, and per-sweep deltas
# that were summed across independent controls.
# ---------------------------------------------------------------------------
from harness import paired_deltas


def test_paired_deltas_pairs_by_round_not_by_pooled_median():
    # Round 1 is a slow window for BOTH arms, round 2 a fast one. Pooling first
    # and dividing the two pooled medians hides that; pairing exposes a clean
    # +20 % in each round. This is report 04 rule 2 in a test.
    a = [10.0, 20.0]   # control, per-round representative
    b = [12.0, 24.0]   # candidate
    r = paired_deltas(a, b)
    assert r["per_round_pct"] == [20.0, 20.0]
    assert r["mean_pct"] == 20.0


def test_paired_deltas_reports_a_range_not_only_a_point():
    r = paired_deltas([10.0, 10.0], [11.0, 13.0])
    assert r["min_pct"] == 10.0
    assert r["max_pct"] == 30.0


def test_paired_deltas_rejects_unequal_round_counts():
    # An unpaired extra round is the control-first mistake wearing a disguise.
    try:
        paired_deltas([1.0, 2.0], [1.0])
    except ValueError as e:
        assert "paired" in str(e).lower()
    else:
        raise AssertionError("unequal round counts must raise")


def test_paired_deltas_rejects_empty():
    try:
        paired_deltas([], [])
    except ValueError:
        pass
    else:
        raise AssertionError("empty input must raise")


def test_paired_deltas_rejects_nonpositive_baseline():
    # A 0.0 tok/s sample already poisoned one median in this project; here it
    # would be a divide-by-zero producing inf rather than a failure.
    try:
        paired_deltas([0.0], [5.0])
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive baseline must raise")


def test_paired_deltas_marks_an_effect_below_the_noise_floor_unresolved():
    # 13.6 % is the measured peak-to-peak restart spread (report 04 s0).
    r = paired_deltas([10.0, 10.0], [11.0, 11.0])   # +10 %, under the floor
    assert r["resolved"] is False


def test_paired_deltas_requires_consistent_sign_to_resolve():
    # +40 % in one round and -10 % in the other averages to a big number that
    # means nothing. Mean size alone must not be enough to call it resolved.
    r = paired_deltas([10.0, 10.0], [14.0, 9.0])
    assert r["resolved"] is False


def test_paired_deltas_resolves_a_large_consistent_effect():
    r = paired_deltas([10.0, 12.0], [20.0, 23.0])
    assert r["resolved"] is True


# ---------------------------------------------------------------------------
# check_tool_call — added 2026-08-19.
#
# The new research names malformed tool calls and omitted required fields as the
# expected failure mode of a lower-fidelity quantization, and requires 100 %
# schema compliance before a candidate may become the default worker. The
# existing corpus only ever extracted a fenced code block, so a model that
# degraded into prose-instead-of-tool-call would have scored the same as one
# that emitted a perfect call. This closes that hole.
# ---------------------------------------------------------------------------
from harness import check_tool_call


def _spec():
    return {"name": "apply_patch", "required": ["path", "edits"]}


def test_tool_call_accepts_a_well_formed_call():
    calls = [{"function": {"name": "apply_patch",
                           "arguments": '{"path": "a.py", "edits": [{"line": 3}]}'}}]
    r = check_tool_call(calls, _spec())
    assert r["ok"] is True
    assert r["errors"] == []


def test_tool_call_flags_prose_instead_of_a_call():
    # The degradation that matters most: the model explains what it would do.
    r = check_tool_call([], _spec())
    assert r["ok"] is False
    assert any("no tool call" in e for e in r["errors"])


def test_tool_call_flags_unparseable_arguments():
    calls = [{"function": {"name": "apply_patch", "arguments": "{path: a.py,}"}}]
    r = check_tool_call(calls, _spec())
    assert r["ok"] is False
    assert any("not valid JSON" in e for e in r["errors"])


def test_tool_call_flags_a_missing_required_field():
    calls = [{"function": {"name": "apply_patch", "arguments": '{"path": "a.py"}'}}]
    r = check_tool_call(calls, _spec())
    assert r["ok"] is False
    assert any("edits" in e for e in r["errors"])


def test_tool_call_flags_the_wrong_function_name():
    calls = [{"function": {"name": "apply_pdatch",
                           "arguments": '{"path": "a.py", "edits": []}'}}]
    r = check_tool_call(calls, _spec())
    assert r["ok"] is False
    assert any("apply_pdatch" in e for e in r["errors"])


def test_tool_call_accepts_arguments_already_decoded_to_a_dict():
    # Some builds hand back a dict rather than a JSON string; that is not a
    # protocol failure and must not be scored as one.
    calls = [{"function": {"name": "apply_patch",
                           "arguments": {"path": "a.py", "edits": []}}}]
    assert check_tool_call(calls, _spec())["ok"] is True


def test_tool_call_reports_every_fault_not_only_the_first():
    # A pass/fail bit cannot tell "one dropped field" from "total collapse",
    # and the research asks for the omission RATE, not a boolean.
    calls = [{"function": {"name": "nope", "arguments": '{}'}}]
    r = check_tool_call(calls, _spec())
    assert len(r["errors"]) >= 3   # wrong name + both missing fields


# ---------------------------------------------------------------------------
# retry_economics — added 2026-08-19.
#
# The research builds its whole "22.6 -> 40.2 merged tasks/h" table on ASSUMED
# values: p2 = min(p1 + 0.10, 0.95) and H = 60 s, both stated as "economic
# assumptions, not measured probabilities". They are measurable on this machine,
# and the difference decides whether a weaker-but-faster worker actually wins.
# ---------------------------------------------------------------------------
from harness import retry_economics


def _rec(attempts, accepted, wall):
    return {"attempts": attempts, "accepted": accepted, "wall_s": wall}


def test_economics_splits_first_pass_from_retry_pass():
    # 2 pass first try, 1 passes on retry, 1 never passes.
    recs = [_rec(1, True, 10), _rec(1, True, 10), _rec(2, True, 25), _rec(2, False, 30)]
    e = retry_economics(recs, escalation_s=90, overhead_s=60)
    assert e["p1"] == 50.0                  # 2 of 4 first-try
    assert e["p2"] == 50.0                  # 1 of the 2 that needed a retry


def test_economics_returns_none_for_p2_when_nothing_needed_a_retry():
    # Reporting 0 % here would read as "retries never work" rather than
    # "no retry was ever attempted". A rate over an empty denominator is a lie.
    e = retry_economics([_rec(1, True, 10)], escalation_s=90, overhead_s=60)
    assert e["p2"] is None


def test_economics_counts_attempts_per_accepted_task_not_per_task():
    recs = [_rec(1, True, 10), _rec(2, True, 20)]
    e = retry_economics(recs, escalation_s=90, overhead_s=60)
    assert e["attempts_per_accepted"] == 1.5


def test_economics_reports_escalations_per_100_tasks():
    recs = [_rec(1, True, 10)] * 3 + [_rec(2, False, 30)]
    e = retry_economics(recs, escalation_s=90, overhead_s=60)
    assert e["escalations_per_100"] == 25.0


def test_economics_charges_escalation_and_overhead_to_wall_time():
    # One task, failed twice, 30 s of worker time, 90 s Q4 escalation, 60 s of
    # fixed review/CI overhead -> 180 s per merged change -> 20 per hour.
    e = retry_economics([_rec(2, False, 30)], escalation_s=90, overhead_s=60)
    assert e["merged_tasks_per_hour"] == 20.0


def test_economics_rejects_an_accepted_task_with_no_attempt():
    try:
        retry_economics([_rec(0, True, 5)], escalation_s=90, overhead_s=60)
    except ValueError:
        pass
    else:
        raise AssertionError("an accepted task with zero attempts must raise")


def test_economics_rejects_empty_input():
    try:
        retry_economics([], escalation_s=90, overhead_s=60)
    except ValueError:
        pass
    else:
        raise AssertionError("empty record set must raise")


# ---------------------------------------------------------------------------
# marginal_rate — added 2026-08-19.
#
# The cost of a broken prefix was first reported here by dividing ONE perturbed
# turn's wall time by its prompt tokens. That is wrong: wall time is prefill
# PLUS the decode of n_predict tokens, so the ratio understates the prefill rate
# and any projection built on it is off by the whole decode.
#
# The slope across several perturbations is the right estimator, because the
# decode component is constant at every point and cancels out of the slope.
# ---------------------------------------------------------------------------
from harness import marginal_rate


def test_marginal_rate_recovers_the_slope_and_ignores_a_constant_offset():
    # 100 tok/s prefill plus a fixed 5 s of decode at every point.
    xs = [1000, 2000, 3000]
    ys = [15.0, 25.0, 35.0]
    r = marginal_rate(xs, ys)
    assert r["rate"] == 100.0
    assert r["offset_s"] == 5.0


def test_marginal_rate_differs_from_the_naive_single_point_ratio():
    # The naive ratio on the first point gives 1000/15 = 66.7 tok/s, which is
    # the number that was published before this function existed.
    r = marginal_rate([1000, 2000, 3000], [15.0, 25.0, 35.0])
    assert round(1000 / 15.0, 1) == 66.7
    assert r["rate"] == 100.0


def test_marginal_rate_projects_to_a_target_context():
    r = marginal_rate([1000, 2000, 3000], [15.0, 25.0, 35.0], project_to=16384)
    # 16384/100 + 5 s of constant overhead
    assert r["projected_s"] == round(16384 / 100.0 + 5.0, 1)


def test_marginal_rate_needs_at_least_three_points():
    # Two points always fit a line perfectly and report no residual, which reads
    # as certainty. Three is the minimum that can disagree with itself.
    try:
        marginal_rate([1000, 2000], [15.0, 25.0])
    except ValueError:
        pass
    else:
        raise AssertionError("fewer than three points must raise")


def test_marginal_rate_rejects_a_flat_or_decreasing_x():
    try:
        marginal_rate([1000, 1000, 1000], [15.0, 25.0, 35.0])
    except ValueError:
        pass
    else:
        raise AssertionError("zero spread in x must raise")


def test_marginal_rate_reports_fit_quality():
    # A perfect line and a scattered one must not look the same in the output.
    tight = marginal_rate([1000, 2000, 3000], [15.0, 25.0, 35.0])
    loose = marginal_rate([1000, 2000, 3000], [15.0, 40.0, 35.0])
    assert tight["r2"] == 1.0
    assert loose["r2"] < 0.9


def test_economics_refuses_a_run_where_no_attempt_actually_reached_the_model():
    # 2026-08-19: a server swap left llama-server still loading, every request
    # came back HTTP 503, and the summary reported 24.0 merged tasks/hour --
    # a perfectly plausible number produced by 30 tasks that never ran, because
    # each one "escalated" and escalation is charged as a constant. A run with
    # no worker time is not a slow run; it is not a run.
    recs = [{"attempts": 1, "accepted": False, "wall_s": 0.0} for _ in range(30)]
    try:
        retry_economics(recs, escalation_s=90, overhead_s=60)
    except ValueError as e:
        assert "no worker time" in str(e).lower()
    else:
        raise AssertionError("a run with zero worker time must raise")


def test_economics_still_accepts_a_genuine_zero_wall_single_task():
    # Guard the guard: one instant task among real ones must not trip it.
    recs = [{"attempts": 1, "accepted": True, "wall_s": 0.0},
            {"attempts": 1, "accepted": True, "wall_s": 12.0}]
    e = retry_economics(recs, escalation_s=90, overhead_s=60)
    assert e["tasks"] == 2


# ---------------------------------------------------------------------------
# parse_layer_split: derive the layer count instead of assuming 65.
#
# 2026-08-19: the MoE arms reported "65 + 0" for a model whose log says
# `qwen35moe.block_count u32 = 40`. The function took the LAST 65 assignment
# lines of 451 (llama.cpp emits several reserve passes) and counted those. The
# arms happened to be fully resident, so the conclusion survived -- but it
# survived by luck: the same code would have printed 65+0 with layers on the
# CPU, because it never knew how many layers the model had.
# ---------------------------------------------------------------------------

_MOE_LOG = """
llama_model_loader: - kv  17:                      qwen35moe.block_count u32              = 40
load_tensors: layer   0 assigned to device CUDA0, is_swa = 0
load_tensors: layer   1 assigned to device CUDA0, is_swa = 0
load_tensors: layer   2 assigned to device CPU, is_swa = 0
"""


def test_layer_split_uses_block_count_from_the_log():
    # The discriminating case: llama.cpp emits several reserve passes, so the
    # log holds far more assignment lines than the model has layers. Here an
    # earlier pass placed everything on the CPU and the FINAL pass placed all 41
    # on the GPU. Reading the last 65 lines mixes the two passes and reports
    # 41+24; reading block_count+1 = 41 reports the truth, 41+0.
    lines = ["llama_model_loader: - kv  17:   qwen35moe.block_count u32   = 40"]
    lines += ["load_tensors: layer %d assigned to device CPU," % i for i in range(41)]
    lines += ["load_tensors: layer %d assigned to device CUDA0," % i for i in range(41)]
    text = chr(10).join(lines)

    assert parse_layer_split(text, total=65) == (41, 24)   # the old behaviour
    assert parse_layer_split(text) == (41, 0)              # block_count-derived


def test_layer_split_still_raises_without_assignment_lines():
    try:
        parse_layer_split("qwen35moe.block_count u32 = 40\n")
    except ValueError as e:
        assert "no layer-assignment" in str(e)
    else:
        raise AssertionError("a log with no assignment lines must raise")


def test_layer_split_falls_back_to_the_explicit_total():
    # Older logs in results/ have no block_count line; the caller's total must
    # still work so historical data stays parseable.
    text = "\n".join("load_tensors: layer %d assigned to device CUDA0," % i
                     for i in range(70))
    gpu, cpu = parse_layer_split(text, total=65)
    assert (gpu, cpu) == (65, 0)


def test_layer_split_reads_the_final_pass_not_the_whole_log():
    # Three reserve passes; only the last describes what was actually loaded.
    a = ["load_tensors: layer %d assigned to device CPU," % i for i in range(5)]
    b = ["load_tensors: layer %d assigned to device CPU," % i for i in range(5)]
    c = ["load_tensors: layer %d assigned to device CUDA0," % i for i in range(5)]
    assert parse_layer_split(chr(10).join(a + b + c)) == (5, 0)


def test_layer_split_default_no_longer_assumes_65_layers():
    # Regression guard for the MoE case: a 41-layer model must not be described
    # with a number borrowed from a 66-layer one.
    lines = ["load_tensors: layer %d assigned to device CUDA0," % i
             for i in range(41)]
    assert parse_layer_split(chr(10).join(lines)) == (41, 0)


# ---------------------------------------------------------------------------
# cached() — artifact identity when a repo is re-published in place.
#
# 2026-08-19T16:39:23Z, mid-session: Unsloth replaced every file in
# Qwen3.8-27B-GGUF with Dynamic V3 builds. Same filenames, new contents, new
# sizes. The cache then held two snapshot directories containing
# `Qwen3.8-27B-UD-IQ2_XXS.gguf`, and a resolver returning hits[0] would have let
# the pre-V3 arm and the V3 arm point at the same file -- a paired, three-round,
# order-counterbalanced comparison of an artifact against itself.
# ---------------------------------------------------------------------------
import os as _os
sys.path.insert(0, str(Path(__file__).parent.parent))
from model_arena import cached as _cached


def _fake_cache(tmp_path, monkeypatch, sizes):
    root = (tmp_path / ".cache" / "huggingface" / "hub"
            / "models--acme--repo" / "snapshots")
    for commit, size in sizes.items():
        d = root / commit
        d.mkdir(parents=True)
        (d / "model.gguf").write_bytes(b"x" * size)
    monkeypatch.setattr(_os.path, "expanduser",
                        lambda p: str(tmp_path) + p.replace("~", ""))
    return root


def test_cached_returns_none_when_the_artifact_is_absent(tmp_path, monkeypatch):
    _fake_cache(tmp_path, monkeypatch, {})
    assert _cached("acme/repo", "model.gguf") is None


def test_cached_resolves_a_single_snapshot(tmp_path, monkeypatch):
    _fake_cache(tmp_path, monkeypatch, {"aaa": 10})
    assert _cached("acme/repo", "model.gguf").endswith("model.gguf")


def test_cached_raises_when_two_snapshots_hold_the_same_filename(tmp_path, monkeypatch):
    # The exact situation the V3 re-publish created.
    _fake_cache(tmp_path, monkeypatch, {"old": 10, "new": 7})
    try:
        _cached("acme/repo", "model.gguf")
    except ValueError as e:
        assert "disambiguate" in str(e)
    else:
        raise AssertionError("two candidates must raise, not pick one")


def test_cached_picks_the_generation_named_by_its_byte_count(tmp_path, monkeypatch):
    _fake_cache(tmp_path, monkeypatch, {"old": 10, "new": 7})
    old = _cached("acme/repo", "model.gguf", 10)
    new = _cached("acme/repo", "model.gguf", 7)
    assert _os.path.getsize(old) == 10
    assert _os.path.getsize(new) == 7
    assert old != new


# ---------------------------------------------------------------------------
# retry_economics: censored attempts, and capability separated from throughput.
#
# Both points came from an independent review panel (2026-08-20):
#
#   * "Treat every length-truncated result as censored, not failed. No arm may
#      be rejected while any evaluated response ended at the token limit."
#     Even at max_tokens 8192 the arms still truncate 1-7 times out of 60, and
#     scoring those as capability failures penalises exactly the artifacts that
#     reason longest -- the same bias a 3072 budget produced, one notch quieter.
#
#   * "A generous budget injects a verbosity tax into tasks-per-hour. Report
#      pass rate and throughput as separate numbers."
#     Four arms tie at 27/30 accepted and differ only in wall clock (2,004s to
#     4,572s). Ranking them by merged_tasks_per_hour alone sums capability and
#     verbosity into one figure and calls it a ranking.
# ---------------------------------------------------------------------------

def test_economics_counts_censored_attempts_separately(_=None):
    recs = [
        {"attempts": 1, "accepted": True,  "wall_s": 10},
        {"attempts": 2, "accepted": False, "wall_s": 40, "censored": True},
        {"attempts": 1, "accepted": False, "wall_s": 30},
    ]
    e = retry_economics(recs, escalation_s=90, overhead_s=60)
    assert e["censored"] == 1
    # p1 is over the tasks that were actually decided, not all three.
    assert e["decided"] == 2


def test_economics_refuses_a_verdict_when_censoring_could_flip_it(_=None):
    # 1 accepted, 1 rejected, 1 censored: the censored task alone decides
    # whether the arm scored 1/3 or 2/3. That is not a result.
    recs = [
        {"attempts": 1, "accepted": True,  "wall_s": 10},
        {"attempts": 2, "accepted": False, "wall_s": 40},
        {"attempts": 2, "accepted": False, "wall_s": 40, "censored": True},
    ]
    e = retry_economics(recs, escalation_s=90, overhead_s=60)
    assert e["censoring_could_change_verdict"] is True


def test_economics_reports_capability_and_throughput_apart(_=None):
    recs = [{"attempts": 1, "accepted": True, "wall_s": 10} for _ in range(3)]
    e = retry_economics(recs, escalation_s=90, overhead_s=60)
    assert e["accepted_of_decided"] == "3/3"
    assert "wall_per_accepted_s" in e


def test_economics_wall_per_accepted_is_the_verbosity_axis(_=None):
    # Same capability, one arm takes twice as long. This is the number that
    # separates the four arms that all scored 27/30.
    fast = [{"attempts": 1, "accepted": True, "wall_s": 10} for _ in range(3)]
    slow = [{"attempts": 1, "accepted": True, "wall_s": 20} for _ in range(3)]
    a = retry_economics(fast, escalation_s=90, overhead_s=60)
    b = retry_economics(slow, escalation_s=90, overhead_s=60)
    assert a["accepted_of_decided"] == b["accepted_of_decided"]
    assert b["wall_per_accepted_s"] == 2 * a["wall_per_accepted_s"]


# ---------------------------------------------------------------------------
# check_output_contract — added 2026-08-20 on a review panel's finding.
#
# "Success is subprocess exit 0 on a unit test. Any recoverable implementation
#  scores the same whether the model obeyed the format constraints or dumped
#  chain-of-thought, wrong filenames, or extra prose."
#
# The corpus prompt already states three hard constraints -- one fenced python
# block, only the requested code, no explanation or usage examples -- and the
# extractor then quietly repairs every violation by taking the largest fenced
# block, or the whole reply when there is no fence at all. Constraint adherence
# is the capability aggressive quantization is said to lose BEFORE closed
# algorithmic coding degrades, and this corpus was structurally blind to it.
#
# Scored as a SEPARATE rate, never folded into pass/fail: changing what counts
# as a passing task mid-project would make every earlier number incomparable.
# ---------------------------------------------------------------------------
from harness import check_output_contract


def test_contract_accepts_a_single_clean_fenced_block():
    text = "```python\ndef f(x):\n    return x + 1\n```"
    r = check_output_contract(text)
    assert r["ok"] is True
    assert r["violations"] == []


def test_contract_flags_prose_outside_the_fence():
    # The extractor silently discards this; the model still disobeyed.
    text = "Here is the implementation:\n\n```python\ndef f(x):\n    return x\n```"
    r = check_output_contract(text)
    assert r["ok"] is False
    assert any("prose outside" in v for v in r["violations"])


def test_contract_flags_more_than_one_block():
    text = "```python\ndef f():\n    pass\n```\n```python\nf()\n```"
    r = check_output_contract(text)
    assert r["ok"] is False
    assert any("2 fenced blocks" in v for v in r["violations"])


def test_contract_flags_a_usage_example_inside_the_block():
    text = ('```python\ndef f(x):\n    return x\n\n'
            'if __name__ == "__main__":\n    print(f(1))\n```')
    r = check_output_contract(text)
    assert r["ok"] is False
    assert any("usage example" in v for v in r["violations"])


def test_contract_flags_no_fence_at_all():
    r = check_output_contract("def f(x):\n    return x\n")
    assert r["ok"] is False
    assert any("no fenced" in v for v in r["violations"])


def test_contract_tolerates_surrounding_whitespace():
    text = "\n\n```python\ndef f():\n    pass\n```\n\n"
    assert check_output_contract(text)["ok"] is True


def test_economics_refuses_a_run_that_mostly_failed_at_the_request_level(_=None):
    # 2026-08-20: an armed queue started mid-run and killed the server the
    # corpus was using. The first four tasks completed, the remaining 26
    # returned HTTP 503 in 0.0 s, and the summary came out looking ordinary --
    # "3/29 accepted, 22.0 merged tasks/hour" -- because the existing guard only
    # fires when NO task recorded any worker time. A run three quarters dead
    # must not summarise.
    recs = ([{"attempts": 1, "accepted": True, "wall_s": 40.0} for _ in range(4)]
            + [{"attempts": 1, "accepted": False, "wall_s": 0.0,
                "request_failed": True} for _ in range(26)])
    try:
        retry_economics(recs, escalation_s=90, overhead_s=60)
    except ValueError as e:
        assert "before reaching the model" in str(e)
    else:
        raise AssertionError("a mostly-dead run must raise, not summarise")


def test_economics_tolerates_a_couple_of_request_failures(_=None):
    recs = ([{"attempts": 1, "accepted": True, "wall_s": 40.0} for _ in range(28)]
            + [{"attempts": 1, "accepted": False, "wall_s": 0.0,
                "request_failed": True} for _ in range(2)])
    e = retry_economics(recs, escalation_s=90, overhead_s=60)
    assert e["request_failures"] == 2


# ---------------------------------------------------------------------------
# completion_timeout_s / vram_settled
#
# Incident, 2026-08-21 01:34-02:35. `F-ot-iq1m-128k` arm `ot-ffn-1` pushed
# 644 MiB of FFN weights to CPU with -ot. Prefill collapsed from 240.6 to
# 8.56 tok/s -- 93,086 tokens would have needed ~10,900 s. post() carried a
# FLAT timeout=3600, so the harness sat on a dead-obvious failure for a full
# hour (01:34:36 -> 02:34:36, to the second) and then raised TimeoutError.
#
# The raise skipped `p.kill()`, which lives at the end of the happy path with
# no try/finally, so the 12 GB server stayed resident. 30 s later the next
# queue step called kill() -- which sleeps a FLAT 5 s -- started its own
# server into VRAM the driver had not yet released, passed /health, and died
# on the first /completion with ConnectionResetError. One slow arm took out
# two steps.
#
# Both flat constants are the fault. A timeout must be sized by how much
# prefill the depth actually implies, and a teardown must wait for the
# hardware rather than for a guess.
# ---------------------------------------------------------------------------

from harness import completion_timeout_s, vram_settled


def test_completion_timeout_scales_with_depth():
    # A flat hour is simultaneously too generous at 16K and meaningless at 256K.
    assert (completion_timeout_s(16384)
            < completion_timeout_s(131072)
            < completion_timeout_s(262144))


def test_completion_timeout_covers_the_slowest_legitimate_prefill():
    # Slowest cold prefill ever measured at 131,072 on a resident arm: 386.9 s
    # (AD-IQ1_M at 65+1). The timeout must clear that with room, or the harness
    # starts truncating real measurements -- the opposite failure.
    assert completion_timeout_s(131072) > 386.9 * 2


def test_completion_timeout_cuts_a_pathological_arm_well_inside_an_hour():
    # ot-ffn-1 at 8.56 tok/s needed ~10,900 s at this depth. Waiting 3,600 s to
    # learn that was the incident. Whatever the budget is, it must be shorter
    # than the flat hour it replaces.
    assert completion_timeout_s(131072) < 3600


def test_completion_timeout_rejects_a_nonpositive_context():
    for bad in (0, -1):
        try:
            completion_timeout_s(bad)
        except ValueError:
            continue
        raise AssertionError(f"ctx={bad} must raise, not return a budget")


def test_vram_settled_is_false_while_the_driver_is_still_releasing():
    # WDDM frees a 12 GB allocation in stages. These readings are still climbing.
    assert vram_settled([167, 2048, 6400]) is False


def test_vram_settled_is_true_once_two_readings_agree():
    assert vram_settled([167, 2048, 10290, 10293]) is True


def test_vram_settled_needs_more_than_one_reading():
    # The incident in miniature: one reading, taken 5 s after the kill, and the
    # caller concluded the GPU was free.
    assert vram_settled([10293]) is False
    assert vram_settled([]) is False


def test_vram_settled_tolerates_small_jitter_but_not_a_release_step():
    # A desktop compositor moves tens of MiB between polls; a model unloading
    # moves thousands. The threshold has to separate those two.
    assert vram_settled([10250, 10293]) is True          # 43 MiB of jitter
    assert vram_settled([8000, 10293]) is False          # 2.2 GiB still landing


def test_vram_settled_cannot_tell_finished_from_not_started():
    # The bug in the first version of the fix, caught before it shipped. kill()
    # polls every 3 s. If the driver has not BEGUN releasing by the second poll,
    # two readings of "still full" agree perfectly and the caller concludes the
    # GPU is free -- which is the 5 s sleep with extra steps.
    #
    # `floor_mib` is what distinguishes them: the caller knows how much was free
    # BEFORE the kill, and a real teardown must beat it. The smallest artifact in
    # this project is 7.80 GiB, so requiring even 1,024 MiB of rise is generous.
    still_full = [167, 167]
    assert vram_settled(still_full) is True                    # stopped moving
    assert vram_settled(still_full, floor_mib=1191) is False   # but never rose


def test_vram_settled_accepts_a_reading_that_cleared_the_floor():
    assert vram_settled([167, 2048, 10290, 10293], floor_mib=1191) is True


def test_vram_settled_floor_applies_to_the_latest_reading_not_the_best():
    # A transient spike mid-release must not count as arrival.
    assert vram_settled([12000, 400, 402], floor_mib=1191) is False


# ---------------------------------------------------------------------------
# compose_developer
#
# Raised 2026-08-21 by the developer: the real worker has karpathy-guidelines
# and tdd injected into its prompt, and run_retry_bench sends a 35-token
# developer message instead. So every quality number this project has is for a
# configuration nobody ships.
#
# Swapping the prompt is not enough on its own. The contract sentence -- "one
# fenced ```python block ... no explanation, no usage examples, no tests" -- is
# what `check_output_contract` grades against. Drop it and the arm changes TWO
# things: skills added AND the format instruction removed, which is the same
# fault that made serve-v3-iq2xxs-fmt.ps1 unreadable.
#
# It also has to come LAST. tdd says "write the failing test first" and
# karpathy says "if uncertain, ask" -- both directly contradict the contract,
# and recency is the only lever available for which one wins.
# ---------------------------------------------------------------------------

from harness import compose_developer, CONTRACT


def test_compose_developer_with_no_skills_is_the_contract_alone():
    assert compose_developer([]) == CONTRACT


def test_compose_developer_keeps_the_contract_when_skills_are_injected():
    out = compose_developer(["Write the failing test first."])
    assert CONTRACT in out


def test_compose_developer_puts_the_contract_last():
    # tdd tells the model to write tests; the grader forbids them. The contract
    # goes last so it is the most recent instruction in the window.
    out = compose_developer(["Write the failing test first."])
    assert out.rstrip().endswith(CONTRACT)


def test_compose_developer_keeps_skill_text_verbatim():
    # A paraphrased skill measures the paraphrase. The whole point is to send
    # what production sends.
    skill = "## 2. Simplicity First\n\n- No features beyond what was asked."
    assert skill in compose_developer([skill])


def test_compose_developer_separates_skills_so_they_do_not_run_together():
    a, b = "First skill body.", "Second skill body."
    out = compose_developer([a, b])
    assert out.index(a) < out.index(b)
    assert "First skill body.Second" not in out


def test_compose_developer_rejects_a_skill_that_is_not_text():
    for bad in (None, 42, ["nested"]):
        try:
            compose_developer([bad])
        except (TypeError, ValueError):
            continue
        raise AssertionError(f"{bad!r} must raise, not be interpolated")


# ---------------------------------------------------------------------------
# filler_repetition_pct
#
# Instrument fault 8, found 2026-08-21 05:35 while checking why 147,456 decoded
# FASTER than 131,072 on the same artifact with the same flags.
#
# `depth_sweep.filler()` repeats one class definition with only a 4-digit index
# changing. At 147,456 that is 962 blocks, and adjacent blocks are 99.5 %
# identical character-for-character. An n-gram decoder drafts from what is
# already in the context, so this is the most favourable text that could
# possibly be constructed for it -- and every n-gram result in this project
# (+135.89 % at 16K, +200.22 % at 131,072, +330.40 % at 147,456) was measured
# on it. Acceptance sits at 99-100 % at every depth, which is the tell.
#
# The numbers are an upper bound on a synthetic best case, not an estimate of
# what real code will do. This function makes the property measurable so a
# filler can be checked rather than assumed.
# ---------------------------------------------------------------------------

from harness import filler_repetition_pct


def test_repetition_of_a_repeating_pattern_is_high():
    # "a b a b a b": the FIRST a and b are new, the remaining four are repeats,
    # so the definition ("duplicates of an EARLIER line") caps below 100 %.
    # The first version of this test asserted 100.0 and was wrong about the
    # metric it was testing — corrected here rather than bending the metric.
    assert filler_repetition_pct("a" + chr(10) + "b" + chr(10)) == 0.0
    assert filler_repetition_pct(("a" + chr(10) + "b" + chr(10)) * 3) == 66.67


def test_repetition_of_all_distinct_lines_is_zero():
    assert filler_repetition_pct("alpha\nbravo\ncharlie\ndelta\n") == 0.0


def test_repetition_catches_the_current_synthetic_filler_shape():
    # Two "blocks" differing only in an index, the shape depth_sweep.filler
    # produces. Everything but the header line repeats.
    block = "class Handler{i:04d}:\n    def process(self, item):\n        return item\n"
    text = block.format(i=1) + block.format(i=2) + block.format(i=3)
    # Nine lines: three distinct headers, two body lines repeated twice each.
    # 4/9 = 44.44 %. The real filler at 147,456 scores 84.53 % because its
    # blocks are longer, so the one line that varies matters less.
    assert filler_repetition_pct(text) == 44.44


def test_repetition_ignores_blank_lines():
    # Blank lines repeat trivially in any text and would inflate every score.
    assert filler_repetition_pct("alpha\n\n\nbravo\n\n\ncharlie\n") == 0.0


def test_repetition_of_empty_text_raises():
    # Returning 0.0 would read as "not repetitive", which is a claim about
    # nothing.
    try:
        filler_repetition_pct("")
    except ValueError:
        return
    raise AssertionError("empty text must raise, not report 0 % repetition")


# ---------------------------------------------------------------------------
# draft_acceptance
#
# Instrument fault 9, found 2026-08-21 06:12 while reading why an arm with
# "acceptance: null" decoded FASTER than one at 100 %.
#
# `depth_sweep.run()` takes five timed generations and reports `tg_med` as the
# median of all five -- but computes `acceptance` from `t`, the timings of the
# FIRST one only. So the two columns describe different requests, and a row
# reading "acceptance 4 %, tg_med 32.4" may be a cold request that drafted badly
# next to four warm ones that did not.
#
# Several claims written earlier the same night rest on that column:
#   * "-ot ssm collapses acceptance from 100 % to 4 %"
#   * "the four-block slice drafts nothing at all"
#   * "acceptance may be a cheap coherence detector"
# All were read off one generation in five.
# ---------------------------------------------------------------------------

from harness import draft_acceptance


def test_acceptance_aggregates_every_generation_not_just_the_first():
    # The first request drafts badly, the next four do not. Reading only the
    # first reports 10 %; the truth over the whole sample is 82 %.
    timings = [{"draft_n": 100, "draft_n_accepted": 10}] + [
        {"draft_n": 100, "draft_n_accepted": 100} for _ in range(4)]
    assert draft_acceptance(timings) == 82.0


def test_acceptance_is_none_when_nothing_drafted_anywhere():
    assert draft_acceptance([{"draft_n": 0}, {}, {"draft_n_accepted": 0}]) is None


def test_acceptance_counts_a_generation_that_drafted_nothing_as_zero_weight():
    # A request with no drafts must not drag the rate down; it has no opinion.
    assert draft_acceptance([{"draft_n": 0}, {"draft_n": 10, "draft_n_accepted": 9}]) == 90.0


def test_acceptance_of_empty_input_is_none():
    assert draft_acceptance([]) is None


def test_acceptance_rejects_more_accepted_than_drafted():
    # Would silently report over 100 %. That is a broken server or a broken
    # reader, and either way it is not a measurement.
    try:
        draft_acceptance([{"draft_n": 5, "draft_n_accepted": 9}])
    except ValueError:
        return
    raise AssertionError("accepted > drafted must raise, not report 180 %")


from harness import line_repetition_pct


def test_line_repetition_is_the_name_the_reasoning_check_uses():
    # `filler_repetition_pct` was written to check a benchmark prompt. It got
    # used on a reasoning trace on 2026-08-21 to settle whether the model was
    # looping, which is the more valuable question -- so the general name is
    # canonical and the original stays as an alias rather than a second copy.
    assert line_repetition_pct is filler_repetition_pct


def test_reasoning_that_never_repeats_scores_zero():
    # The measurement that retracted the "it loops" claim: 6,899 characters of
    # reasoning on the `damerau` task, not one line recurring.
    trace = ("The user wants a function damerau_levenshtein(a, b).\n"
             "Normally a transposition costs 2, but here it says 1.\n"
             "Let me check with a worked example.\n"
             "So this is the optimal string alignment variant.\n")
    assert line_repetition_pct(trace) == 0.0


def test_a_trace_stuck_on_one_thought_scores_high():
    # What looping would actually look like, for contrast.
    stuck = ("Wait, let me reconsider.\nActually that is wrong.\n") * 6
    assert line_repetition_pct(stuck) > 80.0


# --------------------------------------------------------------- -sm tensor

TENSOR_SPLIT_LOG = """
0.00.367.151 I llama_prepare_model_devices: creating a Meta device for tensor parallelism from 2 devices:
0.00.367.578 I llama_prepare_model_devices: using device Meta() (Meta()) (unknown id) - 26241 MiB free
0.00.567.440 D load_tensors: layer   0 assigned to device Meta(), is_swa = 0
0.00.567.441 D load_tensors: layer   1 assigned to device Meta(), is_swa = 0
0.00.567.442 D load_tensors: layer   2 assigned to device Meta(), is_swa = 0
"""

TENSOR_SPLIT_SPILLED = TENSOR_SPLIT_LOG + (
    "0.00.567.443 D load_tensors: layer   3 assigned to device CPU, is_swa = 0\n")


def test_a_meta_device_counts_as_resident():
    """`-sm tensor` aggregates the cards into one virtual device.

    llama.cpp logs `creating a Meta device for tensor parallelism from 2
    devices` and then assigns every layer to `Meta()`. On 2026-08-26 the parser
    refused the row -- "layer split 0+0 does not account for 66 lines;
    unexpected devices: ['Meta']" -- which was the RIGHT failure: it did not
    know what Meta meant and said so instead of guessing.

    A layer on the Meta device is on the GPUs. It is not on the CPU, and
    counting it as CPU would report a fully resident model as spilled, which is
    the same wrong verdict in the other direction.
    """
    assert parse_layer_split(TENSOR_SPLIT_LOG) == (3, 0)


def test_a_meta_device_still_reports_a_spill():
    """The whole point of reading this at all. If Meta made the parser stop
    distinguishing, `-sm tensor` would report 66+0 whatever happened."""
    assert parse_layer_split(TENSOR_SPLIT_SPILLED) == (3, 1)
