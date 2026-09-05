r"""The two small modules cut out of the fork's server file (2026-09-04).

live_timing.report() must reproduce the numbers CORRECTIONS.md §47 fixed:
decode rate = out_toks / time_generate, prompt rate over the UNcached tokens
only, and the end block in llama-server's shape. effort.resolve() must map
what clients send onto the three values the Qwen3.8 template accepts.
"""
import os
import sys

TUNING = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(TUNING, "serving", "exl3"))
import live_timing  # noqa: E402
import effort  # noqa: E402


def test_timings_use_disjoint_prefill_and_generate_and_net_out_cached_tokens():
    final = {"time_prefill": 2.0, "time_generate": 4.0, "cached_tokens": 900,
             "accepted_draft_tokens": 60, "rejected_draft_tokens": 40}
    t = live_timing.timings(final, prompt_toks = 1000, out_toks = 200, wall = 6.5)
    assert t["prompt_n"] == 100 and t["prompt_per_second"] == 50.0
    assert t["predicted_n"] == 200 and t["predicted_per_second"] == 50.0
    assert t["cached_tokens"] == 900 and t["wall_ms"] == 6500.0
    assert (t["draft_accepted"], t["draft_rejected"]) == (60, 40)


def test_report_prints_the_llama_server_block_and_returns_the_timings():
    lines = []
    final = {"time_prefill": 2.0, "time_generate": 4.0, "cached_tokens": 900,
             "accepted_draft_tokens": 60, "rejected_draft_tokens": 40}
    t = live_timing.report(final, 1000, 200, 6.5, "medium", out = lambda s, **k: lines.append(s))
    assert t["prompt_per_second"] == 50.0
    assert [l.split("|")[-1].strip().split(" =")[0] for l in lines] == [
        "prompt eval time", "eval time", "total time", "draft acceptance", "reasoning effort"]
    assert "100 tokens" in lines[0] and "[900 cached]" in lines[0]
    assert "200 tokens" in lines[1] and "50.00 tokens per second" in lines[1]
    assert "0.60000" in lines[3] and lines[4].endswith("reasoning effort = medium")
    assert all("slot print_timing: id  0 | task" in l for l in lines)


def test_rate3_anchors_on_the_last_sample_at_or_before_the_three_second_edge():
    hist = [(0.0, 0), (1.0, 100), (2.0, 200), (5.0, 260)]
    # cutoff 2.0: the anchor is the last sample BEFORE it, (1.0, 100), so the
    # window is at least three seconds, never less: (260-100)/(5-1)
    assert abs(live_timing.LiveTiming.rate3(hist, 5.0) - 40.0) < 1e-9
    hist = [(0.0, 0), (1.9, 100), (2.0, 200), (5.0, 260)]
    assert abs(live_timing.LiveTiming.rate3(hist, 5.0) - (160 / 3.1)) < 1e-9
    assert live_timing.LiveTiming.rate3([(5.0, 1)], 5.05) == 0.0          # too short to rate


def test_effort_resolves_aliases_and_falls_back_to_the_default():
    assert [effort.resolve(v, default = "xhigh") for v in
            (None, "medium", "low", "high", "xhigh", "max", "minimal", "banana", "MEDIUM")] == \
        ["xhigh", "medium", "low", "xhigh", "xhigh", "xhigh", "low", "xhigh", "medium"]
