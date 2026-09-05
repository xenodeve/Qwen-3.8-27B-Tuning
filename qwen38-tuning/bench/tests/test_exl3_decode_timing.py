"""Guards the EXL3 decode figure (issue #71, CORRECTIONS 47).

Incident, 2026-09-03: `exllama3-test-decode.py` computed decode time as
`time_generate - time_prefill`. In the Mia fork `time_generate` is already the
first-token -> last-token interval, disjoint from `time_prefill`
(exllamav3/generator/job.py:674-675), so every warm round was overstated by
tp/tg (~5 % at 144K), and one arm whose warm prefill took 9 s printed 96 tok/s
for a 508-token decode that took 14.4 s.
"""
import pytest

from harness import exl3_decode_seconds


def test_decode_time_is_time_generate_not_minus_prefill():
    final = {"time_prefill": 9.129, "time_generate": 14.424, "new_tokens": 508}
    dec_s, source = exl3_decode_seconds(final, wall=23.585)
    assert source == "generator"
    assert dec_s == pytest.approx(14.424)
    assert 508 / dec_s < 40  # the incident printed 95.93


def test_cold_round_falls_back_to_wall_minus_prefill_when_generator_reports_zero():
    final = {"time_prefill": 292.9, "time_generate": 0.0, "new_tokens": 508}
    dec_s, source = exl3_decode_seconds(final, wall=309.3)
    assert source == "wall_minus_prefill"
    assert dec_s == pytest.approx(309.3 - 292.9)
