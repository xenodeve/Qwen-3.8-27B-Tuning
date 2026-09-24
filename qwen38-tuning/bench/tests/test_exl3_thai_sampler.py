"""Tighten the sampler on Thai-heavy prompts (issue #81).

INCIDENT. A Claude Code session through profile G (turboderp SC 4.0bpw H5) wrote
Thai with dropped U+0E4C ("ดาวน์โหลด" -> "ดาวนโหลด", "เว็บไซต์" -> "เว็บไซต",
"ลิงก์" -> "ลิงค") and Latin substitutions inside Thai runs ("ชื่อ" -> "ชื่o",
"นั้น" -> "นั้n"). Same drift mechanism as #77 (Han in Thai sentences) and #76
(127,996-token U+0E48 loop): where the Thai continuation is diffuse, a
wrong token inside top-k wins at temperature 0.6 / top-k 20.

The module under test is a pure function: given the request messages and the
client-sent sampler values, return the values to serve with. Thai-heavy
prompts get the tightened triple; everything else passes through untouched --
and a client value already tighter than the candidate is never loosened.
"""
import os
import sys

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUNING = os.path.dirname(BENCH)
sys.path.insert(0, os.path.join(TUNING, "serving", "exl3"))
import thai_sampler  # noqa: E402


def msg(text):
    return [{"role": "user", "content": text}]


def test_broken_thai_from_the_incident_counts_as_thai():
    assert thai_sampler.has_thai("ดาวนโหลดวดีโอ")
    assert thai_sampler.has_thai("ชื่o นั้n")
    assert not thai_sampler.has_thai("plain english")
    assert not thai_sampler.has_thai("")


def test_short_thai_does_not_trigger_but_a_thai_paragraph_does():
    assert not thai_sampler.wanted(msg("สวัสดี"))
    assert thai_sampler.wanted(msg("อธิบายวิธีดาวน์โหลดวิดีโอจาก YouTube เป็นภาษาไทยให้ละเอียด"))
    assert not thai_sampler.wanted(msg("def f(x): return x + 1"))
    assert not thai_sampler.wanted(msg("โมเดล前沿 test"))


def test_thai_heavy_prompts_are_capped_at_0_6_by_default():
    """Transcript regime: Claude Code sends temperature 1.0 and Thai breaks
    (เว็บไซตส์, Vietnamese leaks at 1.0 over 64 responses); at 0.6, 144
    responses stayed clean. So a Thai-heavy prompt never serves above 0.6,
    whatever the client sent -- without touching top_p/top_k."""
    heavy = msg("อธิบายวิธีดาวน์โหลดวิดีโอจาก YouTube เป็นภาษาไทยให้ละเอียด")
    assert thai_sampler.params_for(heavy, 1.0, 0.95, 20) == (0.6, 0.95, 20)
    assert thai_sampler.params_for(heavy, 0.6, 0.95, 20) == (0.6, 0.95, 20)
    assert thai_sampler.params_for(heavy, 0.2, 0.9, 10) == (0.2, 0.9, 10)
    assert thai_sampler.params_for(msg("write python"), 1.0, 0.95, 20) == (1.0, 0.95, 20)


def test_tightening_is_opt_in_per_request():
    """Issue #81 verdict: on 48 short Thai prose responses (24 per arm) the
    tightened triple scored 0/5425 vs default 0/5496 Thai chars -- no effect.
    So tightening ships OFF; a request opts in with on=True."""
    heavy = msg("อธิบายวิธีดาวน์โหลดวิดีโอจาก YouTube เป็นภาษาไทยให้ละเอียด")
    assert thai_sampler.params_for(heavy, 0.6, 0.95, 20) == (0.6, 0.95, 20)
    t, p, k = thai_sampler.params_for(heavy, 0.6, 0.95, 20, on=True)
    assert (t, p, k) == (0.3, 0.9, 10)
    assert thai_sampler.params_for(msg("write python"), 0.6, 0.95, 20, on=True) == (0.6, 0.95, 20)


def test_a_client_value_already_tighter_is_never_loosened():
    t, p, k = thai_sampler.params_for(
        msg("อธิบายวิธีดาวน์โหลดวิดีโอจาก YouTube เป็นภาษาไทยให้ละเอียด"),
        0.0, 0.5, 1, on=True)
    assert (t, p, k) == (0.0, 0.5, 1)


def test_explicit_off_is_pure_passthrough():
    """Body thai_sampler: 0 keeps the client values untouched -- the escape
    hatch and the control arm for any future A/B."""
    heavy = msg("อธิบายวิธีดาวน์โหลดวิดีโอจาก YouTube เป็นภาษาไทยให้ละเอียด")
    assert thai_sampler.params_for(heavy, 1.0, 0.95, 20, off=True) == (1.0, 0.95, 20)


def test_env_kill_switch_disables_the_cap_too(monkeypatch):
    heavy = msg("อธิบายวิธีดาวน์โหลดวิดีโอจาก YouTube เป็นภาษาไทยให้ละเอียด")
    monkeypatch.setenv(thai_sampler.ENV, "0")
    assert thai_sampler.params_for(heavy, 1.0, 0.95, 20) == (1.0, 0.95, 20)
    monkeypatch.delenv(thai_sampler.ENV, raising=False)
    assert thai_sampler.params_for(heavy, 1.0, 0.95, 20) == (0.6, 0.95, 20)


def test_env_kill_switch_disables_even_opt_in(monkeypatch):
    heavy = msg("อธิบายวิธีดาวน์โหลดวิดีโอจาก YouTube เป็นภาษาไทยให้ละเอียด")
    monkeypatch.setenv(thai_sampler.ENV, "0")
    assert thai_sampler.params_for(heavy, 0.6, 0.95, 20, on=True) == (0.6, 0.95, 20)
    monkeypatch.delenv(thai_sampler.ENV, raising=False)
    assert thai_sampler.params_for(heavy, 0.6, 0.95, 20, on=True) == (0.3, 0.9, 10)
