"""Tighten the sampler on Thai-heavy prompts (issue #81) -- OPT-IN ONLY.

A Claude Code session through profile G (turboderp SC 4.0bpw H5, default
sampler temperature 0.6 / top-k 20 / top-p 0.95) wrote Thai with dropped
U+0E4C and Latin substitutions inside Thai runs. Same drift mechanism as #77
(Han leaking into Thai sentences) and #76 (a U+0E48 loop): where the Thai
continuation is diffuse, a wrong token inside top-k wins. A prompt line cannot
reach that point; sampler values can.

VERDICT 2026-09-15 (docs/results/17-thai-sampler-2026-09-15.md): on 48 short
Thai prose responses the tightened triple (0.3/0.9/10) scored 0 broken per
5,496 Thai chars against the default's 0 per 5,425 -- no effect. So tightening
ships OFF; a request opts in with body "thai_sampler": 1, and a client value
already tighter than the candidate is never loosened.

The rule when opted in: the prompt carries at least MIN_THAI Thai-block
characters, and the request is served with the tightened triple (T_TIGHT,
P_TIGHT, K_TIGHT). EXL3_THAI_SAMPLER=0 disables even the opt-in for the server.

server.py calls `params_for(messages, temperature, top_p, top_k, on=thai_on)`
once and passes the result to ComboSampler, next to the cjk_guard ban.
"""
import os
import re

ENV = "EXL3_THAI_SAMPLER"
# Thai block: consonants, vowels, tone marks, digits, symbols.
THAI = re.compile(r"[ก-๛]")

MIN_THAI = 20

T_TIGHT = 0.3
P_TIGHT = 0.9
K_TIGHT = 10
T_CAP = 0.6   # transcript regime: Claude Code sends 1.0 and Thai breaks
              # (เว็บไซตส์, Vietnamese leaks at 1.0); at 0.6, 144 responses
              # stayed clean. A Thai-heavy prompt never serves above this.


def has_thai(text):
    return bool(text) and THAI.search(text) is not None


def count_thai(text):
    return len(THAI.findall(text or ""))


def _texts(messages):
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, str):
            yield c
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    yield part["text"]


def prompt_thai_chars(messages):
    return sum(count_thai(t) for t in _texts(messages))


def wanted(messages):
    """Tighten this request? Off for the server with EXL3_THAI_SAMPLER=0; off
    for the request when its prompt carries fewer than MIN_THAI Thai chars."""
    if os.environ.get(ENV, "").strip() == "0":
        return False
    return prompt_thai_chars(messages) >= MIN_THAI


def params_for(messages, temperature, top_p, top_k, on = False, off = False):
    """The sampler triple to serve with.

    Default: a Thai-heavy prompt is capped at T_CAP (0.6) -- the transcript
    regime broke at client temperature 1.0 and held at 0.6. Only the
    temperature is touched; top_p/top_k pass through, and a client value
    already at or under the cap is never raised.
    `on` (body thai_sampler: 1) takes the full tightened triple instead;
    `off` (body thai_sampler: 0) is pure passthrough, the escape hatch.
    """
    if off or not wanted(messages):
        return (temperature, top_p, top_k)
    if on:
        return (min(float(temperature), T_TIGHT),
                min(float(top_p), P_TIGHT),
                min(int(top_k), K_TIGHT))
    return (min(float(temperature), T_CAP), top_p, top_k)
