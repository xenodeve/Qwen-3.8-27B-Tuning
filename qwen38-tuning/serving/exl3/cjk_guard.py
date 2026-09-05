"""Ban Han (Chinese) tokens when the prompt carries none (issue #77, 2026-09-06).

Over the 43 Claude Code streams of the 2026-09-05 quality bench, 4.0bpw H5 dropped
14 Han characters into 3 streams, every one inside a Thai sentence ("โมเดล前沿…",
"ข้อจำกัด这套 และ test", "อีเมล และ协作"). It is sampling drift, not a language
choice: where the next Thai token is diffuse, a Chinese token with the same meaning
sits inside top-k and wins. A prompt line cannot reach that point; a logit ban can.

The rule: if the prompt has no Han character and does not name Chinese/China
(จีน, china, chinese, mandarin), the model may not emit one (thinking included).
A prompt that has one -- pasted text, a request for Chinese, a tool result -- lifts
the ban for that request. EXL3_ALLOW_CJK=1 lifts it for the server.

server.py hands `ban_ids(tokenizer pieces)` to `bias()` once, passes the dict as
ComboSampler's logit_bias when `wanted(messages)`, and counts `count_han()` of every
completion into timings["cjk_chars"] and /health.cjk_chars_total -- the instrument
that shows the ban working, or not.
"""
import os
import re

ENV = "EXL3_ALLOW_CJK"
# CJK Unified Ideographs, Extension A, Compatibility Ideographs, Extensions B-H.
# Not kana, not Hangul, not CJK punctuation: those never appeared in a leak.
HAN = re.compile(r"[一-鿿㐀-䶿豈-﫿\U00020000-\U0003134f]")

def has_han(text):
    return bool(text) and HAN.search(text) is not None


def count_han(text):
    return len(HAN.findall(text or ""))


def _texts(messages):
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, str):
            yield c
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    yield part["text"]


def prompt_has_han(messages):
    return any(has_han(t) for t in _texts(messages))


# 2026-09-06, the developer: talking ABOUT Chinese without typing any ("แปลเป็นภาษาจีน",
# "reply in Chinese") lifts the ban too. Word-bounded in English so "machinations" and
# "chinatown" do not; Thai has no word boundary, so จีน is a plain substring.
MENTION = re.compile(r"จีน|\b(?:china|chinese|mandarin)\b", re.IGNORECASE)


def prompt_mentions_chinese(messages):
    return any(MENTION.search(t) for t in _texts(messages))


def wanted(messages):
    """Ban this request? Off for the server with EXL3_ALLOW_CJK=1; off for the
    request when its prompt carries a Han character or names Chinese/China."""
    if os.environ.get(ENV, "").strip() in ("1", "true", "yes"):
        return False
    return not (prompt_has_han(messages) or prompt_mentions_chinese(messages))


def ban_ids(pieces):
    """Every vocab id whose decoded piece contains a Han character."""
    return [i for i, p in enumerate(pieces) if p and HAN.search(p)]


def bias(ids):
    """The -inf logit_bias dict for ComboSampler."""
    return {int(i): float("-inf") for i in ids}


_bias_cache = {}


def bias_for(tokenizer, messages):
    """ComboSampler's logit_bias for this request: the Han ban when `wanted`,
    else None. The vocab scan (248K pieces) and the dict are built once per tokenizer."""
    if not wanted(messages):
        return None
    b = _bias_cache.get(id(tokenizer))
    if b is None:
        b = bias(ban_ids(tokenizer.get_id_to_piece_list(include_special_tokens = True)))
        _bias_cache[id(tokenizer)] = b
    return b
