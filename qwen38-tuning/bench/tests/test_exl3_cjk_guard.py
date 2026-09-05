"""The EXL3 server bans Han (Chinese) tokens when the prompt has none (issue #77).

INCIDENT. Over the 43 Claude Code streams of the 2026-09-05 quality bench, 4.0bpw
H5 at medium dropped 14 Han characters into 3 streams, every one inside a Thai
sentence: "โมเดล前沿…", "ข้อจำกัด这套 และ test", "อีเมล และ协作". The mechanism is
sampling drift (a Chinese token with the same meaning is inside top-k where the
Thai continuation is diffuse), so a prompt line cannot reach it; a logit ban can.
The rule the developer set: if the prompt carries no Chinese, the model may not
answer in Chinese. A prompt that does carry it (pasted text, a request for
Chinese, a tool result) lifts the ban for that request.

The vocab test decodes the real tokenizer.json with its own byte-level map so it
does not trust the module's regex to judge the module's regex.
"""
import json
import os
import sys

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUNING = os.path.dirname(BENCH)
ROOT = os.path.dirname(TUNING)
sys.path.insert(0, os.path.join(TUNING, "serving", "exl3"))
import cjk_guard  # noqa: E402

TOKENIZER = os.path.join(ROOT, "models", "turboderp-Qwen3.8-27B-EXL3-SC4.0bpw-H5", "tokenizer.json")

LEAKS = ["โมเดล前沿…", "ข้อจำกัด这套 และ test", "อีเมล และ协作 → \"เอกสาร\""]
CLEAN = [
    "ทุ้่่่่ง 8 ข้อผ่านแล้ว — การให้เหตุผล การสร้างโค้ด",
    "def store(items: dict[str, int]) -> None:  # O(1)",
    "ひらがな カタカナ だけ",              # kana only, no kanji
    "한국어 텍스트",                        # Hangul is not Han
    "emoji 🚀 and punctuation 。、「」",   # CJK punctuation is not Han
    "",
]


def test_the_three_leaks_are_seen_and_clean_thai_code_kana_hangul_are_not():
    for s in LEAKS:
        assert cjk_guard.has_han(s), s
    for s in CLEAN:
        assert not cjk_guard.has_han(s), s


def test_count_han_counts_characters_not_runs():
    assert cjk_guard.count_han("โมเดล前沿…") == 2
    assert cjk_guard.count_han("ข้อจำกัด这套 และ test 协作") == 4
    assert cjk_guard.count_han("clean") == 0


def test_prompt_has_han_walks_strings_and_text_parts_of_every_role():
    thai = [{"role": "system", "content": "แชทไทย"},
            {"role": "user", "content": [{"type": "text", "text": "ทำหน้า landing"},
                                         {"type": "image_url", "image_url": {"url": "data:..."}}]},
            {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "content": "ok"}]
    assert not cjk_guard.prompt_has_han(thai)
    assert cjk_guard.prompt_has_han(thai + [{"role": "user", "content": "แปลเป็นจีน: 你好"}])
    assert cjk_guard.prompt_has_han(thai + [{"role": "user", "content": [{"type": "text", "text": "你好"}]}])
    # a tool result that carries Chinese (a file the model read) lifts the ban too
    assert cjk_guard.prompt_has_han(thai + [{"role": "tool", "content": "// 注释"}])


def test_naming_china_or_chinese_lifts_the_ban_without_a_han_character():
    """2026-09-06, the developer: a prompt that talks ABOUT Chinese without typing
    any ("แปลเป็นภาษาจีน", "reply in Chinese") must not be banned either."""
    for p in ["แปลประโยคนี้เป็นภาษาจีน", "ลูกค้าจีนอ่านไม่ออก", "Reply in Chinese, please",
              "our China launch copy", "CHINESE market", "write it in Mandarin"]:
        assert cjk_guard.prompt_mentions_chinese([{"role": "user", "content": p}]), p
        assert not cjk_guard.wanted([{"role": "user", "content": p}]), p
    for p in ["machinations of the board", "the chinatown scene has no word boundary? it does: chinatown",
              "ทำหน้า landing ภาษาไทย", "Japanese and Korean only", "porcelain, not the country"]:
        assert not cjk_guard.prompt_mentions_chinese([{"role": "user", "content": p}]), p


def test_the_env_switch_disables_the_ban_and_only_the_env_switch(monkeypatch):
    monkeypatch.delenv(cjk_guard.ENV, raising = False)
    assert cjk_guard.wanted([{"role": "user", "content": "ไทย"}])
    assert not cjk_guard.wanted([{"role": "user", "content": "你好"}])
    monkeypatch.setenv(cjk_guard.ENV, "1")
    assert not cjk_guard.wanted([{"role": "user", "content": "ไทย"}])
    monkeypatch.setenv(cjk_guard.ENV, "0")
    assert cjk_guard.wanted([{"role": "user", "content": "ไทย"}])


def test_ban_ids_picks_pieces_with_han_and_bias_is_minus_inf():
    pieces = ["hello", "前沿", "と思", "ひら", "ทุ้ง", "<|im_end|>", " 协"]
    ids = cjk_guard.ban_ids(pieces)
    assert ids == [1, 2, 6]
    assert cjk_guard.bias(ids) == {1: float("-inf"), 2: float("-inf"), 6: float("-inf")}


class _Tokenizer:
    """The fork's tokenizer surface the server hands over: the piece list with specials."""
    calls = 0

    def get_id_to_piece_list(self, include_special_tokens = False):
        assert include_special_tokens
        _Tokenizer.calls += 1
        return ["hi", "前", "<|im_end|>", "ทุ้ง", "协作"]


def test_bias_for_is_none_when_the_prompt_has_han_and_scans_the_vocab_once(monkeypatch):
    monkeypatch.delenv(cjk_guard.ENV, raising = False)
    tok = _Tokenizer()
    _Tokenizer.calls = 0
    assert cjk_guard.bias_for(tok, [{"role": "user", "content": "你好"}]) is None
    b = cjk_guard.bias_for(tok, [{"role": "user", "content": "ไทย"}])
    assert b == {1: float("-inf"), 4: float("-inf")}
    assert cjk_guard.bias_for(tok, [{"role": "user", "content": "more"}]) is b   # no per-request rebuild
    assert _Tokenizer.calls == 1


def _decoded_vocab():
    tok = json.load(open(TOKENIZER, encoding = "utf-8"))
    vocab = tok["model"]["vocab"]
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    byte_of = {chr(c): b for b, c in zip(bs, cs)}
    out = {}
    for piece, i in vocab.items():
        try:
            out[i] = bytes(byte_of[c] for c in piece).decode("utf-8")
        except (KeyError, UnicodeDecodeError):
            continue
    return out


@pytest.mark.skipif(not os.path.exists(TOKENIZER), reason = "4.0bpw tokenizer not on this machine")
def test_on_the_real_vocab_the_ban_is_every_han_piece_and_nothing_else():
    """The three leaked words tokenise into banned ids; Thai and code pieces never do.
    Counts pin the vocab we measured on 2026-09-06 (53,429 pure-Han + 1,899 mixed)."""
    decoded = _decoded_vocab()
    n = max(decoded) + 1
    pieces = [decoded.get(i, "") for i in range(n)]
    ids = set(cjk_guard.ban_ids(pieces))
    assert 55_000 <= len(ids) <= 56_000, len(ids)
    for i, s in decoded.items():
        expected = any("一" <= ch <= "鿿" or "㐀" <= ch <= "䶿" or
                       "豈" <= ch <= "﫿" or "\U00020000" <= ch <= "\U0003134f" for ch in s)
        assert (i in ids) == expected, (i, s)
