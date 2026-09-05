"""The EXL3 server stops a generation that has degenerated into repetition (issue #76).

INCIDENT. 2026-09-05 13:19-14:05, 4.0bpw H5 at medium: after a correct page and a
passing gate, the model began its Thai report and locked onto the tone mark U+0E48:
one request produced 127,996 output tokens in 46 min (`draft acceptance 0.998`, the
drafter predicting a token that never changes). The output cap is the whole window
on purpose, so nothing stopped it; Claude Code then compacted and re-prefilled 207K
tokens. The guard here watches the generated TEXT (the fork's iterate() hands the
server text chunks, not ids) and trips when a window of it is one or two characters,
or a short unit repeated end to end. Ordinary prose, Thai included, never looks like
that -- the negatives below are real lines from today's pages.
"""
import os
import sys

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUNING = os.path.dirname(BENCH)
sys.path.insert(0, os.path.join(TUNING, "serving", "exl3"))
import loop_guard  # noqa: E402


def feed(text, **kw):
    g = loop_guard.LoopGuard(**kw)
    for i in range(0, len(text), 7):        # arrive in small chunks, as from the generator
        if g.feed(text[i:i + 7]):
            return g, i + 7
    return g, None


def test_the_incident_trips_within_the_window():
    """The 13:19 message: a few words, then U+0E48 forever."""
    text = "ทุ้่่่่ง 8 ข้อผ่านแล้้" + "่" * 2000
    g, at = feed(text)
    assert at is not None and at <= len("ทุ้่่่่ง 8 ข้อผ่านแล้้") + loop_guard.WINDOW + 8
    assert g.reason and "distinct" in g.reason


def test_a_short_unit_repeated_end_to_end_trips_too():
    text = "The page is done. " + "ab ab " * 400
    g, at = feed(text)
    assert at is not None
    assert "period" in g.reason


def test_thai_prose_with_tone_marks_does_not_trip():
    """Real lines from A-skill-r1/page.html and C-noskill-r1/page.html (2026-09-05)."""
    prose = ("กูเกิลคือบริษัทที่เงียบแต่มีประโยชน์มากที่สุดในโลกออนไลน์ — และเป็นบริษัทที่ ควรค่าแก่การวางใจให้ดูแลความสนใจของคุณที่สุด "
             "เริ่มต้นจากการนับลิงก์ในห้องสมุดสแตนฟอร์ดในปี 1998 คือ โครงสร้างพื้นฐานของอินเทอร์เน็ต "
             "เครื่องมือที่ผู้คนหลายพันล้านคนใช้ ค้นหา เดินทาง ทำงาน และสร้างสรรค์ ตั้งแต่เช้าจนค่ำ ทุก ๆ นาทีทั่วโลก ") * 8
    g, at = feed(prose)
    assert at is None


def test_css_and_html_with_long_runs_of_the_same_character_do_not_trip():
    """A page has `-----` rules, `=====` in comments, long runs of spaces and `0`s in
    data URIs; none of these are loops as long as other characters keep arriving."""
    html = ("<div class=\"stats-band glass-dark reveal\">\n" + " " * 40 + "<span class=\"num\">03 / Scale</span>\n"
            "/* " + "-" * 120 + " */\n" + "data:image/svg+xml;base64,AAAA" + "A" * 200 + "QQ==\n") * 20
    g, at = feed(html)
    assert at is None


def test_a_run_of_one_character_longer_than_the_window_trips_even_inside_html():
    """The line above is the boundary: 200 A's pass, a window full of them does not."""
    html = "<pre>" + "A" * (loop_guard.WINDOW + 50) + "</pre>"
    g, at = feed(html)
    assert at is not None


def test_the_guard_reports_how_far_it_got():
    g, at = feed("x" * 3000)
    assert g.n_chars == at


def test_the_server_feeds_every_chunk_and_cancels_the_job_on_a_loop():
    server = open(os.path.join(TUNING, "serving", "exl3", "server.py"), encoding = "utf-8").read()
    assert "import live_timing, effort, anthropic_routes, watchdog, loop_guard" in server
    assert "guard = loop_guard.LoopGuard()" in server
    assert "if guard.feed(chunk, in_think =" in server   # the sentence-loop rule only inside <think>
    assert "generator.cancel(job)" in server
    assert '"loop": "length"' in server                      # the client sees a normal length stop
    assert 'timings["stop_reason"] = "loop"' in server       # and the log/timings say why
    assert '"loops_stopped": stats["loops_stopped"]' in server   # /health counts them


# --- the second mode, 2026-09-05 19:03: a sentence loop inside <think> ---------------
# "OK, I'm going to write the code now. Let me stop deliberating and just write it."
# ~1,000 times, 127,996 tokens, 45 min; every character distinct enough that the
# two-character rule never fired. Inside thinking a 64-character unit repeating five
# times in the last 4,096 characters is a loop; in content it is not applied, because
# a page repeats card markup legitimately.

def test_a_sentence_loop_in_thinking_trips_on_the_long_unit_rule():
    text = "Let me plan this carefully.\n" + ("OK, let me write the store.py file.\nI'm going to write the code now.\nOK, I'm going to write the code now. Let me stop deliberating and just write it.\n" * 60)
    g = loop_guard.LoopGuard()
    tripped = None
    for i in range(0, len(text), 9):
        if g.feed(text[i:i + 9], in_think = True):
            tripped = i; break
    assert tripped is not None and tripped < 6000
    assert "unit" in g.reason


def test_the_same_text_as_content_does_not_trip_the_long_unit_rule():
    text = ('<div class="card reveal"><span class="tag">SEARCH</span><h3>Google Search</h3></div>\n' * 60)
    g = loop_guard.LoopGuard()
    for i in range(0, len(text), 9):
        assert not g.feed(text[i:i + 9], in_think = False)


def test_ordinary_thinking_does_not_trip():
    """A slice of real thinking from A-skill-r1 (2026-09-05), which weighs options and repeats phrases without looping."""
    text = ("The user wants a landing page for Google in four styles. Let me check the brief table first. Editorial Minimalism needs whitespace and a serif display face; Modern Swiss needs a 12-column grid; Liquid Glass needs backdrop-filter; Visible Grid needs hairlines. "
            "I should pick two font families only. Let me think about the palette: 60-30-10. Now the sections: hero, index, stats, quote, verdict, footer. "
            "I'll write the file now. Actually, let me reconsider the hero: the ruler letters A-L should sit in the gutter, not over the h1. OK. ")
    parts = []
    for i in range(240):
        w = ["hero", "index", "stats", "quote", "verdict", "footer"][i % 6]
        m = ["air", "ink", "grid", "glass"][i % 4]
        f = ["serif", "mono", "sans"][i % 3]
        parts.append(f"Point {i}: the {w} needs {i * 7 % 13} px of {m} and a {f} face; reconsidering section {i % 9}.")
    text = " ".join(parts)
    g = loop_guard.LoopGuard()
    for i in range(0, len(text), 9):
        assert not g.feed(text[i:i + 9], in_think = True), g.reason


def test_markup_drafted_inside_thinking_is_not_a_loop():
    """A-both-r2 (2026-09-05): six section headers drafted in thought share a 64-character
    opening; the healthy run must not be cut. Markup units are ignored by the long-unit rule."""
    block = '<section class="s"><div class="section-head"><span class="index-num">0'
    text = "".join(block + str(i) + '</span><span class="section-label">Part ' + str(i) + '</span></div>...</section>\n' for i in range(12))
    g = loop_guard.LoopGuard()
    for i in range(0, len(text), 9):
        assert not g.feed(text[i:i + 9], in_think = True), g.reason
