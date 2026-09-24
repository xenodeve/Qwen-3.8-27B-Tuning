"""A/B pair: default sampler vs the #81 Thai tightening, one boot, one server.

Arms (since the #81 verdict tightening is opt-in; before the verdict the
semantics were inverted -- body thai_sampler: 0 meant control):
  A  control  -- no thai_sampler key (the client-sent default sampler)
  B  tightened -- thai_sampler: 1 in the request body (0.3 / 0.9 / 10 on
                 Thai-heavy prompts via thai_sampler.params_for).

Each prompt runs BOTH arms; the arm order alternates across prompts so the
pair is not biased by prompt position. Rows land in
qwen38-tuning/bench/results/thai-sampler-<stamp>.jsonl; the verdict is the
broken-Thai count per 1,000 Thai characters, arm by arm.

Usage:  python tools/thai-sampler-pair.py  [--base http://127.0.0.1:8000]
        python tools/thai-sampler-pair.py --seed-prefix [--rounds N]

--seed-prefix (issue #82): arm B prepends a fixed English tool-output block
to the Thai prompt; arm A is the bare Thai prompt. If English context seeds
Thai script errors, B breaks more than A.

--tool-seed (issue #82, regime 2): both arms carry a real tool-role message
with English output (the transcript's post-tool-call shape) and ask for a
LONG Thai summary (8-10 sentences) at reasoning_effort medium; arms differ
only in prevention -- A default sampler, B thai_sampler: 1. This is the
prevention A/B: if A breaks and B does not, the opt-in prevents cascades.

--multiturn (issue #82, regime 3): an English coding-agent system prompt plus
three Thai-task/tool-output turns before the final long Thai summary ask.
Closest to the transcript short of a real 20-turn session.
"""
import argparse
import datetime
import json
import os
import re
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "bench", "results")

# Broken forms observed in the wild (issue #81 incident). Each is a regex that
# matches ONLY the broken form: a missing diacritic at word end is otherwise
# a substring of the correct word ("เว็บไซต" is inside correct "เว็บไซต์"),
# which inflated the first two runs with false positives (2026-09-15).
BROKEN_PATTERNS = [
    (r"ดาวนโหลด", "ดาวน์โหลด"),
    (r"วดีโอ", "วิดีโอ"),
    (r"เว็บไซต(?!์)", "เว็บไซต์"),
    (r"ลิงค(?!์)", "ลิงก์"),
    (r"แลว(?![่-๋แ])", "แล้ว"),   # bare แลว; ดูแลว่า/ดูแลแล้ว stay quiet
    (r"หรื(?!อ)", "หรือ"),
    (r"ไฟล(?!์)", "ไฟล์"),
    (r"โฟลเดอร(?!์)", "โฟลเดอร์"),
    (r"ตอไป", "ต่อไป"),
    (r"เกียวกับ", "เกี่ยวกับ"),
    (r"เครื่่อง", "เครื่อง"),
]
BROKEN_WORDS = None   # retired 2026-09-15, see above

# A Latin letter sandwiched between Thai characters: the "ชื่o / นั้n" fault.
LATIN_IN_THAI = re.compile(r"[ก-๛][A-Za-z][ก-๛]")
THAI = re.compile(r"[ก-๛]")

PROMPTS = [
    "เขียนบทความภาษาไทยสั้น ๆ 3 ประโยค เกี่ยวกับการดาวน์โหลดวิดีโอจาก YouTube",
    "เขียนรีวิวภาษาไทย 3 ประโยค เกี่ยวกับเว็บไซต์ขายของออนไลน์",
    "เขียนภาษาไทย 3 ประโยค อธิบายว่าทำไมควรสำรองไฟล์งานไว้เสมอ",
    "เขียนภาษาไทย 3 ประโยค เกี่ยวกับประโยชน์ของการออกกำลังกายตอนเช้า",
    "เขียนภาษาไทย 3 ประโยค แนะนำวิธีเรียนภาษาอังกฤษด้วยตัวเอง",
    "เขียนภาษาไทย 3 ประโยค เกี่ยวกับอาหารไทยจานโปรดของคุณ",
    "เขียนภาษาไทย 3 ประโยค อธิบายการทำงานของอินเทอร์เน็ตอย่างง่าย",
    "เขียนภาษาไทย 3 ประโยค เกี่ยวกับการใช้เงินอย่างประหยัด",
]

MAX_TOKENS = 400
SLEEP = 2.0

# Issue #82 seed: the shape of a post-tool-call context -- shell output with
# paths, versions and JSON-ish lines, all English. Fixed so every B prompt
# shares the same seed and the A/B comparison stays paired per prompt.
SEED_BLOCK = """Tool result (exec, exit 0):
$ pip install -r requirements.txt
Collecting flask==3.1.3 (from -r requirements.txt (line 1))
  Downloading flask-3.1.3-py3-none-any.whl (103 kB)
Successfully installed flask-3.1.3 yt-dlp-2026.08.19 imageio-ffmpeg-0.4.9
$ ls downloads/
Rick-Astley-213s.mp4  playlist-batch-03.zip
{"endpoint": "/api/info", "status": 200, "duration_ms": 1843}"""

# Regime 2: the transcript's post-tool-call shape. Each topic becomes a
# 3-message exchange; the final ask wants a LONG Thai summary so a seeded
# cascade has room to run.
TOOL_TASKS = [
    "ช่วยติดตั้ง dependencies แล้วรันเทสให้หน่อย",
    "ช่วยดึงข้อมูลวิดีโอจาก URL ที่ให้ไปแล้วสรุปให้หน่อย",
    "ช่วยแปลงไฟล์วิดีโอเป็น MP3 ให้หน่อย",
    "ช่วยเช็คว่า ffmpeg ใช้งานได้แล้วรายงานมา",
    "ช่วยโหลด playlist นี้แล้วรวมเป็น zip ให้หน่อย",
    "ช่วยเทส endpoint ทั้งหมดแล้วบอกผล",
    "ช่วยทำความสะอาดไฟล์ชั่วคราวให้หน่อย",
    "ช่วยเขียน README สรุปวิธีใช้งานให้หน่อย",
]
LONG_ASK = "สรุปผลเป็นภาษาไทย 8-10 ประโยค เกี่ยวกับ{}"

TOPIC_NOUNS = [
    "การดาวน์โหลดวิดีโอจาก YouTube",
    "เว็บไซต์ขายของออนไลน์",
    "การสำรองไฟล์งาน",
    "การออกกำลังกายตอนเช้า",
    "การเรียนภาษาอังกฤษด้วยตัวเอง",
    "อาหารไทย",
    "การทำงานของอินเทอร์เน็ต",
    "การใช้เงินอย่างประหยัด",
]


def build_toolseed(task, topic):
    return [
        {"role": "user", "content": task},
        {"role": "tool", "content": SEED_BLOCK},
        {"role": "user", "content": LONG_ASK.format(topic)},
    ]


# Regime 3: generic English coding-agent system prompt (not any vendor's) plus
# three task/tool turns. The transcript's garbled summaries all came after
# several such turns, never at session start.
SYSTEM_EN = """You are a coding assistant working inside a project directory. You can run shell commands and read or write files with your tools. Always verify with real command output, never claim what you did not run. Reply in the same language the user writes in."""

TOOL_OUTPUTS = [
    SEED_BLOCK,
    """Tool result (exec, exit 0):
$ python app.py
 * Serving Flask app 'app'
 * Running on http://127.0.0.1:5000 (Press CTRL+C to quit)
127.0.0.1 - - [15/Sep/2026 02:43:58] "GET /api/info?url=https://www.youtube.com/watch?v=dQw4w9WgXcQ HTTP/1.1" 200 -
{"title": "Rick Astley - Never Gonna Give You Up", "duration": 213, "qualities": [144, 240, 360, 480, 720, 1080, 1440, 2160]}""",
    """Tool result (read, 84 lines):
1: <!doctype html>
2: <html lang="th">
3: <head>
4:   <meta charset="utf-8" />
14:       <p class="sub">placeholder</p>
77:     <footer class="foot">placeholder</footer>
Wrote 250 lines to app.py in 19s. Background command "Install flask, yt-dlp, imageio-ffmpeg" completed (exit code 0).""",
]

SHORT_THAI_TASKS = [
    "เช็คให้หน่อยว่า dependencies ครบไหม",
    "รันเซิร์ฟเวอร์แล้วเทส endpoint ให้หน่อย",
    "อ่านไฟล์ index.html แล้วบอกโครงสร้าง",
]


def build_multiturn(topic):
    msgs = [{"role": "system", "content": SYSTEM_EN}]
    for k in range(3):
        msgs.append({"role": "user", "content": SHORT_THAI_TASKS[k]})
        msgs.append({"role": "tool", "content": TOOL_OUTPUTS[k]})
    msgs.append({"role": "user", "content": LONG_ASK.format(topic)})
    return msgs


def broken_count(text):
    words = sum(len(re.findall(p, text)) for p, _ in BROKEN_PATTERNS)
    latin = len(LATIN_IN_THAI.findall(text))
    return words + latin


def thai_chars(text):
    return len(THAI.findall(text))


def call(base, prompt, opt_in, messages = None, effort = None, max_tokens = None,
         temperature = 0.6):
    body = {
        "model": "qwen3.8-27b",
        "messages": messages if messages is not None else [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens or MAX_TOKENS,
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 20,
    }
    if effort:
        body["reasoning_effort"] = effort
    if opt_in:
        body["thai_sampler"] = 1
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data = json.dumps(body).encode("utf-8"),
        headers = {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout = 600) as r:
        out = json.loads(r.read().decode("utf-8"))
    choice = out["choices"][0]
    return (choice["message"].get("content") or "", choice.get("finish_reason"),
            out.get("usage", {}).get("completion_tokens"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default = "http://127.0.0.1:8000")
    ap.add_argument("--rounds", type = int, default = 1,
                    help = "how many times to run the whole prompt set")
    ap.add_argument("--seed-prefix", action = "store_true",
                    help = "issue #82: arm B carries a fixed English tool-output block")
    ap.add_argument("--tool-seed", action = "store_true",
                    help = "issue #82 regime 2: real tool-role seed, long Thai summary, "
                           "medium effort; arms differ only in thai_sampler opt-in")
    ap.add_argument("--multiturn", action = "store_true",
                    help = "issue #82 regime 3: English system prompt + three Thai/tool "
                           "turns, then the long Thai summary ask")
    ap.add_argument("--temperature", type = float, default = 0.6,
                    help = "client temperature; 1.0 reproduces the Claude Code "
                           "default the transcript ran under (regime 4)")
    a = ap.parse_args()

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = ("thai-multiturn" if a.multiturn else
           "thai-toolseed" if a.tool_seed else
           ("thai-seed" if a.seed_prefix else "thai-sampler"))
    if abs(a.temperature - 0.6) > 1e-9:
        tag += f"-t{a.temperature:g}"
    out_path = os.path.join(RESULTS, f"{tag}-{stamp}.jsonl")
    topics = TOOL_TASKS if (a.tool_seed or a.multiturn) else PROMPTS
    rows = []
    for r in range(a.rounds):
        for i, topic in enumerate(topics):
            order = [("A", False), ("B", True)] if i % 2 == 0 else [("B", True), ("A", False)]
            for arm, flag in order:
                if a.multiturn:
                    messages = build_multiturn(TOPIC_NOUNS[i % len(TOPIC_NOUNS)])
                    text_prompt = None
                    opt_in = (arm == "B")
                    effort, mt = "medium", 900
                elif a.tool_seed:
                    messages = build_toolseed(topic, TOPIC_NOUNS[i % len(TOPIC_NOUNS)])
                    text_prompt = None
                    opt_in = (arm == "B")
                    effort, mt = "medium", 900
                elif a.seed_prefix:
                    text_prompt = topic if arm == "A" else SEED_BLOCK + "\n\n" + topic
                    messages = None
                    opt_in = False
                    effort, mt = None, None
                else:
                    text_prompt = topic
                    messages = None
                    opt_in = (arm == "B")
                    effort, mt = None, None
                t0 = time.time()
                text, finish, out_toks = call(a.base, text_prompt, opt_in, messages, effort, mt, a.temperature)
                rows.append({
                    "stamp": stamp, "round": r, "prompt": i, "arm": arm,
                    "thai_sampler": 1 if opt_in else 0,
                    "temperature": a.temperature,
                    "seeded": 1 if (a.seed_prefix and arm == "B") else 0,
                    "finish": finish, "out_toks": out_toks,
                    "content_len": len(text),
                    "broken": broken_count(text),
                    "thai_chars": thai_chars(text),
                    "seconds": round(time.time() - t0, 1),
                    "prompt_text": text_prompt,   # the row is auditable end to end
                    "prompt_messages": messages,
                    "effort": effort,
                    "text": text,
                })
                with open(out_path, "a", encoding = "utf-8") as fh:
                    fh.write(json.dumps(rows[-1], ensure_ascii = False) + "\n")
                time.sleep(SLEEP)

    by_arm = {}
    for arm in ("A", "B"):
        sel = [x for x in rows if x["arm"] == arm]
        broken = sum(x["broken"] for x in sel)
        thai = sum(x["thai_chars"] for x in sel)
        by_arm[arm] = (broken, thai, round(broken * 1000 / max(thai, 1), 2))
    print(json.dumps({"results": out_path, "arms": by_arm}, ensure_ascii = False, indent = 2))


if __name__ == "__main__":
    sys.exit(main())
