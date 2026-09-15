# Claude Code Spark1.3 — problem inventory (2026-09-15 → 2026-09-16)

**Purpose.** Everything that actually went wrong during the `✻ Waiting for API
response` investigation, in one place: symptom, root cause, the measurement that
proved it, the fix, and the status. Written after the incident closed, from the
logs and the evidence in the sibling reports — not from memory.

**TL;DR (English).** The banner had **three stacked root causes, all inside our
own wrapper** — a `read()` that withheld every response under 64 KiB until EOF,
a `ping` keepalive that Claude Code's stall tracker ignores, and a stream whose
body was never terminated at the HTTP layer. Two attribution errors were made
and retracted along the way (CORRECTIONS 50, 51). Nine further defects were
found in our own tooling by the verification work itself.

| # | อาการที่เห็น | รากเหตุ | สถานะ |
| --- | --- | --- | --- |
| P1a | banner + คำตอบมาช้ามาก | `read(65536)` กัก stream ไว้จนจบ | แก้ + ยืนยัน |
| P1b | banner ระหว่าง thinking | `ping` ไม่รีเซ็ต stall tracker ของ client | แก้ + ยืนยัน |
| P1c | **banner ทุกเทิร์น แม้คำตอบมาแล้ว** | response ไม่เคยจบที่ชั้น HTTP | แก้ + ยืนยัน |
| P2 | client ทิ้ง stream แล้ว retry non-stream | SSE ผิด protocol (message_start ซ้ำ, block ไม่ปิด) | แก้ + ยืนยัน |
| P3 | permission ตัดสินช้า 4-7s ต่อครั้ง | classifier ขอ model `claude-sonnet-5` → 400 | **เปิดอยู่** |
| P4 | client abort/timeout ระหว่างรอ | idle deadline สั้น + ความเงียบจริงจาก upstream | บรรเทา |
| P5 | request ล้มเป็นครั้งคราว | `APIConnectionError` จาก upstream | ผ่านเอง (มี heal) |
| P6-P14 | บั๊กในเครื่องมือ/การทดสอบของเราเอง | ดูหัวข้อ 4 | แก้ทั้งหมด |

---

## 1. ลำดับเวลา (ย่อ)

| เวลา (local) | เหตุการณ์ |
| --- | --- |
| 2026-09-15 เย็น | ตั้ง profile `claude-spark1.3` (wrapper :4003 → LiteLLM :4002 → Zen Go), เจอ SSE ผิด protocol → retry non-stream (P2) |
| ~22:50 | เขียน observability: timeline กลาง, incident + snapshot, trigger ทันทีเมื่อ client ขึ้น stall |
| 2026-09-16 ~00:10 | trigger ยิงจริงจาก session ผู้ใช้ → เริ่มไล่จากหลักฐาน ไม่ใช่จากทฤษฎี |
| ~00:20 | เจอ P1a (`read` vs `read1`) — first byte 38524 ms → 2502 ms |
| ~01:00 | เจอ P1b (`ping` ไม่รีเซ็ต tracker) → ฉีด thinking heartbeat; ทดสอบกับ client จริง |
| ~01:20 | ทำ auto-remediation (nudge / retry / restart) + ทดสอบกับ upstream ที่จงใจพัง |
| ~01:33 | เจอ P1c (body ไม่เคยจบ) → HTTP/1.1 + chunked + 0-chunk |
| ~01:35 | `claude -p` ออกเองใน 14s, `[Stall]` = 0 → ผู้ใช้ยืนยันว่า banner หาย |
| ~01:45 | ปิด issue #87, เก็บกวาดไฟล์ทดสอบ, เขียน report นี้ |

---

## 2. ปัญหาที่ผู้ใช้เห็น (client-visible)

### P1a — ทุก response เล็กกว่า 64 KiB ถูกส่งลง client ตอนจบเท่านั้น

- **อาการ:** `✻ Waiting for API response` + คำตอบมาช้า (first byte ~38s)
- **รากเหตุ:** `spark13_wrap.py` อ่าน upstream ด้วย `upstream.read(65536)`.
  `BufferedReader.read(n)` **บล็อกจนได้ n bytes หรือ EOF** จึงคายข้อมูลออกมา
  ครั้งเดียวตอน response จบ (`read1(n)` คืนทันทีที่มีข้อมูล)
- **หลักฐาน:** request เดียวกันกับ `:4002` — `read(65536)` อ่าน 1 ครั้งที่
  **4.26 s / 1093 bytes**; `read1(65536)` อ่าน **5 ครั้ง ตั้งแต่ 1.27 s**
  ฝั่ง client: `first byte after 38524ms` → `2502ms` (req `w-74c5cae9`)
- **แก้:** ใช้ `read1(65536)` (มี fallback ถ้า object ไม่มี `read1`)
- **ยืนยัน:** first byte median 1953 ms (min 1453 / max 7391) จาก 90 stream หลัง deploy
- **สถานะ:** ✅

### P1b — keepalive ที่ client ใช้ไม่ได้ (`ping`)

- **อาการ:** banner ระหว่างช่วง model คิด (20-40s)
- **รากเหตุ:** 2 อย่างประกอบกัน — (1) Zen Go ส่ง reasoning มาเป็น **item เดียวจบ**
  (`response.output_item.done`) ไม่ใช่ delta ต่อเนื่อง จึงไม่มี byte ไหลออกระหว่างคิด
  (2) `event: ping` **ไม่รีเซ็ต stall tracker ของ Claude Code** (dispatcher ทิ้ง
  event ที่ `type=="ping"`), และ `bytesTotal` ยังนับ byte ของ ping ด้วย → tracker
  ยิงที่ 15s/30s ในขณะที่เรา "ping อยู่ตลอด"
- **หลักฐาน:** blackhole upstream ที่เงียบ 78s — heartbeat **ON = 0 tick**,
  heartbeat **OFF = 3 tick** (15s/30s/45s) กับ client จริง (`claude -p`)
- **แก้:** ฉีด `content_block_delta {thinking_delta, ""}` (0 ตัวอักษร) เข้า
  thinking block ที่เปิดอยู่ทุก 10s ที่ upstream เงียบ (และยิงทันทีเมื่อ watcher เห็น stall)
- **ยืนยัน:** 78s เงียบ = 0 tick; production หลัง deploy ฉีดจริง 29 ครั้ง ระหว่างที่ผู้ใช้ทำงาน
- **สถานะ:** ✅ (ยังเป็น safety net ที่จำเป็น เพราะ upstream เงียบจริง)

### P1c — response ไม่เคย "จบ" ที่ชั้น HTTP (ต้นเหตุจริงของ banner ที่ทนทุกการแก้)

- **อาการ:** `[Stall] stream_idle_partial` ยิง **ทุกเทิร์น** 15/30/60/120 วิ
  โดย `bytesTotal` คงที่ แม้คำตอบและ tool จะจบแล้ว — ผู้ใช้เห็น banner ค้าง
- **รากเหตุ:** wrapper ส่ง
  `Content-Type: text/event-stream`, `Cache-Control: no-cache, no-transform`,
  `Connection: keep-alive` และ **ไม่มี `Content-Length` และไม่มี
  `Transfer-Encoding`** → body ไม่มีจุดจบ + keep-alive ⇒ client ไม่มีสัญญาณ
  end-of-body เลย จึงคา stream ไว้และ tracker ยิงต่อ
- **หลักฐาน:**
  - log client: tick ที่ 15/30/60/120s ด้วย `bytesTotal` เดิม (35 → 1233 → 1494 → 3610 …)
  - raw wire จาก `:4003` ก่อนแก้: ไม่มี terminator; หลังแก้: `HTTP/1.1 200 OK` +
    `Transfer-Encoding: chunked` ปิดด้วย 0-chunk
  - client จริง: `claude -p` **ไม่ออกเอง ถูก kill ที่ 150s** + 3-4 tick
    → **ออกเองใน 14s + 0 tick** หลังแก้
- **แก้:** `protocol_version = "HTTP/1.1"`, `Transfer-Encoding: chunked`,
  เขียน SSE เป็น chunk ละก้อน, ส่ง 0-chunk ครั้งเดียวจาก `finally` ของ relay
- **ยืนยัน:** หลัง deploy — 60 requests / 41 model streams / 90 stream.end /
  45 terminator / **0 tick** จาก 2,634 บรรทัด log ฝั่ง client
- **สถานะ:** ✅ (ปิด issue #87 ด้วยเกณฑ์ end-to-end user path ผ่าน)

### P2 — SSE ผิด protocol ทำให้ client ทิ้ง stream

- **อาการ:** client retry เป็น non-streaming, ขึ้น banner
- **รากเหตุ:** LiteLLM ส่ง `message_start` ซ้ำ 2 ครั้ง และเปิด thinking block
  แต่ไม่ปิดก่อน `message_delta`
- **หลักฐาน:** wire probe บน `:4002` (event sequence) ก่อนมีตัวซ่อมใน wrapper
- **แก้:** wrapper normalize — dedupe `message_start`, ปิด block ที่ค้างอัตโนมัติ
  (นับเป็น `protocol.fix` ใน timeline)
- **สถานะ:** ✅

### P3 — classifier ของ Claude Code ขอ model ที่ upstream ไม่มี (เปิดอยู่)

- **อาการ:** ตัดสิน permission ช้า ~4-7s ต่อครั้ง
- **รากเหตุ:** Claude Code เรียก auto-mode classifier ด้วย model `claude-sonnet-5`
  → LiteLLM ตอบ `ProxyModelNotFoundError: 400: anthropic_messages: Invalid model
  name passed in model=claude-sonnet-5` → fallback ไป muse-spark แล้วจึงสำเร็จ
- **หลักฐาน:** `call.failure` / `post.failure` / `upstream.http_error status=400`
  ที่ 01:33:44 และ 01:40:09; log client มี
  `classifier_request_started ... model=claude-sonnet-5 outcome=error` แล้ว
  ตามด้วย `model=muse-spark-1.3-contributor outcome=ok`
- **แนวทางแก้:** pin model ของ classifier ใน settings/profile ให้ชี้ muse-spark
- **สถานะ:** ⏳ ยังไม่แก้ (แยกจาก banner; รอเจ้าของงานอนุมัติ)

### P4 — client abort/timeout ระหว่างรอ

- **รากเหตุ:** stall deadline เดิมสั้น + ความเงียบจริงจาก upstream
- **หลักฐาน:** log client `idleDeadlineMs` และการ retry non-streaming ก่อนแก้
- **แก้/บรรเทา:** ยก idle deadline เป็น 30 นาที + P1b/P1c ทำให้ไม่มีช่วงเงียบที่ client มองเห็น
- **สถานะ:** ✅ (บรรเทา)

### P5 — `APIConnectionError` เป็นครั้งคราวจาก upstream

- **หลักฐาน:** `lite call.failure APIConnectionError` ที่ 01:36:20 (model muse-spark)
- **แก้:** มี retry/auto-heal ครอบอยู่แล้ว (ครั้งนี้ผ่านไปเองโดยไม่ต้อง retry)
- **สถานะ:** ✅ (เฝ้าดูต่อ)

---

## 3. สิ่งที่ระบบทำเองได้แล้ว (auto-remediation)

| trigger | action | หลักฐานการทดสอบ |
| --- | --- | --- |
| client ขึ้น `[Stall]` | watcher ยิง `POST /__spark13/heartbeat` → wrapper ฉีด heartbeat ทันที | nudge ที่ 10s ระหว่าง thinking เงียบ 60s → `nudged: 1`, client ได้ delta เพิ่ม |
| upstream เงียบ ≥20s และยังไม่มี output ลง client | ทิ้ง attempt แล้วยิง request เดิมซ้ำ (สูงสุด 2 ครั้ง) | attempt 1 เงียบ → attempt 2 ตอบ → client ได้ข้อความครบ |
| upstream 429/5xx และยังไม่มี output | รอ 2s แล้วยิงซ้ำ | 503 → สำเร็จ ได้ข้อความครบ |
| wrapper/LiteLLM ไม่ listening | restart ผ่าน `spark13_ensure.py` + ตรวจ port ซ้ำ | kill จริง → กลับมา :4003 ใน **3.5s**, `heal.restart_result listening: true` |

กติกา: retry เฉพาะเมื่อ `downstream_events == 0` (ไม่มีทางข้อความซ้ำ/สลับ),
แต่ละ attempt มี queue ของตัวเอง, restart จำกัด 1 ครั้ง/component/60s

---

## 4. ปัญหาในเครื่องมือของเราเอง (เจอเพราะงาน verify)

| # | ปัญหา | อาการที่ทำให้สับสน | แก้ |
| --- | --- | --- | --- |
| P6 | `HTTPResponse.close()` **deadlock** เพราะ reader thread ถือ lock ของ BufferedReader ขณะติดใน `read1()` | retry ไม่เกิดเงียบๆ; relay ค้าง `attempts=2` แต่ไม่มี `heal.retry_start`; health โชว์ in-flight ค้างตลอด | ยกเลิก attempt ด้วย `socket.shutdown(SHUT_RDWR)` |
| P7 | wrapper ไม่ได้ import `socket` | `heal.abandon_shutdown_failed: NameError` | เพิ่ม import |
| P8 | watcher ไม่ได้ import `json` | `/__spark13/health` และ nudge ยิงไม่สำเร็จ, `heal.request_failed` ทุก 5s | เพิ่ม import |
| P9 | cooldown ของ incident ไปบล็อกการแก้ไขด้วย | stall tick เกิดแต่ nudge ไม่ยิง (`trigger.suppressed`) | แยก: incident rate-limit ได้ แต่ remediation ยิงทุกครั้ง |
| P10 | test harness ปิด socket ทั้งที่ยังมี request ค้างใน buffer → Windows ส่ง RST | ดูเหมือน upstream พัง (ConnectionResetError) ทั้งที่โค้ดดี | อ่าน request ให้ครบ (Content-Length) + shutdown แบบ graceful |
| P11 | heredoc ทำให้ `\r\n` ใน byte literal กลายเป็น CR/LF จริง | ไฟล์ syntax error, แก้ผิดรอบ | สร้าง byte จาก `bytes((13,10))` แทน escape |
| P12 | watcher glob ไฟล์ debug ของการทดสอบเอง | incident ปลอมจากไฟล์เทสต์ (`acid.txt`, `proof.txt`) | ลบไฟล์เทสต์ออกจาก `logs/cc-debug/`, เทสต์เขียนลง scratch dir |
| P13 | self-test import module แล้ว `record_pid` ทับ pid ของ LiteLLM จริง | `ensure status` รายงาน pid ผิด (alive=false) | guard `record_pid` เมื่อรันเป็น `__main__` |
| P14 | attribution ผิด 2 ครั้งของผมเอง | สรุปว่า "LiteLLM flush เป็นชุด" และ "tick ท้ายเป็น quirk ของ client" | CORRECTIONS 50 และ 51 (บันทึกไว้ทั้งสองครั้ง) |

---

## 5. วิธีตรวจซ้ำ (ไม่ต้องเดา)

```sh
# 1) framing ของ response ต้องเป็น HTTP/1.1 + chunked และปิดด้วย 0-chunk
python3 - <<'PY'
import json, socket, time
body = json.dumps({"model":"muse-spark-1.3-contributor","max_tokens":16,"stream":True,
                   "messages":[{"role":"user","content":"say hi"}]}).encode()
s = socket.create_connection(("127.0.0.1", 4003), timeout=60)
s.sendall(b"POST /v1/messages HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
          b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
raw = b""
while not raw.endswith(bytes((48,13,10,13,10))):
    raw += s.recv(65536)
print(raw.split(bytes((13,10,13,10)))[0].decode())   # -> HTTP/1.1 200 + chunked
PY

# 2) สถานะ + in-flight + heal config
curl -s http://127.0.0.1:4003/__spark13/health

# 3) trigger/incident ล่าสุด (ต้องไม่มี client.stall_partial ใหม่)
python3 - <<'PY'
import json, os, pathlib
p = pathlib.Path(os.path.expanduser("~/.claude/logs/spark13-events.jsonl"))
rows = []
for l in p.read_text(encoding="utf-8", errors="replace").splitlines():
    try: rows.append(json.loads(l))
    except Exception: pass
tr = [r for r in rows if r.get("event") == "trigger"]
print(len(tr), "triggers; last 5:", [(r["ts"][11:19], r["kind"]) for r in tr[-5:]])
PY

# 4) ฝั่ง client: ต้องไม่มี [Stall] stream_idle หลัง deploy (0 = ผ่าน)
awk '/Stall] stream_idle/ && $0 > "2026-09-15T18:34"' ~/.claude/logs/cc-debug/spark13.txt | wc -l
```

**เกณฑ์ปิดงาน:** end-to-end user path ผ่าน (ผู้ใช้ยืนยันว่าไม่เห็น banner) +
`[Stall] stream_idle` = 0 หลัง deploy — ไม่ปิดจาก health/wire smoke

---

## 6. ยังเปิดอยู่

1. **P3** — classifier ขอ `claude-sonnet-5` → 400 ทุกครั้งที่ตัดสิน permission
   (เสียเวลา 4-7s/ครั้ง) ต้อง pin model
2. **ข้อจำกัดที่ตั้งใจ:** request ที่ upstream เงียบ **หลังจาก** มี content ออกไปแล้ว
   retry ไม่ได้ (client เห็นคำตอบบางส่วนแล้ว) — คุมด้วย heartbeat เท่านั้น
   บันทึกไว้ในรายงาน ไม่ใช่บั๊กที่แก้ได้ด้วย retry

---

## 7. บทเรียน

1. **อ่าน framing จริงบน wire ก่อนโทษฝั่งตรงข้าม** — "client quirk" เป็นข้อสรุปที่
   มาจาก observation ที่ *ขาด* (เราไม่เคยดู bytes ที่ตัวเองส่งออกไป)
2. **keepalive ต้องเป็น event ที่คู่สนทนายอมรับ** — ping ธรรมดาไม่รีเซ็ต watchdog
   ของ peer; effect วัดจากฝั่งผู้ใช้ ไม่ใช่วัดจากว่ายิงสำเร็จหรือไม่
3. **การ retry ต้องมีกติกาป้องกันความเสียหาย** — เฉพาะเมื่อยังไม่มี output ลง client
   และแยก queue ต่อ attempt เพื่อไม่ให้ attempt ที่ถูกทิ้งปนกับอันใหม่
4. **`close()` ไม่ใช่ทางออกของ blocking read** — ใช้ `socket.shutdown` และเข้าใจว่า
   lock ของ buffer อยู่ที่ใคร
5. **ไฟล์เทสต์ต้องไม่ปนกับ log จริง** — ไม่งั้น instrumentation ของเราเองสร้าง
   incident ปลอมและทำให้ debug หลงทาง
