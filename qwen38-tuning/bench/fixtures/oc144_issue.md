# #144 fix(clink): own the whole process tree, so cancellation means nothing is still writing (Deliverable 1 of #20)

> **Deliverable 1 of the PRD #20.** The first of the two un-gated bug fixes — it depends on no spike and on no session machinery, and #20 orders it first because *"without this, every other guarantee below is unenforceable"*.

# EN

## Problem

`clink` spawns a subagent and never owns what that subagent starts.

`clink/agents/base.py:301` calls `asyncio.create_subprocess_exec(...)` with **no `start_new_session=True` (POSIX) and no `creationflags` (Windows)**. The child therefore shares the server's process group, and its own descendants — the shells, language servers and helper processes a coding CLI spawns — belong to nothing the server tracks.

`clink/agents/base.py:319` then calls `process.kill()`, which signals **the direct child only**. Every descendant survives, still holding the working directory open and still able to write to it.

So today "the run was killed" means "we killed the process we happened to have a handle on". #20 story 7 asks for the opposite: *"I want cancellation to terminate every process the subagent started, so that 'cancelled' means nothing is still writing to my repository."*

This is not hypothetical on this repo: `clink` runs Codex and Antigravity with bypass-approvals/sandbox flags against the live tree, so an orphan is an orphan with write access.

## What to build

Spawn each subagent into a **process group** (POSIX) or a **Job Object** (Windows), and terminate the group rather than a PID.

Termination must be **confirmed**, not merely requested — #20 story 8: *"I want cancellation to be acknowledged only after the operating system confirms the process tree is gone, so that I do not proceed while an orphan is mid-write."*

The PTY path is a second spawn site with the same gap: `clink/agents/antigravity.py:190` uses `winpty.PtyProcess.spawn(...)` and its loop at `:194-200` breaks on its own timeout. It must join the same ownership model — #20 story 35 asks for uniform lifecycle across pipe and PTY runners rather than a per-runner accident.

## Acceptance criteria

- [ ] A subagent is spawned into its own process group (POSIX) or Job Object (Windows)
- [ ] Termination targets the group, never a lone PID
- [ ] A descendant spawned by the subagent is gone after cancellation — asserted by a test that spawns a child-of-child and observes it exit
- [ ] Cancellation is acknowledged only after the OS confirms the tree is gone
- [ ] The PTY runner participates in the same ownership model as the pipe runner
- [ ] Cleanup runs on every exit path — success, failure, cancellation and server shutdown (#20 story 32)

## Change inventory

Surveyed 2026-08-19 against `main`; #20's own citations had drifted and these are re-verified.

| Path | What changes | How it is verified |
|---|---|---|
| `clink/agents/base.py:301` | `create_subprocess_exec` gains process-group / Job Object creation | a test asserting a grandchild dies with the group |
| `clink/agents/base.py:319` | `process.kill()` becomes a group termination that waits for confirmation | same test, plus one asserting the call does not return early |
| `clink/agents/antigravity.py:190` | `PtyProcess.spawn` joins the same ownership model | a PTY-path test, skipped where `pywinpty` is absent |
| `tests/test_clink_*.py` | the fake process gains a spawnable descendant | the tests above are the verification |

**Search boundary:** `grep -n "create_subprocess_exec\|start_new_session\|creationflags\|process_group" clink/agents/*.py` returned exactly the one site above. Anything spawning through a path built at runtime would not appear.

## Blocked by

Nothing. This is one of the two un-gated fixes in #20 and can start immediately.

---

# TH

> **Deliverable 1 ของ PRD #20** · หนึ่งในสองการแก้บั๊กที่ไม่ติดประตูอะไร — ไม่ขึ้นกับ spike และไม่ขึ้นกับกลไก session · #20 จัดมันไว้ก่อนเพราะ *"ถ้าไม่มีอันนี้ การรับประกันอื่นทุกข้อข้างล่างบังคับใช้ไม่ได้"*

## ปัญหา

`clink` spawn subagent ขึ้นมาแล้วไม่เคยเป็นเจ้าของสิ่งที่ subagent นั้นสร้างต่อ

`clink/agents/base.py:301` เรียก `asyncio.create_subprocess_exec(...)` โดย **ไม่มี `start_new_session=True` (POSIX) และไม่มี `creationflags` (Windows)** ⇒ child ใช้ process group เดียวกับ server และลูกหลานของมันเอง — shell, language server และ process ผู้ช่วยที่ coding CLI สร้างขึ้น — ไม่สังกัดอะไรที่ server ติดตามอยู่เลย

จากนั้น `clink/agents/base.py:319` เรียก `process.kill()` ซึ่งส่งสัญญาณไปที่ **child โดยตรงเท่านั้น** · ลูกหลานทุกตัวรอด ยังเปิดค้าง working directory ไว้ และยังเขียนลงไปได้

⇒ วันนี้ "run ถูก kill แล้ว" แปลว่า "เรา kill process ที่เราบังเอิญถือ handle อยู่" · #20 story 7 ขอตรงข้าม: *"ผมอยากให้การยกเลิกจบทุก process ที่ subagent สร้างขึ้น เพื่อให้ 'ยกเลิกแล้ว' แปลว่าไม่มีอะไรกำลังเขียนลง repository ของผมอยู่"*

เรื่องนี้ไม่ใช่สมมติใน repo นี้: `clink` รัน Codex และ Antigravity ด้วย flag bypass-approvals/sandbox บน tree จริง ⇒ orphan คือ orphan ที่มีสิทธิ์เขียน

## สร้างอะไร

spawn subagent แต่ละตัวเข้า **process group** (POSIX) หรือ **Job Object** (Windows) และจบการทำงานที่ระดับ group แทนที่จะเป็น PID เดี่ยว

การจบต้องถูก**ยืนยัน** ไม่ใช่แค่ร้องขอ — #20 story 8: *"ผมอยากให้การยกเลิกถูกตอบรับหลังจากระบบปฏิบัติการยืนยันแล้วว่า process tree หายไปแล้วเท่านั้น เพื่อที่ผมจะไม่เดินต่อขณะที่ orphan กำลังเขียนอยู่กลางคัน"*

เส้นทาง PTY เป็นจุด spawn ที่สองที่มีช่องว่างเดียวกัน: `clink/agents/antigravity.py:190` ใช้ `winpty.PtyProcess.spawn(...)` และลูปที่ `:194-200` ออกด้วย timeout ของตัวเอง · มันต้องเข้าร่วมโมเดลความเป็นเจ้าของเดียวกัน — #20 story 35 ขอให้ lifecycle เหมือนกันทั้ง pipe และ PTY runner แทนที่จะเป็นความบังเอิญราย runner

## เกณฑ์การรับงาน

- [ ] subagent ถูก spawn เข้า process group ของตัวเอง (POSIX) หรือ Job Object (Windows)
- [ ] การจบการทำงานเล็งไปที่ group ไม่ใช่ PID เดี่ยว
- [ ] ลูกหลานที่ subagent สร้างขึ้นหายไปหลังการยกเลิก — ยืนยันด้วยเทสต์ที่ spawn ลูกของลูกแล้วสังเกตว่ามันจบ
- [ ] การยกเลิกถูกตอบรับหลัง OS ยืนยันว่า tree หายไปแล้วเท่านั้น
- [ ] PTY runner เข้าร่วมโมเดลความเป็นเจ้าของเดียวกับ pipe runner
- [ ] การเก็บกวาดทำงานบนทุกเส้นทางออก — สำเร็จ ล้มเหลว ยกเลิก และ server ปิดตัว (#20 story 32)

## รายการจุดที่เปลี่ยน

สำรวจ 2026-08-19 เทียบกับ `main` · การอ้างอิงของ #20 เองเคลื่อนไปแล้ว และรายการนี้ตรวจสอบใหม่

| Path | เปลี่ยนอะไร | ยืนยันอย่างไร |
|---|---|---|
| `clink/agents/base.py:301` | `create_subprocess_exec` เพิ่มการสร้าง process group / Job Object | เทสต์ที่ assert ว่าหลานตายไปพร้อม group |
| `clink/agents/base.py:319` | `process.kill()` กลายเป็นการจบระดับ group ที่รอการยืนยัน | เทสต์เดียวกัน บวกอีกตัวที่ assert ว่าการเรียกไม่คืนค่าก่อนเวลา |
| `clink/agents/antigravity.py:190` | `PtyProcess.spawn` เข้าร่วมโมเดลความเป็นเจ้าของเดียวกัน | เทสต์เส้นทาง PTY ข้ามเมื่อไม่มี `pywinpty` |
| `tests/test_clink_*.py` | process ปลอมเพิ่มความสามารถ spawn ลูกหลาน | เทสต์ข้างบนคือการยืนยัน |

**ขอบเขตการค้นหา:** `grep -n "create_subprocess_exec\|start_new_session\|creationflags\|process_group" clink/agents/*.py` คืนจุดเดียวข้างบนพอดี · อะไรที่ spawn ผ่านเส้นทางที่ประกอบขึ้นตอน runtime จะไม่ปรากฏ

## ติดอะไรอยู่

ไม่ติดอะไร · นี่คือหนึ่งในสองการแก้ที่ไม่ติดประตูใน #20 และเริ่มได้ทันที
