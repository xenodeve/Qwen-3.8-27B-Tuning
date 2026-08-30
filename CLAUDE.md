# CLAUDE.md — how to work in this repository

<!-- lang:en -->

This repo is a **measurement project**, not a product. It exists to find the
fastest usable configuration for Qwen3.8-27B on one **RTX 5060 Ti 16 GB**
(Blackwell, from 2026-08-23 -- it replaced an RTX 4070 SUPER 12 GB, and
**every number recorded before that date belongs to the old card**:
`docs/results/09-hardware.md`), and
to record what was measured well enough that a later reader can tell a result
from a guess.

**The agent is the primary engineer here.** Every file below exists so a fresh
session can recover state without re-deriving it.

---

## Session start — read these, in this order

1. **`docs/OPEN-WORK-LEDGER.md`** — what is open, including MD-only items that
   no issue tracks. The 🔴 UNTRACKED rows are the highest miss-risk.
2. **`docs/reports/CORRECTIONS.md`** — forty-three claims this project published
   and later contradicted with its own data. **Read it before quoting any number.**
3. **`docs/agents/traps.md`** — the ways of WORKING that failed here. Corrections
   tells you which figures to distrust; this tells you which of your own
   instincts to. **Fifteen of its nineteen traps produced a plausible number rather
   than an error.**
4. **`docs/results/README.md`** — the register: has X been tried, what happened.
5. **`docs/reports/39-OPTIMISATION-GUIDE.md`** — **read this before proposing any
   speed work.** Eleven levers already settled and not to be re-tested, six
   one-flag sweeps ranked by cost, the trades that need a decision rather than a
   number, and six ways to waste a day — each one something that actually
   happened here. Every line is tagged **MEASURED HERE** with its source file,
   **VENDOR** for an outside claim, or **UNMEASURED**; an untagged number is the
   thing to distrust.
6. **`docs/reports/START-HERE.md`** — the narrative, if you need the why.
7. The specific GitHub issue you are picking up: `gh issue view <n> --comments`.

`DONE.md` and `docs/reports/04-MEASUREMENT-METHODOLOGY.md` §7–§8 only when the
task needs history or you are about to run a benchmark.

---

## The engineering north star

**An instrument that returns a believable number instead of a failure is worse
than one that crashes.** Thirteen documented instrument faults in this project
each produced a plausible wrong figure, and several were published before anyone
noticed. So:

- **No verdict before evidence.** A measurement names the file its number came
  from, or it is a hypothesis and says so.
- **Never compare raw decode across boots.** The spread is measured and the
  **cause is unknown** — do not repeat the old explanation that `--fit` follows
  the boot VRAM: llama.cpp has reported **11,069 MiB free in all 552 logs**, and
  148 of 150 boots on our artifact say *"no changes needed"* (`CORRECTIONS.md`
  §27). **Effects below 13.6 % are noise** — **at ctx 16,384, where that floor was measured.** At 65,536 the same arm with byte-identical counters spans up to **48.9 %** across boots, so re-derive before using it at depth (`CORRECTIONS.md` §23).
  Pair within a round, alternate the order.
- **A verdict at one depth does not transfer to another.** `draft-mtp` is +81 %
  at 16K and −71 % at 131,072 on the same artifact.
- **Retracting is part of the work.** When a measurement contradicts something
  already written, the retraction is not finished until the claim has an entry
  in `docs/reports/CORRECTIONS.md` **and** a rule in
  `scripts/audit-stale-claims.py`.

---

## Repo layout

```text
docs/reports/    findings, numbered — narrative, dated, argues from evidence
docs/results/     the register — has X been tried, what happened
docs/plans/      intent, not results
docs/researchs/  external material, unverified until measured here
docs/agents/     the operating standard: domain, tracker, labels, workflow,
                 and traps.md — the ways of working that failed here
scripts/         tools for the documentation map itself
qwen38-tuning/   the apparatus — bench, scripts, results, grammars
```

Every folder has a `README.md` that says what is in it and what to read first.

## Commands

```powershell
cd qwen38-tuning\bench ; python -m pytest tests\ -q    # 329 tests — the gate
python scripts\check-doc-links.py                      # every link resolves
python scripts\audit-stale-claims.py                   # superseded claims
```

**Run the test gate before trusting any measurement.** A broken instrument
returns a number instead of a failure.

**Two orchestrators cannot share port 8080.** `qwen38-tuning/scripts/swap-model.sh`
takes a lock. An armed queue once killed a running corpus and the summary still
printed a plausible number.

---

## `using-t4` is a standing default, not a pointer

Invoke it at session start **and re-route at every phase boundary** — it is the
map from a task to the skill that owns it:

- after writing code → **`simplify`**
- before merge → **`code-review`** + **`scrutinize`**
- touched auth, secrets or anything that binds a boundary → **`security-review`**
- a run behaves strangely → **`systematic-debugging`**
- before claiming a speedup → **`verification-before-completion`**

**A check at task start does not discharge a later trigger.** Reading the map
once and never returning to it is the one behaviour the map forbids of itself.

`karpathy-guidelines` applies to code written here; `t4-bro` governs every reply
to the developer — Thai chat, English reports, prose by default.

## Delegation

**`clink-subagents` is the delegation default.** The orchestrator's context is
the scarce resource here and the clink back-ends bill against flat
subscriptions, so delegating is the normal move rather than the optimisation.
Two rules do not relax:

- **Verify everything a subagent returns.** A report is a hypothesis until
  checked. In this repo's own history a delegated run reported a task complete
  while its file sat in the wrong directory, and only executing the code found
  it.
- **Never delegate the final verification**, and never delegate a change to
  something you cannot check.

**`clink-masteragent`: invoke it before any `clink` call.** Decided 2026-08-21.
Not loaded at session start — most sessions here never delegate, and its ~19 KB
against `using-t4`'s 9 KB ceiling is not worth paying on those. It owns what may
never be handed out and picks the model from the score table rather than from
memory.

---

## Writing conventions

**Chat is Thai. Reports, code, and commit messages are English.** Identifiers
stay English and byte-exact — a path, a flag, a config key is something the
developer copies and searches.

**GitHub issue, PRD and PR bodies are bilingual**: English plus a full Thai
mirror of the same depth. A mirror is not a summary. This applies to the tracker
only, never to `docs/`.

## Shipping

**PRD → issues → PR. Never a PR without a referenced issue.** Issues are the
source of truth; the ledger is the discovery index over them.

**TDD is mandatory** for anything in `qwen38-tuning/bench/`. Every primitive
there was written red-first and each test names the incident it guards — read
`bench/tests/test_harness.py` before adding one.

**Close an issue with a stated reason and evidence** — a commit, a test, a
measured number. Never silently.

## Notifying the developer

Long unattended runs are normal here; a queue can hold the GPU for hours. Notify
on **a batch finishing** or **a decision that blocks progress** — not on routine
sub-progress. One digest at the end, enumerating what ran and what did not.

<!-- lang:th -->

---

# CLAUDE.md — วิธีทำงานใน repository นี้

repo นี้เป็น **โปรเจกต์วัดผล** ไม่ใช่ผลิตภัณฑ์ มีไว้เพื่อหา config ที่เร็วที่สุด
ที่ใช้งานได้จริงของ Qwen3.8-27B บน **RTX 5060 Ti 16 GB** ใบเดียว
(Blackwell ตั้งแต่ 2026-08-23 มาแทน RTX 4070 SUPER 12 GB — **ตัวเลขทุกตัวที่บันทึกก่อนวันนั้นเป็นของการ์ดใบเก่า**: `docs/results/09-hardware.md`) และเพื่อบันทึกสิ่ง
ที่วัดได้ดีพอที่คนอ่านทีหลังจะแยกออกว่าอันไหนคือผลจริงอันไหนคือการเดา

**agent คือวิศวกรหลักที่นี่** ทุกไฟล์ข้างล่างมีไว้ให้ session ใหม่กู้สถานะได้โดยไม่ต้องหาเอง

## เริ่ม session — อ่านตามลำดับนี้

1. **`docs/OPEN-WORK-LEDGER.md`** — อะไรค้างอยู่ รวมของที่มีแต่ใน MD ไม่มี issue
   แถวที่ติด 🔴 UNTRACKED คือกลุ่มที่หลุดง่ายที่สุด
2. **`docs/reports/CORRECTIONS.md`** — ข้ออ้าง 43 ข้อที่โปรเจกต์นี้เผยแพร่แล้วหักล้าง
   ด้วยข้อมูลตัวเอง **อ่านก่อนยกตัวเลขไหนไปใช้**
3. **`docs/agents/traps.md`** — *วิธีทำงาน* ที่เคยพลาดที่นี่ CORRECTIONS บอกว่าตัวเลขไหน
   ห้ามเชื่อ ส่วนอันนี้บอกว่าสัญชาตญาณข้อไหนของตัวเองห้ามเชื่อ **สิบห้าจากสิบเก้ากับดักในนั้น
   คืนตัวเลขที่ดูสมเหตุสมผลออกมา ไม่ได้แจ้งความผิดพลาด**
4. **`docs/results/README.md`** — ทะเบียนว่าอะไรถูกทดสอบแล้ว ผลเป็นอะไร
5. **`docs/reports/39-OPTIMISATION-GUIDE.md`** — **อ่านก่อนเสนอเรื่องความเร็วทุกครั้ง**
   คันโยกสิบเอ็ดตัวที่จบแล้วและห้ามทดสอบซ้ำ · sweep แบบ flag เดียวหกอันเรียงตามต้นทุน ·
   การแลกที่ต้องตัดสินใจ ไม่ใช่แค่หาตัวเลข · และหกวิธีเสียเวลาทั้งวัน ซึ่งทุกข้อเคยเกิดขึ้นจริงที่นี่
   ทุกบรรทัดติดป้ายว่า **MEASURED HERE** พร้อมไฟล์ที่ตัวเลขมาจาก · **VENDOR** ถ้าเป็นคำอ้างของคนอื่น ·
   หรือ **UNMEASURED** — **ตัวเลขที่ไม่มีป้ายคือตัวที่ต้องไม่เชื่อ**
6. **`docs/reports/START-HERE.md`** — เรื่องเล่าทั้งหมด ถ้าต้องการรู้ว่าทำไม
7. issue ที่กำลังจะทำ: `gh issue view <n> --comments`

## หลักการหลัก

**เครื่องมือที่คืนตัวเลขน่าเชื่อแทนที่จะแจ้งความล้มเหลว แย่กว่าเครื่องมือที่พังไปเลย**
โปรเจกต์นี้มี instrument fault ที่บันทึกไว้ 13 ข้อ แต่ละข้อผลิตตัวเลขผิดที่ดูสมเหตุสมผล
และหลายข้อถูกเผยแพร่ไปก่อนที่ใครจะสังเกต ดังนั้น

- **ไม่มีคำตัดสินก่อนมีหลักฐาน** การวัดต้องระบุไฟล์ที่ตัวเลขมาจาก ไม่งั้นคือสมมติฐานและต้องบอกว่าเป็นสมมติฐาน
- **ห้ามเทียบ decode ดิบข้าม boot** ความแกว่งวัดได้จริงแต่**ยังไม่รู้สาเหตุ** — อย่าใช้คำอธิบายเดิม
  ที่ว่า `--fit` วิ่งตาม VRAM ตอน boot เพราะ llama.cpp รายงาน **11,069 MiB free เหมือนกันทั้ง 552 log**
  และ 148 จาก 150 boot บนไฟล์ของเราบอกว่า *"no changes needed"* (`CORRECTIONS.md` §27)
  **ผลต่ำกว่า 13.6 % คือสัญญาณรบกวน — ที่ ctx 16,384 ซึ่งเป็นความลึกที่เพดานนี้ถูกวัด** ที่ 65,536 arm เดียวกันที่ counter เท่ากันทุกหลักแกว่งได้ถึง **48.9 %** ต้องหาเพดานใหม่ก่อนใช้ที่ความลึก (`CORRECTIONS.md` §23) ให้จับคู่ในรอบเดียวกันและสลับลำดับ
- **คำตัดสินที่ความลึกหนึ่งไม่โอนไปอีกความลึก** `draft-mtp` ได้ +81 % ที่ 16K แต่ −71 % ที่ 131,072 บนไฟล์เดียวกัน
- **การถอนคำเป็นส่วนหนึ่งของงาน** เมื่อผลวัดขัดกับสิ่งที่เขียนไปแล้ว การถอนยังไม่จบ
  จนกว่าจะมีบรรทัดใน `docs/reports/CORRECTIONS.md` **และ** กฎใน `scripts/audit-stale-claims.py`

## คำสั่ง

```powershell
cd qwen38-tuning\bench ; python -m pytest tests\ -q    # 329 test — ด่านหลัก
python scripts\check-doc-links.py                      # ลิงก์ทุกเส้นต้องไปถึง
python scripts\audit-stale-claims.py                   # ข้ออ้างที่ถูกแทนที่แล้ว
```

**รันด่าน test ก่อนเชื่อผลวัดใด ๆ** และ **สอง orchestrator ใช้ port 8080 ร่วมกันไม่ได้**
`qwen38-tuning/scripts/swap-model.sh` ถือ lock ไว้ — เคยมีคิวที่ตั้งไว้ไปฆ่า corpus ที่กำลังรัน
แล้วสรุปผลก็ยังพิมพ์ตัวเลขที่ดูสมเหตุสมผลออกมา

## `using-t4` เป็นค่าตั้งต้น ไม่ใช่แค่ตัวชี้

เรียกตอนเริ่ม session **และเรียกใหม่ทุกครั้งที่ข้ามช่วงงาน** — หลังเขียนโค้ดเรียก
**`simplify`** · ก่อน merge เรียก **`code-review`** กับ **`scrutinize`** · ถ้าแตะ auth
หรือความลับเรียก **`security-review`** · ถ้างานมีอาการแปลกเรียก **`systematic-debugging`**
· ก่อนเคลมว่าเร็วขึ้นเรียก **`verification-before-completion`**

**การเช็คตอนเริ่มงานไม่ได้ปลดล็อกทริกเกอร์ที่มาทีหลัง** การอ่านแผนที่ครั้งเดียวแล้วไม่กลับมาอีก
คือพฤติกรรมเดียวที่แผนที่นั้นห้ามไว้กับตัวเอง

`karpathy-guidelines` ใช้กับโค้ดที่เขียนที่นี่ · `t4-bro` คุมทุกคำตอบที่ส่งให้ developer
— แชทไทย รายงานอังกฤษ ร้อยแก้วเป็นค่าตั้งต้น

## การมอบงาน

**`clink-subagents` คือค่าตั้งต้นของการมอบงาน** เพราะ context ของตัว orchestrator คือ
ทรัพยากรที่หายากที่นี่ และ back-end ของ clink คิดค่าใช้จ่ายจาก subscription แบบเหมา
การมอบงานจึงเป็นท่าปกติ ไม่ใช่ท่าพิเศษ มีสองกฎที่ไม่ผ่อน

- **ตรวจทุกอย่างที่ subagent ส่งกลับมา** รายงานคือสมมติฐานจนกว่าจะตรวจ ในประวัติของ
  repo นี้เอง มีงานที่ delegate ไปแล้วรายงานว่าเสร็จ ทั้งที่ไฟล์ไปอยู่ผิดโฟลเดอร์
  และจับได้ก็ต่อเมื่อเอาโค้ดไปรันจริง
- **ห้ามมอบการตรวจสอบขั้นสุดท้าย** และห้ามมอบสิ่งที่ตรวจเองไม่ได้

**`clink-masteragent`: เรียกก่อนทุกครั้งที่จะ `clink`** ตัดสินใจเมื่อ 2026-08-21
ไม่โหลดตอนเริ่ม session เพราะ session ส่วนใหญ่ที่นี่ไม่ได้ delegate และ ~19 KB ของมัน
เทียบกับเพดาน 9 KB ของ `using-t4` ไม่คุ้มที่จะจ่ายทุกครั้ง

## ธรรมเนียมการเขียน

**แชทเป็นไทย รายงาน โค้ด และ commit message เป็นอังกฤษ** identifier เป็นอังกฤษและตรงทุก byte
เพราะ path, flag, config key คือสิ่งที่ developer ก็อปไปใช้และค้นหา

**issue, PRD และ PR body เป็นสองภาษา** อังกฤษบวกไทยที่สะท้อนความลึกเท่ากัน — ไม่ใช่บทสรุปย่อ
ใช้กับ tracker เท่านั้น ไม่ใช้กับ `docs/`

## การส่งงาน

**PRD → issue → PR ห้ามมี PR ที่ไม่อ้าง issue** issue คือแหล่งความจริง ledger คือดัชนีค้นหา

**TDD บังคับ** สำหรับทุกอย่างใน `qwen38-tuning/bench/` ทุก primitive ที่นั่นเขียน test ก่อน
และแต่ละ test ระบุเหตุการณ์ที่มันป้องกันไว้ — อ่าน `bench/tests/test_harness.py` ก่อนเพิ่มตัวใหม่

**ปิด issue พร้อมเหตุผลและหลักฐาน** — commit, test, หรือตัวเลขที่วัดได้ ห้ามปิดเงียบ ๆ

## การแจ้ง developer

การรันยาวแบบไม่มีคนเฝ้าเป็นเรื่องปกติที่นี่ คิวหนึ่งอาจถือ GPU ไว้หลายชั่วโมง
แจ้งเมื่อ **batch จบ** หรือ **มีการตัดสินใจที่บล็อกงานอยู่** ไม่ใช่ทุกความคืบหน้าย่อย
สรุปครั้งเดียวตอนจบ ระบุให้ครบว่าอะไรรันและอะไรไม่ได้รัน
