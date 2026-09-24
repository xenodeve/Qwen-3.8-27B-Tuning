## Objective

Compare GSQ IQ3_S-MTP with the actual NVFP4 VERY-LOW llama.cpp and EXL3 SC4.0bpw-H5 incumbents. Prioritize time to verified task completion together with quality; prefill/decode are explanatory metrics. Use each artifact's best locally validated recipe at a common window, inspect complete visible thinking and final artifacts, and evaluate language mitigation rather than rejecting a model solely for leakage. Preserve defaults and unrelated user changes.

## Completed bounded evaluation

- Screened GSQ speculation/split configurations and NVFP4 temperature1.0/0.6, then froze recipes before held-out LFU/tree-codec tasks. EXL3 retained its measured incumbent recipe.
- Nine runs, three rotated rounds, 27 responses. Common allocated context147456, medium effort, output budget4096 including thinking. One inference model at a time.
- Replayed captured original session096ebcae-a6ae-4b80-b608-3b60f60a3fac; original file unchanged. Actual prompt depths are138942–139047 for llama.cpp and140688–140762 for EXL3. Current client/tool definitions differ from the historical153948-token incident request; this is not exact reproduction and cannot disprove that incident.
- Code acceptance: EXL3 6/6, GSQ5/6 (one budget-limited), NVFP4 3/6 (two demonstrated bugs, one budget-limited). Successful-only median verified times43.72/79.33/72.39seconds respectively; all-attempt costs249.43/483.67/463.06seconds. Conditional medians must not hide failed work.
- Median cold prefill EXL3/GSQ/NVFP4:496.6/578.6/700.7tok/s; coding decode42.1/34.6/29.4tok/s. NVFP4 wins cold prefill; EXL3 is the recommended foundation for the next warm deep-context optimization experiment, not an automatic default promotion.
- Retained full thinking and final text. Traces expose redundant checking, correct drafts followed by wrong finals, and short thinking followed by broken code. Passing fixtures is not the entire quality assessment; supplemental LFU-storage and tree-depth probes remain separate.
- Conditional Han guarding is part of the tested operating points. No named Han ideographs were found in27 final/thinking responses, but Thai corruption remains and NVFP4 r3 contains Cyrillic `ЭК`. No universal repair or causal language-displacement claim.

## Evidence and remaining scope

Canonical report: `docs/results/20-gsq-time-quality-2026-09-20.md`. Raw evidence: `qwen38-tuning/results/gsq-time-quality-2026-09-20/`, especially frozen configs, validation queue/manifests, summary, responses, streams, verification and full thinking. The initial language-only stop is explicitly superseded by CORRECTIONS52; report19 remains historical evidence, not the current decision.

These are two single-response coding task types repeated across three rounds, not complete repository-editing agents or a global optimum. EXL3 seed application is not established. Keep this issue open for broader agent/tool-loop quality, higher-budget or verifier-controlled completion, Thai repair cost/false positives, multilingual controls and stability. Vision/smaller variants are not measured in this stage. No production default changed.

Final verification: 1741 tests passed, 2 skipped, 9 warnings; compilation passed. All nine runner snapshots match current source text (raw SHA differences are CRLF versus LF). Original session hash unchanged. Experimental listeners are stopped. One pre-existing broken documentation link and historical stale-claim findings remain; global documentation cleanliness is not claimed.

## วัตถุประสงค์

เปรียบเทียบ GSQ IQ3_S-MTP กับ NVFP4 VERY-LOW บน llama.cpp และ EXL3 SC4.0bpw-H5 ที่ใช้จริง ให้ความสำคัญกับเวลาจนงานสำเร็จที่ตรวจได้ควบคู่คุณภาพ โดย prefill/decode เป็นตัวอธิบายเวลา ใช้สูตรที่ดีที่สุดที่มีหลักฐานในเครื่องของแต่ละตัวโดยกำหนดหน้าต่างเท่ากัน ตรวจ thinking ที่มองเห็นและผลลัพธ์สุดท้ายทั้งหมด และประเมินวิธีลดข้อบกพร่องทางภาษาแทนการตัดโมเดลทิ้งเพียงเพราะมีภาษาปน รักษา default และงานเดิมของผู้ใช้

## การประเมินขอบเขตจำกัดที่เสร็จแล้ว

- คัดกรองการตั้งค่า speculation/split ของ GSQ และ temperature1.0/0.6 ของ NVFP4 แล้วตรึงสูตรก่อนทดสอบ LFU/tree codec ที่แยกจากโจทย์จูน ส่วน EXL3 ใช้สูตรหลักที่มีผลวัดเดิม
- รัน9ครั้ง หมุนลำดับ3รอบ ได้27คำตอบ หน้าต่าง147456เท่ากัน effort medium งบคำตอบ4096รวม thinking เปิด inference ทีละโมเดลเท่านั้น
- ใช้ประวัติที่ capture จาก session096ebcae-a6ae-4b80-b608-3b60f60a3fac เดิมโดยไฟล์ต้นทางไม่เปลี่ยน ความลึก prompt จริง138942–139047สำหรับ llama.cpp และ140688–140762สำหรับ EXL3 แต่ client/tool definitions ต่างจาก request เก่าที่เกิดเหตุ153948tokens จึงไม่ใช่การจำลองตรงทั้งหมดและใช้หักล้างเหตุการณ์นั้นไม่ได้
- งานโค้ดผ่าน EXL3 6/6, GSQ5/6 (หมดงบคิด1ครั้ง), NVFP4 3/6 (บั๊กที่พิสูจน์ได้2ครั้ง หมดงบคิด1ครั้ง) มัธยฐานเวลาจนตรวจผ่านเฉพาะงานสำเร็จ43.72/79.33/72.39วินาทีตามลำดับ เวลารวมทุกความพยายาม249.43/483.67/463.06วินาที ต้องไม่ใช้มัธยฐานเฉพาะงานผ่านกลบงานที่ไม่สำเร็จ
- มัธยฐาน cold prefill ของ EXL3/GSQ/NVFP4 คือ496.6/578.6/700.7tok/s และ coding decode42.1/34.6/29.4tok/s NVFP4 ชนะด้าน cold prefill ส่วน EXL3 เป็นฐานที่แนะนำสำหรับการทดลองจูนต่อในงาน warm context ลึก ไม่ใช่การเปลี่ยน default อัตโนมัติ
- เก็บ thinking และคำตอบเต็ม พบทั้งการตรวจซ้ำ ร่างที่ถูกแต่คำตอบสุดท้ายผิด และการคิดสั้นแต่โค้ดพัง การผ่าน fixture ไม่ใช่คุณภาพทั้งหมด จึงแยกผลตรวจเสริมการสะสมข้อมูล LFU และความลึก tree ออกจากคะแนนหลัก
- ใช้ Han guard แบบมีเงื่อนไขเป็นส่วนหนึ่งของสูตรทดสอบ ไม่พบอักขระ Han ที่ระบุชื่อ Unicode ได้ใน27คำตอบทั้งช่อง final/thinking แต่ไทยยังเพี้ยนและ NVFP4 รอบ3มี Cyrillic `ЭК` ไม่ได้อ้างว่าซ่อมภาษาได้ทั้งหมดหรือการแบนทำให้ภาษาอื่นไหลเข้ามา

## หลักฐานและขอบเขตที่เหลือ

รายงานหลัก: `docs/results/20-gsq-time-quality-2026-09-20.md` หลักฐานดิบ: `qwen38-tuning/results/gsq-time-quality-2026-09-20/` โดยเฉพาะ frozen configs, validation queue/manifests, summary, responses, streams, verification และ thinking เต็ม ยกเลิกการหยุดเพราะภาษาเพียงอย่างเดียวชัดเจนใน CORRECTIONS52 แล้ว report19 เป็นหลักฐานประวัติ ไม่ใช่ข้อสรุปปัจจุบัน

ชุดนี้เป็นโจทย์โค้ดตอบครั้งเดียว2ประเภท ทำซ้ำ3รอบ ไม่ใช่ agent แก้ทั้ง repository หรือการหาค่าที่ดีที่สุดในทุกมิติ ยังยืนยันไม่ได้ว่า EXL3 ใช้ seed ที่ร้องขอ เปิด issue ไว้สำหรับทดสอบ agent/tool loop ที่กว้างขึ้น การเพิ่มงบคิดหรือควบคุมจบด้วย verifier ต้นทุน/การแก้ผิดของ Thai repair ชุดควบคุมหลายภาษา และเสถียรภาพ ยังไม่ได้วัด vision/รุ่นเล็กในช่วงนี้ ไม่ได้เปลี่ยน production default

ตรวจขั้นสุดท้าย: test ผ่าน1741 ข้าม2 มีคำเตือน9 การ compile ผ่าน snapshot เครื่องมือทั้ง9ชุดตรงกับข้อความ source ปัจจุบัน (SHA ดิบต่างเพราะ CRLF กับ LF) hash session ต้นฉบับไม่เปลี่ยน หยุด listener ทดลองแล้ว ยังมีลิงก์เอกสารเสียเดิม1จุดและรายการข้ออ้างเก่าจากตัวตรวจ stale claims ไม่ได้อ้างว่าเอกสารทั้งโครงการสะอาด

## Extension: Sharp, Swift and TURBO

The developer requested the three remaining interventions from the four-candidate report. The accepted extension is `docs/plans/2026-09-20-three-candidate-extension.md`:

- Sharp v22.5.0 (pinned revision `85461fc1`) as an explicit template dimension on the same NVFP4 weights, tested Stock/Sharp × medium/xhigh.
- UkisAI Swift Q6_K at pinned revision `eb0e3a7d`, initially Stock template, medium effort, speculation off.
- DavidAU TURBO MTP-Q6_K at pinned revision `c02caef`, initially medium effort and speculation off.

Use a common actual allocation, start at147456 and reduce all new arms if any Q6 arm cannot fully offload/serve. Preserve original-session replay, executable held-out coding, complete thinking, unsuccessful costs and language review. Swift gets math characterization; TURBO gets explicit premature-completion, continuation, duplicate-work, tool-error and loop counters. Short tasks cannot promote TURBO without long-horizon evidence. Current defaults remain unchanged.

## ส่วนขยาย: Sharp, Swift และ TURBO

ผู้พัฒนาขอทดสอบ intervention ที่เหลืออีก3ตัวจากรายงานสี่ candidate แผนที่รับแล้วอยู่ที่ `docs/plans/2026-09-20-three-candidate-extension.md`:

- Sharp v22.5.0 (pin revision `85461fc1`) เป็นมิติ template ที่เลือกได้บน NVFP4 weights ชุดเดิม ทดสอบ Stock/Sharp × medium/xhigh
- UkisAI Swift Q6_K ที่ pin revision `eb0e3a7d` เริ่มด้วย Stock template, effort medium และปิด speculation
- DavidAU TURBO MTP-Q6_K ที่ pin revision `c02caef` เริ่มด้วย effort medium และปิด speculation

ใช้ context allocation จริงเท่ากัน เริ่ม147456 และลดทุก arm ใหม่พร้อมกันถ้า Q6 ตัวใด offload/serve ไม่ครบ รักษาการ replay session เดิม งานโค้ดที่รันตรวจได้ thinking เต็ม ต้นทุนงานที่ไม่สำเร็จ และการตรวจภาษา เพิ่มชุดคณิตเพื่อบันทึกลักษณะ Swift และตัวนับจบก่อนเวลา ต้องสั่งต่อ งานซ้ำ tool error และ loop สำหรับ TURBO งานสั้นอย่างเดียวใช้ promote TURBO ไม่ได้หากไม่มีหลักฐาน long-horizon ไม่เปลี่ยน default ปัจจุบัน
