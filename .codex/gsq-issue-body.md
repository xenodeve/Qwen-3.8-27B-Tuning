## Objective

CURRENT USER CRITERION: prioritize time per verified task together with quality;
prefill/decode rates are explanatory metrics. Use each artifact's best locally
validated configuration at the same allocated context (initially 147456), inspect
full thinking and final text, and evaluate repairable language defects with guards
and their latency/false changes. The previous language-only stop is superseded.
GSQ requires configuration screening before its recipe can be called best.

เกณฑ์ล่าสุดของผู้ใช้: ให้ความสำคัญกับเวลาจนงานสำเร็จที่ตรวจสอบได้ควบคู่คุณภาพ
โดย prefill/decode เป็นตัวอธิบายเวลา ใช้ config ที่ดีที่สุดที่มีหลักฐานในเครื่อง
ของแต่ละตัว ที่หน้าต่าง context เท่ากัน (เริ่ม 147456) ตรวจ thinking และคำตอบเต็ม
และทดลองแก้ข้อบกพร่องทางภาษาด้วย guard พร้อมวัดเวลาและการแก้ผิด ยกเลิกการหยุด
เพราะภาษาไทยเพียงอย่างเดียว ต้องคัดกรอง config ของ GSQ ก่อนเรียกว่าดีที่สุด

Execute the accepted GSQ-RCO evaluation plan against the actual local NVFP4 VERY-LOW llama.cpp artifact and EXL3 SC4.0bpw-H5 Profile G. Preserve current defaults and user changes.

## Plan

Working plan: docs/plans/2026-09-20-gsq-comparison.md. Begin with IQ3_S-MTP compatibility and quality screening, then original-session Thai replay, matched performance/tuning, context validation and stability. Expand to smaller variants and optional tracks only when justified. Record exact artifacts, build, actual prompt depth, active guards, raw requests/responses and failures. Baseline tests have failures before implementation; diagnose their relevance before trusting measurements.

## Acceptance

- Verified artifact identity and reproducible experimental launch.
- Fresh comparisons with both requested incumbents; label cross-engine differences.
- Original session 096ebcae-a6ae-4b80-b608-3b60f60a3fac, with actual context audited and missing request components disclosed.
- Independent Thai assessment; zero regex hits alone is not success.
- Raw evidence retained; documented rejection is a valid outcome.
- No automatic default promotion.

## Initial screening result

IQ3_S-MTP is downloaded and checksum-verified. All three artifacts passed the same two short coding fixtures. GSQ produced Han in a short Thai answer and extensive Thai corruption in original-session replay; native MTP worked but did not prevent that failure. Current replay depth is 138942 tokens in llama.cpp and 140688 in EXL3, versus 153948 in the historical corrupted response. This is not an exact historical-request reproduction. No default was changed. The larger tuning/vision/soak/smaller-variant program is held at the failed Thai quality gate, not claimed complete. Raw evidence and report: docs/results/19-gsq-screen-2026-09-20.md and qwen38-tuning/results/gsq-2026-09-20/. Final suite: 1731 passed, 3 skipped; the busy-GPU skip subsequently passed alone. Compilation passed. One pre-existing broken documentation link remains. All experiment inference processes are stopped. Keep this issue open for the remaining evaluation decision.

## วัตถุประสงค์

ดำเนินแผนประเมิน GSQ-RCO ที่ตกลงไว้ เทียบกับไฟล์ NVFP4 VERY-LOW ของ llama.cpp และ EXL3 SC4.0bpw-H5 Profile G ที่ใช้จริงในเครื่อง รักษาค่าเริ่มต้นและการแก้ไขของผู้ใช้ไว้

## แผน

แผนทำงาน: docs/plans/2026-09-20-gsq-comparison.md เริ่มจากตรวจความเข้ากันได้และคัดกรองคุณภาพ IQ3_S-MTP แล้วทดสอบภาษาไทยจาก session เดิม เปรียบเทียบและจูนประสิทธิภาพ ตรวจ context และเสถียรภาพ ขยายไปรุ่นเล็กและงานเสริมเมื่อมีเหตุผลรองรับ บันทึกไฟล์โมเดล build ความลึก prompt จริง guard ที่เปิด request/response ดิบ และความล้มเหลว พบ test เดิมไม่ผ่านก่อนเริ่มแก้โค้ด ต้องตรวจความเกี่ยวข้องก่อนเชื่อผลวัด

## เกณฑ์รับงาน

- ยืนยันตัวตนไฟล์โมเดลและเปิดตัวทดลองซ้ำได้
- เปรียบเทียบใหม่กับโมเดลหลักทั้งสองตัวที่ระบุ พร้อมแยกความต่างข้าม engine
- ใช้ session 096ebcae-a6ae-4b80-b608-3b60f60a3fac ตรวจ context จริงและแจ้งส่วนประกอบ request ที่ขาด
- ประเมินภาษาไทยอย่างอิสระ regex ไม่พบข้อผิดพลาดอย่างเดียวไม่ถือว่าผ่าน
- เก็บหลักฐานดิบ ผลปฏิเสธพร้อมหลักฐานถือเป็นผลลัพธ์ที่ถูกต้อง
- ไม่เปลี่ยน default อัตโนมัติ

## ผลคัดกรองเบื้องต้น

ดาวน์โหลด IQ3_S-MTP และตรวจ checksum แล้ว ทั้งสามโมเดลผ่านชุดโค้ดสั้นสองโจทย์เดียวกัน GSQ มีจีนปนในคำตอบไทยสั้นและมีภาษาไทยเพี้ยนมากในการทดสอบประวัติ session เดิม native MTP ทำงานได้แต่ไม่ได้ป้องกันอาการนี้ ความลึก replay ปัจจุบันคือ 138942 tokens ใน llama.cpp และ 140688 ใน EXL3 เทียบกับ 153948 ในคำตอบเพี้ยนเดิม จึงไม่ใช่การจำลอง request เดิมตรงทั้งหมด ไม่ได้เปลี่ยน default พักแผนจูนขนาดใหญ่ ทดสอบภาพ soak และรุ่นเล็กไว้ที่ด่านคุณภาพไทยที่ไม่ผ่าน ไม่ได้อ้างว่าเสร็จครบแผน หลักฐานดิบและรายงานอยู่ที่ docs/results/19-gsq-screen-2026-09-20.md และ qwen38-tuning/results/gsq-2026-09-20/ ชุดทดสอบสุดท้ายผ่าน 1731 รายการ ข้าม 3 รายการ และรายการที่ข้ามเพราะ GPU ไม่ว่างผ่านเมื่อตรวจซ้ำเดี่ยว การ compile ผ่าน ยังมีลิงก์เอกสารเสียเดิมหนึ่งจุด หยุด process inference ของการทดลองทั้งหมดแล้ว เปิด issue นี้ไว้สำหรับการตัดสินใจเกี่ยวกับการประเมินส่วนที่เหลือ
