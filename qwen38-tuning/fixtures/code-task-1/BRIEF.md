ใน repo นี้ (`inventory/`) มีงานสองอย่าง

1. แก้ bug: `Store.remove()` ใน `inventory/store.py` ปล่อยให้จำนวนติดลบได้ ให้ raise `ValueError` เมื่อจำนวนที่จะเอาออกมากกว่าที่มี โดยข้อความต้องบอก sku และจำนวนที่เหลืออยู่ และเมื่อ raise แล้วจำนวนต้องไม่เปลี่ยน
2. เพิ่ม feature: `Store.low_stock(threshold)` คืน list ของ sku ที่จำนวน *น้อยกว่า* threshold เรียงจากจำนวนน้อยไปมาก ถ้าจำนวนเท่ากันเรียงตาม sku

เขียน test ของทั้งสองข้อใน `tests/test_store.py` แล้วรัน `python -m pytest -q` ให้ผ่าน แก้เฉพาะ `inventory/store.py` และ `tests/test_store.py`
