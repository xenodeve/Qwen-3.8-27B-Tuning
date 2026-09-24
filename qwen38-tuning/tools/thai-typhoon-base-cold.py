"""Cold test: Typhoon-1B BASE (BF16 safetensors, transformers+CPU) on broken Thai.

Same 7 cases as thai-typhoon-cold.py (GGUF) and thai-wangchan-cold.py.
Greedy, one call per case, sequential only. Stdout ASCII-safe summary;
full rows to --out jsonl (auditable).

Usage: python tools/thai-typhoon-base-cold.py [--model models-typhoon/base] [--threads 8]
"""
import argparse
import datetime
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CASES = [
    ("s1-miss", "การดาวนโหลดวิดีโอจาก YouTube ควรทำอย่างไร", "การดาวน์โหลดวิดีโอจาก YouTube ควรทำอย่างไร"),
    ("s2-miss", "เปิดไฟลในโฟลเดอรแล้วส่งลิงกให้เพื่อน", "เปิดไฟล์ในโฟลเดอร์แล้วส่งลิงก์ให้เพื่อน"),
    ("s3-mix", "ผมจะสแกนโครงส้างโปรเจกตก่อนแล่วบอกร้่องคณภาพ",
     "ผมจะสแกนโครงสร้างโปรเจกต์ก่อนแล้วบอกเรื่องคุณภาพ"),
    ("s4-doub", "การเขีย็นเป็็นยังไงบ้าง", "การเขียนเป็นยังไงบ้าง"),
    ("s5-doub", "ไม่่เป็็น git repo ครับ", "ไม่เป็น git repo ครับ"),
    ("s6-loopfrag", "ผมจะดดู index.html และไฟลลสคริปต์", "ผมจะดู index.html และไฟล์สคริปต์"),
    ("s7-clean", "การดาวน์โหลดวิดีโอจาก YouTube ควรทำอย่างไร",
     "การดาวน์โหลดวิดีโอจาก YouTube ควรทำอย่างไร"),
]

SYS = ("Fix Thai spelling only. Output ONLY the corrected sentence, nothing else. "
       "Do not translate, do not explain, do not change code, paths, URLs or English words.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join("models-typhoon", "base"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(a.model)
    mdl = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    mdl.to("cpu")
    mdl.eval()
    load_s = round(time.perf_counter() - t0, 1)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = a.out or os.path.join(ROOT, "bench", "results",
                                     f"thai-typhoon-base-{stamp}.jsonl")
    summary = {"model": "typhoon-ai/llama3.2-typhoon2-1b-instruct/base",
               "threads": a.threads, "load_s": load_s, "cases": []}
    open(out_path, "w", encoding="utf-8").close()
    for cid, text, want in CASES:
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": "Input: " + text + "\nOutput:"}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_dict=True, return_tensors="pt")
        t1 = time.perf_counter()
        with torch.no_grad():
            out = mdl.generate(**ids, max_new_tokens=128, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        ms = round((time.perf_counter() - t1) * 1000, 1)
        n_in = ids["input_ids"].shape[1]
        gen = tok.decode(out[0][n_in:], skip_special_tokens=True)
        gen = gen.strip().splitlines()
        gen = gen[0].strip() if gen else ""
        row = {"id": cid, "input": text, "want": want, "output": gen,
               "match": gen == want, "ms": ms}
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary["cases"].append({"id": cid, "match": gen == want,
                                "ms": ms, "out": gen})
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    sys.exit(main())
