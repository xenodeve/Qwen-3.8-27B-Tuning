"""Offline, verifier-assisted recovery demonstration; never changes raw scores."""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,'C:/AI/qwen38-tuning/bench')
from run_bench import extract_code

root=Path('C:/AI/qwen38-tuning/results/gsq-time-quality-2026-09-20')
source=root/'tuning-v2/20260920-025208-gsq-mtp-n3-c147456-r1/responses.jsonl'
row=next(json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()
         if json.loads(line)['case']=='session-merge_intervals')
scores=json.loads((source.parent/'coding-verification.json').read_text(encoding='utf-8'))
score=next(item for item in scores if item['case']==row['case'])
assert score['accepted'] is False
start=time.perf_counter()
candidate=extract_code(row['response']['choices'][0]['message']['reasoning_content'])
preparation=time.perf_counter()-start
out=root/'repair-demo'
out.mkdir(exist_ok=True)
recovery={'case':'layer-merge-thinking-candidate','test':row['test'],
          'wall_s':score['time_spent_s']+preparation,
          'time_to_response_s':score['time_spent_s']+preparation,
          'finish_reason':'stop','original_source':str(source),
          'method':'manually reviewed fenced code from visible thinking, tested with the same fixture',
          'limitations':'offline verifier-assisted demonstration, not a generally safe automatic repair or an independent held-out score',
          'response':{'choices':[{'message':{'content':'```python\n'+candidate+'\n```'}}]}}
(out/'responses.jsonl').write_text(json.dumps(recovery,ensure_ascii=False)+'\n',encoding='utf-8')
print(candidate)
