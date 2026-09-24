"""Offline trace analysis of a censored answer, not an automated-runtime claim."""
import json
from pathlib import Path
import re

root=Path('C:/AI/qwen38-tuning/results/gsq-time-quality-2026-09-20')
source=root/'validation/20260920-041027-nvfp4-served-n3-c147456-r2'
row=next(json.loads(line) for line in (source/'responses.jsonl').read_text(encoding='utf-8').splitlines()
         if json.loads(line)['case']=='session-lfu_cache')
text=''
seen=set()
candidates=[]
for line in (source/'session-lfu_cache-stream.jsonl').read_text(encoding='utf-8').splitlines():
    event=json.loads(line)
    for choice in event['event'].get('choices',[]):
        text+=(choice.get('delta') or {}).get('reasoning_content') or ''
    for match in re.finditer(r'```(?:python|py)?\s*\n(.*?)```',text,re.S):
        code=match.group(1)
        if code in seen:
            continue
        seen.add(code)
        candidates.append({'case':f'thinking-candidate-{len(candidates)+1}', 'test':row['test'],
            'wall_s':event['elapsed_s'],'finish_reason':'stop',
            'source':str(source),'kind':'offline visible-thinking candidate; synthetic wrapper, not a natural model final',
            'response':{'choices':[{'message':{'content':'```python\n'+code+'\n```'}}]}})
out=root/'nvfp4-thinking-recovery-demo'
out.mkdir(exist_ok=True)
(out/'responses.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in candidates),encoding='utf-8')
print(json.dumps([{'case':c['case'],'code_available_s':c['wall_s'],'chars':len(c['response']['choices'][0]['message']['content'])} for c in candidates],indent=2))
