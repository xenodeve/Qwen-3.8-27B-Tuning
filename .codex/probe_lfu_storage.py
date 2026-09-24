"""Additional engineering-quality probe; distinct from the frozen functional score."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,'C:/AI/qwen38-tuning/bench')
from run_bench import extract_code

ap=argparse.ArgumentParser()
ap.add_argument('responses',type=Path)
a=ap.parse_args()
row=next(json.loads(line) for line in a.responses.read_text(encoding='utf-8').splitlines()
         if json.loads(line)['case']=='session-lfu_cache')
code=extract_code(row['response']['choices'][0]['message'].get('content') or '')
probe='''
import json
def container_footprint(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, dict):
        return len(value) + sum(container_footprint(v, seen) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return len(value) + sum(container_footprint(v, seen) for v in value)
    if hasattr(value, '__dict__'):
        return container_footprint(vars(value), seen)
    return 0
c = LFUCache(1)
c.put(1, 7)
before = container_footprint(c)
for i in range(1000):
    assert c.get(1) == 7
after = container_footprint(c)
print(json.dumps({'before_slots':before,'after_slots':after,'growth_slots':after-before}))
'''
r=subprocess.run([sys.executable,'-I','-c',code+'\n'+probe],capture_output=True,text=True,timeout=20)
out={'source':str(a.responses),'kind':'additional storage-retention observation; not a retroactive functional-score change',
     'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
(a.responses.parent/'lfu-storage-probe.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
