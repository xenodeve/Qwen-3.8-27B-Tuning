"""Compare the incumbent sampler and a Thai-stabilized candidate serially."""
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path('C:/AI')
out = root/'qwen38-tuning/results/gsq-time-quality-2026-09-20/nvfp4-tuning'
out.mkdir(parents=True, exist_ok=True)
replay = root/'qwen38-tuning/results/gsq-2026-09-20/session-capture/1789842177382847700-request.json'
for temperature in (1.0, 0.6):
    argv = [sys.executable, str(root/'qwen38-tuning/bench/gsq_compare.py'),
            '--artifact','nvfp4','--ctx','147456','--spec','served',
            '--temperature',str(temperature),'--han-guard','--stream',
            '--replay',str(replay),'--replay-tasks','merge_intervals','toposort',
            '--out-root',str(out)]
    before = {p.name for p in out.iterdir() if p.is_dir()}
    print('START temperature='+str(temperature),flush=True)
    started = time.time()
    with (out/f'temperature-{temperature}.log').open('w',encoding='utf-8') as log:
        result = subprocess.run(argv,cwd=root,stdout=log,stderr=subprocess.STDOUT)
    event={'temperature':temperature,'argv':argv,'exit_code':result.returncode,
           'elapsed_s':time.time()-started,
           'runs':sorted({p.name for p in out.iterdir() if p.is_dir()}-before)}
    with (out/'queue.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(event)+'\n')
    print('END '+json.dumps(event),flush=True)
