"""Three rotated validation rounds, with held-out code tasks and shared context."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--nv-temperature', type=float, required=True)
args = parser.parse_args()
root = Path('C:/AI')
out = root/'qwen38-tuning/results/gsq-time-quality-2026-09-20/validation'
out.mkdir(parents=True, exist_ok=True)
replay = root/'qwen38-tuning/results/gsq-2026-09-20/session-capture/1789842177382847700-request.json'
arms = {
    'gsq': ['--artifact','gsq','--spec','mtp-ngram','--draft-n','3',
            '--split-mode','tensor','--ngram-match','24','--temperature','0.6','--han-guard'],
    'nvfp4': ['--artifact','nvfp4','--spec','served','--temperature',str(args.nv_temperature),'--han-guard'],
    'exl3': ['--artifact','exl3','--spec','served','--temperature','0.6'],
}
orders = [('gsq','nvfp4','exl3'), ('nvfp4','exl3','gsq'), ('exl3','gsq','nvfp4')]
for round_number, (order, seed) in enumerate(zip(orders, (17,53,91)), start=1):
    for position, artifact in enumerate(order, start=1):
        argv = [sys.executable,str(root/'qwen38-tuning/bench/gsq_compare.py'),
                *arms[artifact], '--ctx','147456','--round',str(round_number),
                '--seed',str(seed),'--stream','--replay',str(replay),
                '--replay-tasks','lfu_cache','tree_codec','--out-root',str(out)]
        before = {p.name for p in out.iterdir() if p.is_dir()}
        label = f'r{round_number}-p{position}-{artifact}'
        print('START '+label,flush=True)
        started = time.time()
        with (out/(label+'.log')).open('w',encoding='utf-8') as log:
            result = subprocess.run(argv,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        created = sorted({p.name for p in out.iterdir() if p.is_dir()}-before)
        event = {'round':round_number,'order':position,'artifact':artifact,'seed':seed,
                 'argv':argv,'runs':created,'exit_code':result.returncode,'elapsed_s':time.time()-started}
        with (out/'queue.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(event)+'\n')
        print('END '+json.dumps(event),flush=True)
        if result.returncode != 0:
            raise SystemExit('Validation stopped on failed boot/run; inspect evidence before continuing')
