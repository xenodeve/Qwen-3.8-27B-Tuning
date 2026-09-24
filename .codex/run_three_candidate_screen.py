"""Bounded compatibility and configuration screen for issue 92 extension."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path('C:/AI')
RUNNER = ROOT / 'qwen38-tuning/bench/gsq_compare.py'
parser = argparse.ArgumentParser()
parser.add_argument('--ctx', type=int, required=True)
args = parser.parse_args()
OUT = ROOT / f'qwen38-tuning/results/three-candidate-2026-09-20/screen-c{args.ctx}'
OUT.mkdir(parents=True, exist_ok=True)

ARMS = [
    ('swift-stock-medium', ['--artifact','swift','--spec','none','--template','stock',
                            '--effort','medium','--temperature','1.0']),
    ('turbo-stock-medium', ['--artifact','turbo','--spec','none','--template','stock',
                            '--effort','medium','--temperature','0.6']),
    ('nvfp4-current-medium', ['--artifact','nvfp4','--spec','served','--template','current',
                              '--effort','medium','--temperature','1.0']),
    ('nvfp4-stock-medium', ['--artifact','nvfp4','--spec','served','--template','stock',
                            '--effort','medium','--temperature','1.0']),
    ('nvfp4-sharp-medium', ['--artifact','nvfp4','--spec','served','--template','sharp',
                            '--effort','medium','--temperature','1.0']),
    ('nvfp4-stock-xhigh', ['--artifact','nvfp4','--spec','served','--template','stock',
                           '--effort','xhigh','--temperature','1.0']),
    ('nvfp4-sharp-xhigh', ['--artifact','nvfp4','--spec','served','--template','sharp',
                           '--effort','xhigh','--temperature','1.0']),
]


for position, (label, parameters) in enumerate(ARMS, start=1):
    argv = [sys.executable, str(RUNNER), *parameters, '--ctx',str(args.ctx),
            '--max-tokens','8192','--round','1','--seed','17','--stream',
            '--han-guard','--out-root',str(OUT)]
    before = {path.name for path in OUT.iterdir() if path.is_dir()}
    started = time.time()
    print('START ' + label, flush=True)
    with (OUT / (label + '.log')).open('w', encoding='utf-8') as log:
        result = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    created = sorted({path.name for path in OUT.iterdir() if path.is_dir()} - before)
    event = {'position':position, 'label':label, 'argv':argv, 'runs':created,
             'exit_code':result.returncode, 'elapsed_s':time.time()-started}
    with (OUT/'queue.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(event)+'\n')
    print('END '+json.dumps(event),flush=True)
    if result.returncode:
        raise SystemExit('Screen stopped on failed boot/run; inspect evidence before continuing')
