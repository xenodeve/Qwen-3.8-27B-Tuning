"""Artifact-specific MTP screen after the no-spec behavior baseline."""
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
OUT = ROOT / f'qwen38-tuning/results/three-candidate-2026-09-20/spec-screen-c{args.ctx}'
OUT.mkdir(parents=True, exist_ok=True)

ARMS = [
    ('swift-mtp2', 'swift', '1.0', '2'),
    ('swift-mtp3', 'swift', '1.0', '3'),
    ('turbo-mtp2', 'turbo', '0.6', '2'),
    ('turbo-mtp3', 'turbo', '0.6', '3'),
]


for position, (label, artifact, temperature, draft_n) in enumerate(ARMS, start=1):
    argv = [sys.executable, str(RUNNER), '--artifact',artifact,'--spec','mtp',
            '--draft-n',draft_n,'--template','stock','--effort','medium',
            '--temperature',temperature,'--ctx',str(args.ctx),'--max-tokens','8192',
            '--round','1','--seed','17','--stream','--han-guard','--out-root',str(OUT)]
    before = {path.name for path in OUT.iterdir() if path.is_dir()}
    started = time.time()
    print('START '+label,flush=True)
    with (OUT/(label+'.log')).open('w',encoding='utf-8') as log:
        result = subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    created = sorted({path.name for path in OUT.iterdir() if path.is_dir()}-before)
    event = {'position':position,'label':label,'artifact':artifact,'argv':argv,
             'runs':created,'exit_code':result.returncode,'elapsed_s':time.time()-started}
    with (OUT/'queue.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(event)+'\n')
    print('END '+json.dumps(event),flush=True)
    if result.returncode:
        raise SystemExit('Speculation screen stopped; inspect evidence before continuing')
