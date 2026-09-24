"""Sequential candidate screening; never launches two inference processes."""
import json
import argparse
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('C:/AI')
OUT = ROOT / 'qwen38-tuning/results/gsq-time-quality-2026-09-20/tuning-v2'
OUT.mkdir(parents=True, exist_ok=True)
REPLAY = ROOT / 'qwen38-tuning/results/gsq-2026-09-20/session-capture/1789842177382847700-request.json'
parser = argparse.ArgumentParser()
parser.add_argument('--stage', choices=['native', 'ngram'], default='native')
stage = parser.parse_args().stage
arms = ([('tensor-n2', 2, 'tensor', 'mtp', 12), ('tensor-n3', 3, 'tensor', 'mtp', 12),
         ('tensor-n4', 4, 'tensor', 'mtp', 12), ('layer-n3', 3, 'layer', 'mtp', 12)]
        if stage == 'native' else
        [('ngram-only-12', 3, 'tensor', 'ngram', 12),
         ('mtp3-ngram12', 3, 'tensor', 'mtp-ngram', 12),
         ('mtp3-ngram24', 3, 'tensor', 'mtp-ngram', 24)])
for label, draft, split, spec, match in arms:
    before = {p.name for p in OUT.iterdir() if p.is_dir()}
    argv = [sys.executable, str(ROOT / 'qwen38-tuning/bench/gsq_compare.py'),
            '--artifact', 'gsq', '--ctx', '147456', '--spec', spec,
            '--draft-n', str(draft), '--split-mode', split,
            '--ngram-match', str(match),
            '--temperature', '0.6', '--han-guard', '--stream',
            '--replay', str(REPLAY), '--replay-tasks', 'merge_intervals', 'toposort',
            '--out-root', str(OUT)]
    print('START ' + label, flush=True)
    started = time.time()
    with (OUT / (label + '-controller.log')).open('w', encoding='utf-8') as log:
        result = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    created = sorted({p.name for p in OUT.iterdir() if p.is_dir()} - before)
    event = {'label':label,'argv':argv,'exit_code':result.returncode,
             'elapsed_s':time.time()-started,'runs':created}
    with (OUT / 'queue.jsonl').open('a',encoding='utf-8') as out:
        out.write(json.dumps(event) + '\n')
    print('END ' + json.dumps(event), flush=True)
    # A failed candidate remains evidence; the next runner refuses occupied ports.
