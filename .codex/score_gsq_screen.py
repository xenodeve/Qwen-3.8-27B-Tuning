"""Execute the frozen coding assertions on explicitly reviewed response files."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, 'C:/AI/qwen38-tuning/bench')
from run_bench import extract_code
from harness import check_output_contract

ap = argparse.ArgumentParser()
ap.add_argument('responses', type=Path)
args = ap.parse_args()
rows = []
for line in args.responses.read_text(encoding='utf-8').splitlines():
    row = json.loads(line)
    if not row.get('test'):
        continue
    if not row.get('response'):
        rows.append({'case':row['case'],'passed':None,'accepted':False,
                     'status':'request-failed','source':str(args.responses),
                     'error':row.get('error'),'verification_s':0,
                     'time_spent_s':row.get('time_to_response_s',row.get('wall_s')),
                     'single_call_verified_task_s':None})
        continue
    content = row['response']['choices'][0]['message'].get('content') or ''
    started = time.perf_counter()
    code = extract_code(content)
    contract = check_output_contract(content)
    try:
        result = subprocess.run([sys.executable, '-I', '-c', code + '\n' + row['test']],
                                capture_output=True, text=True, timeout=20)
        exit_code, stderr = result.returncode, result.stderr
    except subprocess.TimeoutExpired:
        exit_code, stderr = None, 'TIMEOUT: verification exceeded 20 seconds'
    verification_s = time.perf_counter() - started
    accepted = exit_code == 0 and contract['ok'] and row.get('finish_reason') == 'stop'
    spent = row.get('time_to_response_s', row['wall_s']) + verification_s
    stopped = (row['response'].get('timings') or {}).get('stop_reason')
    status = ('loop-stopped' if stopped == 'loop' else
              'budget-limited' if row.get('finish_reason') == 'length' else
              'accepted' if accepted else
              'functional-failure' if exit_code != 0 else 'format-or-finish-failure')
    rows.append({'case':row['case'], 'passed':exit_code == 0,
                 'exit_code':exit_code, 'stderr':stderr,
                 'source':str(args.responses), 'contract_ok':contract['ok'],
                 'contract_violations':contract['violations'],
                 'accepted':accepted,
                 'status':status,
                 'verification_s':verification_s,
                 'time_spent_s':spent,
                 'single_call_verified_task_s':spent if accepted else None})
output = args.responses.parent / 'coding-verification.json'
output.write_text(json.dumps(rows, indent=2), encoding='utf-8')
print(json.dumps(rows, indent=2))
