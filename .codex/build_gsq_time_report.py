"""Render retained model thinking and task metrics without scoring prose by length."""
import argparse
import json
from pathlib import Path
from gsq_stats import prefill_work

ap = argparse.ArgumentParser()
ap.add_argument('root', type=Path)
args = ap.parse_args()
records = []
sections = ['# Time, quality and visible model thinking',
            'These are external model traces, not a correctness oracle. '
            'Client-observed thinking spans include streaming and buffering. '
            'Unverified or invalid rows must not be ranked as successful tasks.']
for path in sorted(args.root.rglob('responses.jsonl')):
    manifest_path = path.parent / 'manifest.json'
    if not manifest_path.exists():
        continue
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    verification_path = path.parent / 'coding-verification.json'
    verified = ({r['case']: r for r in json.loads(verification_path.read_text(encoding='utf-8'))}
                if verification_path.exists() else {})
    invalid_warm = (path.parent / 'INVALID-WARM-TASKS.md').exists()
    for line in path.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        message = ((row.get('response') or {}).get('choices') or [{'message':{}}])[0].get('message') or {}
        phases = row.get('phases') or {}
        timings = (row.get('response') or {}).get('timings') or {}
        score = verified.get(row['case']) or {}
        valid = not (invalid_warm and row['case'].startswith('session-'))
        record = {'run':path.parent.name, 'stage':path.parent.parent.name,
                  'artifact':row['artifact'], 'case':row['case'], 'valid':valid,
                  'config':manifest.get('candidate_parameters'), 'ctx':manifest.get('ctx'),
                  'actual_prompt_tokens':row.get('prompt_tokens'),
                  'generated_tokens':row.get('generated_tokens'),
                  'cached_tokens':timings.get('cache_n',timings.get('cached_tokens')),
                  'engine_prefill_tps':timings.get('prompt_per_second'),
                  'decode_tps':timings.get('predicted_per_second'),
                  'time_to_response_s':row.get('time_to_response_s',row.get('wall_s')),
                  'thinking_observed_s':phases.get('thinking_observed_s'),
                  'thinking_chars':len(message.get('reasoning_content') or ''),
                  'verification':score or None, 'finish_reason':row.get('finish_reason'),
                  'error':row.get('error'), 'source':str(path)}
        record.update(prefill_work(row.get('response') or {}))
        records.append(record)
        sections.extend(['', '## '+path.parent.name+' / '+row['case'],
                         'Validity: '+('eligible observation' if valid else 'INVALID — conflicting questions'),
                         '```json', json.dumps(record,ensure_ascii=False,indent=2), '```',
                         '<details><summary>Full visible thinking</summary>', '',
                         message.get('reasoning_content') or '(not returned)', '', '</details>',
                         '', '### Final answer', '', message.get('content') or '(empty)'])
(args.root/'observations.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
(args.root/'thinking-and-results.md').write_text('\n'.join(sections)+'\n',encoding='utf-8')
print(json.dumps({'observations':len(records),'report':str(args.root/'thinking-and-results.md')}))
