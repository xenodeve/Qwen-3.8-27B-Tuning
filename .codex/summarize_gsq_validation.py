"""Summarize all planned tasks, retaining failed and censored attempt costs."""
import argparse
import json
from pathlib import Path
from statistics import mean, median


def aggregate(rows):
    accepted = [r for r in rows if r.get('accepted') is True]
    costs = [r['time_spent_s'] for r in rows if r.get('time_spent_s') is not None]
    good_times = [r['single_call_verified_task_s'] for r in accepted]
    return {'planned':len(rows), 'accepted':len(accepted),
            'status_counts':{s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})},
            'all_tasks_completed':len(accepted)==len(rows),
            'mean_verified_task_s':mean(good_times) if good_times and len(accepted)==len(rows) else None,
            'successful_only_median_s':median(good_times) if good_times else None,
            'successful_only_range_s':[min(good_times),max(good_times)] if good_times else None,
            'total_known_spent_s':sum(costs),
            'seconds_per_verified_success_including_failures':sum(costs)/len(accepted)
                if accepted and len(costs)==len(rows) else None}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('root',type=Path)
    ap.add_argument('--allow-partial',action='store_true')
    args=ap.parse_args()
    queue=[json.loads(line) for line in (args.root/'queue.jsonl').read_text(encoding='utf-8').splitlines()]
    if not args.allow_partial and len(queue)!=9:
        raise ValueError(f'Expected 9 completed runs, have {len(queue)}')
    seen=set()
    by_model={}
    episodes=[]
    for event in queue:
        identity=(event['round'],event['artifact'])
        if identity in seen or len(event['runs'])!=1:
            raise ValueError('Duplicate/ambiguous run: '+str(identity))
        seen.add(identity)
        folder=args.root/event['runs'][0]
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('runtime_ctx')!=147456 or not manifest.get('listener',{}).get('owned'):
            raise ValueError('Unverified context/listener in '+str(folder))
        response_path=folder/'responses.jsonl'
        responses={r['case']:r for r in map(json.loads,response_path.read_text(encoding='utf-8').splitlines())}
        score_path=folder/'coding-verification.json'
        scores={r['case']:r for r in json.loads(score_path.read_text(encoding='utf-8'))} if score_path.exists() else {}
        episode={'artifact':event['artifact'],'round':event['round'],'run':folder.name,
                 'cold_response_s':responses.get('original-session-replay',{}).get('time_to_response_s'),
                 'tasks':[]}
        for case in ('session-lfu_cache','session-tree_codec'):
            response=responses.get(case)
            score=scores.get(case)
            if response and not score and not args.allow_partial:
                raise ValueError('Review and score first: '+str(response_path))
            record=dict(score or {'accepted':None,'status':'not-scored' if response else 'not-run','time_spent_s':None})
            if 'status' not in record:
                record['status']=('accepted' if record.get('accepted') else
                                  'budget-limited' if (response or {}).get('finish_reason')=='length' else
                                  'functional-failure' if record.get('passed') is False else 'format-or-finish-failure')
            record.update(case=case,round=event['round'],run=folder.name,
                          response_s=(response or {}).get('time_to_response_s'),
                          phases=(response or {}).get('phases'),
                          engine_timings=((response or {}).get('response') or {}).get('timings'))
            by_model.setdefault(event['artifact'],[]).append(record)
            episode['tasks'].append(record)
        for name in ('lfu-storage-probe.json','tree-depth-probe.json'):
            path=folder/name
            if path.exists():
                probe=json.loads(path.read_text(encoding='utf-8'))
                try:
                    episode[name]=json.loads(probe.get('stdout') or '{}')
                except json.JSONDecodeError:
                    episode[name]={'error':'unparseable probe output'}
        episodes.append(episode)
    output={'complete_run_count':len(queue),'summary':{k:aggregate(v) for k,v in by_model.items()},'episodes':episodes,
            'limits':'Single-response code tasks with external verification; successful-only statistics are conditional. No retry completion time is inferred.'}
    (args.root/'summary.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(output['summary'],indent=2))


if __name__=='__main__':
    main()
