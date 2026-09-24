"""Speculative-decoding workloads against a running server.

rep : edit a real 100-line file (rename one function) and print the WHOLE file
      back -- the copy-heavy shape Claude Code edits have; n-gram drafting's
      best case.
new : write a fresh module (declong prompt), 512 forced tokens -- little to copy.
Each at temperature 1.0 (served default, seeded) and 0. cache_prompt off.
Writes every timings field the server returns (draft_n / draft_n_accepted
included when speculation ran).
  python spec-bench.py <port> <arm> <reps> <out.csv>
"""
import csv, json, os, sys, time, urllib.request

port, arm, reps, out = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
src = open(r'C:\AI\qwen38-tuning\bench\gpu_trace.py', encoding='utf-8').read()
rep_msg = ("Rename the function `sample` to `read_gpu_sample` everywhere in this file, "
           "including every call site. Output the COMPLETE updated file and nothing else.\n\n```python\n"
           + src + "```")
new_msg = json.load(open(r'C:\AI\qwen38-tuning\results\flash-next-optimization\prompts\declong.json',
                         encoding='utf-8'))['messages'][0]['content']
work = {'rep': (rep_msg, 1400, False), 'new': (new_msg, 512, True)}

fields = ['arm', 'work', 'temp', 'rep', 'wall_s', 'prompt_n', 'predicted_n', 'predicted_per_second',
          'draft_n', 'draft_n_accepted', 'accept_rate', 'eff_tps']
new_file = not os.path.exists(out)
with open(out, 'a', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    if new_file:
        w.writeheader()
    for r in range(1, reps + 1):
        for wk, (msg, mx, ign) in work.items():
            for temp in (1.0, 0.0):
                body = {'messages': [{'role': 'user', 'content': msg}], 'max_tokens': mx,
                        'temperature': temp, 'seed': 1000 + r, 'cache_prompt': False,
                        'ignore_eos': ign, 'chat_template_kwargs': {'enable_thinking': False}}
                t0 = time.time()
                req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
                                             json.dumps(body).encode(), {'Content-Type': 'application/json'})
                j = json.load(urllib.request.urlopen(req, timeout=1800))
                wall = time.time() - t0
                t = j['timings']
                dn, da = t.get('draft_n'), t.get('draft_n_accepted')
                row = {'arm': arm, 'work': wk, 'temp': temp, 'rep': r, 'wall_s': round(wall, 2),
                       'prompt_n': t.get('prompt_n'), 'predicted_n': t.get('predicted_n'),
                       'predicted_per_second': round(t.get('predicted_per_second', 0), 2),
                       'draft_n': dn, 'draft_n_accepted': da,
                       'accept_rate': round(da / dn, 3) if dn else '',
                       # generation-only throughput from the server's own clock
                       'eff_tps': round(t['predicted_n'] / (t['predicted_ms'] / 1000), 2)}
                w.writerow(row); fh.flush()
                sd = os.environ.get('SPEC_SAVE_DIR')
                if sd:
                    os.makedirs(sd, exist_ok=True)
                    with open(os.path.join(sd, f'{arm}__{wk}__t{temp}__r{r}.txt'), 'w', encoding='utf-8') as cf:
                        cf.write(j['choices'][0]['message']['content'] or '')
                print(arm, wk, temp, r, row['predicted_per_second'], 'tok/s', 'accept', row['accept_rate'], flush=True)
