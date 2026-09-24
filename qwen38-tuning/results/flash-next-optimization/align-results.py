import csv

path = r'C:\AI\qwen38-tuning\results\flash-next-optimization\results.csv'
rows = open(path, encoding='utf-8-sig').read().splitlines()
hdr = rows[0].split(',')
# rows have one extra cell: tensor_split "9000,16000" was not quoted and split into 2 cells
# recover: after split_mode (index 10), the ts occupies 2 cells, then everything realigns.
out = []
for line in rows[1:]:
    cells = line.split(',')
    if len(cells) != len(hdr) + 1:
        out.append({'note': 'irregular: ' + line})
        continue
    merged = cells[:11] + [cells[11] + ',' + cells[12]] + cells[13:]
    d = dict(zip(hdr, merged))
    out.append(d)

with open(r'C:\AI\qwen38-tuning\results\flash-next-optimization\results-fixed.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=hdr, quoting=csv.QUOTE_ALL)
    w.writeheader()
    for d in out:
        if 'run_id' in d:
            w.writerow(d)

tot = 0
summary = {}
for d in out:
    if 'run_id' not in d:
        continue
    if d['failure_reason']:
        continue
    summary.setdefault(d['run_id'], []).append((float(d['prompt_tps']), float(d['decode_tps']), d['gpu0_used_mb'], d['gpu1_used_mb'], d['ram_used_mb'], d['proc_ws_mb'], d['tensor_split'], d['n_cpu_moe'], d['threads'], d['threads_batch'], d['prompt_tokens'], d['output_tokens'], d['ttft_proxy_s'], d['request_wall_s'], d['total_dedicated_mb']))

for run_id, reqs in summary.items():
    dec = [x[1] for x in reqs]
    pre = [x[0] for x in reqs]
    print('%-22s n=%2d decode mean=%.2f min=%.2f max=%.2f | prefill mean=%.2f | gpu0 %s gpu1 %s | tot %s | ram %s | ws %s' % (
        run_id, len(dec), sum(dec)/len(dec), min(dec), max(dec), sum(pre)/len(pre), reqs[0][2], reqs[0][3], reqs[0][14], reqs[0][4], reqs[0][5]))
