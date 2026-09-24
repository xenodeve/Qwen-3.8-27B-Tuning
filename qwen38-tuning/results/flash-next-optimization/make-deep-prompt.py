"""Build a cold prefill prompt of an exact token count from real source code.

Uses the RUNNING server's /tokenize and /detokenize so the count is the
model's own, then appends a short question. Usage:
  python make-deep-prompt.py <port> <target_tokens> <out.json>
"""
import glob, json, sys, urllib.request

port, target, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
src = sorted(glob.glob(r'C:\AI\llama.cpp-upstream-fn\src\*.cpp')) + \
      sorted(glob.glob(r'C:\AI\llama.cpp-upstream-fn\ggml\src\ggml-cuda\*.cu'))
corpus, need_chars = [], target * 5
for p in src:
    corpus.append(f"// ===== {p.split(chr(92))[-1]} =====\n" + open(p, encoding='utf-8', errors='replace').read())
    if sum(map(len, corpus)) > need_chars:
        break
text = '\n'.join(corpus)

def post(path, body):
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=600))

toks = post('/tokenize', {'content': text})['tokens']
if len(toks) < target:
    sys.exit(f'corpus too small: {len(toks)} < {target}')
body = post('/detokenize', {'tokens': toks[:target]})['content']
question = '\n\nName the three source files above that define the most functions, one per line.'
req = {'messages': [{'role': 'user', 'content': body + question}], 'max_tokens': 64, 'temperature': 0,
       'chat_template_kwargs': {'enable_thinking': False}}
json.dump(req, open(out, 'w', encoding='utf-8'))
print(out, 'tokens', target, 'chars', len(body))
