"""Capture a Claude Code resume request locally without invoking a model.

The original session is read using --resume and never appended to. This captures
the current client's reconstruction, not a claim of byte-identical historical
system prompt or tool schemas. No user code or tool calls are returned.
"""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import threading
import time

ROOT = Path('C:/AI/qwen38-tuning/results/gsq-2026-09-20/session-capture')
ROOT.mkdir(parents=True, exist_ok=True)
SESSION = Path('C:/Users/xenod/.claude/projects/D--Github-YT-Downloader/096ebcae-a6ae-4b80-b608-3b60f60a3fac.jsonl')


class Capture(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if 'count_tokens' in self.path:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"input_tokens":1}')
            return
        ident = str(time.time_ns())
        path = ROOT / f'{ident}-request.json'
        path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'request':str(path),'messages':len(body.get('messages',[])),
                          'tools':len(body.get('tools',[]))}), flush=True)
        message = {'id':'msg_capture_only','type':'message','role':'assistant',
                   'model':body.get('model'),'content':[{'type':'text','text':'CAPTURE_ONLY_NO_MODEL_INFERENCE'}],
                   'stop_reason':'end_turn','stop_sequence':None,'usage':{'input_tokens':0,'output_tokens':0}}
        self.send_response(200)
        self.send_header('Content-Type','text/event-stream' if body.get('stream') else 'application/json')
        self.end_headers()
        if not body.get('stream'):
            self.wfile.write(json.dumps(message).encode())
            return
        events = [('message_start',{'type':'message_start','message':dict(message, content=[],stop_reason=None)}),
                  ('content_block_start',{'type':'content_block_start','index':0,'content_block':{'type':'text','text':''}}),
                  ('content_block_delta',{'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':'CAPTURE_ONLY_NO_MODEL_INFERENCE'}}),
                  ('content_block_stop',{'type':'content_block_stop','index':0}),
                  ('message_delta',{'type':'message_delta','delta':{'stop_reason':'end_turn','stop_sequence':None},'usage':{'output_tokens':0}}),
                  ('message_stop',{'type':'message_stop'})]
        for name, data in events:
            self.wfile.write(('event: '+name+'\ndata: '+json.dumps(data)+'\n\n').encode())
            self.wfile.flush()


before = hashlib.sha256(SESSION.read_bytes()).hexdigest()
settings = {'env':{'ANTHROPIC_BASE_URL':'http://127.0.0.1:18001','ANTHROPIC_AUTH_TOKEN':'local-capture-only',
                  'CLAUDE_CODE_MAX_CONTEXT_TOKENS':'262144','CLAUDE_CODE_AUTO_COMPACT_WINDOW':'262144',
                  'DISABLE_AUTOUPDATER':'1','CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC':'1'},'disableAllHooks':True}
settings_path = ROOT / 'capture-settings.json'
settings_path.write_text(json.dumps(settings),encoding='utf-8')
server = ThreadingHTTPServer(('127.0.0.1',18001),Capture)
threading.Thread(target=server.serve_forever,daemon=True).start()
exe='C:/Users/xenod/AppData/Roaming/npm/node_modules/@anthropic-ai/claude-code/bin/claude.exe'
argv=[exe,'--resume','096ebcae-a6ae-4b80-b608-3b60f60a3fac','--fork-session',
      '--no-session-persistence','--settings',str(settings_path),'--strict-mcp-config',
      '--model','turboderp-Qwen3.8-27B-EXL3-SC4.0bpw-H5','--effort','medium',
      '--output-format','json','-p',
      'อธิบายความคืบหน้าของโปรเจกต์และปัญหาภาษาไทยเป็นภาษาไทย 8 ประโยค ตอบจากประวัติที่มีเท่านั้น ไม่เรียกใช้เครื่องมือและไม่แก้ไขไฟล์']
env=dict(os.environ, PYTHONIOENCODING='utf-8')
env.pop('CLAUDECODE',None)
try:
    result=subprocess.run(argv,cwd='D:/Github/YT Downloader',env=env,capture_output=True,timeout=180)
    (ROOT/'client.stdout').write_bytes(result.stdout)
    (ROOT/'client.stderr').write_bytes(result.stderr)
    after=hashlib.sha256(SESSION.read_bytes()).hexdigest()
    report={'argv':argv,'exit_code':result.returncode,'source_sha256_before':before,
            'source_sha256_after':after,'original_unchanged':before==after,
            'kind':'current-client resume of original session; no model inference',
            'limitations':['current system/tool configuration may differ from incident',
                           'MCP disabled during capture; token count requires engine tokenizer',
                           'count_tokens response is a capture stub, never measurement evidence']}
    (ROOT/'provenance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)
finally:
    server.shutdown()
