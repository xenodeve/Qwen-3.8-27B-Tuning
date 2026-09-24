"""Local Anthropic-to-OpenAI adapter contracts for the real client canary."""
from __future__ import annotations

import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

import pytest
sys.path.insert(0, str(Path(__file__).parents[1]))
from anthropic_adapter import AdapterServer


class FakeOpenAI(BaseHTTPRequestHandler):
    def do_GET(self):
        self.server.seen_headers = dict(self.headers)
        self.server.hits = getattr(self.server, 'hits', 0) + 1
        body = json.dumps({'status':'ok'} if self.path == '/health' else {
            'model_path':'/tmp/pinned.gguf',
            'default_generation_settings':{'n_ctx':65536},
        }).encode()
        self.send_response(200); self.send_header('Content-Length', str(len(body)))
        self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        self.server.raw_request = self.rfile.read(length)
        self.server.seen_headers = dict(self.headers)
        self.server.hits = getattr(self.server, 'hits', 0) + 1
        body = json.loads(self.server.raw_request)
        if getattr(self.server, 'redirect', None):
            self.send_response(307)
            self.send_header('Location', self.server.redirect)
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        if getattr(self.server, 'error_response', False):
            payload = b'{"error":{"message":"fake upstream error"}}'
            self.send_response(503); self.send_header('Content-Length', str(len(payload)))
            self.end_headers(); self.wfile.write(payload)
            return
        self.server.last_request = body
        if body.get('stream'):
            payload = (b'data: {"id":"x","choices":[{"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
                      b'data: {"id":"x","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1}}\n\n'
                      b'data: [DONE]\n\n')
            payload = getattr(self.server, 'stream_payload', payload)
            self.send_response(200); self.send_header('Content-Type','text/event-stream')
            self.send_header('Content-Length', str(len(payload))); self.end_headers()
            if getattr(self.server, 'fragment_body', False):
                import time
                split = payload.index(b'[DONE]') + len(b'[DONE]')
                for part in (payload[:1], payload[1:17], payload[17:split], payload[split:]):
                    self.wfile.write(part); self.wfile.flush(); time.sleep(0.01)
            else:
                self.wfile.write(payload)
            return
        payload = json.dumps({'id':'x','model':'local','choices':[{'message':{'role':'assistant','content':'ok'},'finish_reason':'stop'}], 'usage':{'prompt_tokens':5,'completion_tokens':1}}).encode()
        self.send_response(200); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(payload))); self.end_headers(); self.wfile.write(payload)
    def log_message(self,*args): pass


def start_fake():
    server = ThreadingHTTPServer(('127.0.0.1', 0), FakeOpenAI)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    return server, thread


def request(port, body):
    raw = json.dumps(body).encode()
    req = Request(f'http://127.0.0.1:{port}/v1/messages', data=raw,
                  headers={'Content-Type':'application/json'})
    with urlopen(req, timeout=5) as response:
        return response.status, response.headers, response.read()


def test_nonstream_translates_anthropic_request_and_response():
    upstream, thread = start_fake(); adapter = AdapterServer(f'http://127.0.0.1:{upstream.server_port}').start()
    try:
        status, headers, raw = request(adapter.port, {'model':'local','max_tokens':64,
            'system':'system','messages':[{'role':'user','content':'hello'}]})
        body = json.loads(raw)
        assert status == 200
        assert body['type'] == 'message'
        assert body['content'][0]['text'] == 'ok'
    finally:
        adapter.stop(); upstream.shutdown(); upstream.server_close(); thread.join(timeout=2)


def test_stream_emits_anthropic_terminal_event_and_usage():
    upstream, thread = start_fake(); adapter = AdapterServer(f'http://127.0.0.1:{upstream.server_port}').start()
    try:
        _, _, raw = request(adapter.port, {'model':'local','max_tokens':64,'stream':True,
            'messages':[{'role':'user','content':'hello'}]})
        text = raw.decode()
        assert 'event: message_start' in text
        assert 'event: content_block_delta' in text
        assert 'event: message_stop' in text
        assert 'output_tokens' in text
    finally:
        adapter.stop(); upstream.shutdown(); upstream.server_close(); thread.join(timeout=2)


def test_frozen_profile_transform_reaches_upstream_without_dropping_tools():
    # CLI sampling defaults must not silently replace the selected model profile.
    upstream, thread = start_fake()
    def frozen(body):
        return dict(body, temperature=1.0, seed=29, reasoning_effort='medium')
    adapter = AdapterServer(f'http://127.0.0.1:{upstream.server_port}',
                            request_transform=frozen).start()
    try:
        request(adapter.port, {'model':'local','max_tokens':64, 'temperature':0.2,
            'messages':[{'role':'user','content':'hello'}],
            'tools':[{'name':'Read','description':'Read a file','input_schema':{'type':'object'}}]})
        observed = upstream.last_request
        assert observed['temperature'] == 1.0
        assert observed['seed'] == 29
        assert observed['reasoning_effort'] == 'medium'
        assert observed['tools'][0]['function']['name'] == 'Read'
    finally:
        adapter.stop(); upstream.shutdown(); upstream.server_close(); thread.join(timeout=2)


def test_llama_fragmented_tool_arguments_and_trailing_usage_are_not_cut_off():
    # Real Swift CLI attempt saw only '{': the EXL3 translator expects full calls.
    from anthropic_adapter import LlamaStreamTranslator
    translator = LlamaStreamTranslator(model='local')
    out = translator.start()
    for call in [
        {'index':0,'id':'call-1','type':'function','function':{'name':'Read','arguments':'{'}},
        {'index':0,'function':{'arguments':'"file_path":"inventory/store.py"}'}},
    ]:
        out += translator.feed({'choices':[{'delta':{'tool_calls':[call]},'finish_reason':None}]})
    out += translator.feed({'choices':[{'delta':{},'finish_reason':'tool_calls'}]})
    assert not any(event == 'message_stop' for event, _ in out)
    out += translator.feed({'choices':[], 'usage':{'prompt_tokens':123,'completion_tokens':45}})
    out += translator.finish()
    starts = [data for event, data in out if event == 'content_block_start']
    assert len(starts) == 1
    assert starts[0]['content_block']['name'] == 'Read'
    args = ''.join(data['delta']['partial_json'] for event,data in out
        if event == 'content_block_delta' and data['delta']['type'] == 'input_json_delta')
    assert json.loads(args) == {'file_path':'inventory/store.py'}
    usage = [data['usage'] for event,data in out if event == 'message_delta'][-1]
    assert usage['input_tokens'] == 123 and usage['output_tokens'] == 45
    assert sum(event == 'message_stop' for event,_ in out) == 1


@pytest.mark.parametrize('path,method', [('/health','GET'), ('/props','GET'),
    ('/v1/messages','POST'), ('/v1/messages/count_tokens','POST'), ('/unknown','GET')])
@pytest.mark.parametrize('key', [None, 'wrong'])
def test_nonce_denies_every_route_before_forwarding_or_tokenization(path, method, key):
    # Remote-run ingress must not become an unauthenticated local proxy.
    from urllib.error import HTTPError
    adapter = AdapterServer('http://127.0.0.1:1', client_api_key='caller-nonce').start()
    try:
        headers = {} if key is None else {'X-Api-Key': key}
        req = Request(f'http://127.0.0.1:{adapter.port}{path}',
                      data=b'not-json' if method == 'POST' else None,
                      headers=headers, method=method)
        with pytest.raises(HTTPError) as error:
            urlopen(req, timeout=5)
        assert error.value.code == 401
    finally:
        adapter.stop()


@pytest.mark.parametrize('path', ['/health', '/props', '/v1/messages'])
def test_authenticated_hops_use_independent_nonce_and_never_forward_provider_key(path):
    # A provider secret on ingress must never replace the relay's own nonce.
    upstream, thread = start_fake()
    adapter = AdapterServer(f'http://127.0.0.1:{upstream.server_port}',
                            client_api_key='caller-nonce', upstream_api_key='relay-nonce').start()
    try:
        raw = json.dumps({'model':'local', 'max_tokens':8, 'messages':[]}).encode()
        req = Request(f'http://127.0.0.1:{adapter.port}{path}',
            data=raw if path == '/v1/messages' else None,
            headers={'X-Api-Key':'caller-nonce', 'Authorization':'Bearer provider-secret',
                     'Content-Type':'application/json'})
        with urlopen(req, timeout=5) as response:
            assert response.status == 200
        assert upstream.seen_headers['X-Api-Key'] == 'relay-nonce'
        assert 'Authorization' not in upstream.seen_headers
        assert 'provider-secret' not in str(upstream.seen_headers)
    finally:
        adapter.stop(); upstream.shutdown(); upstream.server_close(); thread.join(timeout=2)


def test_adapter_declines_upstream_redirect_without_second_hop():
    # A loopback URL cannot authorize following redirects to arbitrary hosts.
    from urllib.error import HTTPError
    upstream, thread = start_fake()
    upstream.redirect = f'http://127.0.0.1:{upstream.server_port}/redirected'
    adapter = AdapterServer(f'http://127.0.0.1:{upstream.server_port}').start()
    try:
        with pytest.raises(HTTPError) as error:
            request(adapter.port, {'model':'local','max_tokens':8,'messages':[]})
        assert error.value.code == 502
        assert upstream.hits == 1
    finally:
        adapter.stop(); upstream.shutdown(); upstream.server_close(); thread.join(timeout=2)


@pytest.mark.parametrize('fragment_body', [False, True])
def test_observation_preserves_reasoning_tool_fragments_usage_and_entire_http_body(tmp_path, monkeypatch, fragment_body):
    # Semantic DONE previously abandoned trailing body bytes; reasoning must not disappear.
    import hashlib
    import stream_observation
    seen = []
    base = stream_observation.StreamObservation
    class RecordingObservation(base):
        def feed(self, data, ns):
            seen.append(data)
            return super().feed(data, ns)
    monkeypatch.setattr(stream_observation, 'StreamObservation', RecordingObservation)
    chunks = [
        {'choices':[{'delta':{'reasoning_content':'คิดก่อน'},'finish_reason':None}]},
        {'choices':[{'delta':{'content':'ตอบ'},'finish_reason':None}]},
        {'choices':[{'delta':{'tool_calls':[{'index':0,'id':'call-1','function':{'name':'Read','arguments':'{'}}]},'finish_reason':None}]},
        {'choices':[{'delta':{'tool_calls':[{'index':0,'function':{'arguments':'"path":"x"}'}}]},'finish_reason':'tool_calls'}]},
        {'choices':[], 'usage':{'prompt_tokens':13,'completion_tokens':7}},
    ]
    wire = b''.join(b'data: ' + json.dumps(c, ensure_ascii=False).encode() + b'\r\n\r\n' for c in chunks)
    wire += b'data: [DONE]\r\n\r\n: trailer\r\n\r\n'
    upstream, thread = start_fake(); upstream.stream_payload = wire
    upstream.fragment_body = fragment_body
    directory = tmp_path / 'missing-parent' / 'observations'
    adapter = AdapterServer(f'http://127.0.0.1:{upstream.server_port}', observation_directory=directory).start()
    try:
        _, _, raw = request(adapter.port, {'model':'local','max_tokens':64,'stream':True,'messages':[]})
        events = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith(b'data: ')]
        deltas = [e['delta'] for e in events if e['type'] == 'content_block_delta']
        assert ''.join(d.get('thinking','') for d in deltas) == 'คิดก่อน'
        assert ''.join(d.get('text','') for d in deltas) == 'ตอบ'
        assert ''.join(d.get('partial_json','') for d in deltas) == '{"path":"x"}'
        assert [e['usage'] for e in events if e['type'] == 'message_delta'][-1]['output_tokens'] == 7
        assert b''.join(seen) == wire
        if fragment_body:
            assert len(seen) > 1
        assert adapter.observation_errors == []
        assert len(adapter.observations) == 1
        record = adapter.observations[0]
        assert record['request_sha256'] == hashlib.sha256(upstream.raw_request).hexdigest()
        canonical = json.dumps(upstream.last_request, sort_keys=True, ensure_ascii=False,
                               separators=(',', ':'), allow_nan=False).encode()
        assert record['request_content_sha256'] == hashlib.sha256(canonical).hexdigest()
        assert record['request_content_hash_algorithm'] == 'sha256-canonical-json-v1'
        path = directory / record['request_id'] / 'timing.json'
        assert json.loads(path.read_text()) == record
        assert 'คิดก่อน' not in path.read_text()
        assert record['timing_ns']['eof'] is not None
        assert record['timing_ns']['end'] == record['timing_ns']['eof']
        assert record['complete'] is True
    finally:
        adapter.stop(); upstream.shutdown(); upstream.server_close(); thread.join(timeout=2)


def test_observation_persistence_failure_cannot_claim_success_or_overwrite(tmp_path, monkeypatch):
    # A saved run must never silently replace another request's evidence.
    import anthropic_adapter
    from types import SimpleNamespace
    from http.client import IncompleteRead
    monkeypatch.setattr(anthropic_adapter.uuid, 'uuid4', lambda: SimpleNamespace(hex='fixed-request'))
    directory = tmp_path / 'fixed-request'
    directory.mkdir()
    evidence = directory / 'timing.json'
    evidence.write_text('original evidence')
    upstream, thread = start_fake()
    adapter = AdapterServer(f'http://127.0.0.1:{upstream.server_port}', observation_directory=tmp_path).start()
    try:
        with pytest.raises(IncompleteRead):
            request(adapter.port, {'model':'local','max_tokens':8,'stream':True,'messages':[]})
        assert evidence.read_text() == 'original evidence'
        assert adapter.observations == []
        assert adapter.observation_errors == [{'request_id':'fixed-request','error':'observation_persistence_failed'}]
    finally:
        adapter.stop(); upstream.shutdown(); upstream.server_close(); thread.join(timeout=2)


def test_empty_configured_nonce_is_rejected_instead_of_allowing_missing_header():
    # Empty configuration must not accidentally turn an auth boundary off.
    with pytest.raises(ValueError, match='nonce'):
        AdapterServer('http://127.0.0.1:1', client_api_key='')


def test_stream_upstream_error_keeps_anthropic_error_event_contract():
    # Optional observation must not turn the existing SSE errors into JSON responses.
    upstream, thread = start_fake(); upstream.error_response = True
    adapter = AdapterServer(f'http://127.0.0.1:{upstream.server_port}').start()
    try:
        status, _, raw = request(adapter.port, {'model':'local','max_tokens':8,'stream':True,'messages':[]})
        assert status == 200
        assert b'event: error' in raw
        assert b'fake upstream error' in raw
    finally:
        adapter.stop(); upstream.shutdown(); upstream.server_close(); thread.join(timeout=2)


def test_adapter_is_loopback_and_rejects_unsupported_upstream_scheme():
    with pytest.raises(ValueError): AdapterServer('https://example.com')
