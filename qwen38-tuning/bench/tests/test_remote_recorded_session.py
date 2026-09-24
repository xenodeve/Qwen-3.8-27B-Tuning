"""Offline regressions for the FP8 control's credential boundary."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_child_environment_excludes_secret_names_and_values(tmp_path):
    # A generic inherited variable must not smuggle the gateway credential.
    from remote_recorded_session import remote_client_env
    env = remote_client_env({'PATH': 'bin', 'GITHUB_TOKEN': 'other-secret',
                             'INNOCENT': 'prefix-gateway-secret-suffix',
                             'HTTPS_PROXY': 'http://proxy',
                             'DATABASE_URL': 'postgres://user:password@example'}, tmp_path,
                            'http://127.0.0.1:12345', 'gateway-secret')
    assert env['PATH'] == 'bin'
    assert 'GITHUB_TOKEN' not in env
    assert 'INNOCENT' not in env
    assert 'HTTPS_PROXY' not in env
    assert 'DATABASE_URL' not in env
    assert env['ANTHROPIC_API_KEY'] == 'local-benchmark-placeholder'


def test_loopback_caller_cannot_borrow_gateway_credential_without_session_token(tmp_path):
    import http.client
    import json
    from remote_recorded_session import GatewayForwarder
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError('unauthenticated request reached credential boundary')
    gateway = GatewayForwarder(tmp_path / 'wire', 'gateway-secret', connection_factory=forbidden)
    gateway.start()
    try:
        for headers in ({}, {'X-Api-Key': 'local-benchmark-placeholder'}):
            connection = http.client.HTTPConnection('127.0.0.1', gateway.port)
            connection.request('POST', '/v1/messages', json.dumps({
                'model': 'qwen3.8-27b-fp8', 'max_tokens': 8192, 'messages': []}), headers)
            response = connection.getresponse()
            assert response.status == 401
            response.read()
            connection.close()
    finally:
        gateway.stop()
    assert calls == []
    assert gateway.rows == []


def test_forwarder_fixes_destination_and_never_records_credentials(tmp_path):
    import http.client
    import json
    from remote_recorded_session import GatewayForwarder
    calls = []

    class Response:
        status = 200
        def getheader(self, name, default=None):
            return 'application/json' if name == 'Content-Type' else default
        def read1(self, size):
            if getattr(self, 'done', False):
                return b''
            self.done = True
            return b'{"model":"qwen3.8-27b-fp8","content":[],"usage":{"input_tokens":3}}'

    class Connection:
        def request(self, method, path, body, headers):
            calls.append((method, path, body, headers))
        def getresponse(self):
            return Response()
        def close(self):
            pass

    def transport(host, **kwargs):
        assert host == 'gateway.9arm.co'
        assert kwargs['context'].check_hostname is True
        return Connection()

    gateway = GatewayForwarder(tmp_path / 'wire', 'gateway-secret', connection_factory=transport)
    gateway.start()
    try:
        for path, model, expected in [('/v1/messages?beta=true', 'qwen3.8-27b-fp8', 200),
                                      ('/v1/messages', 'other', 400),
                                      ('https://evil/v1/messages', 'qwen3.8-27b-fp8', 400),
                                      ('/v1/messages?destination=evil', 'qwen3.8-27b-fp8', 400)]:
            connection = http.client.HTTPConnection('127.0.0.1', gateway.port)
            connection.request('POST', path, json.dumps({'model': model, 'messages': [], 'max_tokens': 8192}),
                               {'Authorization': 'Bearer child-secret', 'X-Api-Key': gateway.client_token})
            response = connection.getresponse()
            assert response.status == expected
            response.read()
            connection.close()
    finally:
        gateway.stop()
    assert len(calls) == 1
    assert calls[0][3]['Authorization'] == 'Bearer gateway-secret'
    forwarded = json.loads(calls[0][2])
    assert forwarded['temperature'] == 1.0
    assert forwarded['top_p'] == 0.95
    assert forwarded['top_k'] == 20
    assert 'seed' not in forwarded and 'min_p' not in forwarded
    assert b'child-secret' not in calls[0][2]
    evidence = b''.join(p.read_bytes() for p in tmp_path.rglob('*') if p.is_file())
    assert b'gateway-secret' not in evidence
    assert b'child-secret' not in evidence
    assert gateway.rows[0]['usable'] is True


def test_openai_gateway_records_reasoning_and_separate_semantic_timing(tmp_path):
    # DSH exposes reasoning_content on9arm/OpenAI; Anthropic compatibility did not.
    import http.client
    import itertools
    import json
    from remote_recorded_session import GatewayForwarder
    payloads = [
        {'choices': [{'delta': {'role': 'assistant'}}]},
        {'choices': [{'delta': {'reasoning_content': 'คิด'}}]},
        {'choices': [{'delta': {'content': '391'}}]},
        {'choices': [{'delta': {}, 'finish_reason': 'stop'}],
         'usage': {'prompt_tokens': 9, 'completion_tokens': 7}},
    ]
    wire = [b'data: ' + json.dumps(p, ensure_ascii=False).encode() + b'\n\n' for p in payloads]
    wire += [b'data: [DONE]\n\n', b'']
    class Response:
        status = 200
        def __init__(self): self.chunks = iter(wire)
        def getheader(self, name, default=None): return 'text/event-stream'
        def read1(self, size): return next(self.chunks)
    class Connection:
        def request(self, method, path, body, headers):
            assert path == '/v1/chat/completions'
            assert headers['Authorization'] == 'Bearer gateway-secret'
        def getresponse(self): return Response()
        def close(self): pass
    clock = itertools.count(1_000_000_000, 10_000_000)
    gateway = GatewayForwarder(tmp_path / 'wire', 'gateway-secret', protocol='openai',
        connection_factory=lambda *a, **k: Connection(), clock_ns=lambda: next(clock))
    gateway.start()
    try:
        client = http.client.HTTPConnection('127.0.0.1', gateway.port)
        client.request('POST', '/v1/chat/completions', json.dumps({
            'model': 'qwen3.8-27b-fp8', 'max_tokens': 8192, 'stream': True, 'messages': []}),
            {'X-Api-Key': gateway.client_token})
        response = client.getresponse()
        assert response.status == 200
        assert response.read() == b''.join(wire)
        client.close()
    finally: gateway.stop()
    row = gateway.rows[0]
    observation = json.loads((tmp_path / 'wire/0/timing.json').read_text())
    assert observation['complete'] is True
    assert observation['counts']['reasoning_chars'] == 3
    assert observation['timing_ns']['first_body'] < observation['timing_ns']['first_reasoning']
    assert observation['timing_ns']['first_reasoning'] < observation['timing_ns']['first_text']
    assert observation['timing_ns']['end'] == observation['timing_ns']['eof']
    assert row['request_sha256'] == observation['request_sha256']
    assert row['observation_ref'] == 'wire/0/timing.json'
    assert 'gateway-secret' not in json.dumps(observation)
    assert row['usable'] is True


def test_remote_recorder_preserves_evidence_without_fabricated_local_provenance(tmp_path):
    import json
    from remote_recorded_session import run_remote_recorded_session
    from recorded_session import verify_evidence
    work = tmp_path / 'work'
    work.mkdir()
    (work / 'solution.txt').write_text('before')

    class Gateway:
        port = 12345
        client_token = 'fake-session-token'
        errors = []
        rows = [{'method': 'POST', 'path': '/v1/messages', 'usable': True,
                 'request': {'model': 'qwen3.8-27b-fp8', 'max_tokens': 8192},
                 'response_body': '{"model":"qwen3.8-27b-fp8","usage":{"input_tokens":12}}'}]
        stopped = False
        def __init__(self, path, credential):
            path.mkdir()
        def start(self):
            pass
        def stop(self):
            self.stopped = True

    def client(argv, env, wd, prompt, history, timeout):
        assert '--bare' in argv and '--restricted' in argv
        assert argv[argv.index('--tools') + 1] == 'Read,Edit,mcp__visible-tests__run_visible_tests'
        assert argv[argv.index('--effort') + 1] == 'medium'
        assert env['CLAUDE_CODE_MAX_CONTEXT_TOKENS'] == '65536'
        assert env['CLAUDE_CODE_MAX_OUTPUT_TOKENS'] == '8192'
        assert all('gateway-secret' not in value for value in env.values())
        (wd / 'solution.txt').write_text('after')
        events = [{'type': 'assistant', 'message': {'id': 'm1', 'model': 'qwen3.8-27b-fp8',
                  'content': [{'type': 'thinking', 'thinking': 'check'}, {'type': 'text', 'text': 'done'}]}},
                  {'type': 'result', 'is_error': False, 'result': 'done'}]
        (history.directory / 'stdout.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
        for event in events:
            history.append('client_event', event)
        return {'status': 'completed', 'returncode': 0}

    result = run_remote_recorded_session(tmp_path / 'session', work, 'unused', 'fix it',
        'gateway-secret', lambda wd: {'passed': True, 'returncode': 0},
        client_mcp_config=tmp_path / 'mcp.json', client_runner=client, gateway_factory=Gateway)
    assert result['outcome'] == 'verified'
    assert result['provenance']['weight_sha256'] is None
    assert result['provenance']['hardware'] is None
    assert result['provenance']['runtime_context_tokens'] is None
    assert result['decision_ready'] is False
    assert verify_evidence(tmp_path / 'session')['complete'] is True
    from session_history import inspect_session
    requests = [event['payload'] for event in inspect_session(tmp_path / 'session')['events']
                if event['kind'] == 'request']
    assert requests[0]['context']['input_tokens'] == 12
    assert requests[0]['context']['runtime_window_tokens'] is None
    assert requests[0]['context']['requested_window_tokens'] == 65536
    assert '+after' in (tmp_path / 'session/workspace.diff').read_text()
    assert 'check' in (tmp_path / 'session/history.md').read_text()


def test_remote_openai_path_preserves_reasoning_through_authenticated_adapter(tmp_path):
    import http.client
    import json
    import urllib.parse
    from remote_recorded_session import GatewayForwarder, run_remote_recorded_session
    from recorded_session import verify_evidence
    work = tmp_path / 'work'; work.mkdir()
    class Response:
        status = 200
        def __init__(self):
            frames = [
                {'model': 'qwen3.8-27b-fp8', 'choices': [{'delta': {'reasoning_content': 'check math'}}]},
                {'choices': [{'delta': {'content': '391'}}]},
                {'choices': [{'delta': {}, 'finish_reason': 'stop'}]},
                {'choices': [], 'usage': {'prompt_tokens': 9, 'completion_tokens': 7}},
            ]
            self.chunks = iter([b'data: '+json.dumps(f).encode()+b'\n\n' for f in frames]+[b'data: [DONE]\n\n', b''])
        def getheader(self, name, default=None): return 'text/event-stream'
        def read1(self, size): return next(self.chunks)
    class Connection:
        def request(self, method, path, body, headers):
            assert headers['Authorization'] == 'Bearer gateway-secret'
            assert path == '/v1/chat/completions'
        def getresponse(self): return Response()
        def close(self): pass
    gateways = []
    def factory(path, credential, **kwargs):
        gateway = GatewayForwarder(path, credential, connection_factory=lambda *a, **k: Connection(), **kwargs)
        gateways.append(gateway)
        return gateway
    def client(argv, env, wd, prompt, history, timeout):
        assert env['ANTHROPIC_API_KEY'] != gateways[0].client_token
        url = urllib.parse.urlsplit(env['ANTHROPIC_BASE_URL'])
        c = http.client.HTTPConnection(url.hostname, url.port)
        c.request('POST', '/v1/messages', json.dumps({'model': 'qwen3.8-27b-fp8',
            'max_tokens': 8192, 'stream': True, 'messages': [{'role': 'user', 'content': prompt}]}),
            {'X-Api-Key': env['ANTHROPIC_API_KEY'], 'Content-Type': 'application/json'})
        response = c.getresponse(); body = response.read().decode(); c.close()
        assert response.status == 200
        assert 'check math' in body and 'message_stop' in body
        events = [{'type': 'assistant', 'message': {'id': 'test-message', 'model': 'qwen3.8-27b-fp8',
                   'content': [{'type': 'thinking', 'thinking': 'check math'}, {'type': 'text', 'text': '391'}]}},
                  {'type': 'result', 'is_error': False, 'result': '391'}]
        (history.directory / 'stdout.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
        for e in events: history.append('client_event', e)
        return {'status': 'completed', 'returncode': 0}
    result = run_remote_recorded_session(tmp_path / 'session', work, 'unused', 'Compute 17*23',
        'gateway-secret', lambda wd: {'passed': True, 'returncode': 0},
        client_mcp_config='{"mcpServers":{}}', client_runner=client,
        gateway_factory=factory, gateway_protocol='openai')
    assert result['outcome'] == 'verified', result['errors']
    assert result['provenance']['gateway_protocol'] == 'openai'
    assert verify_evidence(tmp_path / 'session')['complete']
    assert (tmp_path / 'session/frontend-wire-requests.json').is_file()
    assert b'gateway-secret' not in b''.join(p.read_bytes() for p in (tmp_path/'session').rglob('*') if p.is_file())


import pytest


@pytest.mark.parametrize('status,chunks,usable', [
    (302, [b'gateway-secret'], False),
    (429, [b'{"error":"capacity exhausted"}'], False),
    (200, [b'data: {"type":"message_start"}\n\n'], False),
    (200, [b'gateway-', b'secret'], False),
    (200, [b'data: {"type":"message_stop"}\n\n'], True),
])
def test_transport_rejects_redirect_echo_and_truncated_stream(tmp_path, status, chunks, usable):
    import http.client
    import json
    from remote_recorded_session import GatewayForwarder

    class Response:
        def __init__(self):
            self.status = status
            self.chunks = iter(chunks + [b''])
        def getheader(self, name, default=None):
            return 'text/event-stream'
        def read1(self, size):
            return next(self.chunks)

    class Connection:
        closed = False
        def request(self, *args, **kwargs):
            pass
        def getresponse(self):
            return Response()
        def close(self):
            self.closed = True

    upstream = Connection()
    gateway = GatewayForwarder(tmp_path / 'wire', 'gateway-secret',
                               connection_factory=lambda *args, **kwargs: upstream)
    gateway.start()
    try:
        connection = http.client.HTTPConnection('127.0.0.1', gateway.port)
        connection.request('POST', '/v1/messages', json.dumps({
            'model': 'qwen3.8-27b-fp8', 'max_tokens': 8192, 'stream': True, 'messages': []}),
                           {'X-Api-Key': gateway.client_token})
        body = connection.getresponse().read()
        assert b'gateway-secret' not in body
        connection.close()
    finally:
        gateway.stop()
    assert upstream.closed
    assert gateway.rows[0]['usable'] is usable
    if status == 429:
        assert b'capacity exhausted' in (tmp_path / 'wire/0/response.bin').read_bytes()
        assert b'capacity exhausted' not in body
    assert b'gateway-secret' not in b''.join(p.read_bytes() for p in tmp_path.rglob('*') if p.is_file())


def test_stop_joins_owned_inflight_handler_before_evidence_is_sealed(tmp_path):
    import http.client
    import json
    import threading
    import time
    from remote_recorded_session import GatewayForwarder
    entered, released = threading.Event(), threading.Event()

    class Connection:
        def request(self, *args, **kwargs):
            pass
        def getresponse(self):
            entered.set()
            assert released.wait(3)
            time.sleep(.1)
            raise OSError('fake stopped transport')
        def close(self):
            released.set()

    gateway = GatewayForwarder(tmp_path / 'wire', 'gateway-secret',
                               connection_factory=lambda *args, **kwargs: Connection())
    gateway.start()
    connection = http.client.HTTPConnection('127.0.0.1', gateway.port)
    connection.request('POST', '/v1/messages', json.dumps({
        'model': 'qwen3.8-27b-fp8', 'max_tokens': 8192, 'messages': []}),
                       {'X-Api-Key': gateway.client_token})
    assert entered.wait(3)
    gateway.stop()
    assert (tmp_path / 'wire/0/result.json').is_file()
    connection.close()
