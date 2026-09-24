"""Remote-only FP8 control; credentials belong exclusively to the final hop."""
import http.client
import hashlib
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import ssl
import secrets
import threading
import time

from agent_session_client import build_client_env
from recorded_session import _write_json
from stream_observation import StreamObservation

MODEL = 'qwen3.8-27b-fp8'
GATEWAY = 'https://gateway.9arm.co'
CONTEXT, OUTPUT, TURNS, TIMEOUT, EFFORT = 65536, 8192, 64, 1800, 'medium'
ROUTES = {'/v1/messages', '/v1/messages?beta=true', '/v1/messages/count_tokens',
          '/v1/messages/count_tokens?beta=true'}


class GatewayForwarder:
    """Owned loopback capture boundary; no caller-selected remote destination.

    The injectable HTTPS connection factory is only an offline transport seam.
    Headers are rebuilt, never copied or captured. Secret-bearing responses fail
    closed; exceptions are recorded by category, never by message or traceback.
    """
    def __init__(self, directory, credential, *, protocol='anthropic',
                 connection_factory=http.client.HTTPSConnection, clock_ns=time.perf_counter_ns):
        if protocol not in ('anthropic', 'openai'):
            raise ValueError('unsupported gateway protocol')
        self.protocol = protocol
        self._clock_ns = clock_ns
        if not credential or '\r' in credential or '\n' in credential:
            raise ValueError('invalid gateway credential')
        self.directory = Path(directory)
        self.directory.mkdir()
        self._credential = credential
        self.client_token = secrets.token_urlsafe(32)
        self._factory = connection_factory
        self._tls_context = ssl.create_default_context()
        self.rows = []
        self.errors = []
        self._lock = threading.Lock()
        self._connections = set()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.0'
            def log_message(self, *args):
                pass
            def do_POST(self):
                owner._forward(self)
            def do_GET(self):
                self.send_error(405, 'method not allowed')

        self._server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self._server.daemon_threads = False
        self.port = self._server.server_port
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        if self._thread.is_alive():
            self._server.shutdown()
            self._thread.join()
        with self._lock:
            for connection in self._connections:
                connection.close()
        self._server.server_close()

    def _forward(self, handler):
        supplied = handler.headers.get_all('x-api-key') or []
        if len(supplied) != 1 or not secrets.compare_digest(supplied[0], self.client_token):
            handler.send_error(401, 'local session authentication required')
            handler.close_connection = True
            return
        connection = None
        sent_headers = False
        row = None
        observer = None
        eof_ns = None
        observation_error = None
        started = time.monotonic()
        try:
            routes = ROUTES if self.protocol == 'anthropic' else {'/v1/chat/completions'}
            if handler.path not in routes or handler.headers.get('Transfer-Encoding'):
                raise ValueError('request policy')
            lengths = handler.headers.get_all('Content-Length') or []
            if len(lengths) != 1 or not lengths[0].isdigit():
                raise ValueError('request length')
            length = int(lengths[0])
            if not 0 < length <= 16 * 1024 * 1024:
                raise ValueError('request size')
            handler.connection.settimeout(60)
            body = handler.rfile.read(length)
            if len(body) != length or self._credential.encode() in body:
                raise ValueError('request body')
            request = json.loads(body)
            if not isinstance(request, dict) or request.get('model') != MODEL:
                raise ValueError('request model')
            client_body = body
            if '/count_tokens' not in handler.path:
                if request.get('max_tokens') != OUTPUT:
                    raise ValueError('output policy')
                request.update(temperature=1.0, top_p=0.95, top_k=20)
                body = json.dumps(request, ensure_ascii=False, allow_nan=False).encode('utf-8')
            with self._lock:
                index = len(self.rows)
                row = {'method': 'POST', 'path': handler.path, 'request': request,
                       'usable': False, 'response_body': '', 'status': None,
                       'first_byte_s': None, 'request_index': index,
                       'request_sha256': hashlib.sha256(body).hexdigest(),
                       'observation_ref': f'wire/{index}/timing.json'}
                self.rows.append(row)
            target = self.directory / str(index)
            target.mkdir()
            for name, payload in (('client-request.bin', client_body), ('request.bin', body)):
                with (target / name).open('xb') as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            connection = self._factory('gateway.9arm.co', timeout=60,
                                       context=self._tls_context)
            with self._lock:
                self._connections.add(connection)
            # Credential injection is AFTER durable secret-free request capture.
            headers = {'Content-Type': 'application/json',
                       'anthropic-version': '2023-06-01',
                       'Authorization': 'Bearer ' + self._credential}
            observer = StreamObservation(self.protocol, uuid.uuid4().hex,
                row['request_sha256'], self._clock_ns())
            connection.request('POST', handler.path, body=body, headers=headers)
            response = connection.getresponse()
            observer.headers(self._clock_ns())
            row['status'] = response.status
            accepted = 200 <= response.status < 300
            content_type = response.getheader('Content-Type', '').split(';')[0]
            accepted = accepted and content_type in ('application/json', 'text/event-stream')
            row['response_content_type'] = content_type if content_type in ('application/json', 'text/event-stream') else 'other'
            if accepted:
                handler.send_response(response.status)
                handler.send_header('Content-Type', content_type)
                handler.send_header('Connection', 'close')
                handler.end_headers()
                sent_headers = True
            # Delay only a secret-length suffix to catch echoes across chunks.
            secret = self._credential.encode()
            pending = b''
            chunks = []
            with (target / 'response.bin').open('xb') as stream:
                while True:
                    chunk = response.read1(65536)
                    received_ns = self._clock_ns()
                    if chunk:
                        observer.feed(chunk, received_ns)
                    else:
                        eof_ns = received_ns
                        observer.eof(eof_ns)
                    if chunk and row['first_byte_s'] is None:
                        row['first_byte_s'] = time.monotonic() - started
                    pending += chunk
                    if secret in pending:
                        raise RuntimeError('credential echo refused')
                    size = max(0, len(pending) - len(secret) + 1) if chunk else len(pending)
                    safe, pending = pending[:size], pending[size:]
                    stream.write(safe)
                    stream.flush()
                    chunks.append(safe)
                    if accepted:
                        handler.wfile.write(safe)
                        handler.wfile.flush()
                    if not chunk:
                        break
                os.fsync(stream.fileno())
            row['response_body'] = b''.join(chunks).decode('utf-8')
            if not accepted:
                raise RuntimeError('upstream rejected or redirected')
            observer.finish(eof_ns)
            if content_type == 'text/event-stream':
                if not observer.snapshot()['complete']:
                    raise RuntimeError('incomplete event stream')
            else:
                json.loads(row['response_body'])
            row['usable'] = True
        except Exception as error:
            observation_error = 'transport_error'
            self.errors.append(type(error).__name__)
            if not sent_headers:
                handler.send_error(400 if isinstance(error, (ValueError, json.JSONDecodeError)) else 502,
                                   'gateway request failed')
        finally:
            if connection is not None:
                connection.close()
                with self._lock:
                    self._connections.discard(connection)
            if observer is not None and row is not None:
                try:
                    if observer.snapshot()['timing_ns']['end'] is None:
                        observer.finish(eof_ns if eof_ns is not None else self._clock_ns(),
                                        error=observation_error)
                    observation = observer.snapshot()
                    observation['request_content_sha256'] = hashlib.sha256(json.dumps(
                        request, sort_keys=True, ensure_ascii=False, separators=(',', ':'),
                        allow_nan=False).encode('utf-8')).hexdigest()
                    observation['request_content_hash_algorithm'] = 'sha256-canonical-json-v1'
                    observation['transport_usable'] = row['usable']
                    _write_json(self.directory / str(row['request_index']) / 'timing.json', observation)
                except Exception:
                    row['usable'] = False
                    self.errors.append('observation_persistence_failure')
            if row is not None:
                row['elapsed_s'] = time.monotonic() - started
                _write_json(self.directory / str(row['request_index']) / 'result.json', row)
            handler.close_connection = True


def remote_client_env(base, config, endpoint, credential):
    """Drop inherited credentials, including aliases of the gateway secret."""
    allowed = {'PATH', 'PATHEXT', 'SYSTEMROOT', 'WINDIR', 'COMSPEC', 'TEMP', 'TMP',
               'USERPROFILE', 'HOME', 'APPDATA', 'LOCALAPPDATA', 'PROGRAMDATA',
               'PROGRAMFILES', 'PROGRAMFILES(X86)', 'SYSTEMDRIVE', 'HOMEDRIVE',
               'HOMEPATH', 'LANG', 'LC_ALL', 'PYTHONIOENCODING'}
    clean = {key: value for key, value in base.items()
             if key.upper() in allowed and not (credential and credential in value)}
    return build_client_env(clean, config, endpoint)


def run_remote_recorded_session(directory, workdir, executable, prompt, credential,
                                verifier, *, client_mcp_config,
                                client_runner=None, gateway_factory=GatewayForwarder,
                                gateway_protocol='anthropic',
                                client_allowed_tools='Read,Edit,mcp__visible-tests__run_visible_tests',
                                recorder_source_paths=()):
    """Record one restricted remote attempt without admitting it as local evidence."""
    import difflib
    import shutil
    import uuid
    from agent_session_client import build_client_argv, run_client
    from recorded_session import digest, inventory, decide_outcome, _load_tap_module
    from session_history import SessionHistory
    from session_analysis import analyze_events, render_history

    if gateway_protocol not in ('anthropic', 'openai'):
        raise ValueError('unsupported gateway protocol')
    directory, workdir = Path(directory).resolve(), Path(workdir).resolve()
    if directory.is_relative_to(workdir) or workdir.is_relative_to(directory):
        raise ValueError('evidence and workspace must be separate')
    if not workdir.is_dir():
        raise ValueError('workspace must exist')
    started = time.monotonic()
    history = SessionHistory(directory, session_id=str(uuid.uuid4()))
    provenance = {'kind': 'remote', 'requested_model': MODEL, 'gateway': GATEWAY,
        'route': 'controlled claude-9arm, not everyday launcher',
        'gateway_protocol': gateway_protocol,
        'quantization': 'provider model label FP8; unverified', 'weight_sha256': None,
        'hardware': None, 'runtime': None, 'runtime_context_tokens': None,
        'seed': None, 'min_p': None, 'prefill_tps': None, 'decode_tps': None}
    _write_json(directory / 'provenance.json', provenance)
    history.append('provenance', provenance)
    history.append('user_prompt', prompt)
    before = inventory(workdir)
    _write_json(directory / 'workspace-before.json', before)
    shutil.copytree(workdir, directory / 'workspace-initial')
    sources = directory / 'recorder-source'
    sources.mkdir()
    paths = [Path(__file__).with_name(name) for name in (
        'remote_recorded_session.py', 'recorded_session.py', 'agent_session_client.py',
        'session_history.py', 'session_analysis.py', 'stream_observation.py', 'anthropic_adapter.py')]
    if gateway_protocol == 'openai':
        root = Path(__file__).resolve().parents[1]
        paths += [root / 'serving/exl3/anthropic_compat.py',
                  root / 'tools/llama-tap/relay.py', root / 'tools/llama-tap/read_capture.py']
    paths.extend(Path(path) for path in recorder_source_paths)
    if len({path.name for path in paths}) != len(paths):
        raise ValueError('duplicate recorder source name')
    for path in paths:
        shutil.copyfile(path, sources / path.name)
    config = directory / 'client-config'
    config.mkdir()
    settings = config / 'settings.json'
    _write_json(settings, {})
    errors = []
    client = {'status': 'evidence_failure', 'returncode': None}
    verification = None
    gateway = adapter = frontend_tap = None
    client_s = verification_s = 0.0
    try:
        if gateway_protocol == 'openai':
            gateway = gateway_factory(directory / 'wire', credential, protocol='openai')
        else:
            gateway = gateway_factory(directory / 'wire', credential)
        gateway.start()
        endpoint_port, client_token = gateway.port, gateway.client_token
        if gateway_protocol == 'openai':
            from anthropic_adapter import AdapterServer
            client_token = secrets.token_urlsafe(32)
            def transform(body):
                body.update(reasoning_effort=EFFORT, stream_options={'include_usage': True})
                return body
            adapter = AdapterServer(f'http://127.0.0.1:{gateway.port}',
                client_api_key=client_token, upstream_api_key=gateway.client_token,
                request_transform=transform).start()
            frontend_tap = _load_tap_module('relay').Tap(0, adapter.port, str(directory / 'frontend-wire'))
            frontend_tap.start()
            endpoint_port = frontend_tap.listen_port
        env = remote_client_env(os.environ, config, f'http://127.0.0.1:{endpoint_port}', credential)
        env['ANTHROPIC_API_KEY'] = client_token
        env.update(CLAUDE_CODE_MAX_CONTEXT_TOKENS=str(CONTEXT), CLAUDE_CODE_MAX_OUTPUT_TOKENS=str(OUTPUT),
                   CLAUDE_CODE_MAX_TURNS=str(TURNS))
        argv = build_client_argv(executable, MODEL, history.session_id, settings,
            mcp_config=client_mcp_config, allowed_tools=client_allowed_tools)
        argv += ['--effort', EFFORT]
        history.append('effective_client_policy', {'argv': argv, 'context': CONTEXT,
                                                  'output_cap': OUTPUT, 'turn_cap': TURNS})
        t0 = time.monotonic()
        try:
            client = (client_runner or run_client)(argv, env, workdir, prompt, history, timeout=TIMEOUT)
        finally:
            client_s = time.monotonic() - t0
    except Exception as error:
        errors.append({'phase': 'client', 'type': type(error).__name__})
    finally:
        for component, owned in (('frontend_tap', frontend_tap), ('adapter', adapter), ('gateway', gateway)):
            if owned is not None:
                try:
                    owned.stop()
                except Exception as error:
                    errors.append({'phase': component + '_cleanup', 'type': type(error).__name__})
        if gateway is not None:
            errors.extend({'phase': 'wire', 'type': value} for value in gateway.errors)
        if frontend_tap is not None:
            errors.extend({'phase': 'frontend_wire', 'type': value} for value in frontend_tap.capture_errors)
            try:
                frontend_rows = _load_tap_module('read_capture').rows(str(directory / 'frontend-wire'))
                _write_json(directory / 'frontend-wire-requests.json', frontend_rows)
                frontend_inference = [row for row in frontend_rows if row.get('method') == 'POST'
                    and row.get('path', '').split('?')[0] == '/v1/messages']
                if not frontend_inference or not all(row.get('usable') for row in frontend_inference):
                    errors.append({'phase': 'frontend_wire', 'type': 'incomplete_frontend_capture'})
            except Exception as error:
                errors.append({'phase': 'frontend_wire_parse', 'type': type(error).__name__})
    rows = gateway.rows if gateway is not None else []
    _write_json(directory / 'wire-requests.json', rows)
    inference = [row for row in rows if '/count_tokens' not in row['path']]
    from recorded_session import wire_usage
    for index, row in enumerate(inference):
        try:
            usage = wire_usage(row.get('response_body') or '')
            context = {'requested_window_tokens': CONTEXT, 'runtime_window_tokens': None,
                'input_tokens': usage['input_tokens'], 'cached_input_tokens': usage['cached_input_tokens'],
                'max_output_tokens': row['request'].get('max_tokens'),
                'token_count_source': 'provider_usage' if usage['input_tokens'] is not None else None,
                'history_policy': 'unknown', 'source': 'pal-fp8-remote-v1'}
            history.record_request(str(index), {'format': 'claude-cli', 'case_id': 'pal',
                'stage': 'first-pass', 'round': 1, 'attempt': 1}, context, row['request'])
            history.append('request_result', {'request_id': str(index), 'usage': usage,
                'http_status': row.get('status'), 'usable': row.get('usable'),
                'first_byte_s': row.get('first_byte_s'), 'response_ref': 'wire-requests.json'})
        except Exception as error:
            errors.append({'phase': 'request_normalization', 'type': type(error).__name__})
    # Snapshot before the parent-only verifier; candidate submission is immutable evidence.
    after = inventory(workdir)
    _write_json(directory / 'workspace-after.json', after)
    shutil.copytree(workdir, directory / 'workspace-final')
    changes = sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name))
    with (directory / 'workspace.diff').open('x', encoding='utf-8') as diff:
        for name in changes:
            try:
                old = (directory / 'workspace-initial' / name).read_text(encoding='utf-8').splitlines() if name in before else []
                new = (directory / 'workspace-final' / name).read_text(encoding='utf-8').splitlines() if name in after else []
                diff.write('\n'.join(difflib.unified_diff(old, new, fromfile='before/' + name,
                                                        tofile='after/' + name, lineterm='')) + '\n')
            except UnicodeDecodeError:
                diff.write('Binary changed: ' + name + '\n')
    analysis = None
    try:
        events = [json.loads(line) for line in (directory / 'stdout.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        analysis = analyze_events(events)
        _write_json(directory / 'analysis-private.json', analysis)
        errors.extend({'phase': 'analysis', 'type': str(value)} for value in analysis.get('errors', []))
        initial = {'type': 'user', 'message': {'content': [{'type': 'text', 'text': prompt}]}}
        (directory / 'history.md').write_text(render_history([initial] + events), encoding='utf-8')
    except Exception as error:
        errors.append({'phase': 'analysis', 'type': type(error).__name__})
    if client['status'] == 'completed':
        t0 = time.monotonic()
        try:
            verification = verifier(workdir)
        except Exception as error:
            errors.append({'phase': 'verification', 'type': type(error).__name__})
        verification_s = time.monotonic() - t0
    _write_json(directory / 'verification-private.json', verification)
    history.append('verification', {'result': verification, 'elapsed_s': verification_s})
    outcome = decide_outcome(client, inference, verification, errors)
    summary = {'outcome': outcome, 'session_id': history.session_id, 'client': client,
        'provenance': provenance, 'decision_ready': False,
        'decision_blockers': ['primary_review_pending', 'effective_client_context_check_pending',
                              'remote_provenance_unknown'],
        'workspace_changes': changes, 'client_s': client_s, 'verification_s': verification_s,
        'elapsed_s': time.monotonic() - started, 'errors': errors}
    _write_json(directory / 'summary.json', summary)
    _write_json(directory / 'evidence-inventory.json', inventory(directory, exclude=('events.jsonl', 'manifest.json')))
    history.append('evidence_inventory', {'path': 'evidence-inventory.json',
                  'sha256': digest(directory / 'evidence-inventory.json')})
    history.finish(outcome if outcome in ('verified', 'censored') else 'failed',
                   {'outcome': outcome, 'summary': 'summary.json'})
    return summary
