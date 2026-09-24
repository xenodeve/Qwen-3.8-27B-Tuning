"""One recorded local-client attempt: raw evidence, resources and verification.

The caller owns the pre-launched loopback model and supplies frozen provenance.
This module never starts/stops a model or treats a client exit as task success.
Raw files belong in an access-restricted, git-ignored directory outside workdir.
"""
import hashlib
import difflib
import traceback
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time
import uuid
import urllib.request

from agent_session_client import build_client_argv, build_client_env, run_client
from session_history import SessionHistory, inspect_session


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inventory(directory, exclude=()):
    result = {}
    for path in sorted(Path(directory).rglob('*')):
        if path.relative_to(directory).as_posix() in exclude:
            continue
        if path.is_symlink():
            raise ValueError('symlink in evidence/workspace inventory')
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = {
                'bytes': path.stat().st_size, 'sha256': digest(path)}
    return result


def wire_usage(body):
    usage = {}
    events = []
    if body.lstrip().startswith('{'):
        events.append(json.loads(body))
    else:
        for line in body.splitlines():
            if line.startswith('data:'):
                value = line[5:].strip()
                if value and value != '[DONE]':
                    events.append(json.loads(value))
    for event in events:
        if not isinstance(event, dict):
            continue
        message = event.get('message') or {}
        if isinstance(message, dict) and isinstance(message.get('usage'), dict):
            usage.update(message['usage'])
        if isinstance(event.get('usage'), dict):
            usage.update(event['usage'])
    if 'prompt_tokens' in usage:
        total = usage['prompt_tokens']
        cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
    elif 'input_tokens' in usage:
        total = usage['input_tokens'] + (usage.get('cache_read_input_tokens') or 0) + (usage.get('cache_creation_input_tokens') or 0)
        cached = usage.get('cache_read_input_tokens')
    else:
        total = cached = None
    return {'input_tokens': total, 'cached_input_tokens': cached,
            'output_tokens': usage.get('completion_tokens', usage.get('output_tokens')),
            'raw_usage': usage}


def decide_outcome(client, wire_rows, verification, errors):
    if errors or client.get('status') == 'evidence_failure':
        return 'invalid'
    if client.get('status') == 'censored':
        return 'censored'
    if not wire_rows or not all(row.get('usable') is True for row in wire_rows):
        return 'invalid'
    if client.get('status') != 'completed':
        return 'failed'
    if not verification or type(verification.get('returncode')) is not int:
        return 'invalid'
    if verification.get('passed') is not True or verification.get('returncode') != 0:
        return 'failed'
    return 'verified'


def _load_tap_module(name):
    path = Path(__file__).resolve().parents[1] / 'tools' / 'llama-tap' / (name + '.py')
    spec = importlib.util.spec_from_file_location('recorded_session_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def observe_server(port):
    result = {}
    # Bypass ambient proxy configuration for this loopback-only instrument.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for route in ('health', 'props'):
        with opener.open(f'http://127.0.0.1:{port}/{route}', timeout=5) as response:
            result[route] = json.load(response)
    return result


def run_recorded_session(directory, workdir, executable, model, upstream_port,
                         prompt, specification, verifier, *, timeout=3600,
                         client_runner=run_client, sampler_factory=None,
                         server_probe=observe_server, client_mcp_config=None,
                         client_allowed_tools=None, recorder_source_paths=()):
    """Collect all evidence around an isolated client; verifier runs in parent.

    Failure to persist evidence raises rather than returning a success. Metadata
    supplied by the caller is recorded separately from observed wire settings.
    """
    started = time.monotonic()
    directory, workdir = Path(directory).resolve(), Path(workdir).resolve()
    if directory.is_relative_to(workdir) or workdir.is_relative_to(directory):
        raise ValueError('private evidence and model workspace must be separate')
    if not workdir.is_dir():
        raise ValueError('workspace must already be prepared')
    if type(upstream_port) is not int or not 1 <= upstream_port <= 65535:
        raise ValueError('invalid loopback upstream port')
    frozen = json.loads(json.dumps(specification, allow_nan=False))
    for artifact in frozen['artifacts']:
        if digest(artifact['path']) != artifact['sha256']:
            raise ValueError('artifact checksum mismatch')
    test, context = frozen['test'], frozen['context']
    observed_server = server_probe(upstream_port)
    props = observed_server['props']
    actual_window = props.get('default_generation_settings', {}).get('n_ctx')
    if actual_window != context['requested_window_tokens'] or actual_window != context['runtime_window_tokens']:
        raise ValueError('runtime context does not match frozen allocation')
    if observed_server.get('health', {}).get('status') != 'ok':
        raise ValueError('server not healthy')
    if not props.get('model_path') or Path(props['model_path']).resolve() != Path(frozen['artifacts'][0]['path']).resolve():
        raise ValueError('runtime model path differs from hashed artifact')
    history = SessionHistory(directory, session_id=str(uuid.uuid4()))
    _write_json(directory / 'server-observed.json', observed_server)
    _write_json(directory / 'specification.json', frozen)
    before = inventory(workdir)
    _write_json(directory / 'workspace-before.json', before)
    shutil.copytree(workdir, directory / 'workspace-initial', symlinks=True)
    history.append('provenance', {'specification_sha256': digest(directory / 'specification.json'),
                                  'test': test, 'context': context, 'model': model})
    sources = directory / 'recorder-source'; sources.mkdir()
    recorder_paths = [Path(__file__).with_name(name) for name in (
        'recorded_session.py', 'agent_session_client.py', 'session_history.py',
        'session_analysis.py', 'session_telemetry.py', 'gpu_device.py', 'campaign_register.py')]
    recorder_paths += [Path(path) for path in recorder_source_paths]
    recorder_paths += [Path(__file__).resolve().parents[1] / 'tools' / 'llama-tap' / name
                       for name in ('relay.py', 'read_capture.py')]
    names = [path.name for path in recorder_paths]
    if any(not path.is_file() for path in recorder_paths) or len(names) != len(set(names)):
        raise ValueError('recorder source paths must be unique existing files')
    for path in recorder_paths:
        shutil.copyfile(path, sources / path.name)
    config = directory / 'client-config'; config.mkdir()
    settings = config / 'settings.json'; _write_json(settings, {})
    tap = _load_tap_module('relay').Tap(0, upstream_port, str(directory / 'wire'))
    if sampler_factory is None:
        from session_telemetry import TelemetrySampler, sample_resources
        devices = frozen.get('gpu_uuids')
        if not devices:
            raise ValueError('frozen gpu_uuids required for resource telemetry')
        sampler_factory = lambda: TelemetrySampler(
            sample=lambda: sample_resources(device_uuids=devices),
            sink_path=directory / 'resource-live.jsonl')
    sampler = sampler_factory()
    errors = []
    client = {'status': 'evidence_failure', 'returncode': None}
    verification = None
    resources = None
    client_s = verification_s = 0.0
    try:
        sampler.start()
        if hasattr(sampler, 'wait_ready'):
            initial_resources = sampler.wait_ready()
            history.append('resource_preflight', initial_resources)
            if (initial_resources.get('suspected_game_processes') or initial_resources.get('errors')
                    or not initial_resources.get('gpus')):
                raise RuntimeError('resource preflight is unknown or has a competing workload')
        tap.start()
        env = build_client_env(os.environ, config, f'http://127.0.0.1:{tap.listen_port}')
        argv_kwargs = {}
        if client_mcp_config is not None:
            argv_kwargs['mcp_config'] = client_mcp_config
        if client_allowed_tools is not None:
            argv_kwargs['allowed_tools'] = client_allowed_tools
        argv = build_client_argv(executable, model, history.session_id, settings, **argv_kwargs)
        history.append('client_launch', {'argv': argv, 'workdir': str(workdir),
                                          'endpoint': env['ANTHROPIC_BASE_URL'], 'timeout_s': timeout})
        t0 = time.monotonic()
        client = client_runner(argv, env, workdir, prompt, history, timeout=timeout)
        client_s = time.monotonic() - t0
    except Exception as error:
        errors.append({'phase': 'client', 'type': type(error).__name__})
    finally:
        try:
            tap.stop()
            errors.extend({'phase': 'wire', 'type': value} for value in tap.capture_errors)
        except Exception as error:
            errors.append({'phase': 'wire_shutdown', 'type': type(error).__name__})
        try:
            resources = sampler.stop()
        except Exception as error:
            errors.append({'phase': 'telemetry', 'type': type(error).__name__})
    _write_json(directory / 'resource-samples.json', resources)
    try:
        wire = _load_tap_module('read_capture').rows(str(directory / 'wire'))
    except Exception as error:
        wire = []
        errors.append({'phase': 'wire_parse', 'type': type(error).__name__})
    _write_json(directory / 'wire-requests.json', wire)
    inference = []
    measured_depths = []
    for row in wire:
        if row.get('method') != 'POST' or (row.get('path') or '').split('?', 1)[0] not in (
                '/v1/messages', '/v1/chat/completions'):
            history.append('auxiliary_request', row)
            continue
        inference.append(row)
        try:
            usage = wire_usage(row.get('response_body') or '')
            request = row.get('request')
            if not isinstance(request, dict):
                errors.append({'phase': 'request', 'type': 'missing_request_object'})
                continue
            observed = dict(context, input_tokens=usage['input_tokens'],
                            cached_input_tokens=usage['cached_input_tokens'],
                            max_output_tokens=request.get('max_tokens'),
                            token_count_source='server_usage' if usage['input_tokens'] is not None else None,
                            declared_history_policy=context.get('history_policy', 'unknown'),
                            history_policy='unknown',
                            source=context.get('source', 'unknown'))
            rid = f"{row['connection']}:{row['request_index']}"
            history.record_request(rid, test, observed, request)
            if usage['input_tokens'] is not None:
                measured_depths.append(usage['input_tokens'])
            history.append('request_result', {'request_id': rid, 'usage': usage,
                'http_status': row.get('status'), 'usable': row.get('usable'),
                'first_byte_s': row.get('first_byte_s'), 'response_ref': 'wire-requests.json'})
        except Exception as error:
            errors.append({'phase': 'request_normalization', 'type': type(error).__name__})
    events = []
    try:
        with (directory / 'stdout.jsonl').open(encoding='utf-8') as stream:
            events = [json.loads(line) for line in stream if line.strip()]
        from session_analysis import analyze_events, render_history
        analysis = analyze_events(events)
        _write_json(directory / 'analysis-private.json', analysis)
        errors.extend({'phase': 'analysis', 'type': str(value)} for value in analysis.get('errors', []))
        initial_prompt = {'type': 'user', 'source': 'runner_initial_prompt',
                          'message': {'content': [{'type': 'text', 'text': prompt}]}}
        (directory / 'history.md').write_text(render_history([initial_prompt] + events), encoding='utf-8')
    except Exception as error:
        analysis = None
        errors.append({'phase': 'client_analysis', 'type': type(error).__name__})
    if client['status'] == 'completed':
        t0 = time.monotonic()
        try:
            verification = verifier(workdir)
            if not isinstance(verification, dict):
                raise ValueError('verifier must return structured evidence')
        except Exception as error:
            verification = {'passed': False, 'returncode': None, 'error_type': type(error).__name__}
        verification_s = time.monotonic() - t0
    history.append('verification', {'result': verification, 'elapsed_s': verification_s})
    _write_json(directory / 'verification-private.json', verification)
    after = inventory(workdir)
    _write_json(directory / 'workspace-after.json', after)
    shutil.copytree(workdir, directory / 'workspace-final', symlinks=True)
    with (directory / 'workspace.diff').open('x', encoding='utf-8') as diff:
        for name in sorted(before.keys() | after.keys()):
            if before.get(name) == after.get(name):
                continue
            try:
                old = (directory / 'workspace-initial' / name).read_text(encoding='utf-8').splitlines() if name in before else []
                new = (directory / 'workspace-final' / name).read_text(encoding='utf-8').splitlines() if name in after else []
                diff.write('\n'.join(difflib.unified_diff(old, new, fromfile='before/' + name,
                           tofile='after/' + name, lineterm='')) + '\n')
            except UnicodeDecodeError:
                diff.write(f'Binary file changed: {name}; see exact workspace snapshots\n')
    changes = {'added': sorted(after.keys() - before.keys()),
               'removed': sorted(before.keys() - after.keys()),
               'modified': sorted(key for key in before.keys() & after.keys() if before[key] != after[key])}
    outcome = decide_outcome(client, inference, verification, errors)
    metrics = ({key: analysis.get(key) for key in ('assistant_messages', 'tool_calls',
                'tool_errors', 'thinking_chars', 'answer_chars', 'thinking_states', 'model_names')}
               if analysis is not None else None)
    blockers = ['manual_quality_review_pending', 'campaign_coverage_check_pending']
    samples = resources.get('samples', []) if isinstance(resources, dict) else []
    if not samples or any(not sample.get('gpus') or sample.get('errors') for sample in samples):
        blockers.append('resource_telemetry_incomplete')
    if any(sample.get('suspected_game_processes') for sample in samples):
        blockers.append('suspected_competing_workload')
    if len(measured_depths) != len(inference):
        blockers.append('request_token_depth_incomplete')
    if errors:
        blockers.append('evidence_errors')
    summary = {'session_id': history.session_id, 'outcome': outcome, 'test': test,
               'decision_ready': False, 'decision_blockers': blockers,
               'context_high_water_tokens': max(measured_depths) if measured_depths else None,
               'context': context, 'client': client, 'analysis': metrics,
               'verification': ({key: verification.get(key) for key in ('passed','returncode','error_type')}
                                if verification else None),
               'verification_ref': 'verification-private.json', 'workspace_changes': changes,
               'journal_write_s': history.recording_write_s,
               'telemetry_collection_s': resources.get('collection_s') if isinstance(resources, dict) else None,
               'client_s': client_s, 'verification_s': verification_s,
               'elapsed_s': time.monotonic() - started, 'errors': errors,
               'unmeasured': ['guard_false_changes', 'recorder_counterfactual_overhead',
                              'manual_quality_review', 'per_request_stage_correlation']}
    _write_json(directory / 'summary.json', summary)
    files = inventory(directory, exclude=('events.jsonl', 'manifest.json'))
    _write_json(directory / 'evidence-inventory.json', files)
    history.append('evidence_inventory', {'path': 'evidence-inventory.json',
                  'sha256': digest(directory / 'evidence-inventory.json')})
    history.finish(outcome if outcome in ('verified', 'censored') else 'failed',
                   {'outcome': outcome, 'summary': 'summary.json'})
    return summary


def verify_evidence(directory):
    directory = Path(directory).resolve()
    session = inspect_session(directory)
    if not session['complete']:
        return {'complete': False, 'errors': ['journal_not_final_or_corrupt']}
    errors = []
    try:
        refs = [event['payload'] for event in session['events'] if event['kind'] == 'evidence_inventory']
        if len(refs) != 1 or refs[0]['path'] != 'evidence-inventory.json':
            raise ValueError('missing or ambiguous inventory reference')
        target = directory / 'evidence-inventory.json'
        if target.is_symlink() or digest(target) != refs[0]['sha256']:
            raise ValueError('inventory digest mismatch')
        files = json.loads(target.read_text(encoding='utf-8'))
        for name, expected in files.items():
            path = directory / name
            if path.is_symlink() or not path.resolve().is_relative_to(directory):
                raise ValueError('unsafe inventory path')
            if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
                errors.append('artifact_mismatch:' + name)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        errors.append(type(error).__name__)
    return {'complete': not errors, 'errors': errors}


def run_registered_attempt(register, attempt_id, *args, runner=run_recorded_session, **kwargs):
    """Account for every scheduled cell, including failures before model work."""
    register.mark(attempt_id, 'running')
    started = time.monotonic()
    try:
        result = runner(*args, **kwargs)
    except Exception as error:
        # Private diagnostic; public progress stores only the failure category.
        diagnostic = register.path / (uuid.uuid4().hex + '-failure.txt')
        diagnostic.write_text(traceback.format_exc(), encoding='utf-8')
        result = {'outcome': 'invalid', 'elapsed_s': time.monotonic() - started,
                  'error_type': type(error).__name__, 'diagnostic_ref': diagnostic.name}
    evidence = {'session_id': result.get('session_id'), 'elapsed_s': result.get('elapsed_s'),
                'verifier_evidence_ref': str(Path(args[0]) / 'summary.json') if args else None}
    outcome = result['outcome']
    register.mark(attempt_id, outcome, evidence=evidence,
                  reason=None if outcome == 'verified' else result.get('error_type', outcome))
    return result
