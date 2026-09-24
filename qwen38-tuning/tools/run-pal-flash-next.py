"""Flash-Next vs GSQ IQ3_S-MTP on the frozen PAL task, ABBA in one sitting (2026-09-24).

Same task, prompt, hidden suite, sandbox image, context 65,536, output 8192,
64 turns and 1800 s as run-pal-remaining.py (result 25). Changes, recorded in
the frozen protocol: cells gsq-a, flash_next-a, flash_next-b, gsq-b; client is
the only installed Claude Code (~/.local/bin/claude.exe, 2.1.281) -- the pinned
2.1.258 file is gone since 2026-09-23; no legacy supervisor lease, any
.port8080.lock refuses the run; Flash-Next boots the served 128k profile argv at
65,536 with its env and the working-set trim the profile performs.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import ctypes

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'qwen38-tuning/bench'))


def load_predecessor(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pal = load_predecessor('run-q4-pal-workflow')
incumbents = load_predecessor('run-preliminary-incumbents')
gsq = pal.gsq
PAL_PROMPT, PAL_SHA, PAL_HIDDEN, CLIENT, MCP_TOOL = (
    pal.PAL_PROMPT, pal.PAL_SHA, pal.PAL_HIDDEN, pal.CLIENT, pal.MCP_TOOL)
CLIENT = Path.home() / '.local/bin/claude.exe'
CLIENT_SHA = gsq.digest(CLIENT)
IMAGE_ID = 'sha256:3a368bcdf5c27140d15e03f9143ed3530763d0d9d937fc34c2a68e7dfd094dd1'
CANDIDATES = ('gsq-a', 'flash_next-a', 'flash_next-b', 'gsq-b')
CONTEXT, OUTPUT, TURNS, TIMEOUT = 65536, 8192, 64, 1800


def candidate_argv(key):
    if key not in ('gsq', 'flash_next'):
        raise ValueError('unknown candidate')
    if key == 'flash_next':
        return gsq.apply_template(gsq.flash_next_argv(CONTEXT, 18080), 'stock')
    if key == 'exl3':
        return gsq.exl3_argv(CONTEXT)
    if key == 'nvfp4':
        argv = gsq.production_nvfp4_argv(CONTEXT)
    else:
        argv = gsq.apply_template(gsq.llama_argv(key, CONTEXT, 'mtp-ngram', '8500,15468', 24), 'stock')
    return gsq.replace_flag(gsq.replace_flag(argv, '--port', 18080), '-lv', 4)


def check_nvfp4_budget(pre):
    budgets = [gpu['memory_free_mib'] - (2500 if gpu['memory_used_mib'] > 500 else 512)
               for gpu in pre['gpus']]
    demand = 14173 + int(CONTEXT * 18 / 1024) + 2048
    if len(budgets) != 2 or min(budgets) < 1024 or sum(budgets) < demand:
        raise RuntimeError(f'NVFP4 source-profile memory refusal: budgets={budgets}, demand={demand}')


def clone_snapshot(work):
    """No git invocation: candidate-controlled config must never execute hooks."""
    work = Path(work)
    files = pal.inventory(work)
    metadata = {}
    for name in ('HEAD', 'config'):
        path = work / '.git' / name
        metadata['git_' + name.lower()] = path.read_text(encoding='utf-8') if path.is_file() else None
    return {'files': files, **metadata}


def clone_changes(before, after):
    return sorted(name for name in before['files'].keys() | after['files'].keys()
                  if before['files'].get(name) != after['files'].get(name))


def assert_ports_free(ports=(8000, 8080, 18080)):
    for port in ports:
        with socket.socket() as connection:
            connection.settimeout(.2)
            if connection.connect_ex(('127.0.0.1', port)) == 0:
                raise RuntimeError(f'occupied port {port}')


def cleanup_owned(adapter, tap, model, ports):
    errors = []
    for name, obj in (('adapter', adapter), ('tap', tap), ('model', model)):
        if obj is None:
            continue
        try:
            obj.stop()
        except Exception as error:
            errors.append({'component': name, 'error': repr(error)})
    # A broken stop implementation must not prevent the independent PID fallback.
    process = model.process if model is not None else None
    if process is not None and process.poll() is None:
        try:
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], check=False,
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        except Exception as error:
            errors.append({'component': 'model_pid', 'error': repr(error)})
    deadline = time.monotonic() + 15
    while True:
        try:
            assert_ports_free(ports)
            break
        except RuntimeError as error:
            if time.monotonic() >= deadline:
                errors.append({'component': 'ports', 'error': repr(error)})
                break
            time.sleep(.25)
    return errors


def classify_summary(summary):
    if summary['outcome'] == 'invalid':
        return 'evidence_failure'
    if summary['client']['status'] == 'censored':
        return 'model_timeout'
    if summary['outcome'] != 'verified':
        return 'model_failure'
    return 'original_verified_audit_pending'


def file_record(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': gsq.digest(path)}


def exl3_files():
    directory = incumbents.EXL3
    shards = sorted(directory.glob('model*.safetensors'))
    if [path.name for path in shards] != sorted(incumbents.EXPECTED_SHARDS):
        raise ValueError('EXL3 shard set mismatch')
    files = []
    for shard in shards:
        record = file_record(shard)
        if record['sha256'] != incumbents.EXPECTED_SHARDS[shard.name]:
            raise ValueError('EXL3 shard hash mismatch')
        files.append(record)
    files.extend(file_record(path) for path in sorted(directory.iterdir())
                 if path.is_file() and path.suffix in ('.json', '.model', '.jinja'))
    return files


@contextmanager
def original_guard(root):
    before = pal.pal_state()
    pal._write_json(root / 'pal-original-before.json', before)
    try:
        if before['head'] != PAL_SHA:
            raise RuntimeError('original PAL HEAD mismatch')
        yield before
    finally:
        after = pal.pal_state()
        pal._write_json(root / 'pal-original-after.json', after)
        if before != after:
            raise RuntimeError('original PAL repository changed')


def run_task(cell, template, model_name, adapter_port, files, observed, uuids, times=None,
             observation_directory=None):
    work, session = cell / 'work', cell / 'session'
    shutil.copytree(template, work)
    suites = cell / 'suites.json'
    pal._write_json(suites, {'user-config-focused': {
        'args': ['tests/test_user_config_dir_rename.py', '-q'],
        'evidence_files': ['clink/registry.py', 'tests/test_user_config_dir_rename.py']}})
    mcp_path = cell / 'mcp.json'
    pal._write_json(mcp_path, pal.mcp_config(ROOT / 'qwen38-tuning/bench/fixture_test_tool.py',
        work, pal.SANDBOX_IMAGE, cell / 'visible-test-evidence', suites))
    times = times if times is not None else {}
    snapshots = {}

    def runner(argv, env, wd, prompt, history, timeout):
        snapshots['before'] = clone_snapshot(wd)
        pal._write_json(session / 'clone-before-client.json', snapshots['before'])
        env = dict(env, CLAUDE_CODE_MAX_CONTEXT_TOKENS=str(CONTEXT),
                   CLAUDE_CODE_MAX_OUTPUT_TOKENS=str(OUTPUT), CLAUDE_CODE_MAX_TURNS=str(TURNS))
        argv = list(argv) + ['--effort', 'medium']
        history.append('effective_client_policy', {'argv': argv, 'context': CONTEXT,
                                                  'output_cap': OUTPUT, 'turn_cap': TURNS})
        times['start'] = time.monotonic()
        try:
            return pal.run_client(argv, env, wd, prompt, history, timeout=timeout)
        finally:
            times['client_end'] = time.monotonic()
            snapshots['after'] = clone_snapshot(wd)
            pal._write_json(session / 'clone-after-client.json', snapshots['after'])
            visible = cell / 'visible-test-evidence'
            if visible.is_dir():
                shutil.copytree(visible, session / 'visible-test-evidence')
            if observation_directory is not None:
                pal.inventory(observation_directory)  # refuse links before copying
                shutil.copytree(observation_directory, session / 'upstream-observations')

    def verifier(wd):
        try:
            return pal.verify_pal_workspace(wd, template, pal.SANDBOX_IMAGE, PAL_HIDDEN,
                cell / 'verification', visible_test_journal=cell / 'visible-test-evidence/visible-tests.jsonl')
        finally:
            times['end'] = time.monotonic()

    spec = {'artifacts': files, 'gpu_uuids': uuids,
        'test': {'format': 'claude-cli', 'case_id': 'pal', 'stage': 'first-pass', 'round': 1, 'attempt': 1},
        'context': {'requested_window_tokens': CONTEXT, 'runtime_window_tokens': CONTEXT,
            'input_tokens': None, 'cached_input_tokens': None, 'max_output_tokens': OUTPUT,
            'token_count_source': None, 'history_policy': 'client-default-compaction',
            'source': 'pal-flash-next-vs-gsq-v1'}}
    summary = pal.run_recorded_session(session, work, CLIENT, model_name, adapter_port,
        PAL_PROMPT, spec, verifier, timeout=TIMEOUT, client_runner=runner,
        server_probe=lambda _port: observed, client_mcp_config=mcp_path,
        client_allowed_tools=MCP_TOOL, recorder_source_paths=[Path(__file__),
            ROOT / 'qwen38-tuning/bench/fixture_test_tool.py',
            ROOT / 'qwen38-tuning/bench/pal_workflow_screen.py',
            ROOT / 'qwen38-tuning/bench/anthropic_adapter.py',
            ROOT / 'qwen38-tuning/bench/stream_observation.py'])
    times.setdefault('end', times.get('client_end', time.monotonic()))
    allowed = {'clink/registry.py', 'tests/test_user_config_dir_rename.py'}
    changes = clone_changes(snapshots['before'], snapshots['after']) if len(snapshots) == 2 else None
    summary['clone_integrity'] = {'unexpected_changes': [p for p in changes if p not in allowed]
                                 if changes is not None else None,
                                 'source': 'pre-client to post-client, before parent verifier'}
    return summary, times


def verify_identities():
    if gsq.digest(CLIENT) != CLIENT_SHA:
        raise RuntimeError('client identity drift')
    image = subprocess.check_output(['docker', 'image', 'inspect', pal.SANDBOX_IMAGE,
                                     '--format', '{{.Id}}'], text=True).strip()
    if image != IMAGE_ID:
        raise RuntimeError('image identity drift')
    return {'client_path': str(CLIENT), 'client_sha256': CLIENT_SHA,
            'sandbox_image': pal.SANDBOX_IMAGE, 'sandbox_image_id': image}


def start_exl3(model, argv, cwd, env, log_path, primary):
    # Assign ownership before polling: caller's finally can stop even a failed boot.
    model._log_handle = log_path.open('xb')
    model.process = subprocess.Popen(argv, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
        stdout=model._log_handle, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    model.started_utc = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    deadline = time.monotonic() + 600
    while True:
        if model.process.poll() is not None:
            raise RuntimeError(f'EXL3 exited {model.process.returncode}')
        try:
            health = incumbents.get_json(8000, '/health')
            break
        except Exception:
            if time.monotonic() >= deadline:
                raise TimeoutError('EXL3 health timeout')
            time.sleep(1)
    return {'pid': model.process.pid, 'port': 8000, 'started_utc': model.started_utc,
            **incumbents.exl3_observation(health, incumbents.EXL3, primary, CONTEXT)}


def check_lease(lease):
    if (ROOT / 'qwen38-tuning/.port8080.lock').exists():
        raise RuntimeError('a swap-model.sh orchestrator holds .port8080.lock')


def run_cell(cell_id, root, template, lease, uuids):
    cell = root / cell_id
    key = cell_id.rsplit('-', 1)[0]
    cell.mkdir()
    model = adapter = tap = None
    started = time.monotonic()
    times = {}
    result = {'status': 'evidence_failure', 'times': times}
    ports = [8000, 8080, 18080]
    try:
        check_lease(lease)
        assert_ports_free()
        verify_identities()
        pre = pal.sample_resources(uuids)
        pal._write_json(cell / 'gpu-preflight.json', pre)
        if pre['errors'] or pre['suspected_game_processes'] or not pre['gpus']:
            raise RuntimeError('resource preflight failed')
        argv = candidate_argv(key)
        is_exl3 = key == 'exl3'
        if is_exl3:
            files = exl3_files()
            primary = Path(files[0]['path'])
            port, cwd, model_name = 8000, ROOT / 'exllamav3-mia', incumbents.EXL3.name
            metadata = {'quant': 'SC4.0bpw-H5', 'upstream_revision': 'b4e3574d5665efb5d8031c05a578837e6700a912'}
        else:
            if key == 'nvfp4':
                check_nvfp4_budget(pre)
            primary = Path(argv[argv.index('-m') + 1])
            files = [file_record(primary)]
            gsq.verify_artifact_digest(key, files[0]['sha256'])
            port, cwd, model_name = 18080, ROOT, argv[argv.index('--alias') + 1]
            metadata = gsq.artifact_metadata(key)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(uuids), PYTHONIOENCODING='utf-8')
        if key == 'flash_next':
            env.update(gsq.FLASH_NEXT_ENV)
        if is_exl3:
            env['EXL3_RESTART_FLAG'] = str(cell / 'restart-flag.json')
        pal._write_json(cell / 'launch.json', {'artifact': key, 'artifact_metadata': metadata,
            'files': files, 'argv': argv, 'engine_sha256': gsq.digest(argv[0]),
            'cwd': str(cwd), 'profile': 'frozen predecessor operating point'})
        model = pal.ModelSession(argv, cwd, port, env=env, log_path=cell / 'server.log')
        if is_exl3:
            observed = start_exl3(model, argv, cwd, env, cell / 'server.log', primary)
        else:
            observed = model.start()
            log_text = (cell / 'server.log').read_text(encoding='utf-8', errors='replace')
            if key == 'flash_next':
                if not gsq.flash_next_layers_ok(log_text):
                    raise RuntimeError('Flash-Next not 49/49 + 50/50')
                observed['layers'] = list(gsq.FLASH_NEXT_LAYERS)
                k32 = ctypes.WinDLL('kernel32'); psapi = ctypes.WinDLL('psapi')
                k32.OpenProcess.restype = ctypes.c_void_p
                handle = k32.OpenProcess(0x1F0FFF, False, model.process.pid)
                observed['working_set_trim'] = bool(psapi.EmptyWorkingSet(ctypes.c_void_p(handle)))
                k32.CloseHandle(ctypes.c_void_p(handle))
            else:
                observed.update(gsq.layer_evidence(log_text, argv))
                if observed['layers'] != [66, 0]:
                    raise RuntimeError('CPU layer spill')
        observed['listener'] = gsq.listener_evidence(model.process.pid, port)
        pal._write_json(cell / 'boot.json', observed)
        tap = pal._load_tap_module('relay').Tap(0, port, str(cell / 'backend-wire'))
        tap.start()
        ports.append(tap.listen_port)
        if is_exl3:
            transform = lambda body: pal.frozen_request(body, None)
        else:
            pieces = gsq.gguf_pieces(str(primary))
            transform = lambda body: pal.frozen_request(body, gsq.conditional_han_bias(pieces, body['messages']))
        adapter = pal.AdapterServer(f'http://127.0.0.1:{tap.listen_port}', request_transform=transform,
                                    observation_directory=cell / 'backend-observations')
        adapter.start()
        ports.append(adapter.port)
        times['setup_s'] = time.monotonic() - started
        summary, _ = run_task(cell, template, model_name, adapter.port, files, observed, uuids, times,
                              observation_directory=cell / 'backend-observations')
        if adapter.observation_errors:
            raise RuntimeError('upstream observation persistence failed')
        result.update(status=classify_summary(summary), summary=summary)
        integrity = pal.verify_evidence(cell / 'session')
        result['integrity'] = integrity
        # Stop transports before parsing; retain objects for independent final cleanup if stop fails.
        adapter.stop()
        adapter = None
        tap.stop()
        capture_errors = list(tap.capture_errors)
        tap = None
        rows = pal._load_tap_module('read_capture').rows(str(cell / 'backend-wire'))
        pal._write_json(cell / 'backend-requests.json', rows)
        inference = [r for r in rows if r.get('method') == 'POST'
                     and r.get('path', '').split('?')[0] == '/v1/chat/completions']
        events = [json.loads(line) for line in (cell / 'session/stdout.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        context_ok = pal.client_context_valid(events, CONTEXT)
        result['client_context_ok'] = context_ok
        if (not integrity['complete'] or capture_errors or not inference
                or not all(r.get('usable') for r in inference)
                or context_ok is False
                or (summary['client']['status'] == 'completed' and context_ok is not True)):
            result.update(status='evidence_failure', reason='recording/backend/context evidence incomplete')
        if summary['clone_integrity']['unexpected_changes'] is None:
            result.update(status='evidence_failure', reason='clone inventory missing')
        elif summary['clone_integrity']['unexpected_changes']:
            result.update(status='model_failure', reason='clone scope violation')
        check_lease(lease)
    except Exception as error:
        (cell / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        result.update(status='evidence_failure', reason=str(error), error_type=type(error).__name__)
    finally:
        cleanup_started = time.monotonic()
        cleanup = cleanup_owned(adapter, tap, model, ports)
        times['cleanup_s'] = time.monotonic() - cleanup_started
        times['total_s'] = time.monotonic() - started
        times['task_wall_s'] = (times.get('end', times.get('client_end', time.monotonic())) - times['start']
                                if 'start' in times else None)
        result['cleanup_errors'] = cleanup
        if cleanup:
            result.update(status='evidence_failure', reason='owned cleanup fault; campaign stopped')
        pal._write_json(cell / 'result.json', result)
    return result


def run_candidates(root, template, candidates, lease, uuids):
    progress = {key: {'status': 'planned'} for key in candidates}
    pal._write_json(root / 'progress.json', progress)
    for index, key in enumerate(candidates):
        progress[key] = {'status': 'running'}
        pal._write_json(root / 'progress.json', progress)
        progress[key] = run_cell(key, root, template, lease, uuids)
        if progress[key]['status'] == 'evidence_failure':
            for later in candidates[index + 1:]:
                progress[later] = {'status': 'not_run', 'reason': 'prior evidence/ownership fault'}
            pal._write_json(root / 'progress.json', progress)
            break
        pal._write_json(root / 'progress.json', progress)
    return progress


def freeze_sources(root):
    paths = set((ROOT / 'qwen38-tuning/bench').rglob('*.py'))
    paths.update((ROOT / 'qwen38-tuning/serving/exl3').rglob('*.py'))
    paths.update((ROOT / 'qwen38-tuning/tools/llama-tap').glob('*.py'))
    paths.update(Path(__file__).with_name(name) for name in (
        'run-pal-remaining.py', 'run-q4-pal-workflow.py', 'run-preliminary-incumbents.py', 'run-preliminary-cli.py'))
    paths.update((ROOT / 'qwen38-tuning/templates').glob('*.jinja'))
    paths.add(ROOT / 'qwen38-tuning/scripts/worker-q4-dual.ps1')
    records = {}
    for source in sorted(paths):
        relative = source.relative_to(ROOT)
        target = root / 'sources' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        records[relative.as_posix()] = file_record(target)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--legacy-owner', type=int)
    parser.add_argument('--pal-template', type=Path, required=True)
    parser.add_argument('--candidate', choices=CANDIDATES)
    args = parser.parse_args(argv)
    candidates = [args.candidate] if args.candidate else list(CANDIDATES)
    protocol = {'id': 'pal-flash-next-vs-gsq-v1', 'client_version': subprocess.run([str(CLIENT), '--version'], capture_output=True, text=True).stdout.strip(),
        'flash_next_digests': dict(gsq.FLASH_NEXT_OTHER_DIGESTS), 'flash_next_env': dict(gsq.FLASH_NEXT_ENV), 'candidates': candidates, 'context': CONTEXT,
        'output': OUTPUT, 'timeout_s': TIMEOUT, 'max_turns': TURNS, 'effort': 'medium',
        'seed_requested': 29, 'retries': 0, 'pal_sha': PAL_SHA, 'prompt': PAL_PROMPT,
        'prompt_sha256': hashlib.sha256(PAL_PROMPT.encode('utf-8')).hexdigest(),
        'expected_client_sha256': CLIENT_SHA, 'expected_image_id': IMAGE_ID,
        'hidden_verifier': 'original8 inside task timer; overlap audit external and pending',
        'pal_template': str(args.pal_template.resolve())}
    if not args.run:
        print(json.dumps(protocol, indent=2))
        return
    root = Path(tempfile.mkdtemp(prefix='qwen-pal-remaining-'))
    print('PRIVATE_EVIDENCE ' + str(root), flush=True)
    with original_guard(root):
        protocol.update(verify_identities())
        template = args.pal_template.resolve()
        if template == pal.PAL_ORIGINAL.resolve() or template.is_relative_to(pal.PAL_ORIGINAL.resolve()):
            raise RuntimeError('template cannot be original repository')
        baseline = clone_snapshot(template)
        if baseline['git_head'].strip() != PAL_SHA:
            raise RuntimeError('template must be detached at frozen PAL SHA')
        for command, expected in ((['status', '--porcelain=v1', '--untracked-files=all'], ''), (['remote'], '')):
            if subprocess.check_output(['git', '-C', str(template), *command], text=True).strip() != expected:
                raise RuntimeError('template dirty or has remotes')
        protocol['template_snapshot'] = baseline
        protocol['hidden_sha256'] = gsq.digest(PAL_HIDDEN)
        protocol['sources'] = freeze_sources(root)
        pal._write_json(root / 'frozen-protocol.json', protocol)
        lease = None
        check_lease(lease)
        assert_ports_free()
        with pal.CampaignLock(ROOT / 'qwen38-tuning/.agent-campaign.lock', root.name, protocol['id']):
            run_candidates(root, template, candidates, lease, gsq.arena.BOTH_CARDS.split(','))
    print('CAMPAIGN_FINISHED ' + str(root), flush=True)


if __name__ == '__main__':
    main()
