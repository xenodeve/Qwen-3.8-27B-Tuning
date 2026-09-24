"""One controlled claude-9arm FP8 PAL attempt; dry-run unless --run is given."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from time import monotonic as clock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'qwen38-tuning/bench'))
from remote_recorded_session import MODEL, GATEWAY, CONTEXT, OUTPUT, TURNS, TIMEOUT, EFFORT, run_remote_recorded_session
from recorded_session import _write_json, digest, verify_evidence

spec = importlib.util.spec_from_file_location('pal_remaining', Path(__file__).with_name('run-pal-remaining.py'))
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


def protocol(gateway_api='anthropic'):
    if gateway_api not in ('anthropic', 'openai'):
        raise ValueError('unsupported gateway protocol')
    return {'id': 'pal-fp8-remote-v1' if gateway_api == 'anthropic' else 'pal-fp8-openai-observed-v2',
        'gateway_api': gateway_api,
        'measurement_repeat_of': 'pal-fp8-remote-v1' if gateway_api == 'openai' else None,
        'model': MODEL, 'gateway': GATEWAY,
        'pal_sha': previous.PAL_SHA, 'prompt': previous.PAL_PROMPT,
        'prompt_sha256': hashlib.sha256(previous.PAL_PROMPT.encode()).hexdigest(),
        'tools': previous.MCP_TOOL, 'context': CONTEXT, 'output': OUTPUT,
        'timeout_s': TIMEOUT, 'max_turns': TURNS, 'effort': EFFORT, 'retries': 0,
        'route': 'controlled claude-9arm without Headroom/profile hooks',
        'remote_weight_hardware_runtime': 'unknown',
        'hidden_verifier': 'original8; overlap audit by primary separately'}


def read_credential():
    """Read only into parent memory; no profile copy, repr or diagnostic output."""
    try:
        profile = json.loads((Path.home() / '.claude-9arm.json').read_text(encoding='utf-8'))
        env = profile.get('env', {})
        if profile.get('model') != MODEL or env.get('ANTHROPIC_BASE_URL', '').rstrip('/') != GATEWAY:
            raise ValueError()
        credential = env['ANTHROPIC_AUTH_TOKEN']
        if not isinstance(credential, str) or not credential or '\r' in credential or '\n' in credential:
            raise ValueError()
        return credential
    except Exception:
        raise RuntimeError('selected gateway profile invalid') from None


def validate_template(template):
    original = previous.pal.PAL_ORIGINAL.resolve()
    if template == original or template.is_relative_to(original):
        raise ValueError('template cannot be original PAL repository')
    snapshot = previous.clone_snapshot(template)
    if snapshot['git_head'] is None or snapshot['git_head'].strip() != previous.PAL_SHA:
        raise ValueError('template must be detached at frozen PAL baseline')
    for command in (['status', '--porcelain=v1', '--untracked-files=all'], ['remote']):
        if subprocess.check_output(['git', '-C', str(template), *command], text=True).strip():
            raise ValueError('template must be clean and without remotes')
    return snapshot


def run_task(root, template, credential, gateway_api='anthropic'):
    work, session = root / 'work', root / 'session'
    shutil.copytree(template, work)
    suites = root / 'suites.json'
    _write_json(suites, {'user-config-focused': {
        'args': ['tests/test_user_config_dir_rename.py', '-q'],
        'evidence_files': ['clink/registry.py', 'tests/test_user_config_dir_rename.py']}})
    mcp_path = root / 'mcp.json'
    _write_json(mcp_path, previous.pal.mcp_config(
        ROOT / 'qwen38-tuning/bench/fixture_test_tool.py', work, previous.pal.SANDBOX_IMAGE,
        root / 'visible-test-evidence', suites))

    times = {}

    def verifier(wd):
        try:
            return previous.pal.verify_pal_workspace(wd, template, previous.pal.SANDBOX_IMAGE,
                previous.PAL_HIDDEN, root / 'verification',
                visible_test_journal=root / 'visible-test-evidence/visible-tests.jsonl')
        finally:
            for name in ('visible-test-evidence', 'verification'):
                if (root / name).is_dir():
                    shutil.copytree(root / name, session / name)
            times['end'] = clock()

    def client(argv, env, wd, prompt, history, timeout):
        times['start'] = clock()
        try:
            return previous.pal.run_client(argv, env, wd, prompt, history, timeout=timeout)
        finally:
            times['client_end'] = clock()
            visible = root / 'visible-test-evidence'
            if visible.is_dir():
                shutil.copytree(visible, session / 'visible-test-evidence-client')

    summary = run_remote_recorded_session(session, work, previous.CLIENT, previous.PAL_PROMPT,
        credential, verifier, client_mcp_config=mcp_path, client_runner=client,
        client_allowed_tools=previous.MCP_TOOL, gateway_protocol=gateway_api,
        recorder_source_paths=[Path(__file__), Path(previous.__file__),
            Path(previous.pal.__file__), ROOT / 'qwen38-tuning/bench/fixture_test_tool.py',
            ROOT / 'qwen38-tuning/bench/pal_workflow_screen.py'])
    result = {'summary': summary, 'integrity': verify_evidence(session),
              'task_wall_s': times.get('end', times.get('client_end', 0)) - times['start'] if 'start' in times else None,
              'client_execution_s': times['client_end'] - times['start'] if 'client_end' in times else None}
    allowed = {'clink/registry.py', 'tests/test_user_config_dir_rename.py'}
    result['unexpected_changes'] = [name for name in summary['workspace_changes'] if name not in allowed]
    events = [json.loads(line) for line in (session / 'stdout.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    result['client_context_ok'] = previous.pal.client_context_valid(events, CONTEXT)
    result['status'] = summary['outcome']
    if not result['integrity']['complete'] or (summary['client']['status'] == 'completed' and result['client_context_ok'] is not True):
        result['status'] = 'invalid'
    if result['unexpected_changes']:
        result['status'] = 'failed'
    _write_json(root / 'result.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--pal-template', type=Path, required=True)
    parser.add_argument('--gateway-api', choices=('anthropic', 'openai'), default='anthropic')
    args = parser.parse_args(argv)
    frozen = protocol(args.gateway_api)
    if not args.run:
        print(json.dumps(frozen, indent=2))
        return
    prefix = 'qwen-pal-fp8-openai-' if args.gateway_api == 'openai' else 'qwen-pal-fp8-'
    root = Path(tempfile.mkdtemp(prefix=prefix))
    print('PRIVATE_EVIDENCE ' + str(root), flush=True)
    try:
        with previous.original_guard(root):
            frozen.update(previous.verify_identities())
            template = args.pal_template.resolve()
            frozen['template_snapshot'] = validate_template(template)
            frozen['hidden_sha256'] = digest(previous.PAL_HIDDEN)
            _write_json(root / 'frozen-protocol.json', frozen)
            credential = read_credential()
            try:
                run_task(root, template, credential, args.gateway_api)
            finally:
                del credential
    except Exception as error:
        # Never persist exception messages, stack locals, or the profile contents.
        _write_json(root / 'failure.json', {'status': 'invalid', 'error_type': type(error).__name__})
        raise RuntimeError('remote attempt failed; inspect secret-free evidence') from None
    print('CAMPAIGN_FINISHED ' + str(root), flush=True)


if __name__ == '__main__':
    main()
