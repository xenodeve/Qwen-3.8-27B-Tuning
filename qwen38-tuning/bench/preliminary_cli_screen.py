"""Bounded preliminary Claude CLI screen helpers; not a long-horizon runner."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from recorded_session import inventory


_IGNORE = shutil.ignore_patterns('hidden', 'BRIEF.md', 'EXPECT.json', '__pycache__',
                                '.pytest_cache', '*.pyc', '.git')
_ALLOWED = {'inventory/store.py', 'tests/test_store.py'}


def exl3_observation(health, model_dir, shard, context):
    """Normalize native health for the recorder; do not invent a native /props.

    Absolute identity comes from the separately checked owned launch argv and
    hashed shard inventory; native health corroborates directory name/context.
    """
    model_dir, shard = Path(model_dir).resolve(), Path(shard).resolve()
    if (health.get('ok') is not True or health.get('backend') != 'exl3'
            or health.get('model') != model_dir.name
            or health.get('context_length') != context
            or not shard.is_relative_to(model_dir) or not shard.is_file()):
        raise ValueError('EXL3 runtime identity/context mismatch')
    return {'native_health':dict(health), 'health':dict(health, status='ok'),
            'props':{'model_path':str(shard),
                     'model_path_source':'owned_argv_and_native_health_name',
                     'model_directory':str(model_dir),
                     'default_generation_settings':{'n_ctx':health['context_length']}}}


def client_context_valid(events, expected):
    finals = [event for event in events if event.get('type') == 'result']
    if not finals:
        return None
    usage = finals[-1].get('modelUsage')
    return (isinstance(usage, dict) and bool(usage)
            and all(isinstance(value, dict) and value.get('contextWindow') == expected
                    for value in usage.values()))


def frozen_request(body, bias):
    """Apply the result22 operating point to each translated CLI request."""
    result = dict(body, temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
                  seed=29, max_tokens=8192, reasoning_effort='medium',
                  chat_template_kwargs={'reasoning_effort':'medium'},
                  cache_prompt=True, stream_options={'include_usage':True})
    if bias:
        result['logit_bias'] = bias
    else:
        result.pop('logit_bias', None)
    return result


def seed_workspace(fixture, work):
    inventory(fixture)  # Reject links before copying any fixture content.
    shutil.copytree(fixture, work, ignore=_IGNORE)


def verify_workspace(work, fixture, output):
    """Verify copies only; preserve the exact pre-verification agent workspace.

    This is clean-fixture evaluation, not an OS sandbox for hostile programs.
    The child receives no inherited provider credentials or pytest plugins.
    """
    work, fixture, output = Path(work), Path(fixture), Path(output)
    current = inventory(work)
    output.mkdir()
    baseline = output / 'baseline'
    seed_workspace(fixture, baseline)
    initial = inventory(baseline)
    changes = sorted(k for k in initial.keys() | current.keys()
                     if initial.get(k) != current.get(k))
    violations = sorted(set(changes) - _ALLOWED)
    env = {key: value for key, value in os.environ.items()
           if key.upper() in {'SYSTEMROOT', 'WINDIR', 'PATH', 'TEMP', 'TMP', 'COMSPEC'}}
    env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    results = {}
    for name, target in (('hidden', 'hidden_tests'), ('visible', 'tests')):
        clone = output / (name + '-workspace')
        shutil.copytree(work, clone)
        if name == 'hidden':
            shutil.rmtree(clone / 'tests')
            shutil.copytree(fixture / 'hidden', clone / 'hidden_tests')
        xml = output / (name + '.xml')
        command = [sys.executable, '-I', '-B', '-c',
                   'import sys,pytest; sys.path.insert(0,sys.argv[1]); '
                   'raise SystemExit(pytest.main(sys.argv[2:]))', str(clone),
                   str(clone / target), '-q', '-o', 'addopts=', '-p', 'no:cacheprovider',
                   '--junitxml=' + str(xml)]
        started = time.monotonic()
        completed = subprocess.run(command, cwd=clone, env=env, capture_output=True,
                                   timeout=60)
        (output / (name + '.stdout')).write_bytes(completed.stdout)
        (output / (name + '.stderr')).write_bytes(completed.stderr)
        suites = ET.parse(xml).getroot().iter('testsuite')
        counts = {key: 0 for key in ('tests', 'failures', 'errors', 'skipped')}
        for suite in suites:
            for key in counts:
                counts[key] += int(suite.get(key, '0'))
        results[name] = dict(counts, returncode=completed.returncode,
                             elapsed_s=time.monotonic() - started)
    passed = not violations and all(r['returncode'] == 0 and r['tests'] > 0
        and not (r['failures'] or r['errors'] or r['skipped']) for r in results.values())
    result = dict(results, passed=passed, returncode=0 if passed else 1,
                  scope_violations=violations, changed_files=changes,
                  tests_modified='tests/test_store.py' in changes)
    (output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
