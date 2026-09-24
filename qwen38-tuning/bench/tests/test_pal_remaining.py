"""Remaining PAL contract regressions; no client, Docker or GPU is launched."""
import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest

RUNNER = Path(__file__).resolve().parents[2] / 'tools/run-pal-remaining.py'


def load_runner():
    spec = importlib.util.spec_from_file_location('pal_remaining_test_runner', RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('candidate', ['dirk', 'gsq'])
def test_selected_stock_profiles_do_not_inherit_q4_ngram_fallback(candidate, monkeypatch):
    runner = load_runner()
    build = Mock(return_value=['server', '--port', '8080', '-lv', '1'])
    template = Mock(side_effect=lambda argv, name: argv)
    monkeypatch.setattr(runner.gsq, 'llama_argv', build)
    monkeypatch.setattr(runner.gsq, 'apply_template', template)
    argv = runner.candidate_argv(candidate)
    build.assert_called_once_with(candidate, 65536, 'mtp-ngram', '8500,15468', 24)
    assert template.call_args.args[1] == 'stock'
    assert argv[argv.index('--port') + 1] == '18080'


def test_incumbents_keep_native_profiles_and_memory_admission(monkeypatch):
    runner = load_runner()
    nv = Mock(return_value=['server', '--port', '8080', '-lv', '1'])
    ex = Mock(return_value=['python', 'server.py'])
    monkeypatch.setattr(runner.gsq, 'production_nvfp4_argv', nv)
    monkeypatch.setattr(runner.gsq, 'exl3_argv', ex)
    runner.candidate_argv('nvfp4')
    assert runner.candidate_argv('exl3') == ['python', 'server.py']
    nv.assert_called_once_with(65536)
    ex.assert_called_once_with(65536)
    with pytest.raises(RuntimeError, match='memory refusal'):
        runner.check_nvfp4_budget({'gpus': [{'memory_used_mib': 1000, 'memory_free_mib': 9000}] * 2})
    runner.check_nvfp4_budget({'gpus': [{'memory_used_mib': 100, 'memory_free_mib': 16000}] * 2})


def test_clone_snapshot_detects_git_and_excluded_path_changes_without_git(tmp_path, monkeypatch):
    runner = load_runner()
    (tmp_path / '.git').mkdir()
    (tmp_path / '.git/HEAD').write_text(runner.PAL_SHA + '\n')
    (tmp_path / '.git/config').write_text('[core]\n bare = false\n')
    (tmp_path / '.codex').mkdir()
    (tmp_path / '.codex/private').write_text('before')
    before = runner.clone_snapshot(tmp_path)
    (tmp_path / '.git/config').write_text('[remote "bad"]\n url = hostile\n')
    (tmp_path / '.codex/private').write_text('after')
    monkeypatch.setattr(runner.subprocess, 'check_output', Mock(side_effect=AssertionError('git forbidden')))
    after = runner.clone_snapshot(tmp_path)
    assert after['git_config'] != before['git_config']
    assert runner.clone_changes(before, after) == ['.codex/private', '.git/config']


def test_owned_cleanup_is_independent_and_uses_pid_fallback(monkeypatch):
    runner = load_runner()
    adapter = Mock(); adapter.stop.side_effect = RuntimeError('transport')
    tap = Mock(); model = Mock(); model.stop.side_effect = RuntimeError('model stop')
    model.process.pid = 1234
    model.process.poll.side_effect = [None, 0]
    kill = Mock()
    monkeypatch.setattr(runner.subprocess, 'run', kill)
    monkeypatch.setattr(runner, 'assert_ports_free', Mock())
    errors = runner.cleanup_owned(adapter, tap, model, [8000])
    tap.stop.assert_called_once()
    model.stop.assert_called_once()
    assert kill.call_args.args[0] == ['taskkill', '/PID', '1234', '/T', '/F']
    assert {error['component'] for error in errors} == {'adapter', 'model'}


def test_cleanup_waits_for_listener_release_after_owned_process_exits(monkeypatch):
    # 2026-09-21 EXL3: parent exit was observed, but the immediate port probe
    # still connected; an independent later probe found port8000 closed.
    runner = load_runner()
    probe = Mock(side_effect=[RuntimeError('occupied port 8000'), None])
    monkeypatch.setattr(runner, 'assert_ports_free', probe)
    monkeypatch.setattr(runner.time, 'sleep', Mock())
    assert runner.cleanup_owned(None, None, None, [8000]) == []
    assert probe.call_count == 2


def test_cleanup_listener_wait_is_bounded_and_never_kills_by_port(monkeypatch):
    runner = load_runner()
    monkeypatch.setattr(runner, 'assert_ports_free', Mock(side_effect=RuntimeError('occupied')))
    monkeypatch.setattr(runner.time, 'monotonic', Mock(side_effect=[0, 0, 16]))
    monkeypatch.setattr(runner.time, 'sleep', Mock())
    kill = Mock(side_effect=AssertionError('must not kill an unowned listener'))
    monkeypatch.setattr(runner.subprocess, 'run', kill)
    errors = runner.cleanup_owned(None, None, None, [8000])
    assert errors[0]['component'] == 'ports'
    kill.assert_not_called()


def test_outcomes_do_not_relabel_model_timeout_as_infrastructure():
    runner = load_runner()
    assert runner.classify_summary({'outcome': 'censored', 'client': {'status': 'censored'}}) == 'model_timeout'
    assert runner.classify_summary({'outcome': 'failed', 'client': {'status': 'failed'}}) == 'model_failure'
    assert runner.classify_summary({'outcome': 'invalid', 'client': {'status': 'completed'}}) == 'evidence_failure'


def test_exl3_records_both_pinned_shards_primary_first(tmp_path, monkeypatch):
    runner = load_runner()
    for name in runner.incumbents.EXPECTED_SHARDS:
        (tmp_path / name).write_text(name)
    (tmp_path / 'tokenizer.json').write_text('{}')
    monkeypatch.setattr(runner.incumbents, 'EXL3', tmp_path)
    monkeypatch.setattr(runner.gsq, 'digest', lambda path: runner.incumbents.EXPECTED_SHARDS.get(Path(path).name, 'tokenizer-hash'))
    files = runner.exl3_files()
    assert [Path(f['path']).name for f in files[:2]] == ['model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors']
    assert Path(files[2]['path']).name == 'tokenizer.json'
    (tmp_path / 'model-00002-of-00002.safetensors').unlink()
    with pytest.raises(ValueError, match='shard'):
        runner.exl3_files()


def test_task_freezes_policy_and_captures_clone_before_verifier_and_seal(tmp_path, monkeypatch):
    runner = load_runner()
    template = tmp_path / 'template'; template.mkdir()
    (template / '.git').mkdir()
    (template / '.git/HEAD').write_text(runner.PAL_SHA)
    (template / '.git/config').write_text('[core]\n')
    cell = tmp_path / 'cell'; cell.mkdir()
    observations = cell / 'backend-observations'; observations.mkdir()
    (observations / 'timing.json').write_text('{"complete":true}')
    calls = []
    def client(argv, env, wd, prompt, history, timeout):
        assert prompt == runner.pal.PAL_PROMPT
        assert timeout == 1800
        assert env['CLAUDE_CODE_MAX_TURNS'] == '64'
        assert env['CLAUDE_CODE_MAX_CONTEXT_TOKENS'] == '65536'
        assert env['CLAUDE_CODE_MAX_OUTPUT_TOKENS'] == '8192'
        assert argv[-2:] == ['--effort', 'medium']
        (wd / '.git/config').write_text('[remote "bad"]\n')
        journal = cell / 'visible-test-evidence'; journal.mkdir()
        (journal / 'visible-tests.jsonl').write_text('visible log')
        calls.append('client')
        return {'status': 'completed'}
    def verify(wd, *args, **kwargs):
        assert (cell / 'session/clone-after-client.json').is_file()
        calls.append('verifier')
        return {'passed': True, 'returncode': 0}
    def recorded(session, work, client_path, name, port, prompt, spec, verifier, **kwargs):
        session.mkdir()
        assert spec['artifacts'] == [{'path': 'first'}, {'path': 'second'}]
        assert kwargs['server_probe'](port) == {'native': True}
        kwargs['client_runner']([], {}, work, prompt, Mock(), timeout=kwargs['timeout'])
        result = verifier(work)
        assert (session / 'visible-test-evidence/visible-tests.jsonl').read_text() == 'visible log'
        assert (session / 'upstream-observations/timing.json').read_text() == '{"complete":true}'
        calls.append('seal')
        return {'outcome': 'verified', 'client': {'status': 'completed'}, 'verification': result}
    monkeypatch.setattr(runner.pal, 'run_client', client)
    monkeypatch.setattr(runner.pal, 'verify_pal_workspace', verify)
    monkeypatch.setattr(runner.pal, 'run_recorded_session', recorded)
    result, times = runner.run_task(cell, template, 'model', 12345, [{'path': 'first'}, {'path': 'second'}], {'native': True}, [], observation_directory=observations)
    assert calls == ['client', 'verifier', 'seal']
    assert result['clone_integrity']['unexpected_changes'] == ['.git/config']
    assert times['end'] >= times['start']


def test_original_guard_runs_even_when_campaign_raises(tmp_path, monkeypatch):
    runner = load_runner()
    state = {'head': runner.PAL_SHA, 'status': ''}
    check = Mock(return_value=state)
    monkeypatch.setattr(runner.pal, 'pal_state', check)
    with pytest.raises(RuntimeError, match='campaign fault'):
        with runner.original_guard(tmp_path):
            raise RuntimeError('campaign fault')
    assert check.call_count == 2
    assert (tmp_path / 'pal-original-after.json').exists()


def test_native_exl3_launch_never_calls_llama_readiness(tmp_path, monkeypatch):
    runner = load_runner()
    model = Mock(); model.argv = ['python', 'server.py']
    process = Mock(); process.pid = 456; process.poll.return_value = None
    monkeypatch.setattr(runner.subprocess, 'Popen', Mock(return_value=process))
    monkeypatch.setattr(runner.incumbents, 'get_json', Mock(return_value={'native': 'health'}))
    observation = Mock(return_value={'props': {'model_path': 'primary'}, 'health': {'status': 'ok'}})
    monkeypatch.setattr(runner.incumbents, 'exl3_observation', observation)
    actual = runner.start_exl3(model, model.argv, tmp_path, {}, tmp_path / 'server.log', Path('primary'))
    model.start.assert_not_called()
    observation.assert_called_once_with({'native': 'health'}, runner.incumbents.EXL3, Path('primary'), 65536)
    assert actual['pid'] == 456
    model._log_handle.close()


def test_serial_campaign_continues_model_failure_but_stops_evidence_failure(tmp_path, monkeypatch):
    runner = load_runner()
    calls = []
    def cell(key, *args):
        calls.append(key)
        return {'status': 'model_failure' if key == 'dirk' else 'evidence_failure'}
    monkeypatch.setattr(runner, 'run_cell', cell)
    runner.run_candidates(tmp_path, Path('template'), list(runner.CANDIDATES), 'lease', [])
    assert calls == ['dirk', 'gsq']
    import json
    result = json.loads((tmp_path / 'progress.json').read_text())
    assert result['nvfp4']['status'] == 'not_run'


def test_identity_drift_blocks_before_any_model(monkeypatch):
    runner = load_runner()
    monkeypatch.setattr(runner.gsq, 'digest', lambda path: 'wrong')
    with pytest.raises(RuntimeError, match='client identity'):
        runner.verify_identities()
    monkeypatch.setattr(runner.gsq, 'digest', lambda path: runner.CLIENT_SHA)
    monkeypatch.setattr(runner.subprocess, 'check_output', Mock(return_value='sha256:wrong\n'))
    with pytest.raises(RuntimeError, match='image identity'):
        runner.verify_identities()
