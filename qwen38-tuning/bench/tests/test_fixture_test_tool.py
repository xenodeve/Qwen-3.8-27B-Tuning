"""Contracts for the allowlisted, containerized visible-test MCP server."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from fixture_test_tool import FixtureTestTool, mcp_config, run_sandboxed_pytest

IMAGE = 'qwen-pal-test:20260921'


def workspace(tmp_path):
    root = tmp_path / 'work'; root.mkdir()
    (root / 'test_sample.py').write_text('def test_ok():\n    assert True\n', encoding='utf-8')
    return root


def fake_runner(stdout='1 passed', returncode=0):
    def run(workspace, image, arguments, stdout_path, stderr_path, timeout):
        stdout_path.write_text(stdout, encoding='utf-8'); stderr_path.write_text('', encoding='utf-8')
        return {'returncode':returncode, 'timed_out':False, 'elapsed_s':0.01,
                'sandbox_command':{'image':image,'network':'none','workspace_mount':'read-only'}}
    return run


def test_allowlisted_suite_records_bounded_sandbox_evidence(tmp_path):
    work = workspace(tmp_path); evidence = tmp_path / 'private'; evidence.mkdir()
    tool = FixtureTestTool(work, IMAGE, evidence,
                           {'focused':{'args':['test_sample.py','-q'],
                                       'evidence_files':['test_sample.py']}},
                           run_tests=fake_runner())
    result = tool.run({'suite_id':'focused'})
    assert result['returncode'] == 0 and result['passed'] is True
    assert result['sandbox_command']['network'] == 'none'
    assert result['workspace_files']['test_sample.py']['bytes'] > 0
    journal = [json.loads(line) for line in (evidence/'visible-tests.jsonl').read_text().splitlines()]
    assert len(journal)==1 and journal[0]['command_id']=='focused-v1'
    assert (evidence/result['stdout_ref']).is_file()


def test_unknown_suite_extra_arguments_and_outside_workspace_are_denied(tmp_path):
    work=workspace(tmp_path); evidence=tmp_path/'private'; evidence.mkdir()
    tool=FixtureTestTool(work,IMAGE,evidence,{'focused':['test_sample.py','-q']},run_tests=fake_runner())
    for request in ({'suite_id':'other'},{'suite_id':'focused','command':'whoami'},{}):
        with pytest.raises(ValueError):tool.run(request)
    with pytest.raises(ValueError):FixtureTestTool(work,IMAGE,work/'evidence',{'focused':['test_sample.py']})
    with pytest.raises(ValueError):FixtureTestTool(work,IMAGE,evidence,{'bad':['../outside.py']})
    assert not (evidence/'visible-tests.jsonl').exists()


def test_output_limit_is_bounded_but_full_raw_output_is_retained(tmp_path):
    work=workspace(tmp_path); evidence=tmp_path/'private'; evidence.mkdir()
    tool=FixtureTestTool(work,IMAGE,evidence,{'focused':['test_sample.py']},output_limit=128,
                         run_tests=fake_runner('x'*10000))
    result=tool.run({'suite_id':'focused'})
    assert result['stdout_truncated'] is True and len(result['stdout'])==128
    assert len((evidence/result['stdout_ref']).read_text())==10000


def test_timeout_force_removes_named_container(tmp_path, monkeypatch):
    work=workspace(tmp_path);calls=[]
    def fake_run(command, **kwargs):
        calls.append(command)
        if command[1]=='run':
            raise subprocess.TimeoutExpired(command,1)
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr('fixture_test_tool.subprocess.run',fake_run)
    result=run_sandboxed_pytest(work,IMAGE,['test_sample.py'],tmp_path/'out',tmp_path/'err',1)
    assert result['timed_out'] is True
    run_name=calls[0][calls[0].index('--name')+1]
    assert calls[1]==['docker','rm','-f',run_name]
    assert result['container_cleanup_returncode']==0


def test_mcp_config_pins_server_workspace_image_and_evidence(tmp_path):
    work=workspace(tmp_path);evidence=tmp_path/'private';evidence.mkdir();spec=tmp_path/'suites.json';spec.write_text(json.dumps({'focused':['test_sample.py','-q']}))
    config=mcp_config(Path('C:/AI/qwen38-tuning/bench/fixture_test_tool.py'),work,IMAGE,evidence,spec)
    server=config['mcpServers']['visible-tests']
    assert server['command']==sys.executable and server['args'][0].endswith('fixture_test_tool.py')
    assert '--sandbox-image' in server['args'] and IMAGE in server['args']
    assert str(work.resolve()) in server['args'] and str(evidence.resolve()) in server['args']
    assert server['env']=={'PYTHONIOENCODING':'utf-8'}


def _captured_command(tmp_path, monkeypatch, **kwargs):
    work=workspace(tmp_path);calls=[]
    def fake_run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command,0,'','')
    monkeypatch.setattr('fixture_test_tool.subprocess.run',fake_run)
    run_sandboxed_pytest(work,IMAGE,['test_sample.py'],tmp_path/'out',tmp_path/'err',1,**kwargs)
    return calls[0]


def test_init_is_opt_in_and_off_by_default(tmp_path, monkeypatch):
    """openclink #144's oracle kills a process group; with pytest as PID 1 the
    orphaned grandchild stays a zombie and killpg(pgid, 0) still sees it, so a
    correct fix read as 'group survived' (2026-09-24). --init reaps orphans.
    Opt-in only: the frozen PAL verifier behind results 24-34 must not change."""
    assert '--init' not in _captured_command(tmp_path, monkeypatch)
    (tmp_path/'b').mkdir()
    command=_captured_command(tmp_path/'b', monkeypatch, init=True)
    assert command.index('--init') < command.index(IMAGE)
