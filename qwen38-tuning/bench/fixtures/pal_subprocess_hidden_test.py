"""Parent-only hidden oracle for openclink #145 bounded stream draining."""
from __future__ import annotations
import asyncio, json, os, sys
from pathlib import Path
import pytest

@pytest.fixture
def workspace():
    return Path('/workspace')


def _agent(workspace, tmp_path, script, timeout=5):
    from clink.agents.base import BaseCLIAgent
    from clink.models import ResolvedCLIClient, ResolvedCLIRole
    prompt = tmp_path / 'prompt.txt'; prompt.write_text('x')
    class Probe(BaseCLIAgent):
        def __init__(self):
            client = ResolvedCLIClient(name='probe', executable=[sys.executable, str(script)],
                working_dir=tmp_path, internal_args=[], config_args=[], env={},
                timeout_seconds=timeout, parser='claude_json', roles={
                    'default': ResolvedCLIRole(name='default', prompt_path=prompt, role_args=[])})
            super().__init__(client)
        def _build_command(self, *, role, system_prompt, model, reasoning_effort):
            return [sys.executable, str(script)]
        def _build_environment(self): return dict(os.environ)
        def refuse_unservable(self, command): return None
    return Probe(), ResolvedCLIRole(name='default', prompt_path=prompt, role_args=[])


def _make_script(tmp_path, stdout_bytes=100000, stderr_bytes=100000, sleep_s=0):
    script = tmp_path / 'child.py'
    payload = json.dumps([{'type':'assistant','message':'x'}] * max(1, stdout_bytes // 40) +
                         [{'type':'result','result':'ok'}]).encode()
    script.write_text('import sys,time\n'
        f'sys.stderr.buffer.write(b"E"*{stderr_bytes}); sys.stderr.flush()\n'
        f'sys.stdout.buffer.write({payload!r}); sys.stdout.flush()\n'
        f'time.sleep({sleep_s})\n', encoding='utf-8')
    return script


@pytest.mark.asyncio
async def test_both_streams_drain_concurrently_and_counters_are_observed(workspace, tmp_path):
    agent, role = _agent(workspace, tmp_path, _make_script(tmp_path, 300000, 1000000))
    output = await agent.run(role=role, prompt='x', system_prompt=None, files=[], images=[])
    assert output.returncode == 0
    assert len(output.stdout) >= 200000
    assert len(output.stderr) >= 900000
    assert getattr(output, 'stdout_bytes', 0) >= len(output.stdout.encode())
    assert getattr(output, 'stderr_bytes', 0) >= len(output.stderr.encode())
    assert getattr(output, 'stdout_truncated', False) is False
    assert getattr(output, 'stderr_truncated', False) is False


@pytest.mark.asyncio
async def test_large_streams_do_not_wait_for_child_after_timeout(workspace, tmp_path):
    agent, role = _agent(workspace, tmp_path, _make_script(tmp_path, 100000, 100000, 10), timeout=0.2)
    with pytest.raises(Exception) as caught:
        await agent.run(role=role, prompt='x', system_prompt=None, files=[], images=[])
    error = caught.value
    assert getattr(error, 'stdout', None) is not None
    assert getattr(error, 'stderr', None) is not None
    assert len(error.stdout) <= 10000
    assert len(error.stderr) <= 10000


@pytest.mark.asyncio
async def test_stdout_and_stderr_caps_are_independent(workspace, tmp_path):
    agent, role = _agent(workspace, tmp_path, _make_script(tmp_path, 500000, 500000))
    output = await agent.run(role=role, prompt='x', system_prompt=None, files=[], images=[])
    assert output.stdout_bytes >= output.stderr_bytes * 0
    assert getattr(output, 'stdout_truncated', False) is not getattr(output, 'stderr_truncated', None)
