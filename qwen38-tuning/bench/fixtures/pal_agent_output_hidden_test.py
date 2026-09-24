"""Parent-only oracle for openclink issue #145: bounded concurrent streams."""
from __future__ import annotations
import asyncio
import importlib.util
import json
import sys
import textwrap
import time
from pathlib import Path
import pytest

@pytest.fixture
def workspace():
    return Path('/workspace')

def _base(workspace):
    path = workspace / 'clink' / 'agents' / 'base.py'
    spec = importlib.util.spec_from_file_location('candidate_base', path)
    module = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(module)
    return module

def _agent(workspace, tmp_path, code, timeout=1):
    module = _base(workspace)
    script = tmp_path / 'child.py'; script.write_text(textwrap.dedent(code), encoding='utf-8')
    from clink.models import ResolvedCLIClient, ResolvedCLIRole
    client = ResolvedCLIClient(name='probe', executable=[sys.executable], working_dir=tmp_path,
        timeout_seconds=timeout, parser='claude_json', roles={'default': ResolvedCLIRole(
            name='default', prompt_path=tmp_path / 'prompt.txt')})
    (tmp_path / 'prompt.txt').write_text('prompt', encoding='utf-8')
    class Probe(module.BaseCLIAgent):
        def _build_command(self, **kwargs): return [sys.executable, str(script)]
        def _build_environment(self): return dict(PYTHONIOENCODING='utf-8')
    return module, Probe(client)

async def _run(agent):
    return await agent.run(role=agent.client.get_role('default'), prompt='x', files=[], images=[])

def test_timeout_drains_stdout_stderr_concurrently_with_independent_bounded_counters(workspace, tmp_path):
    module, agent = _agent(workspace, tmp_path, """
        import sys, time
        sys.stdout.write('O' * 250000); sys.stdout.flush()
        sys.stderr.write('E' * 250000); sys.stderr.flush()
        time.sleep(10)
    """)
    with pytest.raises(module.CLIAgentError) as caught:
        asyncio.run(_run(agent))
    error = caught.value
    assert error.stdout_bytes >= 250000
    assert error.stderr_bytes >= 250000
    assert error.stdout_truncated is True
    assert error.stderr_truncated is True
    assert len(error.stdout) <= module.MAX_DRAINED_OUTPUT_CHARS
    assert len(error.stderr) <= module.MAX_DRAINED_OUTPUT_CHARS


def test_successful_result_exposes_stream_counters_without_changing_parser_contract(workspace, tmp_path):
    module, agent = _agent(workspace, tmp_path, """
        import json, sys
        sys.stdout.write(json.dumps({'result': 'ok'})); sys.stdout.flush()
        sys.stderr.write('warning'); sys.stderr.flush()
    """)
    result = asyncio.run(_run(agent))
    assert result.parsed.content == 'ok'
    assert result.stdout_bytes > 0 and result.stderr_bytes == len('warning')
    assert result.stdout_truncated is False and result.stderr_truncated is False


def test_timeout_partial_output_survives_kill_and_is_bounded(workspace, tmp_path):
    module, agent = _agent(workspace, tmp_path, """
        import sys, time
        sys.stdout.write('partial'); sys.stdout.flush()
        time.sleep(10)
    """)
    with pytest.raises(module.CLIAgentError) as caught:
        asyncio.run(_run(agent))
    assert 'partial' in caught.value.stdout
    assert caught.value.returncode is None
