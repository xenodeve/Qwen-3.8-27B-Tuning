"""Task #145 runner wrapper over the isolated PAL journal lifecycle."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
_base_path = Path(__file__).with_name('run-pal-journal.py')
spec = importlib.util.spec_from_file_location('pal_task145_base', _base_path)
base = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(base)

base.CANDIDATES = ('gsq', 'exl3')
base.ALLOWED = {'clink/agents/base.py', 'tests/test_agents_output.py'}
base.TASK_SOURCE_FILE = 'clink/agents/base.py'
base.TASK_TEST_FILE = 'tests/test_agents_output.py'
base.JOURNAL_HIDDEN = ROOT / 'qwen38-tuning' / 'bench' / 'fixtures' / 'pal_agent_output_hidden_test.py'
base.JOURNAL_PROMPT = """You are working from scratch in an offline disposable clone of pal-mcp-server. Follow TDD using only the fixed run_visible_tests tool with suite_id \"user-config-focused\"; do not claim tests ran unless that tool returned the result.

Implement openclink issue #145: concurrent, bounded subprocess output draining in clink/agents/base.py. Modify only clink/agents/base.py and tests/test_agents_output.py. Do not implement #144 process-tree ownership or change any other file.

Required contract:
- Drain stdout and stderr concurrently and incrementally while the child is running; a child flooding either or both pipes must not deadlock.
- Maintain independent MAX_DRAINED_OUTPUT_CHARS caps for captured stdout and stderr. Preserve the retained output tail/partial output, expose independent total byte counters and independent boolean truncation flags on both AgentOutput and CLIAgentError.
- On timeout/kill, post-kill draining is bounded; partial output and counters survive even when the child does not terminate promptly. Do not wait forever after kill.
- Preserve existing parser/result semantics and error status. Existing successful JSON parsing must still produce AgentOutput.
- Add focused visible tests; first observe the new behavior fail, implement the smallest fix, then rerun green. No network, installs, remotes, hidden-test access or changes outside the two allowed files."""

_original_prepare = base.prepare_workspace
def prepare_workspace(template, task_root):
    work = shutil.copytree(template, task_root / 'work')
    tests = work / 'tests'; tests.mkdir(exist_ok=True)
    (tests / 'test_agents_output.py').write_text('# harness scaffold; candidate must replace\n', encoding='utf-8')
    return work
base.prepare_workspace = prepare_workspace

if __name__ == '__main__':
    base.main()
