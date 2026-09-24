"""Allowlisted visible-test MCP tool using an OS-isolated Docker runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, Mapping, Sequence
import uuid


SandboxRunner = Callable[[Path, str, Sequence[str], Path, Path, float], dict[str, object]]


def _mount(path: Path, target: str, *, readonly: bool = True) -> str:
    value = f'type=bind,source={path.resolve()},target={target}'
    return value + (',readonly' if readonly else '')


def run_sandboxed_pytest(workspace: Path, image: str, arguments: Sequence[str],
                           stdout_path: Path, stderr_path: Path, timeout: float,
                           *, trusted_test: Path | None = None,
                           init: bool = False) -> dict[str, object]:
    """Run untrusted candidate Python with no host authority or network.

    The candidate tree is read-only. A parent-owned hidden test may be mounted
    read-only at /oracle/test_hidden.py; its output is never returned to a model.
    """
    container_name = 'qwen-pal-test-' + uuid.uuid4().hex
    command = ['docker', 'run', '--rm', '--name', container_name,
               '--network', 'none', '--read-only',
               '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
               '--pids-limit', '64', '--memory', '1g', '--cpus', '2',
               '--tmpfs', '/tmp:rw,noexec,nosuid,size=128m',
               '--mount', _mount(workspace, '/workspace'),
               '-e', 'HOME=/tmp/home', '-e', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD=1',
               '-e', 'PYTHONDONTWRITEBYTECODE=1']
    if init:
        # Reap orphans: without an init, pytest is PID 1 and a killed grandchild
        # stays a zombie that process-group checks still see (openclink #144).
        command.append('--init')
    target_arguments = list(arguments)
    if trusted_test is not None:
        command += ['--mount', _mount(trusted_test, '/oracle/test_hidden.py')]
        target_arguments = ['/oracle/test_hidden.py' if value == str(trusted_test) else value
                            for value in target_arguments]
    source = ('import sys,pytest; sys.path.insert(0,"/workspace"); '
              'raise SystemExit(pytest.main(sys.argv[1:]))')
    command += [image, 'python', '-I', '-B', '-c', source,
                *target_arguments, '--noconftest', '-o', 'addopts=', '-p', 'no:cacheprovider']
    started = time.monotonic()
    timed_out = False
    cleanup_returncode = None
    with stdout_path.open('xb') as stdout_file, stderr_path.open('xb') as stderr_file:
        try:
            completed = subprocess.run(command, stdout=stdout_file, stderr=stderr_file,
                                       timeout=timeout, stdin=subprocess.DEVNULL)
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out, returncode = True, None
            cleanup = subprocess.run(['docker', 'rm', '-f', container_name],
                                     stdin=subprocess.DEVNULL,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            cleanup_returncode = cleanup.returncode
            if cleanup.returncode != 0:
                raise RuntimeError('timed-out sandbox container could not be removed')
    return {'returncode': returncode, 'timed_out': timed_out,
            'container_cleanup_returncode': cleanup_returncode,
            'elapsed_s': time.monotonic() - started,
            'sandbox_command': {'image': image, 'network': 'none',
                                'workspace_mount': 'read-only',
                                'trusted_test_mounted': trusted_test is not None}}


class FixtureTestTool:
    def __init__(self, workspace: str | Path, sandbox_image: str,
                 evidence_dir: str | Path, suites: Mapping[str, Sequence[str]],
                 *, timeout: float = 120, output_limit: int = 64 * 1024,
                 run_tests: SandboxRunner = run_sandboxed_pytest) -> None:
        self.workspace = Path(workspace).resolve()
        self.sandbox_image = sandbox_image
        self.evidence_dir = Path(evidence_dir).resolve()
        if not self.workspace.is_dir() or not isinstance(sandbox_image, str) or not sandbox_image:
            raise ValueError('workspace and sandbox image must be explicit')
        if self.evidence_dir.is_relative_to(self.workspace) or self.workspace.is_relative_to(self.evidence_dir):
            raise ValueError('test evidence must be outside the model workspace')
        if timeout <= 0 or output_limit <= 0:
            raise ValueError('test bounds must be positive')
        self.timeout, self.output_limit, self._run_tests = timeout, output_limit, run_tests
        self.suites: dict[str, list[str]] = {}
        self.evidence_files: dict[str, list[str]] = {}
        for suite_id, definition in suites.items():
            if isinstance(definition, dict):
                if set(definition) != {'args', 'evidence_files'}:
                    raise ValueError('suite object must contain args and evidence_files')
                arguments, observed = definition['args'], definition['evidence_files']
            else:
                arguments, observed = definition, []
            if not isinstance(suite_id, str) or not suite_id or not isinstance(arguments, (list, tuple)) or not arguments:
                raise ValueError('suite definitions must be nonempty')
            values = list(arguments)
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError('suite arguments must be nonempty strings')
            for value in values:
                if value.startswith('-'):
                    continue
                candidate = Path(value)
                if candidate.is_absolute() or '..' in candidate.parts:
                    raise ValueError('suite paths must stay inside the workspace')
            self.suites[suite_id] = values
            if not isinstance(observed, (list, tuple)) or any(
                    not isinstance(value, str) or not value or Path(value).is_absolute()
                    or '..' in Path(value).parts for value in observed):
                raise ValueError('evidence_files must be relative workspace paths')
            self.evidence_files[suite_id] = list(observed)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.journal = self.evidence_dir / 'visible-tests.jsonl'
        self._run_count = 0

    def run(self, request: object) -> dict[str, object]:
        if not isinstance(request, dict) or set(request) != {'suite_id'}:
            raise ValueError('input must contain only suite_id')
        suite_id = request['suite_id']
        if suite_id not in self.suites:
            raise ValueError('unknown suite_id')
        self._run_count += 1
        stdout_path = self.evidence_dir / f'{suite_id}-{self._run_count:04d}.stdout'
        stderr_path = self.evidence_dir / f'{suite_id}-{self._run_count:04d}.stderr'
        workspace_files = {}
        for name in self.evidence_files[suite_id]:
            path = (self.workspace / name).resolve()
            if not path.is_relative_to(self.workspace) or path.is_symlink() or not path.is_file():
                raise ValueError('observed workspace file is missing or unsafe')
            with path.open('rb') as stream:
                file_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
            workspace_files[name] = {'bytes':path.stat().st_size, 'sha256':file_hash}
        execution = self._run_tests(self.workspace, self.sandbox_image, self.suites[suite_id],
                                    stdout_path, stderr_path, self.timeout)
        stdout_text = stdout_path.read_text(encoding='utf-8', errors='replace')
        stderr_text = stderr_path.read_text(encoding='utf-8', errors='replace')
        result = {
            'suite_id': suite_id, 'command_id': suite_id + '-v1',
            'cwd': str(self.workspace), 'workspace_files': workspace_files, **execution,
            'passed': execution['returncode'] == 0 and not execution['timed_out'],
            'stdout': stdout_text[:self.output_limit], 'stderr': stderr_text[:self.output_limit],
            'stdout_truncated': len(stdout_text) > self.output_limit,
            'stderr_truncated': len(stderr_text) > self.output_limit,
            'stdout_ref': stdout_path.name, 'stderr_ref': stderr_path.name,
        }
        with self.journal.open('a', encoding='utf-8', newline='\n') as handle:
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
            handle.flush(); os.fsync(handle.fileno())
        return result


def mcp_config(server_script: str | Path, workspace: str | Path,
               sandbox_image: str, evidence_dir: str | Path,
               suites_path: str | Path) -> dict[str, object]:
    return {'mcpServers': {'visible-tests': {
        'type': 'stdio', 'command': sys.executable,
        'args': [str(Path(server_script).resolve()), '--workspace', str(Path(workspace).resolve()),
                 '--sandbox-image', sandbox_image, '--evidence', str(Path(evidence_dir).resolve()),
                 '--suites', str(Path(suites_path).resolve())],
        'env': {'PYTHONIOENCODING': 'utf-8'},
    }}}


def _reply(request_id: object, result: object = None, error: object = None) -> None:
    message = {'jsonrpc': '2.0', 'id': request_id}
    message['error' if error is not None else 'result'] = error if error is not None else result
    sys.stdout.write(json.dumps(message, ensure_ascii=False, separators=(',', ':')) + '\n');sys.stdout.flush()


def serve(tool: FixtureTestTool) -> None:
    for raw in sys.stdin:
        request_id = None
        try:
            message = json.loads(raw); method = message.get('method'); request_id = message.get('id')
            if method == 'initialize':
                version = (message.get('params') or {}).get('protocolVersion', '2025-06-18')
                _reply(request_id, {'protocolVersion':version,'capabilities':{'tools':{}},
                                    'serverInfo':{'name':'visible-tests','version':'1'}})
            elif method == 'tools/list':
                _reply(request_id, {'tools':[{'name':'run_visible_tests',
                    'description':'Run one fixed visible test suite in the isolated workspace.',
                    'inputSchema':{'type':'object','properties':{'suite_id':{'type':'string','enum':sorted(tool.suites)}},'required':['suite_id'],'additionalProperties':False},
                    'annotations':{'readOnlyHint':False,'destructiveHint':False,'idempotentHint':True,'openWorldHint':False}}]})
            elif method == 'tools/call':
                params=message.get('params') or {}
                if params.get('name')!='run_visible_tests':raise ValueError('unknown tool')
                result=tool.run(params.get('arguments'))
                _reply(request_id, {'content':[{'type':'text','text':json.dumps(result,ensure_ascii=False)}],
                                    'structuredContent':result,'isError':False})
            elif request_id is not None:_reply(request_id,error={'code':-32601,'message':'method not found'})
        except Exception as error:
            if request_id is not None:_reply(request_id, {'content':[{'type':'text','text':f'{type(error).__name__}: {error}'}],'isError':True})


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',required=True);parser.add_argument('--sandbox-image',required=True);parser.add_argument('--evidence',required=True);parser.add_argument('--suites',required=True);args=parser.parse_args()
    suites=json.loads(Path(args.suites).read_text(encoding='utf-8'))
    serve(FixtureTestTool(args.workspace,args.sandbox_image,args.evidence,suites))

if __name__=='__main__':main()
