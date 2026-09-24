"""Real-scenario benchmark: openclink #144 through the developer's own Claude Code.

Per cell (ABBA: gsq-a, thinkingcap-a, thinkingcap-b, gsq-b):
  1. boot the model's SERVED profile at 262,144 (serve-gsq.ps1 / serve-thinkingcap.ps1)
  2. fresh clone of openclink at the frozen SHA, remote removed
  3. `claude -p` (the installed 2.1.281) in the clone, --permission-mode auto,
     stream-json transcript, 90-minute cap, settings mirroring ~/.claude-gsq.json
  4. stop the server; oracle = bench/fixtures/oc144_process_tree_hidden_test.py
     (sandbox with --init) + the repo's own clink tests, on a COPY of the tree
  5. speed from the server log (prefill / decode / draft acceptance per request)

Evidence stays private under %TEMP%; only summary.json is meant for the repo.
Nothing here pushes, comments or touches the original repository.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'qwen38-tuning/bench'))
from fixture_test_tool import run_sandboxed_pytest  # noqa: E402

SHA = '200fcb9262d25e4002e30bf00c4364f55a42354e'
REPO = 'https://github.com/xenodeve/openclink.git'
CLIENT = Path.home() / '.local/bin/claude.exe'
IMAGE = 'qwen-pal-test:20260921'
HIDDEN = ROOT / 'qwen38-tuning/bench/fixtures/oc144_process_tree_hidden_test.py'
ISSUE = ROOT / 'qwen38-tuning/bench/fixtures/oc144_issue.md'
PORT, CTX, CAP_S = 18080, 262144, 5400
PROFILES = {
    'gsq': (ROOT / 'qwen38-tuning/scripts/serve-gsq.ps1', 'Qwen3.8-27B-gsq-262k'),
    'thinkingcap': (ROOT / 'qwen38-tuning/scripts/serve-thinkingcap.ps1', 'ThinkingCap-Qwen3.8-27B-IQ4_XS-262k'),
}
ORDER = ['gsq-a', 'thinkingcap-a', 'thinkingcap-b', 'gsq-b']
PROMPT = ('แก้ issue นี้ใน repository นี้ให้เสร็จตาม acceptance criteria ทุกข้อ '
          'เขียน test ที่พิสูจน์ได้จริง รัน test ให้ผ่าน แล้วสรุปสิ่งที่ทำเป็นภาษาไทย\n'
          'ข้อจำกัด: ห้ามใช้ gh, git push หรือเข้าถึงเครือข่าย ทำงานเฉพาะในโฟลเดอร์นี้\n\n')


def health():
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def stop_servers():
    subprocess.run(['powershell', '-NoProfile', '-Command',
                    'Get-Process llama-server -EA 0 | Stop-Process -Force'], check=False)
    time.sleep(3)


def boot(key, cell):
    script, alias = PROFILES[key]
    log = (cell / 'server.log').open('wb')
    proc = subprocess.Popen(['pwsh', '-NoProfile', '-File', str(script), '-Port', str(PORT)],
                            stdout=log, stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if health():
            return proc, alias
        if proc.poll() is not None:
            raise RuntimeError(f'{key} profile exited {proc.returncode}')
        time.sleep(3)
    raise TimeoutError(f'{key} not healthy in 600 s')


def settings(cell, alias):
    hooks = json.loads((Path.home() / '.claude-gsq.json').read_text(encoding='utf-8'))['hooks']
    data = {'model': alias, 'env': {
        'ANTHROPIC_BASE_URL': f'http://127.0.0.1:{PORT}', 'ANTHROPIC_AUTH_TOKEN': 'none',
        'CLAUDE_CODE_MAX_OUTPUT_TOKENS': '12288', 'CLAUDE_CODE_MAX_CONTEXT_TOKENS': str(CTX),
        'CLAUDE_CODE_AUTO_COMPACT_WINDOW': str(CTX), 'CLAUDE_AUTOCOMPACT_PCT_OVERRIDE': '95',
        'API_TIMEOUT_MS': '3600000', 'CLAUDE_STREAM_IDLE_TIMEOUT_MS': '1800000',
        'CLAUDE_STREAM_FIRST_BYTE_TIMEOUT_MS': '1800000'},
        'showThinkingSummaries': True, 'hooks': hooks}
    path = cell / 'settings.json'
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    return path


def clone(cell):
    work = cell / 'work'
    subprocess.run(['git', 'clone', '-q', REPO, str(work)], check=True)
    subprocess.run(['git', '-C', str(work), 'checkout', '-q', '--detach', SHA], check=True)
    subprocess.run(['git', '-C', str(work), 'remote', 'remove', 'origin'], check=True)
    return work


def run_client(cell, work, settings_path, alias):
    prompt = PROMPT + ISSUE.read_text(encoding='utf-8')
    empty_mcp = cell / 'mcp.json'
    empty_mcp.write_text('{"mcpServers": {}}', encoding='utf-8')
    argv = [str(CLIENT), '-p', prompt, '--settings', str(settings_path), '--model', alias,
            '--permission-mode', 'auto', '--output-format', 'stream-json', '--verbose',
            '--strict-mcp-config', '--mcp-config', str(empty_mcp)]
    env = dict(os.environ)
    for key in ('ANTHROPIC_API_KEY', 'ANTHROPIC_BASE_URL', 'ANTHROPIC_AUTH_TOKEN'):
        env.pop(key, None)
    started = time.monotonic()
    with (cell / 'transcript.jsonl').open('wb') as out, (cell / 'client.err').open('wb') as err:
        proc = subprocess.Popen(argv, cwd=str(work), stdout=out, stderr=err, env=env,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            code = proc.wait(timeout=CAP_S)
            status = 'completed' if code == 0 else f'exit_{code}'
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'], check=False)
            status = 'timeout'
    return status, time.monotonic() - started


def verify(cell, work):
    out = cell / 'verify'
    copy = out / 'work'
    shutil.copytree(work, copy, ignore=shutil.ignore_patterns('.git', '__pycache__', '.venv'))
    hidden = run_sandboxed_pytest(copy, IMAGE, [str(HIDDEN), '-q'], out / 'hidden.out',
                                  out / 'hidden.err', 300, trusted_test=HIDDEN, init=True)
    own = run_sandboxed_pytest(copy, IMAGE, ['tests', '-q', '-k', 'clink', '-x', '--maxfail=50'],
                               out / 'repo.out', out / 'repo.err', 600, init=True)
    def tail(p):
        lines = [l for l in p.read_text(errors='replace').splitlines() if re.search(r'passed|failed|error', l)]
        return lines[-1] if lines else ''
    return {'hidden': tail(out / 'hidden.out'), 'hidden_rc': hidden.get('returncode'),
            'repo_clink_tests': tail(out / 'repo.out'), 'repo_rc': own.get('returncode')}


def speed(cell):
    t = re.sub(r'\x1b\[[0-9;]*m', '', (cell / 'server.log').read_text(encoding='utf-8', errors='replace'))
    pp = [(int(a), float(b)) for a, b in re.findall(r'prompt eval time =\s+[\d.]+ ms /\s+(\d+) tokens.*?([\d.]+) tokens per second', t)]
    tg = [(int(a), float(b)) for a, b in re.findall(r'\n[^\n]*\s eval time =\s+[\d.]+ ms /\s+(\d+) tokens.*?([\d.]+) tokens per second', t)]
    acc = [float(a) for a in re.findall(r'draft acceptance = ([\d.]+)', t)]
    ctx = [int(a) for a in re.findall(r'task\.n_tokens = (\d+)', t)]
    big = sorted(b for a, b in pp if a >= 2000)
    n = sum(a for a, _ in tg)
    w = sum(a / b for a, b in tg if b)
    return {'requests': len(tg), 'prefill_median_ge2k': big[len(big) // 2] if big else None,
            'decode_token_weighted': round(n / w, 2) if w else None, 'output_tokens': n,
            'prefill_tokens': sum(a for a, _ in pp),
            'draft_acceptance_median': sorted(acc)[len(acc) // 2] if acc else None,
            'max_prompt_tokens': max(ctx) if ctx else None,
            'full_reprefill_events': t.count('forcing full prompt re-processing')}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', action='store_true')
    args = ap.parse_args()
    protocol = {'id': 'real-oc144-v1', 'order': ORDER, 'context': CTX, 'cap_s': CAP_S,
                'permission_mode': 'auto', 'client': str(CLIENT),
                'client_version': subprocess.run([str(CLIENT), '--version'], capture_output=True, text=True).stdout.strip(),
                'openclink_sha': SHA, 'profiles': {k: str(v[0]) for k, v in PROFILES.items()},
                'hidden_test': str(HIDDEN), 'issue': str(ISSUE)}
    if not args.run:
        print(json.dumps(protocol, indent=2, ensure_ascii=False))
        return
    if subprocess.run(['tasklist'], capture_output=True, text=True).stdout.count('llama-server'):
        raise SystemExit('a llama-server is running; not ours to stop')
    root = Path(tempfile.mkdtemp(prefix='qwen-real-oc144-'))
    print('PRIVATE_EVIDENCE ' + str(root), flush=True)
    (root / 'protocol.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding='utf-8')
    progress = {c: {'status': 'planned'} for c in ORDER}
    for cell_id in ORDER:
        key = cell_id.rsplit('-', 1)[0]
        cell = root / cell_id
        cell.mkdir()
        progress[cell_id] = {'status': 'running'}
        (root / 'progress.json').write_text(json.dumps(progress, indent=2), encoding='utf-8')
        server = None
        try:
            server, alias = boot(key, cell)
            work = clone(cell)
            status, wall = run_client(cell, work, settings(cell, alias), alias)
            stop_servers()
            result = {'status': status, 'wall_s': round(wall, 1), 'speed': speed(cell),
                      'verify': verify(cell, work)}
        except Exception as error:  # an instrument fault stops the campaign
            result = {'status': 'invalid', 'reason': f'{type(error).__name__}: {error}'}
            progress[cell_id] = result
            (root / 'progress.json').write_text(json.dumps(progress, indent=2), encoding='utf-8')
            stop_servers()
            break
        finally:
            if server is not None and server.poll() is None:
                server.kill()
        progress[cell_id] = result
        (root / 'progress.json').write_text(json.dumps(progress, indent=2), encoding='utf-8')
        print(json.dumps({cell_id: result}, ensure_ascii=False), flush=True)
    stop_servers()
    print('CAMPAIGN_FINISHED ' + str(root), flush=True)


if __name__ == '__main__':
    main()
