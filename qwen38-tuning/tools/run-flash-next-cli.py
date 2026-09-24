"""Flash-Next vs GSQ IQ3_S-MTP on the frozen code1 screen, paired in one sitting.

Same protocol as run-preliminary-cli.py (result 23) with four changes, all
recorded in frozen-protocol.json: candidates GSQ and Flash-Next in ABBA order;
the client is the only installed Claude Code (~/.local/bin/claude.exe, 2.1.281;
the 2.1.258 file result 23 used is gone since 2026-09-23); no legacy supervisor
lease is required, instead any .port8080.lock refuses the run; Flash-Next boots
the served 128k profile argv at 65,536 with its env and working-set trim.
"""
import argparse
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
import gsq_compare as gsq
from anthropic_adapter import AdapterServer
from campaign_lock import CampaignLock
from model_session import ModelSession
from agent_session_client import run_client
from preliminary_cli_screen import frozen_request, seed_workspace, verify_workspace, client_context_valid
from recorded_session import run_recorded_session, verify_evidence, inventory, _write_json
from session_telemetry import sample_resources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Launch the authorized GPU screen')
    args = parser.parse_args()
    order = [('gsq-a','gsq'), ('flash_next-a','flash_next'), ('flash_next-b','flash_next'), ('gsq-b','gsq')]
    fixture = ROOT / 'qwen38-tuning/fixtures/code-task-1'
    executable = Path.home() / '.local/bin/claude.exe'
    uuids = gsq.arena.BOTH_CARDS.split(',')
    prompt = (fixture / 'BRIEF.md').read_text(encoding='utf-8').replace(
        'แล้วรัน `python -m pytest -q` ให้ผ่าน',
        'ไม่ต้องรัน test เอง เพราะรอบนี้มีเฉพาะ Read/Edit; ตัวตรวจภายนอกจะรัน test หลังจบงาน')
    prompt += '\nแก้ไฟล์จริงด้วย tools ไม่ใช่ส่งโค้ดแทนการแก้ไฟล์ แล้วสรุปสิ่งที่แก้เป็นภาษาไทย ห้ามอ้างว่ารัน test แล้ว\n'
    planned = {'protocol':'flash-next-vs-gsq-code1-v1', 'order':[c for c,_ in order], 'context':65536,
        'timeout_s':1200, 'max_turns':32, 'seed_requested':29, 'retries':0,
        'fixture':inventory(fixture), 'prompt':prompt,
        'client_path':str(executable), 'client_sha256':gsq.digest(executable),
        'client_version':subprocess.run([str(executable),'--version'],capture_output=True,text=True).stdout.strip(),
        'flash_next_digests':dict(gsq.FLASH_NEXT_OTHER_DIGESTS), 'flash_next_env':dict(gsq.FLASH_NEXT_ENV),
        'gpu_uuids':uuids, 'scope':'one-task first-pass screen; no long-horizon verdict'}
    if not args.run:
        print(json.dumps(planned, ensure_ascii=True, indent=2)); return
    legacy_lock = ROOT / 'qwen38-tuning/.port8080.lock'
    if legacy_lock.exists():
        raise RuntimeError('A swap-model.sh orchestrator holds .port8080.lock; not ours to override')
    for port in (8000, 8080, 18080):
        with socket.socket() as probe:
            probe.settimeout(0.2)
            if probe.connect_ex(('127.0.0.1', port)) == 0:
                raise RuntimeError(f'Existing inference listener on {port}; not ours to stop')
    root = Path(tempfile.mkdtemp(prefix='qwen-preliminary-cli-'))
    print('PRIVATE_EVIDENCE ' + str(root), flush=True)
    shutil.copyfile(__file__, root / 'runner.py')
    _write_json(root / 'frozen-protocol.json', planned)
    sources = root / 'sources'; sources.mkdir()
    for source in (ROOT / 'qwen38-tuning/bench/preliminary_cli_screen.py',
                   ROOT / 'qwen38-tuning/bench/anthropic_adapter.py',
                   ROOT / 'qwen38-tuning/bench/gsq_compare.py',
                   ROOT / 'qwen38-tuning/serving/exl3/anthropic_compat.py',
                   ROOT / 'qwen38-tuning/serving/exl3/cjk_guard.py'):
        shutil.copyfile(source, sources / source.name)
    progress = {cell_id:{'status':'planned'} for cell_id,_ in order}
    _write_json(root / 'progress.json', progress)
    with CampaignLock(ROOT / 'qwen38-tuning/.agent-campaign.lock', root.name, planned['protocol']):
        for cell_id, key in order:
            cell = root / cell_id; cell.mkdir()
            model = None; adapter = None; backend_tap = None
            setup_start = time.monotonic()
            try:
                if legacy_lock.exists():
                    raise RuntimeError('A legacy lease appeared during the screen')
                preflight = sample_resources(uuids)
                _write_json(cell / 'gpu-preflight.json', preflight)
                if preflight['errors'] or preflight['suspected_game_processes']:
                    raise RuntimeError('Resource preflight failed')
                if key == 'flash_next':
                    # Same stock template as GSQ: the adapter, not the file, handles the
                    # late system message here; Flash-Next's stock == qwen38-stock.jinja.
                    argv = gsq.apply_template(gsq.flash_next_argv(65536, 18080), 'stock')
                else:
                    argv = gsq.apply_template(gsq.llama_argv(key,65536,'mtp-ngram','8500,15468',24),'stock')
                    argv = gsq.replace_flag(argv, '--port', 18080)
                    argv = gsq.replace_flag(argv, '-lv', 4)
                path = Path(argv[argv.index('-m')+1]); actual_hash = gsq.digest(path)
                gsq.verify_artifact_digest(key, actual_hash)
                pieces = gsq.gguf_pieces(str(path))
                manifest = {'artifact':key,'artifact_metadata':gsq.artifact_metadata(key),
                    'model_path':str(path),'sha256':actual_hash,'argv':argv,
                    'engine_sha256':gsq.digest(argv[0]),'fixture':planned['fixture'],
                    'profile':('served 128k profile J at 65536' if key=='flash_next' else 'result22 selected; actual CLI requests with prefix cache'),
                    'psutil_available':preflight['process'] is not None}
                _write_json(cell / 'launch.json', manifest)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(uuids), PYTHONIOENCODING='utf-8')
                if key == 'flash_next':
                    env.update(gsq.FLASH_NEXT_ENV)
                model = ModelSession(argv, ROOT, 18080, env=env, log_path=cell / 'server.log')
                observed = model.start()
                observed['listener'] = gsq.listener_evidence(model.process.pid,18080)
                log_text = (cell/'server.log').read_text(encoding='utf-8',errors='replace')
                if key == 'flash_next':
                    if not gsq.flash_next_layers_ok(log_text): raise RuntimeError('Flash-Next not 49/49 + 50/50')
                    observed['layers'] = list(gsq.FLASH_NEXT_LAYERS)
                    # As the served profile does after /health: drop the mmap copy of
                    # uploaded GPU weights from the working set (doc 32 section 2.5).
                    k32 = ctypes.WinDLL('kernel32'); psapi = ctypes.WinDLL('psapi')
                    k32.OpenProcess.restype = ctypes.c_void_p
                    handle = k32.OpenProcess(0x1F0FFF, False, model.process.pid)
                    observed['working_set_trim'] = bool(psapi.EmptyWorkingSet(ctypes.c_void_p(handle)))
                    k32.CloseHandle(ctypes.c_void_p(handle))
                else:
                    observed.update(gsq.layer_evidence(log_text,argv))
                    if observed['layers'] != [66,0]: raise RuntimeError('Not fully resident 66+0')
                _write_json(cell / 'boot.json', observed)
                from recorded_session import _load_tap_module
                backend_tap = _load_tap_module('relay').Tap(0,18080,str(cell/'backend-wire'))
                backend_tap.start()
                def transform(body):
                    return frozen_request(body, gsq.conditional_han_bias(pieces,body['messages']))
                adapter = AdapterServer(f'http://127.0.0.1:{backend_tap.listen_port}',request_transform=transform).start()
                work = cell / 'work'; seed_workspace(fixture,work)
                setup_s = time.monotonic()-setup_start
                times = {}
                def client_runner(argv, env, workdir, prompt, history, timeout):
                    env = dict(env, CLAUDE_CODE_MAX_CONTEXT_TOKENS='65536',
                               CLAUDE_CODE_MAX_OUTPUT_TOKENS='8192',CLAUDE_CODE_MAX_TURNS='32')
                    argv = list(argv)+['--effort','medium']
                    history.append('effective_client_policy',{'argv':argv,'context':65536,'output_cap':8192,'turn_cap':32})
                    times['start'] = time.monotonic()
                    return run_client(argv,env,workdir,prompt,history,timeout)
                def verifier(workdir):
                    try:return verify_workspace(workdir,fixture,cell/'verification')
                    finally:times['end']=time.monotonic()
                spec = {'artifacts':[{'path':str(path),'sha256':actual_hash}],
                    'gpu_uuids':uuids,'test':{'format':'claude-cli','case_id':'code1',
                        'stage':'first-pass','round':1,'attempt':1},
                    'context':{'requested_window_tokens':65536,'runtime_window_tokens':65536,
                        'input_tokens':None,'cached_input_tokens':None,'max_output_tokens':8192,
                        'token_count_source':None,'history_policy':'client-default-compaction',
                        'source':'preliminary-cli-code1-v1'}}
                progress[cell_id]={'status':'running'}; _write_json(root/'progress.json',progress)
                summary=run_recorded_session(cell/'session',work,executable,
                    argv[argv.index('--alias')+1],adapter.port,prompt,spec,verifier,
                    timeout=1200,client_runner=client_runner)
                if 'end' not in times:times['end']=time.monotonic()
                integrity=verify_evidence(cell/'session')
                if summary['outcome'] == 'invalid':
                    raise RuntimeError('Shared recording instrument invalid: ' + str(summary['errors']))
                adapter.stop();adapter=None
                backend_tap.stop()
                backend_errors=list(backend_tap.capture_errors)
                backend_tap=None
                backend_rows=_load_tap_module('read_capture').rows(str(cell/'backend-wire'))
                _write_json(cell/'backend-requests.json',backend_rows)
                inference=[r for r in backend_rows if r.get('method')=='POST' and r.get('path','').split('?')[0]=='/v1/chat/completions']
                if backend_errors or not inference or not all(r.get('usable') for r in inference):
                    raise RuntimeError('Backend capture incomplete')
                if not integrity['complete']:raise RuntimeError('Final session integrity failed')
                result_events=[json.loads(line) for line in (cell/'session/stdout.jsonl').read_text(encoding='utf-8').splitlines()]
                context_ok=client_context_valid(result_events,65536)
                if context_ok is False or (summary['client']['status']=='completed' and context_ok is not True):
                    raise RuntimeError('Client context evidence missing or mismatched')
                status=summary['outcome']
                progress[cell_id]={'status':status,'artifact':key,'task_wall_s':times['end']-times['start'],
                    'setup_s':setup_s,'client_s':summary['client_s'],'verification_s':summary['verification_s'],
                    'client_context_ok':context_ok,'integrity':integrity,
                    'verification':summary['verification'],'session':str(cell/'session')}
                print(json.dumps({cell_id:progress[cell_id]}),flush=True)
            except Exception as error:
                (cell/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
                progress[cell_id]={'status':'invalid','artifact':key,'error_type':type(error).__name__,
                               'elapsed_s':time.monotonic()-setup_start,'reason':str(error)}
                print(json.dumps({cell_id:progress[cell_id]}),flush=True)
                # Shared instrument uncertainty is not model failure; stop rather than rank it.
                for later,_ in order[[c for c,_ in order].index(cell_id)+1:]:
                    progress[later]={'status':'not_run','reason':'prior invalid cell needs investigation'}
                break
            finally:
                if adapter is not None:adapter.stop()
                if backend_tap is not None:backend_tap.stop()
                if model is not None:model.stop()
                _write_json(root/'progress.json',progress)
    print('SCREEN_FINISHED '+str(root),flush=True)


if __name__ == '__main__':
    main()
