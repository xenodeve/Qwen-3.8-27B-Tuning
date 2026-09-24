"""Run Q4 code1 admission then an isolated PAL workflow through real Claude Code."""
import argparse, json, os, shutil, socket, subprocess, sys, tempfile, time, traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'qwen38-tuning/bench'))
import gsq_compare as gsq
from anthropic_adapter import AdapterServer
from campaign_lock import CampaignLock
from model_session import ModelSession
from agent_session_client import run_client
from fixture_test_tool import mcp_config
from pal_workflow_screen import verify_pal_workspace,source_inventory
from preliminary_cli_screen import frozen_request,seed_workspace,verify_workspace,client_context_valid
from recorded_session import run_recorded_session,verify_evidence,_write_json,_load_tap_module,inventory
from session_telemetry import sample_resources

PAL_SHA='200fcb9262d25e4002e30bf00c4364f55a42354e'
PAL_ORIGINAL=Path('D:/Github/pal-mcp-server')
PAL_HIDDEN=ROOT/'qwen38-tuning/bench/fixtures/pal_registry_hidden_test.py'
SANDBOX_IMAGE='qwen-pal-test:20260921'
CLIENT=Path('C:/Program Files/nodejs/node_modules/@anthropic-ai/claude-code/bin/claude.exe')
FILE_TOOLS='Read,Write,Edit,Glob,Grep'
MCP_TOOL=FILE_TOOLS+',mcp__visible-tests__run_visible_tests'

CODE1_PROMPT=(ROOT/'qwen38-tuning/fixtures/code-task-1/BRIEF.md').read_text(encoding='utf-8').replace(
    'แล้วรัน `python -m pytest -q` ให้ผ่าน','ไม่ต้องรัน test เอง เพราะรอบนี้มีเฉพาะ Read/Edit; ตัวตรวจภายนอกจะรัน test หลังจบงาน')+'\nแก้ไฟล์จริงด้วย tools แล้วสรุปเป็นภาษาไทย ห้ามอ้างว่ารัน test แล้ว\n'
PAL_PROMPT='''You are working from scratch in an offline disposable clone of pal-mcp-server. Follow TDD using only the fixed run_visible_tests tool with suite_id "user-config-focused"; do not claim tests ran unless that tool returned the result. Modify only clink/registry.py and tests/test_user_config_dir_rename.py.\n\nImplement this real registry behavior:\n- An unsupported CLI definition in either the current or legacy USER config directory must be skipped, with one warning naming both the unsupported client and source file path. Other valid clients must remain available.\n- The existing legacy-directory deprecation warning must remain. Current user config must still override legacy.\n- Unsupported BUNDLED definitions and unsupported explicit CLI_CLIENTS_CONFIG_PATH files/directories must remain fatal.\n- Schema-invalid files, supported-but-incomplete configs, and missing role prompts must remain fatal. Existing malformed-JSON handling is out of scope and must not be broadened.\n- Do not broadly catch RegistryLoadError. Only the unsupported-client case whose source is a user config directory may be downgraded.\n\nAdd focused visible tests, first observe the appropriate new test fail, implement the smallest source-provenance-aware fix, rerun the fixed suite to green, then summarize the evidence and changes concisely. No network, installs, remotes or changes outside the two allowed files.'''


def pal_state():
    return {'head':subprocess.check_output(['git','-C',str(PAL_ORIGINAL),'rev-parse','HEAD'],text=True).strip(),
            'status':subprocess.check_output(['git','-C',str(PAL_ORIGINAL),'status','--porcelain=v1','--untracked-files=all'],text=True)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');ap.add_argument('--legacy-owner',type=int);ap.add_argument('--pal-template',type=Path,required=True);ap.add_argument('--candidate',choices=['turbo_mtp_q4km','swift_q4km','turbo','swift']);ap.add_argument('--task',choices=['code1','pal','both'],default='both');args=ap.parse_args()
    all_candidates=[('turbo_mtp_q4km','9500,14500','ngram'),('swift_q4km','9500,14500','mtp-ngram')]
    if args.candidate in ('turbo','swift'):
        all_candidates=[(args.candidate,'9500,14500','ngram' if args.candidate=='turbo' else 'mtp-ngram')]
    candidates=[item for item in all_candidates if args.candidate in (None,item[0])];uuids=gsq.arena.BOTH_CARDS.split(',')
    task_names=('code1','pal') if args.task=='both' else (args.task,)
    protocol={'id':'q4-code1-pal-v1','tasks':list(task_names),'candidates':[{'id':k,'spec':spec} for k,_,spec in candidates],'context':65536,'effort':'medium','seed_requested':29,'code1_prompt':CODE1_PROMPT,'pal_prompt':PAL_PROMPT,'pal_sha':PAL_SHA,'pal_original_before':pal_state(),'pal_template':str(args.pal_template.resolve()),'pal_template_inventory':source_inventory(args.pal_template),'hidden_sha256':gsq.digest(PAL_HIDDEN),'client_sha256':gsq.digest(CLIENT),'sandbox_image':SANDBOX_IMAGE,'sandbox_image_id':subprocess.check_output(['docker','image','inspect',SANDBOX_IMAGE,'--format','{{.Id}}'],text=True).strip(),'turbo_mtp_qualification':'C:/Users/xenod/AppData/Local/Temp/qwen-q4-clean-qualification','turbo_mtp_decision':'reject MTP runtime for workflow: 71.6% acceptance and faster, but outputs diverged and Thai repair truncated; use ngram on same MTP weights'}
    if protocol['pal_original_before']['head']!=PAL_SHA:raise RuntimeError('PAL original HEAD changed')
    if subprocess.check_output(['git','-C',str(args.pal_template),'status','--porcelain=v1','--untracked-files=all'],text=True):raise RuntimeError('PAL template dirty')
    if subprocess.check_output(['git','-C',str(args.pal_template),'remote'],text=True).strip():raise RuntimeError('PAL template still has remote')
    if not args.run:print(json.dumps(protocol,indent=2));return
    legacy=ROOT/'qwen38-tuning/.port8080.lock';lease=legacy.read_text() if legacy.exists() else None
    if args.legacy_owner is None or lease is None or lease.split()[0]!=str(args.legacy_owner):raise RuntimeError('matching live legacy lease required')
    subprocess.run(['C:/Program Files/Git/usr/bin/bash.exe','-c','kill -0 "$1"','lease',str(args.legacy_owner)],check=True)
    for port in (8000,8080,18080):
        with socket.socket() as s:
            s.settimeout(.2)
            if s.connect_ex(('127.0.0.1',port))==0:raise RuntimeError(f'occupied port {port}')
    root=Path(tempfile.mkdtemp(prefix='qwen-q4-pal-'));print('PRIVATE_EVIDENCE '+str(root),flush=True);_write_json(root/'frozen-protocol.json',protocol);shutil.copyfile(__file__,root/'runner.py')
    progress={k:{'status':'planned','tasks':{}} for k,_,_ in candidates};_write_json(root/'progress.json',progress)
    with CampaignLock(ROOT/'qwen38-tuning/.agent-campaign.lock',root.name,protocol['id']):
      for key,split,spec_mode in candidates:
        cell=root/key;cell.mkdir();model=None;setup_started=time.monotonic()
        try:
          if not legacy.exists() or legacy.read_text()!=lease:raise RuntimeError('legacy lease changed')
          pre=sample_resources(uuids);_write_json(cell/'gpu-preflight.json',pre)
          if pre['errors'] or pre['suspected_game_processes']:raise RuntimeError('resource preflight failed')
          argv=gsq.apply_template(gsq.llama_argv(key,65536,spec_mode,split,24),'stock');argv=gsq.replace_flag(argv,'--port',18080);argv=gsq.replace_flag(argv,'-lv',4)
          model_path=Path(argv[argv.index('-m')+1]);model_hash=gsq.digest(model_path);gsq.verify_artifact_digest(key,model_hash);pieces=gsq.gguf_pieces(str(model_path));env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(uuids),PYTHONIOENCODING='utf-8')
          _write_json(cell/'launch.json',{'artifact':key,'metadata':gsq.artifact_metadata(key),'model_path':str(model_path),'sha256':model_hash,'bytes':model_path.stat().st_size,'argv':argv,'engine_sha256':gsq.digest(argv[0]),'split_source':'frozen Q6 family operating point','spec_mode':spec_mode})
          model=ModelSession(argv,ROOT,18080,env=env,log_path=cell/'server.log');boot=model.start();boot['listener']=gsq.listener_evidence(model.process.pid,18080);boot.update(gsq.layer_evidence((cell/'server.log').read_text(encoding='utf-8',errors='replace'),argv))
          if boot['layers']!=[66,0]:raise RuntimeError('CPU layer spill')
          _write_json(cell/'boot.json',boot);setup_s=time.monotonic()-setup_started
          for task_name in task_names:
            task_root=cell/task_name;task_root.mkdir();tap=adapter=None
            try:
              tap=_load_tap_module('relay').Tap(0,18080,str(task_root/'backend-wire'));tap.start()
              adapter=AdapterServer(f'http://127.0.0.1:{tap.listen_port}',request_transform=lambda body:frozen_request(body,gsq.conditional_han_bias(pieces,body['messages']))).start()
              if task_name=='code1':
                work=task_root/'work';seed_workspace(ROOT/'qwen38-tuning/fixtures/code-task-1',work);prompt=CODE1_PROMPT;timeout=1200;turns='32';mcp_path=None;tools=FILE_TOOLS
                verifier=lambda wd:verify_workspace(wd,ROOT/'qwen38-tuning/fixtures/code-task-1',task_root/'verification')
              else:
                work=task_root/'work';shutil.copytree(args.pal_template,work);prompt=PAL_PROMPT;timeout=1800;turns='64';tools=MCP_TOOL
                suites=task_root/'suites.json';suites.write_text(json.dumps({'user-config-focused':{'args':['tests/test_user_config_dir_rename.py','-q'],'evidence_files':['clink/registry.py','tests/test_user_config_dir_rename.py']}}),encoding='utf-8');mcp_path=task_root/'mcp.json';mcp_path.write_text(json.dumps(mcp_config(ROOT/'qwen38-tuning/bench/fixture_test_tool.py',work,SANDBOX_IMAGE,task_root/'visible-test-evidence',suites)),encoding='utf-8')
                def pal_verifier(wd):
                  journal=task_root/'visible-test-evidence/visible-tests.jsonl'
                  result=verify_pal_workspace(wd,args.pal_template,SANDBOX_IMAGE,PAL_HIDDEN,
                                               task_root/'verification',visible_test_journal=journal)
                  evidence_dir=task_root/'visible-test-evidence'
                  if evidence_dir.is_dir():shutil.copytree(evidence_dir,task_root/'session/visible-test-evidence')
                  return result
                verifier=pal_verifier
              times={}
              def runner(argv2,env2,wd,prompt2,history,timeout):
                env2=dict(env2,CLAUDE_CODE_MAX_CONTEXT_TOKENS='65536',CLAUDE_CODE_MAX_OUTPUT_TOKENS='8192',CLAUDE_CODE_MAX_TURNS=turns);argv2=list(argv2)+['--effort','medium'];history.append('effective_client_policy',{'argv':argv2,'context':65536,'turn_cap':int(turns)});times['start']=time.monotonic();return run_client(argv2,env2,wd,prompt2,history,timeout)
              def timed_verifier(wd):
                try:return verifier(wd)
                finally:times['end']=time.monotonic()
              spec={'artifacts':[{'path':str(model_path),'sha256':model_hash}],'gpu_uuids':uuids,'test':{'format':'claude-cli','case_id':task_name,'stage':'first-pass','round':1,'attempt':1},'context':{'requested_window_tokens':65536,'runtime_window_tokens':65536,'input_tokens':None,'cached_input_tokens':None,'max_output_tokens':8192,'token_count_source':None,'history_policy':'client-default-compaction','source':protocol['id']}}
              summary=run_recorded_session(task_root/'session',work,CLIENT,argv[argv.index('--alias')+1],adapter.port,prompt,spec,timed_verifier,timeout=timeout,client_runner=runner,client_mcp_config=mcp_path,client_allowed_tools=tools,recorder_source_paths=[ROOT/'qwen38-tuning/bench/fixture_test_tool.py',ROOT/'qwen38-tuning/bench/pal_workflow_screen.py'])
              if 'end' not in times:times['end']=time.monotonic()
              if summary['outcome']=='invalid':raise RuntimeError('recorded session evidence invalid: '+str(summary['errors']))
              integrity=verify_evidence(task_root/'session');adapter.stop();adapter=None;tap.stop();tap_errors=list(tap.capture_errors);tap=None;rows=_load_tap_module('read_capture').rows(str(task_root/'backend-wire'));_write_json(task_root/'backend-requests.json',rows);inference=[r for r in rows if r.get('method')=='POST' and r.get('path','').split('?')[0]=='/v1/chat/completions'];events=[json.loads(line) for line in (task_root/'session/stdout.jsonl').read_text(encoding='utf-8').splitlines()];ctx=client_context_valid(events,65536)
              if not integrity['complete'] or tap_errors or not inference or not all(r.get('usable') for r in inference) or (summary['client']['status']=='completed' and ctx is not True):raise RuntimeError('task evidence invalid')
              progress[key]['tasks'][task_name]={'status':summary['outcome'],'task_wall_s':times['end']-times['start'],'client_s':summary['client_s'],'verification':summary['verification'],'session':str(task_root/'session'),'integrity':integrity}
              print(json.dumps({key:{task_name:progress[key]['tasks'][task_name]}}),flush=True)
              if task_name=='code1' and summary['outcome']!='verified':
                progress[key]['tasks']['pal']={'status':'not_run','reason':'code1 Q4 admission failed'};break
            finally:
              cleanup_errors=[]
              for name,obj in (('adapter',adapter),('tap',tap)):
                if obj is None:continue
                try:obj.stop()
                except Exception as error:cleanup_errors.append({'component':name,'error':type(error).__name__})
              if cleanup_errors:
                (task_root/'cleanup-error.json').write_text(json.dumps(cleanup_errors,indent=2),encoding='utf-8')
                raise RuntimeError('task transport cleanup incomplete')
              _write_json(root/'progress.json',progress)
          progress[key]['status']='completed';progress[key]['setup_s']=setup_s
        except Exception as e:
          (cell/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');progress[key]={'status':'invalid','reason':str(e),'error_type':type(e).__name__,'tasks':progress.get(key,{}).get('tasks',{})}
          for later,_,_ in candidates[candidates.index((key,split,spec_mode))+1:]:progress[later]={'status':'not_run','reason':'prior shared/candidate invalid','tasks':{}}
          break
        finally:
          if model is not None:
            try:model.stop()
            except Exception:
              (cell/'model-cleanup-error.txt').write_text(traceback.format_exc(),encoding='utf-8')
              process=model.process
              if process is not None and process.poll() is None:
                subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],check=False,
                               stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                  process.kill();process.wait(timeout=10)
              progress[key]={'status':'invalid','reason':'model cleanup required fallback','tasks':progress.get(key,{}).get('tasks',{})}
              for later,_,_ in candidates[candidates.index((key,split,spec_mode))+1:]:progress[later]={'status':'not_run','reason':'owned model cleanup fault','tasks':{}}
              break
          _write_json(root/'progress.json',progress)
    after=pal_state();_write_json(root/'pal-original-after.json',after)
    if after!=protocol['pal_original_before']:raise RuntimeError('original PAL repository changed')
    print('CAMPAIGN_FINISHED '+str(root),flush=True)

if __name__=='__main__':main()
