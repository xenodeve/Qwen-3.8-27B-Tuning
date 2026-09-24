"""Extend preliminary code1 screen to current llama NVFP4 and EXL3 incumbents."""
import json, os, sys, tempfile, time, traceback
from pathlib import Path
import shutil, socket, subprocess, urllib.request

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'qwen38-tuning/bench'))
import gsq_compare as gsq
from anthropic_adapter import AdapterServer
from campaign_lock import CampaignLock
from model_session import ModelSession
from agent_session_client import run_client
from preliminary_cli_screen import frozen_request,seed_workspace,verify_workspace,client_context_valid,exl3_observation
from recorded_session import run_recorded_session,verify_evidence,inventory,_write_json,_load_tap_module
from session_telemetry import sample_resources

EXL3=ROOT/'models/turboderp-Qwen3.8-27B-EXL3-SC4.0bpw-H5'
EXPECTED_SHARDS={'model-00001-of-00002.safetensors':'2633b0cbcc4b250a2837f78fac315d2f4d6c9ce3decc51a625bbf30a2a5b214f','model-00002-of-00002.safetensors':'23879b3c29049377af1c14ba78a1a998d0bc7053bd9237f4a352821730a83c12'}

def get_json(port,route):
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 with opener.open(f'http://127.0.0.1:{port}/{route.lstrip("/")}',timeout=5) as response:return json.load(response)

def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--legacy-owner',type=int);a=p.parse_args()
 order=['nvfp4','exl3'];fixture=ROOT/'qwen38-tuning/fixtures/code-task-1';client=Path('C:/Program Files/nodejs/node_modules/@anthropic-ai/claude-code/bin/claude.exe');uuids=gsq.arena.BOTH_CARDS.split(',')
 prompt=(fixture/'BRIEF.md').read_text(encoding='utf-8').replace('แล้วรัน `python -m pytest -q` ให้ผ่าน','ไม่ต้องรัน test เอง เพราะรอบนี้มีเฉพาะ Read/Edit; ตัวตรวจภายนอกจะรัน test หลังจบงาน')+'\nแก้ไฟล์จริงด้วย tools ไม่ใช่ส่งโค้ดแทนการแก้ไฟล์ แล้วสรุปสิ่งที่แก้เป็นภาษาไทย ห้ามอ้างว่ารัน test แล้ว\n'
 protocol={'protocol':'preliminary-cli-code1-incumbents-v1','extension_of':'preliminary-cli-code1-v1','order':order,'context':65536,'timeout_s':1200,'max_turns':32,'seed_requested':29,'retries':0,'fixture':inventory(fixture),'prompt':prompt,'client_path':str(client),'client_sha256':gsq.digest(client),'gpu_uuids':uuids,'scope':'same task/prompt/client/verifier; engine-native language guards; no winner claim'}
 if not a.run:print(json.dumps(protocol,indent=2));return
 lock=ROOT/'qwen38-tuning/.port8080.lock';lease=lock.read_text() if lock.exists() else None
 if a.legacy_owner is None or lease is None or lease.split()[0]!=str(a.legacy_owner):raise RuntimeError('matching live lease required')
 subprocess.run(['C:/Program Files/Git/usr/bin/bash.exe','-c','kill -0 "$1"','lease',str(a.legacy_owner)],check=True)
 for port in (8000,8080,18080):
  with socket.socket() as s:
   s.settimeout(.2)
   if s.connect_ex(('127.0.0.1',port))==0:raise RuntimeError(f'occupied port {port}')
 root=Path(tempfile.mkdtemp(prefix='qwen-preliminary-incumbents-'));print('PRIVATE_EVIDENCE '+str(root),flush=True);shutil.copyfile(__file__,root/'runner.py');_write_json(root/'frozen-protocol.json',protocol);sources=root/'sources';sources.mkdir()
 for source in (ROOT/'qwen38-tuning/bench/preliminary_cli_screen.py',ROOT/'qwen38-tuning/bench/anthropic_adapter.py',ROOT/'qwen38-tuning/bench/recorded_session.py',ROOT/'qwen38-tuning/bench/agent_session_client.py',ROOT/'qwen38-tuning/bench/session_history.py',ROOT/'qwen38-tuning/bench/session_analysis.py',ROOT/'qwen38-tuning/serving/exl3/anthropic_compat.py',ROOT/'qwen38-tuning/serving/exl3/cjk_guard.py',ROOT/'qwen38-tuning/serving/exl3/thai_sampler.py',ROOT/'qwen38-tuning/serving/exl3/loop_guard.py',ROOT/'qwen38-tuning/serving/exl3/server.py',ROOT/'qwen38-tuning/scripts/worker-q4-dual.ps1',ROOT/'qwen38-tuning/templates/qwen38-late-system.jinja'):
  shutil.copyfile(source,sources/source.name)
 progress={k:{'status':'planned'} for k in order};_write_json(root/'progress.json',progress)
 with CampaignLock(ROOT/'qwen38-tuning/.agent-campaign.lock',root.name,protocol['protocol']):
  for key in order:
   cell=root/key;cell.mkdir();model=adapter=tap=None;start=time.monotonic()
   try:
    if not lock.exists() or lock.read_text()!=lease:raise RuntimeError('lease changed')
    pre=sample_resources(uuids);_write_json(cell/'gpu-preflight.json',pre)
    if pre['errors'] or pre['suspected_game_processes']:raise RuntimeError('resource preflight failed')
    is_exl3=key=='exl3';port=8000 if is_exl3 else 18080
    if is_exl3:
     argv=gsq.exl3_argv(65536);cwd=ROOT/'exllamav3-mia';shards=sorted(EXL3.glob('model*.safetensors'));files=[]
     for shard in shards:
      h=gsq.digest(shard)
      if EXPECTED_SHARDS.get(shard.name)!=h:raise ValueError('EXL3 shard mismatch')
      files.append({'path':str(shard),'bytes':shard.stat().st_size,'sha256':h})
     primary=shards[0];model_name=EXL3.name;metadata={'quant':'SC4.0bpw-H5','upstream_revision':'b4e3574d5665efb5d8031c05a578837e6700a912'}
     provenance=[]
     for artifact_file in sorted(EXL3.iterdir()):
      if artifact_file.is_file() and artifact_file.suffix in ('.json','.model','.jinja'):
       provenance.append({'path':str(artifact_file),'bytes':artifact_file.stat().st_size,'sha256':gsq.digest(artifact_file)})
     files.extend(provenance)
     env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(uuids),PYTHONIOENCODING='utf-8',EXL3_RESTART_FLAG=str(cell/'restart-flag.json'))
    else:
     argv=gsq.production_nvfp4_argv(65536);argv=gsq.replace_flag(argv,'--port',18080);argv=gsq.replace_flag(argv,'-lv',4);cwd=ROOT;primary=Path(argv[argv.index('-m')+1]);h=gsq.digest(primary);gsq.verify_artifact_digest('nvfp4',h);files=[{'path':str(primary),'bytes':primary.stat().st_size,'sha256':h}];model_name=argv[argv.index('--alias')+1];metadata=gsq.artifact_metadata('nvfp4');env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(uuids),PYTHONIOENCODING='utf-8')
     budgets=[]
     for gpu in pre['gpus']:
      reserve=2500 if gpu['memory_used_mib']>500 else 512
      budgets.append(gpu['memory_free_mib']-reserve)
     demand=14173+int(65536*18/1024)+2048
     if len(budgets)!=2 or min(budgets)<1024 or sum(budgets)<demand:
      raise RuntimeError(f'NVFP4 source-profile memory refusal: budgets={budgets}, demand={demand}')
    _write_json(cell/'launch.json',{'artifact':key,'artifact_metadata':metadata,'files':files,'argv':argv,'engine_sha256':gsq.digest(argv[0]),'fixture':protocol['fixture'],'profile':'current source default (NVFP4) or validated served recipe (EXL3)','psutil_available':pre['process'] is not None})
    if is_exl3:
     model=ModelSession(argv,cwd,port,env=env,log_path=cell/'server.log')
     # ModelSession readiness is llama-shaped; launch owned process then use native health.
     opts={'cwd':str(cwd),'env':env,'stdin':subprocess.DEVNULL};log=(cell/'server.log').open('xb');opts.update(stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP);model._log_handle=log;model.process=subprocess.Popen(argv,**opts);model.started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
     deadline=time.monotonic()+600
     while True:
      if model.process.poll() is not None:raise RuntimeError(f'EXL3 exited {model.process.returncode}')
      try:health=get_json(port,'/health');break
      except Exception:
       if time.monotonic()>=deadline:raise TimeoutError('EXL3 health timeout')
       time.sleep(1)
     observed={'pid':model.process.pid,'port':port,'started_utc':model.started_utc,**exl3_observation(health,EXL3,primary,65536)}
    else:
     model=ModelSession(argv,cwd,port,env=env,log_path=cell/'server.log');observed=model.start();observed.update(gsq.layer_evidence((cell/'server.log').read_text(encoding='utf-8',errors='replace'),argv));
     if observed['layers']!=[66,0]:raise RuntimeError('NVFP4 spill')
    observed['listener']=gsq.listener_evidence(model.process.pid,port);_write_json(cell/'boot.json',observed)
    tap=_load_tap_module('relay').Tap(0,port,str(cell/'backend-wire'));tap.start()
    if is_exl3: transform=lambda body:frozen_request(body,None)
    else:
     pieces=gsq.gguf_pieces(str(primary));transform=lambda body:frozen_request(body,gsq.conditional_han_bias(pieces,body['messages']))
    adapter=AdapterServer(f'http://127.0.0.1:{tap.listen_port}',request_transform=transform).start();work=cell/'work';seed_workspace(fixture,work);setup=time.monotonic()-start;times={}
    def runner(argv2,env2,wd,prompt2,history,timeout):
     env2=dict(env2,CLAUDE_CODE_MAX_CONTEXT_TOKENS='65536',CLAUDE_CODE_MAX_OUTPUT_TOKENS='8192',CLAUDE_CODE_MAX_TURNS='32');argv2=list(argv2)+['--effort','medium'];history.append('effective_client_policy',{'argv':argv2,'context':65536,'output_cap':8192,'turn_cap':32});times['start']=time.monotonic();return run_client(argv2,env2,wd,prompt2,history,timeout)
    def verifier(wd):
     try:return verify_workspace(wd,fixture,cell/'verification')
     finally:times['end']=time.monotonic()
    spec={'artifacts':[{'path':str(primary),'sha256':gsq.digest(primary)}],'gpu_uuids':uuids,'test':{'format':'claude-cli','case_id':'code1','stage':'first-pass','round':1,'attempt':1},'context':{'requested_window_tokens':65536,'runtime_window_tokens':65536,'input_tokens':None,'cached_input_tokens':None,'max_output_tokens':8192,'token_count_source':None,'history_policy':'client-default-compaction','source':protocol['protocol']}}
    progress[key]={'status':'running'};_write_json(root/'progress.json',progress);summary=run_recorded_session(cell/'session',work,client,model_name,adapter.port,prompt,spec,verifier,timeout=1200,client_runner=runner,server_probe=lambda _p:observed)
    if 'end' not in times:times['end']=time.monotonic()
    integrity=verify_evidence(cell/'session')
    if summary['outcome']=='invalid':raise RuntimeError('recording invalid '+str(summary['errors']))
    adapter.stop();adapter=None;tap.stop();errors=list(tap.capture_errors);tap=None;rows=_load_tap_module('read_capture').rows(str(cell/'backend-wire'));_write_json(cell/'backend-requests.json',rows);inference=[r for r in rows if r.get('method')=='POST' and r.get('path','').split('?')[0]=='/v1/chat/completions']
    if errors or not inference or not all(r.get('usable') for r in inference):raise RuntimeError('backend capture incomplete')
    events=[json.loads(l) for l in (cell/'session/stdout.jsonl').read_text(encoding='utf-8').splitlines()];context_ok=client_context_valid(events,65536)
    if not integrity['complete']:raise RuntimeError('integrity failed')
    if context_ok is False or (summary['client']['status']=='completed' and context_ok is not True):raise RuntimeError('client context evidence missing or mismatched')
    progress[key]={'status':summary['outcome'],'task_wall_s':times['end']-times['start'],'setup_s':setup,'client_s':summary['client_s'],'verification_s':summary['verification_s'],'client_context_ok':context_ok,'integrity':integrity,'verification':summary['verification'],'session':str(cell/'session')};print(json.dumps({key:progress[key]}),flush=True)
   except Exception as e:
    (cell/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');progress[key]={'status':'invalid','error_type':type(e).__name__,'elapsed_s':time.monotonic()-start,'reason':str(e)};print(json.dumps({key:progress[key]}),flush=True)
    for later in order[order.index(key)+1:]:progress[later]={'status':'not_run','reason':'prior invalid'}
    break
   finally:
    cleanup=[]
    for name,obj in (('adapter',adapter),('tap',tap),('model',model)):
     if obj is None:continue
     try:obj.stop()
     except Exception as stop_error:cleanup.append({'component':name,'error':type(stop_error).__name__})
    if cleanup:
     (cell/'cleanup-errors.json').write_text(json.dumps(cleanup,indent=2),encoding='utf-8')
     progress[key]={'status':'invalid','reason':'owned cleanup incomplete','cleanup_errors':cleanup}
     for later in order[order.index(key)+1:]:progress[later]={'status':'not_run','reason':'owned cleanup incomplete'}
     _write_json(root/'progress.json',progress)
     break
    _write_json(root/'progress.json',progress)
 print('SCREEN_FINISHED '+str(root),flush=True)
if __name__=='__main__':main()
