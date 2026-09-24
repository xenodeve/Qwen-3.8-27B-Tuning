"""Integrated evidence gates: a client exit is not a verified benchmark result."""
import hashlib
import json
import sys
from pathlib import Path

import pytest
sys.path.insert(0, str(Path(__file__).parents[1]))
import recorded_session as recorded


def test_wire_usage_keeps_cached_input_and_window_distinct():
    body = ('data: {"type":"message_start","message":{"usage":'
            '{"input_tokens":10,"cache_read_input_tokens":90,"cache_creation_input_tokens":2}}}\n\n'
            'data: {"type":"message_delta","usage":{"output_tokens":15}}\n\n'
            'data: {"type":"message_stop"}\n\n')
    usage = recorded.wire_usage(body)
    assert usage['input_tokens'] == 102
    assert usage['cached_input_tokens'] == 90
    assert usage['output_tokens'] == 15
    assert recorded.wire_usage('data: [DONE]\n\n')['input_tokens'] is None


def test_file_inventory_records_changes_and_removals(tmp_path):
    (tmp_path / 'a.py').write_text('before')
    before = recorded.inventory(tmp_path)
    (tmp_path / 'a.py').write_text('after')
    (tmp_path / 'b.py').write_text('new')
    after = recorded.inventory(tmp_path)
    assert before['a.py']['sha256'] != after['a.py']['sha256']
    assert after['b.py']['bytes'] == 3


def test_incomplete_wire_cannot_be_verified_by_passing_tests():
    assert recorded.decide_outcome({'status':'completed'}, [{'usable':False}],
                                   {'passed':True,'returncode':0}, []) == 'invalid'
    assert recorded.decide_outcome({'status':'completed'}, [{'usable':True}],
                                   {'passed':True,'returncode':0}, []) == 'verified'
    assert recorded.decide_outcome({'status':'completed'}, [],
                                   {'passed':True,'returncode':0}, []) == 'invalid'
    assert recorded.decide_outcome({'status':'completed'}, [{'usable':True}],
                                   {'passed':False,'returncode':None,'error_type':'OSError'}, []) == 'invalid'
    assert recorded.decide_outcome({'status':'completed'}, [{'usable':True}],
                                   {'passed':False,'returncode':1}, []) == 'failed'
    assert recorded.decide_outcome({'status':'censored'}, [{'usable':False}],
                                   None, []) == 'censored'


def test_full_offline_session_collects_all_evidence_groups(tmp_path):
    import urllib.request
    import http.server
    import threading
    from session_history import inspect_session
    class Server(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            value = ({'status':'ok'} if self.path == '/health' else
                     {'model_path':str(artifact), 'default_generation_settings':{'n_ctx':65536},
                      'build_info':'test-engine', 'chat_template':'test-template'})
            body = json.dumps(value).encode()
            self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers()
            self.wfile.write(body)
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            body = (b'data: {"type":"message_start","message":{"usage":{"input_tokens":10,"cache_read_input_tokens":90}}}\n\n'
                    b'data: {"type":"message_stop"}\n\n')
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args): pass
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Server)
    worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
    class Sampler:
        def start(self): return self
        def stop(self): return {'samples':[{'gpu':None,'errors':['unavailable']}], 'collection_s':0.01}
    work = tmp_path / 'work'; work.mkdir()
    (work / 'solution.txt').write_text('before')
    artifact = tmp_path / 'model.gguf'; artifact.write_bytes(b'model')
    spec = {'test':{'format':'long-horizon-agent','case_id':'case','stage':'fix','round':1,'attempt':1},
            'context':{'requested_window_tokens':65536,'runtime_window_tokens':65536,
                       'history_policy':'original','source':'fixture'},
            'artifacts':[{'path':str(artifact),'sha256':hashlib.sha256(b'model').hexdigest()}],
            'config':{'effort':'medium','seed':29,'guard':'none'}}
    seen_client = {}
    def fake_client(argv, env, workdir, prompt, history, timeout):
        seen_client['argv'] = list(argv)
        body = {'messages':[{'role':'user','content':prompt}], 'max_tokens':1024}
        request = urllib.request.Request(env['ANTHROPIC_BASE_URL']+'/v1/messages',
                data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        urllib.request.urlopen(env['ANTHROPIC_BASE_URL']+'/health', timeout=2).read()
        urllib.request.urlopen(request, timeout=2).read()
        (Path(workdir) / 'solution.txt').write_text('after')
        events = [{'type':'assistant','message':{'id':'m1','content':[
            {'type':'thinking','thinking':'check'}, {'type':'text','text':'done'}]}},
            {'type':'result','is_error':False,'usage':{'output_tokens':2}, 'result':'private-response-marker'}]
        history.append('user_prompt', prompt)
        for event in events: history.append('client_event', event)
        (history.directory/'stdout.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
        return {'status':'completed','returncode':0}
    try:
        result = recorded.run_recorded_session(tmp_path/'run',work,'unused','model',server.server_port,
            'fix it',spec,lambda path:{'passed':(path/'solution.txt').read_text()=='after','returncode':0},
            client_runner=fake_client,sampler_factory=Sampler,
            client_mcp_config=tmp_path/'fixed-mcp.json',
            client_allowed_tools='Read,Edit,mcp__visible-tests__run_visible_tests',
            recorder_source_paths=[Path(recorded.__file__).with_name('fixture_test_tool.py'),
                                   Path(recorded.__file__).with_name('pal_workflow_screen.py')])
    finally:
        server.shutdown();server.server_close();worker.join(timeout=2)
    assert seen_client['argv'][seen_client['argv'].index('--mcp-config') + 1] == str(tmp_path/'fixed-mcp.json')
    assert seen_client['argv'][seen_client['argv'].index('--tools') + 1] == 'Read,Edit,mcp__visible-tests__run_visible_tests'
    assert result['outcome'] == 'verified'
    assert (tmp_path/'run'/'recorder-source'/'fixture_test_tool.py').exists()
    assert (tmp_path/'run'/'recorder-source'/'pal_workflow_screen.py').exists()
    assert (tmp_path/'run'/'history.md').exists()
    assert 'fix it' in (tmp_path/'run'/'history.md').read_text(encoding='utf-8')
    assert (tmp_path/'run'/'resource-samples.json').exists()
    assert (tmp_path/'run'/'evidence-inventory.json').exists()
    assert (tmp_path/'run'/'workspace-initial'/'solution.txt').read_text() == 'before'
    assert (tmp_path/'run'/'workspace-final'/'solution.txt').read_text() == 'after'
    assert '+after' in (tmp_path/'run'/'workspace.diff').read_text()
    summary = json.loads((tmp_path/'run'/'summary.json').read_text())
    assert 'private-response-marker' not in json.dumps(summary)
    assert (tmp_path/'run'/'analysis-private.json').exists()
    assert summary['decision_ready'] is False
    assert 'resource_telemetry_incomplete' in summary['decision_blockers']
    assert summary['context_high_water_tokens'] == 100
    assert summary['workspace_changes']['modified'] == ['solution.txt']
    inspected = inspect_session(tmp_path/'run')
    assert inspected['complete'] is True
    requests = [e['payload'] for e in inspected['events'] if e['kind']=='request']
    assert requests[0]['context']['input_tokens'] == 100
    assert requests[0]['context']['runtime_window_tokens'] == 65536
    assert requests[0]['context']['history_policy'] == 'unknown'
    assert requests[0]['context']['declared_history_policy'] == 'original'
    assert requests[0]['test']['format'] == 'long-horizon-agent'
    assert recorded.verify_evidence(tmp_path/'run')['complete'] is True
    (tmp_path/'run'/'stdout.jsonl').write_text('changed')
    assert recorded.verify_evidence(tmp_path/'run')['complete'] is False


def test_runtime_window_mismatch_blocks_client_before_inference(tmp_path):
    work = tmp_path / 'work'; work.mkdir()
    artifact = tmp_path / 'model.gguf'; artifact.write_bytes(b'model')
    spec = {'test':{'format':'agent','case_id':'case','stage':'fix','round':1,'attempt':1},
            'context':{'requested_window_tokens':65536,'runtime_window_tokens':65536},
            'artifacts':[{'path':str(artifact),'sha256':hashlib.sha256(b'model').hexdigest()}]}
    called = []
    with pytest.raises(ValueError, match='runtime context'):
        recorded.run_recorded_session(tmp_path / 'run', work, 'unused', 'model', 8080,
            'prompt', spec, lambda path: {}, client_runner=lambda *a,**k: called.append(True),
            server_probe=lambda port: {'health':{'status':'ok'},'props':{
                'model_path':str(artifact),'default_generation_settings':{'n_ctx':32768}}})
    assert called == []


def test_registered_attempt_retains_preflight_failure_and_missing_cells(tmp_path):
    from campaign_register import CampaignRegister
    register = CampaignRegister(tmp_path / 'campaign', {'attempts':[
        {'attempt_id':name,'model':'model','case_id':'case','round':1,'config_id':'config'}
        for name in ('a','b')]})
    def fail(*args, **kwargs): raise ValueError('runtime context mismatch')
    result = recorded.run_registered_attempt(register, 'a', runner=fail)
    assert result['outcome'] == 'invalid'
    summary = register.summary()
    assert summary['counts']['invalid'] == 1
    assert summary['missing_attempt_ids'] == ['b']
    assert summary['complete'] is False


def test_competing_workload_blocks_client_before_model_request(tmp_path):
    work = tmp_path/'work'; work.mkdir()
    artifact=tmp_path/'model.gguf'; artifact.write_bytes(b'model')
    spec={'test':{'format':'agent','case_id':'case','stage':'fix','round':1,'attempt':1},
          'context':{'requested_window_tokens':65536,'runtime_window_tokens':65536},
          'artifacts':[{'path':str(artifact),'sha256':hashlib.sha256(b'model').hexdigest()}]}
    sample={'gpus':[{'uuid':'GPU-test'}], 'errors':[], 'suspected_game_processes':['javaw.exe']}
    class Sampler:
        def start(self): return self
        def wait_ready(self, timeout=7): return sample
        def stop(self): return {'samples':[sample], 'collection_s':0}
    called=[]
    result=recorded.run_recorded_session(tmp_path/'run',work,'unused','model',8080,'prompt',spec,
        lambda path:{},client_runner=lambda *a,**k:(called.append(True) or {'status':'completed','returncode':0}),sampler_factory=Sampler,
        server_probe=lambda port:{'health':{'status':'ok'},'props':{'model_path':str(artifact),
                                  'default_generation_settings':{'n_ctx':65536}}})
    assert called == []
    assert result['outcome'] == 'invalid'
    assert 'suspected_competing_workload' in result['decision_blockers']


def test_artifact_hash_mismatch_prevents_client_launch(tmp_path):
    work = tmp_path / 'work'; work.mkdir()
    artifact = tmp_path / 'model.gguf'; artifact.write_bytes(b'model')
    spec = {'test':{'format':'long-horizon-agent','case_id':'case','stage':'fix','round':1,'attempt':1},
            'context':{'requested_window_tokens':65536,'runtime_window_tokens':65536},
            'artifacts':[{'path':str(artifact),'sha256':'0'*64}]}
    called = []
    with pytest.raises(ValueError, match='checksum'):
        recorded.run_recorded_session(tmp_path / 'run', work, 'unused', 'model', 8080,
            'prompt', spec, lambda path: {}, client_runner=lambda *a,**k: called.append(True))
    assert called == []
