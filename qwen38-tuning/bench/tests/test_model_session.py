"""Offline ownership and lifecycle contracts for the model process."""
from __future__ import annotations

import socket
import sys
import time
from pathlib import Path

import pytest
sys.path.insert(0, str(Path(__file__).parents[1]))
from model_session import ModelSession, SessionStartError


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def fake_server_args(port, ready=True):
    code = f'''import http.server,json
port={port}
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        value = {{'status':'ok'}} if self.path == '/health' else {{'props':True}}
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self,*args): pass
http.server.ThreadingHTTPServer(('127.0.0.1', port), H).serve_forever()
'''
    return [sys.executable, '-u', '-c', code]


def test_session_starts_health_checks_and_stops_only_owned_child(tmp_path):
    port = free_port()
    session = ModelSession(fake_server_args(port), cwd=tmp_path, port=port, health_timeout=5)
    evidence = session.start()
    try:
        assert evidence['pid'] > 0
        assert evidence['port'] == port
        assert evidence['health']['status'] == 'ok'
        assert session.process.poll() is None
    finally:
        stopped = session.stop()
    assert stopped['owned_pid'] == evidence['pid']
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1', port)) != 0


def test_session_writes_owned_server_log_without_pipe_backpressure(tmp_path):
    port = free_port(); log = tmp_path / 'server.log'
    session = ModelSession(fake_server_args(port), cwd=tmp_path, port=port,
                           log_path=log, health_timeout=5)
    session.start()
    try:
        assert log.exists()
    finally:
        session.stop()
    assert log.read_bytes() == log.read_bytes()



def test_occupied_port_refuses_before_starting_process(tmp_path):
    port = free_port()
    listener = socket.socket(); listener.bind(('127.0.0.1', port)); listener.listen(1)
    try:
        session = ModelSession(fake_server_args(port), cwd=tmp_path, port=port)
        with pytest.raises(SessionStartError, match='occupied'):
            session.start()
        assert session.process is None
    finally:
        listener.close()


def test_health_timeout_terminates_owned_process(tmp_path):
    port = free_port()
    session = ModelSession([sys.executable, '-u', '-c', 'import time; time.sleep(30)'],
                           cwd=tmp_path, port=port, health_timeout=0.2, poll_interval=0.02)
    with pytest.raises(SessionStartError, match='health'):
        session.start()
    assert session.process is not None
    assert session.process.poll() is not None


def test_stop_is_idempotent_without_owned_process(tmp_path):
    session = ModelSession(fake_server_args(free_port()), cwd=tmp_path, port=free_port())
    assert session.stop()['owned_pid'] is None
