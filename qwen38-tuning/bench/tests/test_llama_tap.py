r"""`llama-tap` -- read what Unsloth Studio puts on the wire, not what it typed.

WHY THIS EXISTS, IN THREE RETRACTIONS.

Everything this project knew about Studio's configuration came from its
`llama-server` argv and its Python source. Both shipped wrong claims:

  CORRECTIONS 36  `-Beta` dropped `--reasoning-effort` because their command
                  line has none. They send it PER REQUEST. Our server ran at
                  xhigh for an afternoon with decode looking healthy.
  CORRECTIONS 37  the sampler was quoted off `--help`. `/props` says the
                  artifact's own general.sampling.* wins.
  CORRECTIONS 38  `n-match 24` was read as independent agreement. It sits
                  beside n-min 48 and n-max 64; all three are defaults.

One shape: **a command line is not a configuration.**

THE ONE RULE THE INSTRUMENT MUST NOT BREAK. It forwards bytes and observes
them. It never re-serialises a request. This repo's north star is that an
instrument returning a believable number instead of a failure is worse than one
that crashes -- a tap that quietly reformats a request would make Studio's
traffic look like whatever our parser thinks JSON should be.

So the first test here is not "did we capture it", it is "is what arrived
upstream byte-identical to what was sent".
"""
import json
import os
import socket
import subprocess
import sys
import threading
import time

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(BENCH))
TOOLS = os.path.join(ROOT, "qwen38-tuning", "tools", "llama-tap")


def _load(name):
    """Import by FILE, not by sys.path.

    `bench/tap.py` already existed -- an HTTP-level recording proxy for our
    own harness -- and every other test file in this directory puts `bench/`
    on sys.path. `import tap` therefore resolved to whichever ran first, and
    this module's tests passed alone and failed in the suite with
    `BaseRequestHandler.__init__() got an unexpected keyword argument`.
    The relay is now `relay.py`, and loading by path means the next module
    added anywhere in this repo cannot shadow it either.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "llama_tap_" + name, os.path.join(TOOLS, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    sys.path.insert(0, TOOLS)          # relay is imported by shim at runtime
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(TOOLS)
    return mod


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def win_argv(command_line):
    """What CreateProcess will hand the child, decoded by Windows itself.

    Re-implementing this rule set is how the quoting gets lost; the shim never
    parses, and the test that checks the shim should not parse either.
    """
    import ctypes
    from ctypes import wintypes
    n = ctypes.c_int(0)
    ctypes.windll.shell32.CommandLineToArgvW.restype = ctypes.POINTER(
        wintypes.LPWSTR)
    p = ctypes.windll.shell32.CommandLineToArgvW(command_line,
                                                 ctypes.byref(n))
    try:
        return [p[i] for i in range(n.value)]
    finally:
        ctypes.windll.kernel32.LocalFree(p)


class Upstream:
    """A stand-in for llama-server that records exactly what reached it."""

    def __init__(self, reply=b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"):
        self.port = free_port()
        self.reply = reply
        self.seen = b""
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", self.port))
        self._sock.listen(8)
        self._stop = False
        self._t = threading.Thread(target=self._serve, daemon=True)
        self._t.start()

    def _serve(self):
        while not self._stop:
            try:
                c, _ = self._sock.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(c,), daemon=True).start()

    def _handle(self, c):
        c.settimeout(5)
        buf = b""
        try:
            while b"\r\n\r\n" not in buf:
                d = c.recv(65536)
                if not d:
                    break
                buf += d
            head, _, rest = buf.partition(b"\r\n\r\n")
            n = 0
            for line in head.split(b"\r\n"):
                if line.lower().startswith(b"content-length:"):
                    n = int(line.split(b":", 1)[1])
            while len(rest) < n:
                d = c.recv(65536)
                if not d:
                    break
                rest += d
            self.seen += buf.split(b"\r\n\r\n")[0] + b"\r\n\r\n" + rest
            c.sendall(self.reply)
        except Exception:
            pass
        finally:
            c.close()

    def close(self):
        self._stop = True
        try:
            self._sock.close()
        except OSError:
            pass


def send(port, payload, read_all=True, timeout=5):
    c = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    c.sendall(payload)
    got = b""
    if read_all:
        c.settimeout(timeout)
        try:
            while True:
                d = c.recv(65536)
                if not d:
                    break
                got += d
        except socket.timeout:
            pass
    c.close()
    return got


# --------------------------------------------------------------- transparency

BODY = (b'{"messages":[{"role":"user","content":"hi"}],"temperature":1.0,'
        b'"chat_template_kwargs":{"x":1}}')
# Counted, not typed. A hand-written Content-Length that is one byte long makes
# the upstream block until its read timeout and then answer nothing, and the
# transparency test passes anyway because "nothing" arrives unchanged. The
# first version of this file did exactly that.
REQ = (b"POST /v1/chat/completions HTTP/1.1\r\n"
       b"Host: 127.0.0.1\r\n"
       b"Authorization: Bearer sk-secret-do-not-record\r\n"
       b"Content-Type: application/json\r\n"
       b"Content-Length: " + str(len(BODY)).encode() + b"\r\n\r\n" + BODY)


@pytest.fixture
def tapped(tmp_path):
    tap = _load('relay')
    up = Upstream()
    t = tap.Tap(listen_port=free_port(), upstream_port=up.port,
                out_dir=str(tmp_path))
    t.start()
    yield t, up, tmp_path
    t.stop()
    up.close()


def test_what_reaches_upstream_is_byte_identical(tapped):
    """THE rule. Everything else here is worthless if this fails."""
    t, up, _ = tapped
    send(t.listen_port, REQ)
    time.sleep(0.3)
    assert up.seen == REQ, (up.seen, REQ)


def test_the_response_reaches_the_client_unchanged(tapped):
    t, up, _ = tapped
    got = send(t.listen_port, REQ)
    assert got == up.reply, got


def test_both_directions_are_recorded(tapped):
    t, _, out = tapped
    send(t.listen_port, REQ)
    time.sleep(0.3)
    names = sorted(p.name for p in out.iterdir())
    assert any(n.endswith(".req.bin") for n in names), names
    assert any(n.endswith(".rsp.bin") for n in names), names


def test_the_capture_does_not_hold_the_bearer_token(tapped):
    """The tap is pointed at somebody's private chat traffic. Recording their
    prompts is the point; recording a credential is not, and a capture is a
    file that gets copied into an issue by whoever is debugging."""
    t, up, out = tapped
    send(t.listen_port, REQ)
    time.sleep(0.3)
    req = b"".join(p.read_bytes() for p in out.glob("*.req.bin"))
    assert b"sk-secret-do-not-record" not in req, req
    assert b"REDACTED" in req, req
    # ... and redacting the RECORD must not have redacted the TRAFFIC.
    assert b"sk-secret-do-not-record" in up.seen


def test_the_body_survives_redaction_intact(tapped):
    """The redactor works line by line, so a body must pass through whole."""
    t, _, out = tapped
    send(t.listen_port, REQ)
    time.sleep(0.3)
    req = b"".join(p.read_bytes() for p in out.glob("*.req.bin"))
    assert b'"chat_template_kwargs":{"x":1}' in req, req


# ------------------------------------------------------------------ the reader

SSE = (b"HTTP/1.1 200 OK\r\n"
       b"Content-Type: text/event-stream\r\n\r\n"
       b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
       b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n'
       b"data: [DONE]\n\n")


def test_the_reader_recovers_the_request_json(tmp_path):
    read_capture = _load('read_capture')
    tap = _load('relay')
    up = Upstream(reply=SSE)
    t = tap.Tap(listen_port=free_port(), upstream_port=up.port,
                out_dir=str(tmp_path))
    t.start()
    try:
        send(t.listen_port, REQ)
        time.sleep(0.3)
    finally:
        t.stop()
        up.close()
    rows = read_capture.rows(str(tmp_path))
    assert rows, "the reader found no exchanges in a capture that has one"
    r = rows[0]
    assert r["method"] == "POST"
    assert r["path"] == "/v1/chat/completions"
    assert r["request"]["temperature"] == 1.0
    assert r["request"]["chat_template_kwargs"] == {"x": 1}
    assert r["status"] == 200
    assert r["sse_events"] == 3


def test_the_reader_never_reports_the_token_either(tmp_path):
    read_capture = _load('read_capture')
    tap = _load('relay')
    up = Upstream()
    t = tap.Tap(listen_port=free_port(), upstream_port=up.port,
                out_dir=str(tmp_path))
    t.start()
    try:
        send(t.listen_port, REQ)
        time.sleep(0.3)
    finally:
        t.stop()
        up.close()
    blob = json.dumps(read_capture.rows(str(tmp_path)))
    assert "sk-secret-do-not-record" not in blob


# ------------------------------------------------------------------- the shim

JSON_ARG = '--chat-template-kwargs {"enable_thinking": true, "preserve_thinking": true}'


def test_the_shim_rewrites_the_port_and_nothing_else():
    shim = _load('shim')
    line = (r'"C:\x\llama-server.exe" -m C:\m.gguf --port 49297 --parallel 1 '
            + JSON_ARG)
    out = shim.rewrite(line, real=r"C:\real\llama-server.exe", new_port=57000)
    assert "--port 57000" in out
    assert "--port 49297" not in out
    assert JSON_ARG in out, (
        "the one argument that cannot survive a sys.argv round trip was "
        "changed: " + out)


def test_the_shim_keeps_a_json_argument_byte_for_byte():
    """WHY THE SHIM READS GetCommandLineW AND NOT sys.argv.

    Studio passes `--chat-template-kwargs {"enable_thinking": true, ...}` as ONE
    argument. Route it through a .cmd and rebuild the command line from
    sys.argv and cmd.exe has already re-split it on spaces; llama.cpp then sees
    `{"enable_thinking":` alone and the JSON is gone. The tap would be changing
    the configuration it exists to observe.
    """
    shim = _load('shim')
    line = 'x.exe --port 1 ' + JSON_ARG
    assert JSON_ARG in shim.rewrite(line, real="r.exe", new_port=2)


def test_the_shim_puts_the_real_binary_first():
    shim = _load('shim')
    out = shim.rewrite('"C:\\x\\llama-server.exe" -m m.gguf --port 1',
                       real=r"C:\real\llama-server.exe", new_port=2)
    assert out.startswith('"C:\\real\\llama-server.exe" '), out
    assert "llama-server.exe\" \"" not in out


def test_a_command_line_with_no_port_is_passed_through():
    """`--help` and `--version` are how Studio probes for flag support
    (`supports_slot_save = _is_real("--slot-save-path")`). A probe that got a
    proxy's opinion instead of the binary's would change what Studio then
    launches with."""
    shim = _load('shim')
    line = 'x.exe --help'
    out = shim.rewrite(line, real="r.exe", new_port=2)
    assert out == 'r.exe --help', out
    assert shim.wants_tap(line) is False
    assert shim.wants_tap('x.exe -m a.gguf --port 8080') is True


def test_the_shim_cmd_forwards_the_tail_as_raw_text():
    """WRITTEN BACKWARDS FIRST, AND BUILDING IT SAID SO.

    This test originally asserted `%*` was ABSENT, on the belief that it would
    re-split the JSON argument. It does not: `%*` is a LITERAL substitution of
    the remaining command text, quotes and braces intact. The parser that
    destroys the argument is the C runtime's, on the way into `sys.argv` -- and
    the shim never uses `sys.argv`. Asserting the absence of `%*` would have
    banned the one mechanism that preserves the thing it was written to protect.

    `--llama-tap-args` marks where Studio's text begins, so the shim never has
    to count leading interpreter tokens.
    """
    p = os.path.join(TOOLS, "llama-server.cmd")
    assert os.path.exists(p), p
    body = open(p, encoding="ascii").read()
    assert "shim.py" in body
    assert "%*" in body, body
    assert "--llama-tap-args" in body, body


def test_the_marker_takes_studio_text_verbatim():
    shim = _load('shim')
    raw = r'python "C:\t\shim.py" --llama-tap-args ' + '-m m.gguf --port 5 ' + JSON_ARG
    assert shim.args_after_marker(raw) == '-m m.gguf --port 5 ' + JSON_ARG


def test_the_cmd_really_survives_the_json_argument(tmp_path):
    """The end of the argument-mangling question, run rather than reasoned.

    A stand-in for shim.py prints what reached it; the real .cmd is invoked the
    way Studio invokes its binary -- `subprocess` list form, no shell.

    WHAT ARRIVES IS NOT THE LITERAL JSON, and that is correct. Windows carries
    arguments as ONE string, so `subprocess` encodes the braces and quotes on
    the way in:

        --chat-template-kwargs "{\"enable_thinking\": true, ...}"

    cmd forwards that text unchanged and the shim hands it to the next
    CreateProcess, which decodes it back. So the property to assert is not
    "the string looks the same" -- it is "Windows' own parser recovers the
    original argument", and `CommandLineToArgvW` is that parser.
    """
    probe = tmp_path / "shim.py"
    probe.write_text("\n".join([
        "import ctypes",
        "ctypes.windll.kernel32.GetCommandLineW.restype = ctypes.c_wchar_p",
        "print(ctypes.windll.kernel32.GetCommandLineW())",
        ""]), encoding="ascii")
    cmd = tmp_path / "llama-server.cmd"
    cmd.write_bytes(open(os.path.join(TOOLS, "llama-server.cmd"), "rb").read())
    r = subprocess.run([str(cmd), "-m", "m.gguf", "--port", "49297",
                        "--chat-template-kwargs",
                        '{"enable_thinking": true, "preserve_thinking": true}'],
                       capture_output=True, text=True, timeout=60)
    shim = _load('shim')
    assert r.returncode == 0, r.stderr
    tail = shim.args_after_marker(r.stdout.strip())
    argv = win_argv("x.exe " + tail)
    assert argv[argv.index("--chat-template-kwargs") + 1] == \
        '{"enable_thinking": true, "preserve_thinking": true}', argv
    assert argv[argv.index("--port") + 1] == "49297", argv


def test_the_capture_directory_is_ignored_by_git():
    """A capture is somebody's private chat traffic."""
    ig = open(os.path.join(ROOT, ".gitignore"), encoding="utf-8").read()
    assert "llama-tap" in ig, ig


# ------------------------------------------------------- it runs end to end

def test_the_shim_launches_something_and_the_tap_carries_it(tmp_path):
    """The shim, a real child process, and a real request through the relay.

    The child is a Python one-liner rather than llama-server: this asserts the
    plumbing, not the model.
    """
    shim = _load('shim')
    port = free_port()
    fake = os.path.join(str(tmp_path), "fake.py")
    with open(fake, "w", encoding="utf-8") as fh:
        fh.write(
            "import sys,http.server\n"
            "port=int(sys.argv[sys.argv.index('--port')+1])\n"
            "class H(http.server.BaseHTTPRequestHandler):\n"
            "    def do_POST(self):\n"
            "        n=int(self.headers.get('Content-Length',0))\n"
            "        b=self.rfile.read(n)\n"
            "        self.send_response(200)\n"
            "        self.send_header('Content-Length',str(len(b)))\n"
            "        self.end_headers()\n"
            "        self.wfile.write(b)\n"
            "    def log_message(self,*a): pass\n"
            "http.server.HTTPServer(('127.0.0.1',port),H).serve_forever()\n")
    line = '%s "%s" --port %d' % (sys.executable, fake, port)
    p = shim.run(line, real=None, out_dir=str(tmp_path), wait=False)
    try:
        deadline = time.time() + 30
        got = b""
        while time.time() < deadline:
            try:
                got = send(port, REQ, timeout=2)
                if got:
                    break
            except OSError:
                time.sleep(0.3)
        assert got.endswith(b'"chat_template_kwargs":{"x":1}}'), got
    finally:
        shim.terminate(p)
    reqs = list(tmp_path.glob("*.req.bin"))
    assert reqs, "the shim started the tap but nothing was recorded"
