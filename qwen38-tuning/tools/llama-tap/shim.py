r"""Sit where Unsloth Studio thinks `llama-server` is, and tap what goes past.

Studio finds its binary through `LLAMA_SERVER_PATH` first
(`studio/backend/core/inference/llama_cpp.py:6544`) and checks only that the
path is a file -- no extension, no executable bit on Windows. So pointing that
variable at `llama-server.cmd` beside this file is the whole installation, and
unsetting it is the whole uninstall. **Studio's own install is never touched.**

WHAT IT DOES

    Studio  ->  llama-server.cmd  ->  shim.py
                                        |- starts the REAL binary on a free port
                                        |- starts relay.py on the port Studio asked for
                                        `- waits, and returns the child's exit code

WHY IT READS THE RAW COMMAND LINE

Studio passes

    --chat-template-kwargs {"enable_thinking": true, "preserve_thinking": true}

as ONE argument. Going through `cmd.exe` and rebuilding from `sys.argv` splits
it on spaces, and llama-server would then be launched with a configuration this
tool invented -- the tap changing the thing it exists to observe, which is the
one failure `relay.py` refuses on the socket and must equally refuse here.

`GetCommandLineW()` returns the line as Windows holds it. The shim rewrites the
`--port` value textually and hands the string to `CreateProcess` unparsed, so
every other byte is the one Studio wrote.

`llama-server.cmd` forwards the tail as `%*`, which is a LITERAL substitution of
the remaining command text -- quotes, braces and spaces intact. That is the
opposite of `sys.argv`, which has already been through the C runtime's parser
by the time Python sees it. `--llama-tap-args` marks where Studio's own text
begins so the shim never has to count leading tokens.

ENVIRONMENT

    LLAMA_TAP_REAL   the real binary (default: Studio's own cmake build)
    LLAMA_TAP_OUT    capture directory (default: qwen38-tuning/logs/llama-tap)
"""
import os
import re
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REAL = os.path.join(
    os.path.expanduser("~"), ".unsloth", "llama.cpp", "build", "bin",
    "Release", "llama-server.exe")
DEFAULT_OUT = os.path.abspath(os.path.join(_HERE, "..", "..", "logs", "llama-tap"))

_PORT = re.compile(r"(?<![\w-])--port[ \t]+(\d+)")
MARKER = "--llama-tap-args"


def args_after_marker(raw):
    """Studio's argument text, taken verbatim from after `--llama-tap-args`.

    Counting leading tokens would work today and break the first time the
    interpreter is invoked differently (`py -3`, an absolute python path with
    spaces, `-X utf8`). A marker cannot be miscounted.
    """
    i = raw.find(MARKER)
    return "" if i < 0 else raw[i + len(MARKER):].strip()


def raw_command_line():
    """This process's command line, exactly as Windows holds it."""
    import ctypes
    ctypes.windll.kernel32.GetCommandLineW.restype = ctypes.c_wchar_p
    return ctypes.windll.kernel32.GetCommandLineW()


def _strip_program(line):
    """Drop the leading program token, quoted or not, and return the rest."""
    line = line.lstrip()
    if line.startswith('"'):
        end = line.find('"', 1)
        return line[end + 1:].lstrip() if end >= 0 else ""
    i = line.find(" ")
    return line[i + 1:].lstrip() if i >= 0 else ""


def _quote(path):
    return '"%s"' % path if " " in path or "\\" in path else path


def wants_tap(line):
    """A launch has a `--port`; a capability probe does not.

    Studio probes flag support by running the binary and grepping
    (`supports_slot_save = _is_real("--slot-save-path")`). A probe that got a
    proxy's answer instead of the binary's would change what Studio then
    launches with, so probes go straight through.
    """
    return _PORT.search(line) is not None


def listen_port(line):
    m = _PORT.search(line)
    return int(m.group(1)) if m else None


def child_line(args, real, new_port):
    """The child's command line from Studio's argument TEXT, not from argv."""
    if new_port is not None and _PORT.search(args):
        args = _PORT.sub("--port %d" % new_port, args, count=1)
    return (_quote(real) + " " + args).rstrip()


def rewrite(line, real, new_port):
    """The child's command line: same arguments, new program, new port."""
    rest = _strip_program(line)
    if real is not None:
        head = _quote(real)
    else:
        # No real binary named: the caller supplied the whole child command in
        # `line` already (this is how the end-to-end test drives it).
        head = line[:len(line) - len(rest)].strip()
    if new_port is not None and _PORT.search(rest):
        rest = _PORT.sub("--port %d" % new_port, rest, count=1)
    return (head + " " + rest).rstrip()


def free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def run(line, real=None, out_dir=None, wait=True):
    """Start the child and, if this is a launch, the tap in front of it."""
    out_dir = out_dir or os.environ.get("LLAMA_TAP_OUT") or DEFAULT_OUT
    front = listen_port(line)
    if front is None:
        child = subprocess.Popen(rewrite(line, real, None))
        return child if not wait else child.wait()

    back = free_port()
    child = subprocess.Popen(rewrite(line, real, back))

    sys.path.insert(0, _HERE)
    import relay
    t = relay.Tap(listen_port=front, upstream_port=back, out_dir=out_dir)
    # The relay binds BEFORE the model finishes loading, which is deliberate:
    # Studio polls the port for readiness, and a tap that appeared late would
    # let one probe reach a closed socket and be reported as a failed launch.
    # A connection arriving before llama-server is up is closed, not answered
    # -- see relay.Tap._session.
    t.start()
    child._llama_tap = t                       # so terminate() can stop it
    if not wait:
        return child
    try:
        return child.wait()
    finally:
        t.stop()


def terminate(child, timeout=15):
    t = getattr(child, "_llama_tap", None)
    if t is not None:
        t.stop()
    child.terminate()
    deadline = time.time() + timeout
    while child.poll() is None and time.time() < deadline:
        time.sleep(0.1)
    if child.poll() is None:
        child.kill()
    return child.poll()


def main():
    real = os.environ.get("LLAMA_TAP_REAL") or DEFAULT_REAL
    if not os.path.isfile(real):
        # Refuse rather than fall back to something that happens to be on PATH.
        # `docs/agents/traps.md` has the entry for killing a llama-server by
        # name; launching one by name is the same mistake facing forward.
        sys.stderr.write(
            "llama-tap: LLAMA_TAP_REAL is not a file: %s\n"
            "Set it to the llama-server this tap should wrap.\n" % real)
        return 2
    args = args_after_marker(raw_command_line())
    line = child_line(args, real, None)
    if not wants_tap(line):
        # A capability probe (`--help`, `--version`). Straight through, so what
        # Studio reads about this build is the build's own answer.
        return subprocess.call(line)
    return run(line, real=None, wait=True)


if __name__ == "__main__":
    sys.exit(main())
