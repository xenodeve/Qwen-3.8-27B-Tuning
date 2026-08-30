r"""A byte-transparent TCP relay that records HTTP traffic to llama-server.

NOT `bench/tap.py`, AND THAT ONE CAME FIRST. This was written without
checking the register, which is the miss `CLAUDE.md` puts at session-start
item 4. They are different instruments and both are wanted:

  bench/tap.py   an HTTP proxy that PARSES a request and re-issues it with
                 urllib, emitting one JSONL row per request with timings and
                 a `--mark` label so a harness can attribute rows to tasks.
                 It rebuilds headers, so what reaches upstream is its idea of
                 the request -- correct for measuring OUR harness, where we
                 wrote the client.
  relay.py       forwards bytes and never rebuilds anything, so it can be
                 pointed at SOMEBODY ELSE'S client without the audit becoming
                 a measurement of our parser. It also redacts credentials and
                 comes with a shim that gets it between Studio and the server
                 Studio launches on a port of its own choosing.

Use bench/tap.py to label our own runs. Use this to audit a client we do not
control.

WHAT IT IS FOR

Three claims this project published about Unsloth Studio's configuration were
retracted within a day of being written (`docs/reports/CORRECTIONS.md` 36, 37,
38). All three were read off a command line or out of source. The one that cost
real time -- `--reasoning-effort` -- was invisible in both, because Studio sends
it inside every request body rather than at launch.

So: read the wire.

THE RULE

**It forwards bytes and observes them. It never re-serialises anything.**

A tap that parsed a request and wrote it out again would hand llama-server
whatever our parser believes JSON looks like, and the measurement would be of
our parser. This repo's north star is that an instrument returning a believable
number instead of a failure is worse than one that crashes; a rewriting tap is
exactly that instrument. Every byte the client sends is `sendall`-ed onward
unchanged, and the RECORDING is a separate, lossy, redacted copy.

WHAT IT WRITES, per connection, into --out:

    NNNN.req.bin   client -> upstream, credentials redacted
    NNNN.rsp.bin   upstream -> client, verbatim
    NNNN.rsp.idx   one JSON object per chunk: {"t": monotonic, "n": bytes}
                   -- this is what makes first-token latency readable later

Read them with `read_capture.py`. Standalone use, against our own server:

    python tap.py --listen 8081 --upstream 8080 --out ../../logs/llama-tap
"""
import argparse
import json
import os
import re
import socket
import threading
import time

# Header lines whose VALUE must never reach a capture file. Matched at the start
# of a line, case-insensitively, in the request direction only.
#
# Studio's `self._auth_headers` carries an API key when one is configured, and
# `~/.unsloth/studio/run/desktop_backend.json` holds a session token. A capture
# is a file someone pastes into an issue while debugging, so the credential must
# not be in it -- while the traffic itself must still carry it, or upstream
# rejects the request. Those two requirements are why redaction happens on the
# copy and never on the socket.
_SECRET_LINE = re.compile(
    rb"^(authorization|x-api-key|api-key|cookie|set-cookie|proxy-authorization)"
    rb"[ \t]*:[ \t]*", re.IGNORECASE)


class _Redactor:
    """Line-oriented, so it survives arbitrary chunk boundaries.

    A header is a line; a JSON body has no CRLF-delimited `Authorization:`
    line, so working line by line needs no HTTP framing at all. It cannot know
    where headers stop, which makes it conservative rather than wrong: a body
    line that happens to begin `cookie:` is redacted too. Losing that is
    cheaper than a parser that has to track Content-Length and chunked encoding
    correctly to know when it may stop looking.
    """

    def __init__(self):
        self._tail = b""

    def feed(self, data):
        buf = self._tail + data
        out = []
        while True:
            i = buf.find(b"\n")
            if i < 0:
                break
            line, buf = buf[:i + 1], buf[i + 1:]
            m = _SECRET_LINE.match(line)
            if m:
                line = m.group(0) + b"[REDACTED]\r\n"
            out.append(line)
        # A partial line is held back: the secret may straddle the boundary.
        # Cap it so an SSE body with no newline cannot grow without limit.
        if len(buf) > 1 << 20:
            out.append(buf)
            buf = b""
        self._tail = buf
        return b"".join(out)

    def flush(self):
        tail, self._tail = self._tail, b""
        return tail


class Tap:
    """Listens on `listen_port`, forwards to `upstream_port`, records both."""

    def __init__(self, listen_port, upstream_port, out_dir,
                 upstream_host="127.0.0.1", listen_host="127.0.0.1"):
        self.listen_port = listen_port
        self.upstream_port = upstream_port
        self.upstream_host = upstream_host
        self.listen_host = listen_host
        self.out_dir = out_dir
        self._n = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sock = None
        self._thread = None

    def start(self):
        os.makedirs(self.out_dir, exist_ok=True)
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.listen_host, self.listen_port))
        self._sock.listen(64)
        self._thread = threading.Thread(target=self._accept, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        try:
            self._sock.close()
        except (OSError, AttributeError):
            pass

    def _accept(self):
        while not self._stop.is_set():
            try:
                client, _ = self._sock.accept()
            except OSError:
                return
            with self._lock:
                self._n += 1
                n = self._n
            threading.Thread(target=self._session, args=(client, n),
                             daemon=True).start()

    def _session(self, client, n):
        base = os.path.join(self.out_dir, "%04d" % n)
        try:
            up = socket.create_connection(
                (self.upstream_host, self.upstream_port), timeout=30)
        except OSError:
            # Upstream is not up yet. Closing is the honest answer -- inventing
            # a 503 here would be the tap speaking for llama-server, and a
            # client that retried on it would be measuring us.
            client.close()
            return
        up.settimeout(None)
        client.settimeout(None)
        done = threading.Event()
        a = threading.Thread(target=self._pump, daemon=True, args=(
            client, up, base + ".req.bin", _Redactor(), None, done))
        b = threading.Thread(target=self._pump, daemon=True, args=(
            up, client, base + ".rsp.bin", None, base + ".rsp.idx", done))
        a.start()
        b.start()
        done.wait()
        for s in (client, up):
            try:
                s.close()
            except OSError:
                pass

    @staticmethod
    def _pump(src, dst, record, redactor, index, done):
        t0 = time.monotonic()
        try:
            with open(record, "wb") as fh:
                idx = open(index, "w", encoding="ascii") if index else None
                try:
                    while True:
                        try:
                            data = src.recv(65536)
                        except OSError:
                            # The OTHER direction finished and `_session` closed
                            # both sockets under us. Breaking rather than
                            # unwinding is what gets the redactor's held-back
                            # tail written: a request body has no trailing
                            # newline, so it sits in the line buffer until
                            # flush, and letting this raise past the `finally`
                            # dropped every body from the capture while the
                            # traffic itself was perfect. Silent, and the
                            # reader just reported `request: null`.
                            break
                        if not data:
                            break
                        # FORWARD FIRST, RECORD SECOND. If writing the capture
                        # ever raises, the traffic has already gone; the
                        # instrument fails without taking the thing it measures
                        # with it.
                        dst.sendall(data)
                        fh.write(redactor.feed(data) if redactor else data)
                        if idx:
                            idx.write(json.dumps(
                                {"t": round(time.monotonic() - t0, 6),
                                 "n": len(data)}) + "\n")
                            idx.flush()
                        fh.flush()
                finally:
                    if redactor:
                        fh.write(redactor.flush())
                    if idx:
                        idx.close()
        except OSError:
            pass
        finally:
            try:
                dst.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            done.set()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--listen", type=int, required=True)
    ap.add_argument("--upstream", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t = Tap(listen_port=a.listen, upstream_port=a.upstream, out_dir=a.out)
    t.start()
    print("llama-tap: %d -> %d, recording to %s" % (a.listen, a.upstream, a.out),
          flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        t.stop()


if __name__ == "__main__":
    main()
