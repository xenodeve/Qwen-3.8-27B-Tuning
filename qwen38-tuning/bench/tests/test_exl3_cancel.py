"""A dead client must stop the GPU job (issue #83).

INCIDENT. ESC in Claude Code closes the SSE connection to /v1/messages, but
the model keeps working -- GPU usage stays high until the generation ends on
its own, /health sticks at busy:true, and the next request queues behind the
dead one under gen_lock. Traced: the cancel signal dies twice (the outer
_stream swallows ConnectionResetError and only closes a socket; the inner
chat_completions swallows it too while its worker thread keeps iterating),
and the non-streaming path never observes the disconnect at all, because
aiohttp never cancels a handler on client disconnect.

The fix under test: a per-request threading.Event threaded through
generate_full and checked every iterate next to the loop-guard check; both
chat_completions paths poll cancel.client_gone(request) and set the event.
"""
import os
import sys

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUNING = os.path.dirname(BENCH)
SERVING = os.path.join(TUNING, "serving", "exl3")
sys.path.insert(0, SERVING)
import cancel  # noqa: E402
import anthropic_compat as ac  # noqa: E402


class FakeTransport:
    def __init__(self, closing):
        self._closing = closing

    def is_closing(self):
        return self._closing


class FakeRequest:
    def __init__(self, transport):
        self.transport = transport


def test_live_connection_reads_as_present():
    assert cancel.client_gone(FakeRequest(FakeTransport(False))) is False


def test_closing_transport_reads_as_gone():
    assert cancel.client_gone(FakeRequest(FakeTransport(True))) is True


def test_missing_transport_reads_as_present_never_kills_live_work():
    assert cancel.client_gone(object()) is False


def test_a_detached_transport_means_the_client_is_gone():
    """Measured 2026-09-15 on aiohttp 3.14.2 AND the server's own 3.14.3:
    after the client closes, request.transport becomes None within 0.5 s
    while is_closing() stays False forever (it only reflects a LOCAL close).
    The first version of this probe read None as 'present' and was blind
    through whole prefills -- the reopened #83."""
    assert cancel.client_gone(FakeRequest(None)) is True


def test_a_broken_probe_reads_as_present():
    class BadTransport:
        def is_closing(self):
            raise RuntimeError("gone sideways")

    class BadRequest:
        @property
        def transport(self):
            raise RuntimeError("no transport")

    assert cancel.client_gone(FakeRequest(BadTransport())) is False
    assert cancel.client_gone(BadRequest()) is False


def read_server():
    with open(os.path.join(SERVING, "server.py"), encoding = "utf-8") as fh:
        return fh.read()


def test_the_server_imports_the_cancel_probe():
    lines = [l for l in read_server().splitlines()
             if l.startswith("import live_timing")]
    names = [n.strip() for n in lines[0].split("#")[0].replace("import ", "").split(",")]
    assert len(lines) == 1 and "cancel" in names


def test_generate_full_takes_a_cancel_event():
    assert "cancel_event" in read_server()


def test_the_iterate_loop_stops_the_job_on_cancel():
    server = read_server()
    assert "cancel_event.is_set()" in server
    assert 'reason = "cancelled"' in server


def test_both_chat_paths_poll_for_a_dead_client():
    assert read_server().count("client_gone(request)") >= 2


def test_cancels_are_counted_and_visible_on_health():
    server = read_server()
    assert '"cancelled": 0' in server or "'cancelled': 0" in server
    assert 'stats["cancelled"]' in server


def test_pump_stops_on_should_stop_without_stranding_the_read():
    """The outer /v1/messages leg: during prefill the upstream is silent for
    tens of seconds, so ESC is noticed only via should_stop -- and the
    abandoned read must be cancelled, not left to log noise later."""
    import asyncio

    async def silent():
        await asyncio.sleep(30)
        yield b"data: [DONE]\n"

    async def run():
        gen = silent()
        tr = ac.StreamTranslator(model = "m")
        out = []
        async for ev in ac.pump(gen, tr, ping_s = 5.0,
                                should_stop = lambda: True):
            out.append(ev)
        # The read pump cancelled must have settled: closing it here must
        # not raise "aclose(): asynchronous generator is already running"
        # (the ESC traceback of 2026-09-15).
        await gen.aclose()
        return out

    out = asyncio.run(asyncio.wait_for(run(), timeout = 5))
    assert out == []


def test_pump_without_should_stop_keeps_its_ping_cadence():
    """The 1 s probe slices must not change the wire: pings still go out
    every ping_s of silence."""
    import asyncio
    import json as _json

    async def silent_then_done():
        await asyncio.sleep(0.25)
        yield b"data: [DONE]\n"

    async def run():
        tr = ac.StreamTranslator(model = "m")
        out = []
        async for ev, _ in ac.pump(silent_then_done(), tr, ping_s = 0.1):
            out.append(ev)
        return out

    names = [e for e in asyncio.run(run())]
    assert names.count("ping") >= 2


def test_the_outer_stream_hands_pump_a_should_stop_probe():
    body = open(os.path.join(SERVING, "anthropic_routes.py"),
                encoding = "utf-8").read()
    assert "should_stop" in body
    assert "client_gone" in body
