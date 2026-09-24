"""Notice a dead HTTP client so a cancelled request stops burning GPU (issue #83).

ESC in Claude Code closes the SSE connection; aiohttp never cancels the
handler, so without a probe the ExLlamaV3 job runs to max_tokens behind
gen_lock. Both chat_completions paths poll `client_gone(request)` and set
their per-request event; generate_full checks it every iterate next to the
loop-guard check and calls generator.cancel(job).
"""
POLL_S = 1.0   # bound on wasted GPU time after a disconnect


def client_gone(request):
    """True when the HTTP client already went away. Measured 2026-09-15
    (localhost, aiohttp 3.14.2 and the server's own 3.14.3): after the peer
    closes, request.transport becomes None within 0.5 s -- while
    transport.is_closing() stays False forever, because it only reflects a
    LOCAL close, never a peer FIN. A missing attribute still reads as
    'present': the probe must never kill a live request on uncertainty, but
    a detached transport IS the disconnect signal, including through whole
    prefills where no write is ever attempted (#83)."""
    try:
        transport = request.transport
    except Exception:
        return False
    if transport is None:
        return True
    try:
        return bool(transport.is_closing())
    except Exception:
        return False
