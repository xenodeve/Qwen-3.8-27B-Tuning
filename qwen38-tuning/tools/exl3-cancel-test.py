"""Live cancel test for issue #83: ESC must stop the GPU job.

Simulates Claude Code pressing ESC mid-stream: opens POST /v1/messages
(streaming, max_tokens 2000, a prompt that generates long), reads a few SSE
chunks, then hard-closes the socket. PASS when /health.busy flips false
within ~3 s, completion_tokens_total freezes, and cancelled counts the job.

Usage:  python tools/exl3-cancel-test.py [--base http://127.0.0.1:8000]
"""
import argparse
import http.client
import json
import sys
import time
import urllib.parse
import urllib.request

PROMPT = ("Write a long 1500-word essay about the history of video compression, "
          "from MPEG-1 to AV1, with technical detail in every paragraph. "
          "Do not stop early.")

# Prefill regime (#83 reopen): a prompt big enough that prefill alone takes
# ~15 s at ~700 tok/s. Cutting before the first chunk proves the cancel
# lands while NO write was ever attempted -- the case the first fix missed.
PREFILL_PARA = ("The quick brown fox jumps over the lazy dog while the river keeps "
                "flowing under the old wooden bridge near the quiet village. ")


def health(base):
    with urllib.request.urlopen(base.rstrip("/") + "/health", timeout = 10) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default = "http://127.0.0.1:8000")
    ap.add_argument("--phase", choices = ("decode", "prefill"), default = "decode",
                    help = "decode: cut after 5 chunks; prefill: cut 3 s in with "
                           "~10k prompt tokens, before any chunk can arrive")
    a = ap.parse_args()
    u = urllib.parse.urlparse(a.base)
    host, port = u.hostname, u.port or 80

    model = health(a.base)["model"]
    if a.phase == "prefill":
        prompt = PREFILL_PARA * 450 + "\nSummarize the above in Thai, in detail."
        max_tokens, want_chunks, cut_after = 500, None, 3.0
    else:
        prompt, max_tokens, want_chunks, cut_after = PROMPT, 2000, 5, None
    body = json.dumps({
        "model": model, "max_tokens": max_tokens, "stream": True,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    conn = http.client.HTTPConnection(host, port, timeout = 60)
    conn.request("POST", "/v1/messages", body = body,
                 headers = {"Content-Type": "application/json",
                            "Accept": "text/event-stream"})
    resp = conn.getresponse()
    assert resp.status == 200, f"stream open failed: {resp.status}"
    # NOTE: no socket timeout here -- http.client's chunked reader corrupts
    # itself on a timeout mid-chunk ("cannot read from timed out object").
    # The read loop therefore wakes on line arrival; in prefill that is the
    # 5 s SSE ping, so the cut lands ~5 s in, still deep in prefill.
    chunks = 0
    t0 = time.time()
    while True:
        if cut_after is not None and time.time() - t0 >= cut_after:
            break
        line = resp.readline()
        if not line:
            break
        if line.startswith(b"data: "):
            chunks += 1
            if want_chunks is not None and chunks >= want_chunks:
                break
    t_cut = time.time()
    conn.close()   # ESC: the client is gone
    print(f"[{a.phase}] cut after {chunks} SSE chunks "
          f"({round(t_cut - t0, 1)} s in) at t={t_cut:.1f}")

    busy_free_at, frozen, cancelled = None, None, None
    samples = []
    for i in range(30):
        time.sleep(0.5)
        h = health(a.base)
        samples.append((round(time.time() - t_cut, 1), h["busy"],
                        h["completion_tokens_total"], h.get("cancelled"))
        )
        if busy_free_at is None and not h["busy"]:
            busy_free_at = round(time.time() - t_cut, 1)
        if i >= 28:
            frozen = (samples[-1][2] == samples[-6][2])
            cancelled = h.get("cancelled")
    for s in samples[::4]:
        print(f"t+{s[0]:>5}s busy={s[1]!s:>5} completion_tokens={s[2]} cancelled={s[3]}")
    print(f"busy flipped false at t+{busy_free_at}s; "
          f"tokens frozen over last 2.5s: {frozen}; cancelled={cancelled}")
    ok = (busy_free_at is not None and frozen and (cancelled or 0) >= 1)
    # Honest bounds: detection is polled every 1 s on both legs, but the
    # worker only checks between ExLlamaV3 iterate() calls, and one prefill
    # chunk can run seconds. Decode must die fast; prefill gets room for one
    # in-flight chunk. Before #83 neither ever died.
    bound = 3.0 if a.phase == "decode" else 8.0
    ok = ok and busy_free_at <= bound
    if a.phase == "prefill" and chunks > 1:
        # chunk 1 is always message_start (emitted before the upstream call);
        # anything beyond it means decode had begun -- not a prefill cut.
        print("REGIME INVALID: data arrived before the cut -- not a prefill cut")
        ok = False
    print("CANCEL TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
