r"""Turn a `relay.py` capture into rows a person can read.

The capture is raw bytes on purpose (see `relay.py`). This is the lossy half: it
parses what it can and says so when it cannot, rather than silently dropping an
exchange it did not understand -- an unparsed request still gets a row, with
`request: null` and the reason.

    python read_capture.py ../../logs/llama-tap            # JSONL
    python read_capture.py ../../logs/llama-tap --summary  # the flags only

`--summary` answers the question the whole tool exists for: **which settings
arrive per request rather than on the command line.**
"""
import argparse
import json
import os
import sys

# The fields that are invisible in `--help`, invisible in the argv, and decide
# how the server behaves. CORRECTIONS 36 is the first of these; the rest are
# from Studio's own payload builder,
# `studio/backend/core/inference/llama_cpp.py:26462`.
PER_REQUEST = (
    "temperature", "top_p", "top_k", "min_p", "repeat_penalty",
    "presence_penalty", "frequency_penalty", "max_tokens", "seed", "stop",
    "chat_template_kwargs", "logit_bias", "cache_prompt", "n_probs",
    "timings_per_token", "return_progress", "continue_final_message",
    "add_generation_prompt", "response_format", "tools", "tool_choice",
)


def _split_head(blob):
    i = blob.find(b"\r\n\r\n")
    return (blob[:i], blob[i + 4:]) if i >= 0 else (blob, b"")


def _headers(head):
    out = {}
    for line in head.split(b"\r\n")[1:]:
        k, _, v = line.partition(b":")
        if _:
            out[k.strip().lower().decode("ascii", "replace")] = \
                v.strip().decode("utf-8", "replace")
    return out


def _requests(blob):
    """Successive requests on one keep-alive connection.

    Only Content-Length framing is handled. A chunked REQUEST body would end
    the walk -- llama-server's clients do not send one, and guessing at a frame
    we have never seen would be the tap inventing data.
    """
    out = []
    while blob[:4] in (b"POST", b"GET ", b"PUT ", b"HEAD", b"DELE", b"OPTI"):
        head, rest = _split_head(blob)
        if not head:
            break
        h = _headers(head)
        n = int(h.get("content-length") or 0)
        body, blob = rest[:n], rest[n:]
        line = head.split(b"\r\n", 1)[0].decode("ascii", "replace").split(" ")
        out.append({"method": line[0],
                    "path": line[1] if len(line) > 1 else "",
                    "headers": h, "body": body})
        if not h.get("content-length"):
            break
    return out


def _responses(blob):
    out = []
    while blob.startswith(b"HTTP/"):
        head, rest = _split_head(blob)
        h = _headers(head)
        status = int(head.split(b" ")[1]) if b" " in head else 0
        n = h.get("content-length")
        if n is not None:
            body, blob = rest[:int(n)], rest[int(n):]
        else:
            # Streamed: chunked or event-stream. The rest of the connection is
            # this response -- llama-server does not pipeline another after it.
            body, blob = rest, b""
        out.append({"status": status, "headers": h, "body": body})
    return out


def _sse_events(body):
    return sum(1 for line in body.split(b"\n") if line.startswith(b"data:"))


def _first_byte(idx_path):
    """Seconds from upstream connect to its first byte back.

    Not time-to-first-token: the tap cannot see when llama-server started
    working, only when bytes crossed it. It is a ceiling on prefill+first token
    and is labelled that way rather than given a name it has not earned.
    """
    try:
        with open(idx_path, encoding="ascii") as fh:
            for line in fh:
                return json.loads(line)["t"]
    except (OSError, ValueError):
        return None


def rows(capture_dir):
    out = []
    names = sorted(n[:-8] for n in os.listdir(capture_dir)
                   if n.endswith(".req.bin"))
    for stem in names:
        base = os.path.join(capture_dir, stem)
        req_blob = open(base + ".req.bin", "rb").read()
        try:
            rsp_blob = open(base + ".rsp.bin", "rb").read()
        except OSError:
            rsp_blob = b""
        reqs, rsps = _requests(req_blob), _responses(rsp_blob)
        if not reqs:
            out.append({"connection": stem, "method": None, "path": None,
                        "request": None,
                        "note": "no HTTP request recognised in %d bytes"
                                % len(req_blob)})
            continue
        for i, q in enumerate(reqs):
            body, note = None, None
            if q["body"]:
                try:
                    body = json.loads(q["body"])
                except ValueError:
                    note = "body is not JSON (%d bytes)" % len(q["body"])
            r = rsps[i] if i < len(rsps) else None
            out.append({
                "connection": stem,
                "method": q["method"],
                "path": q["path"],
                "request": body,
                "note": note,
                "status": r["status"] if r else None,
                "content_type": (r["headers"].get("content-type") if r else None),
                "response_bytes": len(r["body"]) if r else 0,
                "sse_events": _sse_events(r["body"]) if r else 0,
                "first_byte_s": _first_byte(base + ".rsp.idx") if i == 0 else None,
            })
    return out


def summary(rs):
    """Only the fields that travel per request, only where they were sent."""
    seen = {}
    for r in rs:
        body = r.get("request") or {}
        for k in PER_REQUEST:
            if k in body:
                seen.setdefault(k, []).append(body[k])
    lines = []
    for k in PER_REQUEST:
        if k not in seen:
            continue
        vals = seen[k]
        uniq = []
        for v in vals:
            s = json.dumps(v, sort_keys=True, ensure_ascii=False)
            if s not in uniq:
                uniq.append(s)
        lines.append("%-24s %3d sent  %s" % (
            k, len(vals), " | ".join(uniq[:4])[:140]))
    absent = [k for k in PER_REQUEST if k not in seen]
    lines.append("")
    lines.append("never sent: " + (", ".join(absent) if absent else "(none)"))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("capture_dir")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args(argv)
    rs = rows(a.capture_dir)
    if a.summary:
        print("%d exchanges in %s" % (len(rs), a.capture_dir))
        print(summary(rs))
    else:
        for r in rs:
            sys.stdout.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
