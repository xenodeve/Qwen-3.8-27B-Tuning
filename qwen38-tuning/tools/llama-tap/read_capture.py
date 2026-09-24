r"""Turn a `relay.py` capture into rows a person can read.

The capture is raw bytes on purpose (see `relay.py`).  This reader parses HTTP
framing conservatively: a framing error becomes row metadata, never a guessed
exchange.

    python read_capture.py ../../logs/llama-tap            # JSONL
    python read_capture.py ../../logs/llama-tap --summary  # the flags only
"""
import argparse
import json
import os
import sys


PER_REQUEST = (
    "temperature", "top_p", "top_k", "min_p", "repeat_penalty",
    "presence_penalty", "frequency_penalty", "max_tokens", "seed", "stop",
    "chat_template_kwargs", "logit_bias", "cache_prompt", "n_probs",
    "timings_per_token", "return_progress", "continue_final_message",
    "add_generation_prompt", "response_format", "tools", "tool_choice",
)


_REQUEST_METHODS = (b"POST ", b"GET ", b"PUT ", b"HEAD ", b"DELETE ",
                    b"OPTIONS ", b"PATCH ")


def _split_head(blob):
    marker = blob.find(b"\r\n\r\n")
    if marker < 0:
        return blob, b"", False
    return blob[:marker], blob[marker + 4:], True


def _headers(head):
    """Return decoded headers and decoding/format problems without raising."""
    out, notes = {}, []
    for raw in head.split(b"\r\n")[1:]:
        key, sep, value = raw.partition(b":")
        if not sep:
            notes.append("error: malformed header")
            continue
        try:
            name = key.strip().decode("ascii").lower()
        except UnicodeDecodeError:
            name = key.strip().decode("ascii", "replace").lower()
            notes.append("error: header name is not ASCII")
        try:
            text = value.strip().decode("utf-8")
        except UnicodeDecodeError:
            text = value.strip().decode("utf-8", "replace")
            notes.append("error: header value is not valid UTF-8")
        out[name] = text
    return out, notes


def _content_length(headers):
    value = headers.get("content-length")
    if value is None:
        return None, None
    try:
        length = int(value)
    except ValueError:
        return None, "error: invalid content-length"
    if length < 0:
        return None, "error: negative content-length"
    return length, None


def _transfer_encoding(headers):
    value = headers.get("transfer-encoding")
    if value is None:
        return False, None
    codings = [coding.strip().lower() for coding in value.split(",")]
    if codings == ["chunked"]:
        return True, None
    return False, "error: unsupported transfer-encoding %r" % value


def _chunked(blob):
    """Read a chunked body, including extensions and trailers, without guessing."""
    body = bytearray()
    offset = 0
    while True:
        line_end = blob.find(b"\r\n", offset)
        if line_end < 0:
            return bytes(body), len(blob), False, "error: incomplete chunk size"
        line = blob[offset:line_end]
        size_text = line.split(b";", 1)[0].strip()
        try:
            size = int(size_text, 16)
        except ValueError:
            return bytes(body), line_end + 2, False, "error: invalid chunk size"
        if size < 0:
            return bytes(body), line_end + 2, False, "error: negative chunk size"
        offset = line_end + 2
        if size == 0:
            if blob[offset:offset + 2] == b"\r\n":
                return bytes(body), offset + 2, True, None
            trailer_end = blob.find(b"\r\n\r\n", offset)
            if trailer_end < 0:
                return bytes(body), len(blob), False, "error: incomplete chunk trailers"
            trailers = blob[offset:trailer_end]
            _, notes = _headers(b"HTTP/1.1 200 OK\r\n" + trailers)
            note = "; ".join(notes) if notes else None
            return bytes(body), trailer_end + 4, True, note
        end = offset + size
        if end + 2 > len(blob):
            return bytes(body) + blob[offset:], len(blob), False, \
                "error: truncated chunk data"
        body.extend(blob[offset:end])
        if blob[end:end + 2] != b"\r\n":
            return bytes(body), end, False, "error: chunk data lacks CRLF"
        offset = end + 2


def _requests(blob):
    out = []
    while blob.startswith(_REQUEST_METHODS):
        head, rest, head_complete = _split_head(blob)
        line = head.split(b"\r\n", 1)[0]
        try:
            parts = line.decode("ascii").split(" ")
        except UnicodeDecodeError:
            parts = ["", ""]
        headers, notes = _headers(head)
        complete = head_complete and not notes
        body = b""
        consumed = len(blob) - len(rest) if head_complete else len(blob)
        chunked, transfer_error = _transfer_encoding(headers)
        conflict = chunked and "content-length" in headers
        if not head_complete:
            notes.append("error: incomplete request headers")
        elif transfer_error:
            body = rest
            consumed = len(blob)
            complete = False
            notes.append(transfer_error)
        elif chunked:
            body, used, framed, note = _chunked(rest)
            consumed += used
            complete = complete and framed
            if note:
                notes.append(note)
        else:
            length, error = _content_length(headers)
            if error:
                notes.append(error)
                complete = False
            else:
                length = length or 0
                body = rest[:length]
                consumed += len(body)
                if len(body) != length:
                    complete = False
                    notes.append("error: truncated request body (%d of %d bytes)" %
                                 (len(body), length))
        if conflict:
            complete = False
            notes.append("error: transfer-encoding conflicts with content-length")
        out.append({"method": parts[0] if parts else "",
                    "path": parts[1] if len(parts) > 1 else "",
                    "headers": headers, "body": body, "complete": complete,
                    "notes": notes})
        if consumed <= 0 or not complete:
            break
        blob = blob[consumed:]
    return out


def _response(blob, method):
    if not blob.startswith(b"HTTP/"):
        return None, blob
    head, rest, head_complete = _split_head(blob)
    headers, notes = _headers(head)
    status = 0
    try:
        status = int(head.split(b" ", 2)[1])
    except (IndexError, ValueError):
        notes.append("error: malformed response status")
    complete = head_complete and not notes
    body = b""
    consumed = len(blob) - len(rest) if head_complete else len(blob)
    chunked, transfer_error = _transfer_encoding(headers)
    conflict = chunked and "content-length" in headers
    if not head_complete:
        notes.append("error: incomplete response headers")
    elif method == "HEAD" or status in (204, 304) or 100 <= status < 200:
        pass
    elif transfer_error:
        body = rest
        consumed = len(blob)
        complete = False
        notes.append(transfer_error)
    elif chunked:
        body, used, framed, note = _chunked(rest)
        consumed += used
        complete = complete and framed
        if note:
            notes.append(note)
    else:
        length, error = _content_length(headers)
        if error:
            notes.append(error)
            complete = False
        elif length is not None:
            body = rest[:length]
            consumed += len(body)
            if len(body) != length:
                complete = False
                notes.append("error: truncated response body (%d of %d bytes)" %
                             (len(body), length))
        else:
            # No delimiter permits neither a safe split nor a completion claim.
            body = rest
            consumed = len(blob)
            complete = None
            notes.append("unknown: response has no HTTP body framing")
    if conflict:
        complete = False
        notes.append("error: transfer-encoding conflicts with content-length")
    return {"status": status, "headers": headers, "body": body,
            "complete": complete, "notes": notes}, blob[consumed:]


def _responses(blob, methods):
    """One final response per request; interim 1xx responses do not consume it."""
    out = []
    for method in methods:
        response = None
        while blob.startswith(b"HTTP/"):
            candidate, remainder = _response(blob, method)
            if candidate is None:
                break
            blob = remainder
            if 100 <= candidate["status"] < 200 and candidate["status"] != 101:
                continue
            response = candidate
            break
        out.append(response)
        if response is None:
            break
    return out


def _sse_events(body):
    return sum(1 for line in body.split(b"\n") if line.startswith(b"data:"))


def _first_byte(idx_path):
    try:
        with open(idx_path, encoding="ascii") as fh:
            for line in fh:
                return json.loads(line)["t"]
    except (OSError, ValueError, KeyError):
        return None


def _metadata(base):
    try:
        with open(base + ".meta.json", encoding="utf-8") as fh:
            record = json.load(fh)
    except OSError:
        return {"state": "unknown", "errors": [], "note":
                "unknown: capture metadata is missing"}
    except (ValueError, UnicodeDecodeError):
        return {"state": "unknown", "errors": ["invalid metadata"], "note":
                "error: capture metadata is invalid"}
    if not isinstance(record, dict):
        return {"state": "unknown", "errors": ["invalid metadata"], "note":
                "error: capture metadata is invalid"}
    errors = record.get("capture_errors") or []
    state = record.get("state") or "unknown"
    if not isinstance(errors, list) or not isinstance(state, str):
        return {"state": "unknown", "errors": ["invalid metadata"], "note":
                "error: capture metadata is invalid"}
    if state != "closed" or errors:
        return {"state": state, "errors": errors, "note":
                "error: capture is not final or has capture errors"}
    return {"state": state, "errors": [], "note": None}


def _body_text(body, notes):
    try:
        return body.decode("utf-8"), None
    except UnicodeDecodeError:
        notes.append("error: response body is not valid UTF-8")
        return None, "response body is not valid UTF-8"


def _sse_state(text, notes):
    """Return whether an SSE stream reached a terminal event and stayed valid."""
    if text is None:
        return False, False
    terminal = False
    valid = True
    event_type = None
    for line in text.splitlines():
        if not line:
            event_type = None
            continue
        if line.startswith("event:"):
            event_type = line[6:].strip()
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            terminal = True
            continue
        try:
            message = json.loads(data)
        except ValueError:
            valid = False
            notes.append("error: malformed SSE JSON data")
            continue
        if event_type == "message_stop" or (
                isinstance(message, dict) and message.get("type") == "message_stop"):
            terminal = True
    return terminal, valid


def rows(capture_dir):
    out = []
    names = sorted(n[:-8] for n in os.listdir(capture_dir)
                   if n.endswith(".req.bin"))
    for stem in names:
        base = os.path.join(capture_dir, stem)
        with open(base + ".req.bin", "rb") as fh:
            req_blob = fh.read()
        try:
            with open(base + ".rsp.bin", "rb") as fh:
                rsp_blob = fh.read()
        except OSError:
            rsp_blob = b""
        meta = _metadata(base)
        reqs = _requests(req_blob)
        rsps = _responses(rsp_blob, [request["method"] for request in reqs])
        if not reqs:
            out.append({"connection": stem, "request_index": 0, "method": None,
                        "path": None, "request": None, "status": None,
                        "request_complete": False, "response_complete": None,
                        "stream_complete": None, "usable": False,
                        "capture_state": meta["state"],
                        "note": "no HTTP request recognised in %d bytes" %
                                len(req_blob)})
            continue
        for index, request in enumerate(reqs):
            notes = list(request["notes"])
            body = None
            request_json_valid = False
            if request["body"]:
                try:
                    body = json.loads(request["body"])
                    request_json_valid = True
                except (ValueError, UnicodeDecodeError):
                    notes.append("error: request body is not valid JSON (%d bytes)" %
                                 len(request["body"]))
            response = rsps[index] if index < len(rsps) else None
            stream_complete = None
            stream_valid = True
            if response is None:
                notes.append("unknown: no response captured for request")
                response_complete = None
                response_text = None
                decode_error = None
                is_sse = False
            else:
                notes.extend(response["notes"])
                response_complete = response["complete"]
                response_text, decode_error = _body_text(response["body"], notes)
                content_type = response["headers"].get("content-type", "")
                is_sse = content_type.split(";", 1)[0].strip().lower() == \
                    "text/event-stream"
                if is_sse:
                    stream_complete, stream_valid = _sse_state(response_text, notes)
            request_complete = request["complete"]
            if meta["note"]:
                notes.append(meta["note"])
            if meta["state"] != "closed" or meta["errors"]:
                request_complete = False
                if response is not None:
                    response_complete = False
            usable = (request_json_valid and request_complete is True and
                      response_complete is True and meta["state"] == "closed" and
                      not meta["errors"] and decode_error is None and stream_valid and
                      (not is_sse or stream_complete is True))
            note = "; ".join(dict.fromkeys(notes)) or None
            row = {
                "connection": stem,
                "request_index": index,
                "method": request["method"],
                "path": request["path"],
                "request": body,
                "note": note,
                "status": response["status"] if response else None,
                "content_type": (response["headers"].get("content-type")
                                 if response else None),
                "response_bytes": len(response["body"]) if response else 0,
                "sse_events": _sse_events(response["body"]) if response else 0,
                "first_byte_s": _first_byte(base + ".rsp.idx") if index == 0 else None,
                "request_complete": request_complete,
                "response_complete": response_complete,
                "capture_state": meta["state"],
                "response_body": response_text,
                "stream_complete": stream_complete,
                "usable": usable,
            }
            if decode_error:
                row["response_decode_error"] = decode_error
            out.append(row)
    return out


def summary(rs):
    """Only the fields that travel per request, only where they were sent."""
    seen = {}
    for row in rs:
        body = row.get("request")
        if not isinstance(body, dict):
            continue
        for key in PER_REQUEST:
            if key in body:
                seen.setdefault(key, []).append(body[key])
    lines = []
    for key in PER_REQUEST:
        if key not in seen:
            continue
        vals = seen[key]
        unique = []
        for value in vals:
            text = json.dumps(value, sort_keys=True, ensure_ascii=False)
            if text not in unique:
                unique.append(text)
        lines.append("%-24s %3d sent  %s" % (
            key, len(vals), " | ".join(unique[:4])[:140]))
    absent = [key for key in PER_REQUEST if key not in seen]
    lines.append("")
    lines.append("never sent: " + (", ".join(absent) if absent else "(none)"))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("capture_dir")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args(argv)
    result = rows(args.capture_dir)
    if args.summary:
        print("%d exchanges in %s" % (len(result), args.capture_dir))
        print(summary(result))
    else:
        for row in result:
            sys.stdout.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
