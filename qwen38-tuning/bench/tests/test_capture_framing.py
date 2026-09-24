"""Offline framing tests for llama-tap capture reading."""
import importlib.util
import json
import os
import sys


BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(BENCH))
READER = os.path.join(ROOT, "qwen38-tuning", "tools", "llama-tap",
                      "read_capture.py")
CLOSED = {"state": "closed", "capture_errors": []}


def _reader():
    spec = importlib.util.spec_from_file_location("capture_framing_reader", READER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _capture(tmp_path, request, response, metadata=None):
    (tmp_path / "one.req.bin").write_bytes(request)
    (tmp_path / "one.rsp.bin").write_bytes(response)
    if metadata is not None:
        (tmp_path / "one.meta.json").write_text(json.dumps(metadata),
                                                  encoding="utf-8")
    return _reader().rows(str(tmp_path))


def _request(path, body=b"{}"):
    return (b"POST " + path.encode() + b" HTTP/1.1\r\n"
            b"Content-Type: application/json\r\nContent-Length: " +
            str(len(body)).encode() + b"\r\n\r\n" + body)


def _response(status=200, body=b"", headers=b""):
    return (b"HTTP/1.1 " + str(status).encode() + b" OK\r\n" + headers +
            b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" +
            body)


def test_content_length_keepalive_rows_have_unique_indices_and_complete_bodies(tmp_path):
    rows = _capture(
        tmp_path,
        _request("/one", b'{"n":1}') + _request("/two", b'{"n":2}'),
        _response(body=b"first") + _response(body=b"second"),
        CLOSED,
    )
    assert [(row["request_index"], row["status"], row["response_body"])
            for row in rows] == [(0, 200, "first"), (1, 200, "second")]
    assert all(row["request_complete"] is True for row in rows)
    assert all(row["response_complete"] is True for row in rows)


def test_chunked_extensions_and_trailers_do_not_consume_next_response(tmp_path):
    chunked = (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
               b"3;part=one\r\nabc\r\n2\r\nde\r\n0\r\nX-End: yes\r\n\r\n")
    rows = _capture(tmp_path, _request("/one") + _request("/two"),
                    chunked + _response(body=b"next"),
                    CLOSED)
    assert rows[0]["response_body"] == "abcde"
    assert rows[0]["response_complete"] is True
    assert rows[1]["response_body"] == "next"
    assert rows[1]["status"] == 200


def test_interim_response_does_not_shift_response_pairing(tmp_path):
    rows = _capture(tmp_path, _request("/one") + _request("/two"),
                    b"HTTP/1.1 100 Continue\r\n\r\n" +
                    _response(body=b"one") + _response(body=b"two"),
                    CLOSED)
    assert [row["response_body"] for row in rows] == ["one", "two"]


def test_chunked_request_body_is_decoded_without_losing_its_response(tmp_path):
    request = (b"POST /chunked HTTP/1.1\r\nContent-Type: application/json\r\n"
               b"Transfer-Encoding: chunked\r\n\r\n"
               b"3;name=json\r\n{\"n\r\n4\r\n\":3}\r\n0\r\n\r\n")
    rows = _capture(tmp_path, request, _response(body=b"ok"), CLOSED)
    assert rows[0]["request"] == {"n": 3}
    assert rows[0]["request_complete"] is True
    assert rows[0]["response_body"] == "ok"


def test_no_body_statuses_do_not_consume_the_next_response(tmp_path):
    response = (b"HTTP/1.1 204 No Content\r\nContent-Length: 4\r\n\r\n" +
                b"HTTP/1.1 304 Not Modified\r\nContent-Length: 4\r\n\r\n" +
                _response(body=b"next"))
    rows = _capture(tmp_path, _request("/204") + _request("/304") +
                    _request("/next"), response, CLOSED)
    assert [row["response_bytes"] for row in rows] == [0, 0, 4]
    assert [row["response_body"] for row in rows] == ["", "", "next"]


def test_head_and_statuses_without_bodies_leave_next_response_intact(tmp_path):
    request = (b"HEAD /head HTTP/1.1\r\nContent-Length: 0\r\n\r\n" +
               _request("/get"))
    response = (b"HTTP/1.1 200 OK\r\nContent-Length: 9\r\n\r\n" +
                _response(body=b"next"))
    rows = _capture(tmp_path, request, response,
                    CLOSED)
    assert rows[0]["response_bytes"] == 0
    assert rows[0]["response_complete"] is True
    assert rows[1]["response_body"] == "next"


def test_unframed_persistent_and_legacy_sse_preserve_body_but_are_unknown(tmp_path):
    body = b'data: {"text":"thai \\u0e44"}\n\ndata: [DONE]\n\n'
    rows = _capture(tmp_path, _request("/stream"),
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n" + body,
                    CLOSED)
    assert rows[0]["response_body"] == body.decode("utf-8")
    assert rows[0]["response_complete"] is None
    assert "unknown" in rows[0]["note"]


def test_truncated_and_malformed_framing_is_explicit_not_silent(tmp_path):
    rows = _capture(
        tmp_path,
        b"POST /bad HTTP/1.1\r\nContent-Length: 9\r\n\r\n{}",
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nz\r\nnot-a-chunk",
    )
    assert rows[0]["request_complete"] is False
    assert rows[0]["response_complete"] is False
    assert "error" in rows[0]["note"]


def test_invalid_header_utf8_is_explicit(tmp_path):
    rows = _capture(
        tmp_path,
        b"POST /utf8 HTTP/1.1\r\nX-Name: \xff\r\nContent-Length: 0\r\n\r\n",
        _response(),
        CLOSED,
    )
    assert rows[0]["request_complete"] is False
    assert "UTF-8" in rows[0]["note"]


def test_invalid_json_and_invalid_content_length_are_explicit(tmp_path):
    request = (b"POST /json HTTP/1.1\r\nContent-Length: 1\r\n\r\n{" +
               b"POST /length HTTP/1.1\r\nContent-Length: nope\r\n\r\n")
    rows = _capture(tmp_path, request, _response(body=b"ok"))
    assert "JSON" in rows[0]["note"]
    assert rows[1]["request_complete"] is False
    assert "content-length" in rows[1]["note"]


def test_failed_capture_never_claims_complete_even_when_framed(tmp_path):
    rows = _capture(tmp_path, _request("/failed"), _response(body=b"ok"),
                    {"state": "failed", "capture_errors": ["socket error"]})
    assert rows[0]["request_complete"] is False
    assert rows[0]["response_complete"] is False
    assert rows[0]["capture_state"] == "failed"



def _sse_response(body, chunked=False):
    headers = b"Content-Type: text/event-stream\r\n"
    if chunked:
        return (b"HTTP/1.1 200 OK\r\n" + headers +
                b"Transfer-Encoding: chunked\r\n\r\n" +
                (b"%x\r\n" % len(body)) + body + b"\r\n0\r\n\r\n")
    return _response(body=body, headers=headers)


def test_metadata_json_array_is_invalid_metadata_not_a_reader_crash(tmp_path):
    rows = _capture(tmp_path, _request("/metadata"), _response(body=b"ok"), [])
    assert rows[0]["capture_state"] == "unknown"
    assert "metadata is invalid" in rows[0]["note"]


def test_malformed_headers_and_unsupported_transfer_encoding_are_not_complete(tmp_path):
    malformed = (b"POST /bad-header HTTP/1.1\r\nBroken Header\r\n"
                 b"Content-Length: 0\r\n\r\n")
    rows = _capture(tmp_path, malformed, _response(), CLOSED)
    assert rows[0]["request_complete"] is False
    assert "malformed header" in rows[0]["note"]

    gzip = (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: gzip\r\n\r\n"
            b"not silently empty")
    rows = _capture(tmp_path, _request("/gzip"), gzip, CLOSED)
    assert rows[0]["response_complete"] is False
    assert rows[0]["response_bytes"] == len(b"not silently empty")
    assert "unsupported transfer-encoding" in rows[0]["note"]


def test_transfer_encoding_and_content_length_conflict_is_not_complete(tmp_path):
    response = (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n"
                b"Content-Length: 2\r\n\r\n2\r\nok\r\n0\r\n\r\n")
    rows = _capture(tmp_path, _request("/conflict"), response, CLOSED)
    assert rows[0]["response_complete"] is False
    assert rows[0]["response_body"] == "ok"
    assert "transfer-encoding conflicts" in rows[0]["note"]


def test_fully_framed_sse_without_terminal_is_not_usable(tmp_path):
    rows = _capture(tmp_path, _request("/sse"),
                    _sse_response(b'data: {"type":"message_delta"}\n\n',
                                  chunked=True), CLOSED)
    assert rows[0]["response_complete"] is True
    assert rows[0]["stream_complete"] is False
    assert rows[0]["usable"] is False


def test_malformed_sse_data_before_done_is_not_usable(tmp_path):
    body = b"data: {broken}\n\ndata: [DONE]\n\n"
    rows = _capture(tmp_path, _request("/sse-bad"), _sse_response(body), CLOSED)
    assert rows[0]["response_complete"] is True
    assert rows[0]["stream_complete"] is True
    assert rows[0]["usable"] is False
    assert "malformed SSE JSON" in rows[0]["note"]


def test_anthropic_message_stop_makes_a_valid_sse_usable(tmp_path):
    body = b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    rows = _capture(tmp_path, _request("/messages"), _sse_response(body), CLOSED)
    assert rows[0]["stream_complete"] is True
    assert rows[0]["usable"] is True


def test_scalar_request_json_does_not_crash_summary(tmp_path):
    rows = _capture(tmp_path, _request("/scalar", b"1"), _response(body=b"ok"),
                    CLOSED)
    assert rows[0]["request"] == 1
    assert "never sent:" in _reader().summary(rows)


def test_missing_metadata_marks_connection_close_capture_nonfinal(tmp_path):
    rows = _capture(tmp_path, _request("/close"),
                    b"HTTP/1.0 200 OK\r\n\r\nlegacy")
    assert rows[0]["response_body"] == "legacy"
    assert rows[0]["capture_state"] == "unknown"
    assert rows[0]["response_complete"] is False
    assert "metadata is missing" in rows[0]["note"]
