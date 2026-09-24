"""Offline regressions: never turn missing stream evidence into plausible timing."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stream_observation import StreamObservation


def frame(obj):
    return b'data: ' + json.dumps(obj, ensure_ascii=False).encode() + b'\n\n'


def observer(protocol='openai'):
    return StreamObservation(protocol, 'request-1', 'a' * 64, 100)


def test_terminal_is_not_http_eof_or_request_end():
    # A protocol sentinel cannot establish that the HTTP response ended.
    o = observer()
    o.headers(110)
    o.feed(frame({'choices': [{'index': 0, 'delta': {'content': 'Hi'}, 'finish_reason': 'stop'}]}) + b'data: [DONE]\n\n', 120)
    s = o.snapshot()
    assert s['timing_ns']['first_text'] == 20
    assert s['timing_ns']['terminal'] == 20
    assert s['timing_ns']['eof'] is None
    assert not s['complete']
    o.eof(130)
    assert not o.snapshot()['complete']
    o.finish(140)
    s = o.snapshot()
    assert s['complete']
    assert s['timing_ns']['end'] == 40
    assert s['counts']['text_chars'] == 2
    assert s['finish_reason'] == 'stop'


def test_every_split_preserves_utf8_crlf_multiline_and_reasoning():
    # Delivery read boundaries are not SSE or Unicode boundaries.
    wire = (b': ping\r\n\r\n' + b'data: {"choices": [\r\ndata: {"index": 0, "delta": {"reasoning_content": "", "reasoning": "' + 'คิด'.encode() + b'", "content": "ok"}}]}\r\n\r\n' + frame({'choices': [], 'usage': {'completion_tokens': 2}}) + b'data: [DONE]\n\n')
    for split in range(len(wire) + 1):
        o = observer()
        o.feed(wire[:split], 120)
        o.feed(wire[split:], 130)
        o.eof(140)
        o.finish(150)
        s = o.snapshot()
        assert s['complete'], (split, s)
        assert s['counts']['reasoning_chars'] == 3
        assert s['counts']['text_chars'] == 2
        assert s['counts']['events'] == 3
        assert s['timing_ns']['first_semantic'] == s['timing_ns']['first_reasoning']
        assert s['chunks'][-1]['end_offset'] == len(wire)
        assert s['timing_ns']['terminal'] in (20, 30)


def test_anthropic_empty_thinking_duplicate_start_and_tool_arguments():
    # Empty thinking and repeated startup metadata must not masquerade as output.
    o = observer('anthropic')
    start = {'type': 'message_start', 'message': {'content': []}}
    o.feed(frame(start) + frame(start) + frame({'type': 'ping'}), 110)
    o.feed(frame({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'thinking', 'thinking': ''}}), 120)
    assert o.snapshot()['timing_ns']['first_semantic'] is None
    o.feed(frame({'type': 'content_block_stop', 'index': 0}), 125)
    o.feed(frame({'type': 'content_block_start', 'index': 1, 'content_block': {'type': 'tool_use', 'id': 'secret', 'name': 'secret', 'input': {}}}), 130)
    o.feed(frame({'type': 'content_block_delta', 'index': 1, 'delta': {'type': 'input_json_delta', 'partial_json': '{"x":1}'}}), 140)
    o.feed(frame({'type': 'content_block_stop', 'index': 1}) + frame({'type': 'message_delta', 'delta': {'stop_reason': 'tool_use'}}) + frame({'type': 'message_stop'}), 150)
    o.eof(160)
    o.finish(170)
    s = o.snapshot()
    assert s['complete']
    assert s['counts']['empty_thinking_blocks'] == 1
    assert s['timing_ns']['first_reasoning'] is None
    assert s['timing_ns']['first_tool'] == 30
    assert s['timing_ns']['first_semantic'] == 30
    assert s['timing_ns']['first_arguments'] == 40
    assert s['counts']['tool_argument_chars'] == 7
    assert 'secret' not in json.dumps(s)


def test_openai_tool_fragments_and_trailing_usage_are_observed():
    o = observer()
    o.feed(frame({'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0, 'id': 'secret', 'function': {'name': 'secret', 'arguments': ''}}]}}]}), 110)
    o.feed(frame({'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': '{}'}}]}, 'finish_reason': 'tool_calls'}]}), 120)
    assert o.snapshot()['timing_ns']['terminal'] is None
    o.feed(frame({'choices': [], 'usage': {'completion_tokens': 5}}), 130)
    o.feed(b'data: [DONE]\n\n', 140)
    o.eof(150)
    o.finish(160)
    s = o.snapshot()
    assert s['complete']
    assert s['counts']['events'] == 4
    assert s['counts']['tool_argument_chars'] == 2
    assert s['timing_ns']['first_tool'] == 10
    assert s['timing_ns']['first_arguments'] == 20


@pytest.mark.parametrize('at', [True, False, 99, 1.5, '120', None])
def test_wrong_timestamp_invalidates_without_invented_elapsed(at):
    o = observer()
    o.headers(at)
    assert 'invalid_timestamp' in o.snapshot()['errors']
    assert o.snapshot()['timing_ns']['headers'] is None
    assert not o.snapshot()['complete']


def test_backwards_order_and_calls_after_end_are_invalid():
    o = observer()
    o.headers(120)
    o.feed(b'data: [DONE]\n\n', 119)
    assert o.snapshot()['timing_ns']['terminal'] is None
    assert 'invalid_timestamp' in o.snapshot()['errors']
    o.finish(130)
    o.feed(b'data: [DONE]\n\n', 140)
    assert 'invalid_lifecycle' in o.snapshot()['errors']


@pytest.mark.parametrize('wire,code', [(b'data: {', 'truncated_event'), (b'data: bad\n\n', 'invalid_json'), (b'data: \xff\n\n', 'invalid_utf8'), (frame({'choices': []}), 'missing_terminal')])
def test_truncation_and_parse_failures_cannot_complete(wire, code):
    o = observer()
    o.feed(wire, 110)
    o.eof(120)
    o.finish(130)
    assert code in o.snapshot()['errors']
    assert not o.snapshot()['complete']


@pytest.mark.parametrize('obj', [
    {'choices': [{'index': 0, 'delta': {'content': 'x'}}, {'index': 1, 'delta': {'content': 'y'}}]},
    {'choices': [{'index': 1, 'delta': {'content': 'x'}}]},
    {'choices': [{'index': 0, 'delta': {'content': ['unsupported']}}]},
    {'type': 'response.output_text.delta', 'delta': 'unsupported'},
    {'error': {'message': 'secret'}},
])
def test_ambiguous_or_unsupported_provider_events_invalidate(obj):
    o = observer()
    o.feed(frame(obj) + b'data: [DONE]\n\n', 110)
    o.eof(120)
    o.finish(130)
    assert o.snapshot()['errors']
    assert not o.snapshot()['complete']


def test_secret_provider_strings_error_and_clock_are_not_persisted():
    secret = 'https://credential:SECRET@provider.invalid/path'
    o = StreamObservation('openai', 'request-1', 'a' * 64, 100, {'implementation': secret, 'resolution': 1e-9, 'monotonic': True, 'adjustable': False})
    o.feed(frame({'choices': [{'index': 0, 'delta': {'content': secret, 'reasoning_text': secret}, 'finish_reason': secret}], 'model': secret}), 110)
    o.feed(b'data: [DONE]\n\n', 120)
    o.eof(130)
    o.finish(140, error=RuntimeError(secret))
    s = o.snapshot()
    assert secret not in json.dumps(s)
    assert s['finish_reason'] == 'unknown'
    assert s['errors'] == ['request_error']
    assert not s['complete']
    assert s['clock']['resolution_ns'] == 1
    assert set(s) == {'schema_version', 'protocol', 'request_id', 'request_sha256', 'boundary', 'clock', 'timing_ns', 'chunks', 'counts', 'complete', 'errors', 'finish_reason'}


def test_conflicting_finish_reasons_and_duplicate_terminal_invalidate():
    o = observer()
    for reason in ('stop', 'length'):
        o.feed(frame({'choices': [{'index': 0, 'delta': {}, 'finish_reason': reason}]}), 110)
    o.feed(b'data: [DONE]\n\ndata: [DONE]\n\n', 120)
    o.eof(130)
    o.finish(140)
    assert 'conflicting_finish_reason' in o.snapshot()['errors']
    assert 'duplicate_terminal' in o.snapshot()['errors']


@pytest.mark.parametrize('protocol,wire', [
    ('anthropic', frame({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'thinking_delta', 'thinking': 'x'}})),
    ('anthropic', frame({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'image'}})),
    ('anthropic', frame({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'thinking'}})),
    ('openai', frame({'choices': [{'index': 0, 'delta': {'function_call': {'name': 'old'}}}]})),
    ('openai', frame({'choices': [{'index': 0, 'delta': {'tool_calls': [42]}}]})),
])
def test_unsupported_or_unclosed_blocks_do_not_look_complete(protocol, wire):
    o = observer(protocol)
    o.feed(wire, 110)
    o.feed(frame({'type': 'message_stop'}) if protocol == 'anthropic' else b'data: [DONE]\n\n', 120)
    o.eof(130)
    o.finish(140)
    assert o.snapshot()['errors']
    assert not o.snapshot()['complete']


def test_nonempty_anthropic_start_is_counted_once_and_all_splits_work():
    start = frame({'type': 'message_start', 'message': {'content': [{'type': 'thinking', 'thinking': 'คิด'}]}})
    wire = start + start + frame({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'thinking_delta', 'thinking': '!'}}) + frame({'type': 'content_block_stop', 'index': 0}) + frame({'type': 'message_stop'})
    for split in range(len(wire) + 1):
        o = observer('anthropic')
        o.feed(wire[:split], 110)
        o.feed(wire[split:], 120)
        o.eof(130)
        o.finish(140)
        assert o.snapshot()['complete']
        assert o.snapshot()['counts']['reasoning_chars'] == 4
        assert o.snapshot()['counts']['empty_thinking_blocks'] == 0


def test_offsets_are_body_bytes_and_coalesced_events_share_observation_time():
    o = observer()
    wire = frame({'choices': [{'delta': {'content': 'ก'}}]}) + frame({'choices': [{'delta': {'reasoning': 'r'}}]})
    o.feed(wire, 150)
    s = o.snapshot()
    assert s['chunks'] == [{'start_offset': 0, 'end_offset': len(wire), 'elapsed_ns': 50}]
    assert s['timing_ns']['first_body'] == s['timing_ns']['first_text'] == s['timing_ns']['first_reasoning'] == 50
    s['chunks'][0]['end_offset'] = -1
    s['counts']['text_chars'] = -1
    assert o.snapshot()['chunks'][0]['end_offset'] == len(wire)
    assert o.snapshot()['counts']['text_chars'] == 1


@pytest.mark.parametrize('kwargs', [dict(started_ns=True), dict(protocol='secret'), dict(request_id='https://secret'), dict(request_sha256='secret')])
def test_invalid_constructor_metadata_raises_fixed_code(kwargs):
    args = dict(protocol='openai', request_id='r', request_sha256='a' * 64, started_ns=100)
    args.update(kwargs)
    with pytest.raises(ValueError) as exc:
        StreamObservation(**args)
    assert 'secret' not in str(exc.value)
