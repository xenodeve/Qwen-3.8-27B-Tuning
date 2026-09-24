"""Offline comparison guards: plausible rates must never hide missing evidence."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from session_comparison import build_comparison


def row(output=100, **extra):
    return dict(response_body=json.dumps({'usage': {'prompt_tokens': 20,
                'completion_tokens': output}}), usable=True, **extra)


def observation(key, end=2_000_000_000, complete=True):
    return {'schema_version': 1, 'request_id': key, 'request_sha256': key,
            'protocol': 'openai', 'boundary': 'upstream_http_body',
            'clock': {'implementation': 'perf_counter'}, 'complete': complete, 'errors': [],
            'timing_ns': {'headers': 100_000_000, 'first_body': 200_000_000,
                          'first_semantic': 500_000_000, 'terminal': end,
                          'eof': end, 'end': end}, 'chunks': [], 'counts': {}}


def test_weighted_tps_requires_unique_complete_identity_not_position():
    result = build_comparison([row(100, request_sha256='a'), row(300, request_sha256='b'),
                               row(999, request_sha256='c'), row(999)], [],
                              [observation('b', 6_000_000_000), observation('c', complete=False),
                               observation('a')], {})
    assert result['timing']['effective_request_tps']['value'] == 50
    assert result['timing']['effective_request_tps']['coverage_count'] == 2
    assert result['timing']['effective_output_tokens']['value'] == 400
    assert result['timing']['effective_wall_seconds']['value'] == 8
    assert result['coverage']['incomplete_observations'] == 1
    assert result['timing']['headers_seconds']['value']['n'] == 2
    ambiguous = build_comparison([row(request_sha256='a')], [],
                                 [observation('a'), observation('a')], {})
    assert ambiguous['timing']['effective_request_tps']['value'] is None
    assert ambiguous['coverage']['ambiguous_matches'] == 1


def test_semantic_categories_and_post_first_estimate_are_distinct_from_native():
    obs = observation('a')
    obs['timing_ns'].update(first_reasoning=500_000_000, first_text=1_000_000_000,
                           first_tool=None, first_arguments=None)
    result = build_comparison([row(150, request_sha256='a')], [], [obs], {})
    assert result['timing']['first_reasoning_seconds']['value']['median'] == .5
    assert result['timing']['first_text_seconds']['value']['median'] == 1.0
    assert result['timing']['post_first_semantic_tps_estimate']['value'] == 100.0
    assert result['timing']['post_first_semantic_tps_estimate']['status'] == 'estimated'
    assert result['timing']['effective_request_tps']['value'] == 75.0
    assert result['native']['decode_tps']['value'] is None


def test_legacy_walls_do_not_become_semantic_or_native_rates():
    result = build_comparison([row(first_byte_s=.2, elapsed_s=2)], [], [], {'outcome': 'failed'})
    assert result['timing']['first_semantic_seconds']['value'] is None
    assert result['timing']['effective_request_tps']['value'] is None
    assert result['legacy']['elapsed_seconds']['value']['median'] == 2
    assert result['native']['decode_tps']['value'] is None
    assert result['quality']['outcome'] == 'failed'


def envelope(event, seconds):
    return {'kind': 'client_event', 'elapsed_seconds': seconds, 'payload': event}


def test_tool_identity_union_and_private_content_never_escape():
    def use(tool_id, uuid, seconds):
        return envelope({'type': 'assistant', 'uuid': uuid, 'message': {'id': 'm', 'content': [
            {'type': 'tool_use', 'id': tool_id, 'name': 'Bash', 'input': {'secret': 'DO_NOT_PUBLISH'}}]}}, seconds)
    def done(tool_id, seconds):
        return envelope({'type': 'user', 'message': {'content': [
            {'type': 'tool_result', 'tool_use_id': tool_id, 'content': 'DO_NOT_PUBLISH'}]}}, seconds)
    events = [use('a', 'u1', 1), use('b', 'u2', 2), use('a', 'u1', 3), done('a', 4), done('b', 5), done('b', 6)]
    result = build_comparison([], events, [], {'journal_complete': True}, [{'elapsed_s': 9}])
    assert result['tools']['latency_seconds']['value']['n'] == 2
    assert result['tools']['interval_sum_seconds']['value'] == 6
    assert result['tools']['interval_union_seconds']['value'] == 4
    assert result['tools']['overlap_seconds']['value'] == 2
    assert result['tests']['elapsed_seconds']['value'] == 9
    assert result['compactions']['status'] == 'not_observed'
    assert 'DO_NOT_PUBLISH' not in json.dumps(result)
    incomplete = build_comparison([], [], [], {})
    assert incomplete['compactions']['value'] is None


def test_native_counters_are_weighted_and_cache_fields_separate():
    body = {'usage': {'input_tokens': 10, 'cache_read_input_tokens': 20,
                     'cache_creation_input_tokens': 30, 'output_tokens': 8,
                     'output_tokens_details': {'reasoning_tokens': 3}},
            'timings': {'predicted_n': 8, 'predicted_ms': 200, 'prompt_n': 10, 'prompt_ms': 100}}
    result = build_comparison([{'response_body': json.dumps(body)}], [], [], {})
    assert result['usage']['input_tokens']['value'] == 60
    assert result['usage']['cache_read_tokens']['value'] == 20
    assert result['usage']['cache_write_tokens']['value'] == 30
    assert result['usage']['reasoning_tokens']['value'] == 3
    assert result['native']['decode_tps']['value'] == 40


def test_cli_validates_inventory_hashes_and_refuses_overwrite(tmp_path):
    import hashlib
    import subprocess
    from recorded_session import digest, inventory
    from session_history import SessionHistory
    directory = tmp_path / 'session'
    history = SessionHistory(directory)
    (directory / 'summary.json').write_text('{"outcome":"failed"}', encoding='utf-8')
    (directory / 'wire-requests.json').write_text(json.dumps([row()]), encoding='utf-8')
    (directory / 'evidence-inventory.json').write_text(json.dumps(inventory(directory,
        exclude=('events.jsonl', 'manifest.json'))), encoding='utf-8')
    history.append('evidence_inventory', {'path': 'evidence-inventory.json',
                   'sha256': digest(directory / 'evidence-inventory.json')})
    history.finish('failed')
    output = tmp_path / 'derived.json'
    cli = Path(__file__).resolve().parents[2] / 'tools/report-session-comparison.py'
    command = [sys.executable, str(cli), '--session', str(directory), '--output', str(output)]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text(encoding='utf-8'))
    assert result['inputs']['wire_rows']['sha256'] == hashlib.sha256((directory / 'wire-requests.json').read_bytes()).hexdigest()
    assert result['quality']['outcome'] == 'failed'
    original = output.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert output.read_bytes() == original
    inside = command[:-1] + [str(directory / 'new.json')]
    assert subprocess.run(inside, capture_output=True).returncode != 0
    (directory / 'summary.json').write_text('{}', encoding='utf-8')
    output.unlink()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert not output.exists()


@pytest.mark.parametrize('patch', [
    {'timing_ns': {'end': True}},
    {'timing_ns': {'headers': 10, 'first_body': 1, 'end': 20}},
    {'timing_ns': {'first_body': None, 'first_text': 3, 'end': 20}},
    {'timing_ns': {'end': 20, 'first_semantic': 30}},
])
def test_impossible_observations_are_excluded(patch):
    obs = observation('a')
    obs.update(patch)
    result = build_comparison([row(request_sha256='a')], [], [obs], {})
    assert result['timing']['effective_request_tps']['value'] is None


def test_missing_terminal_or_transport_admission_cannot_yield_comparable_rate():
    missing = observation('a')
    missing['timing_ns']['terminal'] = None
    assert build_comparison([row(request_sha256='a')], [], [missing], {})['timing']['effective_request_tps']['value'] is None
    unknown = row(request_sha256='b'); del unknown['usable']
    assert build_comparison([unknown], [], [observation('b')], {})['timing']['effective_request_tps']['value'] is None


def test_conflicting_uuids_exclude_original_tool_interval():
    start = {'type': 'assistant', 'uuid': 'x', 'message': {'id': 'm', 'content': [
        {'type': 'tool_use', 'id': 'a', 'name': 'Bash', 'input': {}}]}}
    conflict = {'type': 'assistant', 'uuid': 'x', 'message': {'id': 'm', 'content': [
        {'type': 'tool_use', 'id': 'b', 'name': 'Bash', 'input': {}}]}}
    end = {'type': 'user', 'message': {'content': [{'type': 'tool_result', 'tool_use_id': 'a'}]}}
    result = build_comparison([], [envelope(start, 1), envelope(conflict, 2), envelope(end, 3)], [], {})
    assert result['tools']['latency_seconds']['value'] is None


def test_canonical_fallback_is_explicit_and_repeated_requests_are_ambiguous():
    import hashlib
    request = {'messages': [], 'model': 'private-model'}
    obs = observation('exact_bytes_hash')
    obs['request_content_sha256'] = hashlib.sha256(json.dumps(request, sort_keys=True,
        ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()
    obs['request_content_hash_algorithm'] = 'sha256-canonical-json-v1'
    assert build_comparison([row(request=request)], [], [obs], {})['coverage']['matched_observations'] == 1
    assert build_comparison([row(request=request), row(request=request)], [], [obs], {})['coverage']['ambiguous_matches'] == 2
    del obs['request_content_hash_algorithm']
    assert build_comparison([row(request=request)], [], [obs], {})['coverage']['matched_observations'] == 0


def test_malformed_intermediate_usage_cannot_be_hidden_by_final_update():
    body = 'data: {"usage":{"output_tokens":true}}\n\ndata: {"usage":{"output_tokens":10}}\n\n'
    result = build_comparison([{'response_body': body}], [], [], {})
    assert result['coverage']['invalid_usage'] == 1


def test_delivery_counts_and_gap_metrics_are_not_token_gaps():
    obs = observation('a')
    obs['counts'] = {'text_chars': 5, 'reasoning_chars': 0, 'empty_thinking_blocks': 1,
                     'tool_argument_chars': 2, 'events': 3}
    obs['chunks'] = [{'start_offset': 0, 'end_offset': 10, 'elapsed_ns': 200_000_000},
                     {'start_offset': 10, 'end_offset': 20, 'elapsed_ns': 600_000_000}]
    result = build_comparison([row(request_sha256='a')], [], [obs], {})
    assert result['delivery']['reasoning_chars']['value'] == 0
    assert result['delivery']['empty_thinking_blocks']['value'] == 1
    assert result['timing']['body_gap_seconds']['value']['median'] == .4
    assert result['coverage']['observations'] == 1
    obs['chunks'][1]['start_offset'] = 11
    invalid = build_comparison([row(request_sha256='a')], [], [obs], {})
    assert invalid['timing']['effective_request_tps']['value'] is None


def test_duplicate_wire_identity_is_counted_once_and_conflicts_excluded():
    first = row(10, connection=1, request_index=0)
    result = build_comparison([first, dict(first)], [], [], {})
    assert result['usage']['output_tokens']['value'] == 10
    assert result['coverage']['duplicate_rows'] == 1
    conflicting = build_comparison([first, row(20, connection=1, request_index=0)], [], [], {})
    assert conflicting['usage']['output_tokens']['value'] is None
    assert conflicting['coverage']['conflicting_rows'] == 2


def test_multiline_sse_usage_is_not_lost_by_linewise_json_parsing():
    body = 'data: {"usage":\r\ndata: {"prompt_tokens":20,"completion_tokens":10}}\r\n\r\ndata: [DONE]\r\n\r\n'
    result = build_comparison([{'usable': True, 'response_body': body}], [], [], {})
    assert result['usage']['output_tokens']['value'] == 10
    assert result['usage']['input_tokens']['value'] == 20
    assert result['coverage']['invalid_usage'] == 0


def test_regressing_cumulative_usage_is_rejected():
    body = 'data: {"usage":{"output_tokens":10}}\n\ndata: {"usage":{"output_tokens":5}}\n\n'
    assert build_comparison([{'response_body': body}], [], [], {})['coverage']['invalid_usage'] == 1


def test_explicit_observation_reference_can_use_session_relative_sidecar_alias():
    obs = observation('a')
    obs['_observation_refs'] = ['observations/a/timing.json', 'a/timing.json']
    result = build_comparison([row(observation_ref='observations/a/timing.json')], [], [obs], {})
    assert result['coverage']['matched_observations'] == 1


def test_repeated_usage_is_cumulative_and_malformed_tokens_are_rejected():
    repeated = 'data: {"usage":{"input_tokens":10,"output_tokens":5}}\n\ndata: {"usage":{"output_tokens":8}}\n\n'
    result = build_comparison([{'response_body': repeated}], [], [], {})
    assert result['usage']['output_tokens']['value'] == 8
    assert result['usage']['output_tokens']['coverage_count'] == 1
    for bad in (True, -1, 1.5, '12'):
        result = build_comparison([row(bad)], [], [], {})
        assert result['usage']['output_tokens']['value'] is None
        assert result['coverage']['invalid_usage'] == 1
