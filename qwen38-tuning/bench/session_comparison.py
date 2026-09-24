"""Pure, content-free comparison of recorded evidence; never a quality scorer."""
from __future__ import annotations

import hashlib
import json
import math
from statistics import median

from recorded_session import wire_usage
from session_analysis import analyze_events


def _integer(value):
    return type(value) is int and value >= 0


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _metric(value, unit, source, boundary, count=0, status=None):
    return dict(value=value, unit=unit, status=status or ('observed' if value is not None else 'unavailable'),
                source=source, boundary=boundary, coverage_count=count)


def _usage(row):
    """Merge cumulative provider usage once, validating before arithmetic."""
    body = row.get('response_body') or ''
    try:
        previous = {}
        events = _events(body)
        for event in events:
            if not isinstance(event, dict):
                continue
            message = event.get('message')
            raw_items = [event.get('usage')]
            if isinstance(message, dict):
                raw_items.append(message.get('usage'))
            for raw_item in raw_items:
                if raw_item is None:
                    continue
                if not isinstance(raw_item, dict):
                    raise ValueError('invalid_usage')
                for key, value in raw_item.items():
                    if key.endswith('_tokens'):
                        if not _integer(value) or value < previous.get(key, 0):
                            raise ValueError('invalid_count')
                        previous[key] = value
                    if key.endswith('_tokens_details') and value is not None:
                        if not isinstance(value, dict) or any(not _integer(v) for k, v in value.items()
                                                            if k.endswith('_tokens')):
                            raise ValueError('invalid_details')
                        previous[key] = value
        raw = previous
        if raw.get('prompt_tokens') is not None:
            input_tokens = raw['prompt_tokens']
        elif raw.get('input_tokens') is not None:
            input_tokens = raw['input_tokens'] + (raw.get('cache_read_input_tokens') or 0) + (raw.get('cache_creation_input_tokens') or 0)
        else:
            input_tokens = None
        output_tokens = raw.get('completion_tokens', raw.get('output_tokens'))
        return {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cache_read_tokens': raw.get('cache_read_input_tokens'),
            'cache_write_tokens': raw.get('cache_creation_input_tokens'),
            'reasoning_tokens': raw.get('reasoning_tokens', (raw.get('completion_tokens_details') or
                                raw.get('output_tokens_details') or {}).get('reasoning_tokens')),
        }
    except (ValueError, TypeError, AttributeError, KeyError):
        return None


def _distribution(values, source='stream_observation', boundary='upstream_http_body'):
    ordered = sorted(values)
    value = ({'n': len(ordered), 'median': median(ordered),
              'p95': ordered[math.ceil(.95 * len(ordered)) - 1]} if ordered else None)
    return _metric(value, 'seconds', source, boundary, len(ordered))


def _matches(row, observations):
    ref = row.get('observation_ref')
    exact = row.get('request_sha256')
    if ref:
        matches = [i for i, obs in enumerate(observations)
                   if ref in (obs.get('observation_ref'), obs.get('request_id'))
                   or ref in obs.get('_observation_refs', [])]
        return [i for i in matches if not exact or observations[i].get('request_sha256') == exact]
    if exact:
        return [i for i, obs in enumerate(observations) if obs.get('request_sha256') == exact]
    if isinstance(row.get('request'), dict):
        canonical = json.dumps(row['request'], sort_keys=True, ensure_ascii=False,
                               separators=(',', ':'), allow_nan=False).encode('utf-8')
        digest = hashlib.sha256(canonical).hexdigest()
        return [i for i, obs in enumerate(observations)
                if obs.get('request_content_hash_algorithm') == 'sha256-canonical-json-v1'
                and obs.get('request_content_sha256') == digest]
    return []


def _valid_observation(obs):
    if (obs.get('schema_version') != 1 or obs.get('boundary') != 'upstream_http_body'
            or obs.get('complete') is not True or obs.get('errors')):
        return False
    times = obs.get('timing_ns')
    if (not isinstance(times, dict) or any(not _integer(times.get(key))
            for key in ('terminal', 'eof', 'end')) or times['end'] == 0):
        return False
    if any(value is not None and not _integer(value) for value in times.values()):
        return False
    chain = [times.get(key) for key in ('headers', 'first_body', 'first_semantic', 'terminal', 'eof', 'end')]
    known = [value for value in chain if value is not None]
    if known != sorted(known):
        return False
    for key in ('first_text', 'first_reasoning', 'first_tool', 'first_arguments'):
        value = times.get(key)
        if value is not None and (times.get('first_body') is None
                                  or not times['first_body'] <= value <= times['end']):
            return False
    counts = obs.get('counts', {})
    if not isinstance(counts, dict) or any(not _integer(value) for value in counts.values()):
        return False
    chunks = obs.get('chunks', [])
    if not isinstance(chunks, list):
        return False
    offset, stamp = 0, 0
    for chunk in chunks:
        if not isinstance(chunk, dict) or any(not _integer(chunk.get(key))
                for key in ('start_offset', 'end_offset', 'elapsed_ns')):
            return False
        if (chunk['start_offset'] != offset or chunk['end_offset'] <= offset
                or not stamp <= chunk['elapsed_ns'] <= times['end']):
            return False
        offset, stamp = chunk['end_offset'], chunk['elapsed_ns']
    return True


def _timing(rows, usages, observations):
    candidates = [_matches(row, observations) for row in rows]
    use_counts = {}
    for matches in candidates:
        for index in matches:
            use_counts[index] = use_counts.get(index, 0) + 1
    matched = [(row, usage, observations[indices[0]]) for row, usage, indices in zip(rows, usages, candidates)
               if len(indices) == 1 and use_counts[indices[0]] == 1]
    valid = [(row, usage, obs) for row, usage, obs in matched
             if row.get('usable') is True and obs.get('transport_usable') is not False and _valid_observation(obs)]
    timing = {}
    for key in ('headers', 'first_body', 'first_semantic', 'first_text', 'first_reasoning',
                'first_tool', 'first_arguments', 'terminal', 'eof', 'end'):
        timing[key + '_seconds'] = _distribution([obs['timing_ns'][key] / 1e9
                           for _, _, obs in valid if obs['timing_ns'].get(key) is not None])
    for name, start, end in (('stream_seconds', 'first_body', 'eof'),
                             ('terminal_to_eof_seconds', 'terminal', 'eof')):
        timing[name] = _distribution([(obs['timing_ns'][end] - obs['timing_ns'][start]) / 1e9
            for _, _, obs in valid if obs['timing_ns'].get(start) is not None
            and obs['timing_ns'].get(end) is not None])
    eligible = [(usage['output_tokens'], obs['timing_ns']['end'] / 1e9)
                for _, usage, obs in valid if usage and usage['output_tokens'] is not None]
    numerator = sum(tokens for tokens, _ in eligible)
    denominator = sum(seconds for _, seconds in eligible)
    for name, value, unit in (('effective_request_tps', numerator / denominator if denominator else None, 'tokens/second'),
                              ('effective_output_tokens', numerator if eligible else None, 'tokens'),
                              ('effective_wall_seconds', denominator if eligible else None, 'seconds')):
        timing[name] = _metric(value, unit, 'provider_usage+stream_observation', 'upstream_http_body', len(eligible))
    post_first = [(usage['output_tokens'], (obs['timing_ns']['end'] - obs['timing_ns']['first_semantic']) / 1e9)
                  for _, usage, obs in valid if usage and usage['output_tokens'] is not None
                  and obs['timing_ns'].get('first_semantic') is not None
                  and obs['timing_ns']['end'] > obs['timing_ns']['first_semantic']]
    post_seconds = sum(seconds for _, seconds in post_first)
    timing['post_first_semantic_tps_estimate'] = _metric(
        sum(tokens for tokens, _ in post_first) / post_seconds if post_seconds else None,
        'tokens/second', 'provider_usage+stream_observation', 'first_semantic_to_http_eof',
        len(post_first), 'estimated' if post_seconds else 'unavailable')
    timing['post_first_semantic_tps_estimate']['limitations'] = [
        'Includes first-chunk tokens and possibly unexposed reasoning in the numerator.',
        'DSH-style formula, not native decode or proof of identical UI timing boundaries.']
    gaps = []
    for _, _, obs in valid:
        chunks = obs.get('chunks', [])
        gaps.extend((later['elapsed_ns'] - earlier['elapsed_ns']) / 1e9
                    for earlier, later in zip(chunks, chunks[1:]))
    timing['body_gap_seconds'] = _distribution(gaps)
    delivery = {}
    for key in ('text_chars', 'reasoning_chars', 'tool_argument_chars', 'empty_thinking_blocks', 'events'):
        values = [obs['counts'][key] for _, _, obs in valid if key in obs.get('counts', {})]
        delivery[key] = _metric(sum(values) if values else None,
                               'characters' if key.endswith('_chars') else 'count',
                               'stream_observation', 'upstream_http_body', len(values))
    return timing, delivery, {'observations': len(observations),
                   'failed_rows': sum(row.get('usable') is False for row in rows),
                   'matched_observations': len(matched), 'valid_complete_observations': len(valid),
                   'unmatched_rows': sum(not indices for indices in candidates),
                   'ambiguous_matches': sum(bool(indices) and (len(indices) != 1 or use_counts[indices[0]] != 1)
                                            for indices in candidates),
                   'incomplete_observations': sum(obs.get('complete') is not True for obs in observations),
                   'invalid_matched_observations': len(matched) - len(valid)}


def _events(body):
    if body.lstrip().startswith('{'):
        return [json.loads(body)]
    events, data_lines = [], []
    def flush():
        if not data_lines:
            return
        payload = '\n'.join(data_lines)
        data_lines.clear()
        if payload and payload != '[DONE]':
            events.append(json.loads(payload))
    for raw in body.splitlines():
        line = raw[:-1] if raw.endswith('\r') else raw
        if line.startswith('data:'):
            value = line[5:]
            data_lines.append(value[1:] if value.startswith(' ') else value)
        elif not line:
            flush()
    flush()
    return events


def _native(rows):
    result = {name: _metric(None, unit, 'not_supplied', 'server_only') for name, unit in
              (('gpu_memory', 'bytes'), ('queue_seconds', 'seconds'))}
    for label, prefix in (('decode', 'predicted'), ('prefill', 'prompt')):
        pairs = []
        for row in rows:
            try:
                timings = [event['timings'] for event in _events(row.get('response_body') or '')
                           if isinstance(event, dict) and isinstance(event.get('timings'), dict)]
                if not timings:
                    continue
                final = timings[-1]
                n, ms = final.get(prefix + '_n'), final.get(prefix + '_ms')
                if _integer(n) and _number(ms) and ms > 0:
                    pairs.append((n, ms / 1000))
            except (ValueError, TypeError, AttributeError):
                continue
        result[label + '_tps'] = _metric(sum(n for n, _ in pairs) / sum(s for _, s in pairs) if pairs else None,
                                         'tokens/second', 'response_body.timings', 'server_reported', len(pairs))
        result[label + '_seconds'] = _metric(sum(s for _, s in pairs) if pairs else None, 'seconds',
                                             'response_body.timings', 'server_reported', len(pairs))
    return result


def _tool_timeline(envelopes, complete):
    events = [entry.get('payload') for entry in envelopes if entry.get('kind') == 'client_event']
    analysis = analyze_events(events)
    seen = {}
    starts, ends, conflicts = {}, {}, set()
    duplicate_events = 0
    for entry in envelopes:
        if entry.get('kind') != 'client_event' or not isinstance(entry.get('payload'), dict):
            continue
        event, stamp = entry['payload'], entry.get('elapsed_seconds')
        kind = event.get('type')
        message = event.get('message')
        if kind not in ('assistant', 'user') or not isinstance(message, dict):
            continue
        identity = event.get('uuid') or (message.get('id') if kind == 'assistant' else None)
        fingerprint = json.dumps(message, sort_keys=True, ensure_ascii=False)
        blocks = message.get('content')
        if not isinstance(blocks, list):
            continue
        if identity:
            key = (kind, identity)
            if key in seen:
                duplicate_events += 1
                if seen[key] != fingerprint:
                    original_blocks = json.loads(seen[key]).get('content', [])
                    for block in blocks + original_blocks:
                        if isinstance(block, dict):
                            conflicts.add(block.get('id') or block.get('tool_use_id'))
                continue
            seen[key] = fingerprint
        for block in blocks:
            if not isinstance(block, dict):
                continue
            target = starts if kind == 'assistant' and block.get('type') == 'tool_use' else (
                     ends if kind == 'user' and block.get('type') == 'tool_result' else None)
            if target is None:
                continue
            tool_id = block.get('id') if target is starts else block.get('tool_use_id')
            if not isinstance(tool_id, str) or not tool_id:
                continue
            value = json.dumps(block, sort_keys=True, ensure_ascii=False)
            if tool_id in target:
                if target[tool_id][1] != value:
                    conflicts.add(tool_id)
                continue
            target[tool_id] = (stamp, value, block.get('is_error') is True)
    intervals = []
    for tool_id in starts.keys() & ends.keys() - conflicts:
        start, end = starts[tool_id][0], ends[tool_id][0]
        if _number(start) and _number(end) and end >= start:
            intervals.append((start, end))
        else:
            conflicts.add(tool_id)
    union = 0
    right = None
    for start, end in sorted(intervals):
        union += end - start if right is None or start > right else max(0, end - right)
        right = max(right or 0, end)
    total = sum(end - start for start, end in intervals)
    metric = lambda value, unit, n: _metric(value, unit, 'session_history', 'observed_client_tool_use_to_result', n)
    tools = {'latency_seconds': _distribution([end - start for start, end in intervals],
                                               'session_history', 'observed_client_tool_use_to_result'),
             'interval_sum_seconds': metric(total if intervals else None, 'seconds', len(intervals)),
             'interval_union_seconds': metric(union if intervals else None, 'seconds', len(intervals)),
             'overlap_seconds': metric(total - union if intervals else None, 'seconds', len(intervals)),
             'calls': metric(len(starts), 'calls', len(starts)),
             'errors': metric(sum(value[2] for key, value in ends.items() if key not in conflicts), 'calls', len(ends)),
             'conflicting_ids': len(conflicts), 'duplicate_events': duplicate_events,
             'unresolved_calls': len(starts.keys() - ends.keys()), 'analysis_error_count': len(analysis['errors'])}
    compactions = len(analysis['compactions'])
    compact = _metric(compactions if compactions or complete else None, 'markers', 'session_history',
                       'observed_client_journal', len(events),
                       'observed' if compactions else 'not_observed' if complete else 'unavailable')
    thinking = _metric(analysis['thinking_chars'] if events else None, 'characters', 'session_analysis',
                        'exposed_client_content', len(events))
    return tools, compact, thinking


def build_comparison(wire_rows, journal_envelopes, observations, session_summary, test_journal=None):
    """Return public aggregates only. Input objects are never mutated."""
    original_count = len(wire_rows)
    groups = {}
    for index, row in enumerate(wire_rows):
        if row.get('request_id'):
            identity = ('request', row['request_id'])
        elif row.get('connection') is not None and row.get('request_index') is not None:
            identity = ('wire', row['connection'], row['request_index'])
        else:
            identity = ('unidentified', index)
        groups.setdefault(identity, []).append(row)
    wire_rows, duplicate_rows, conflicting_rows = [], 0, 0
    for group in groups.values():
        if all(item == group[0] for item in group):
            wire_rows.append(group[0])
            duplicate_rows += len(group) - 1
        else:
            conflicting_rows += len(group)
    usages = [_usage(row) for row in wire_rows]
    usage = {}
    for key in ('input_tokens', 'output_tokens', 'cache_read_tokens', 'cache_write_tokens', 'reasoning_tokens'):
        values = [item[key] for item in usages if item is not None and item[key] is not None]
        usage[key] = _metric(sum(values) if values else None, 'tokens', 'provider_usage',
                             'reported_request_usage', len(values))
    timing, delivery, coverage = _timing(wire_rows, usages, observations)
    coverage.update(wire_rows=original_count, unique_rows=len(wire_rows), invalid_usage=usages.count(None),
                    duplicate_rows=duplicate_rows, conflicting_rows=conflicting_rows)
    legacy = {name: _distribution([row[key] for row in wire_rows if _number(row.get(key))],
                                 'wire_rows', 'legacy_recorder_origin')
              for name, key in (('elapsed_seconds', 'elapsed_s'), ('first_byte_seconds', 'first_byte_s'))}
    tools, compactions, thinking = _tool_timeline(journal_envelopes, session_summary.get('journal_complete') is True)
    test_times = [item['elapsed_s'] for item in (test_journal or []) if _number(item.get('elapsed_s'))]
    costs = {key: _metric(session_summary[key] if _number(session_summary.get(key)) else None,
                         'seconds', 'session_summary', 'recorded_' + key, int(_number(session_summary.get(key))))
             for key in ('client_s', 'verification_s', 'elapsed_s', 'setup_s', 'cleanup_s')}
    inputs = [item['input_tokens'] for item in usages if item and item['input_tokens'] is not None]
    context = {'max_input_tokens': _metric(max(inputs) if inputs else None, 'tokens', 'provider_usage',
                                           'reported_request_usage', len(inputs))}
    for key in ('requested_window_tokens', 'runtime_window_tokens'):
        value = (session_summary.get('context') or {}).get(key)
        context[key] = _metric(value if _integer(value) else None, 'tokens', 'session_summary.context',
                               'declared_allocation', int(_integer(value)))
    return {'schema_version': 1, 'usage': usage, 'timing': timing, 'legacy': legacy, 'delivery': delivery,
            'tools': tools, 'compactions': compactions, 'exposed_thinking': thinking, 'costs': costs,
            'context': context,
            'tests': {'elapsed_seconds': _metric(sum(test_times) if test_times else None, 'seconds',
                      'fixed_test_journal', 'independent_fixed_test_execution', len(test_times))},
            'coverage': coverage,
            'quality': {'outcome': session_summary.get('outcome'), 'source': 'session_summary',
                        'external_audit': 'caller_owned_not_scored'},
            'native': _native(wire_rows),
            'limitations': ['Distributions are descriptive for the valid matched subset; p95 uses nearest rank.',
                           'HTTP body reads are delivery times, not token generation or server emission.',
                           'Reported usage may include unseen tokens; hidden thinking is never inferred.']}
