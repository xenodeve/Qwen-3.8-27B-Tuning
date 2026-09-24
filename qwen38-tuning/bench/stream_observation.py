"""Pure observations of upstream HTTP body delivery, never token generation."""
import json
import math
import re
import time


FINISH_REASONS = frozenset(('stop', 'length', 'tool_calls', 'function_call', 'content_filter',
                            'end_turn', 'max_tokens', 'stop_sequence', 'tool_use',
                            'pause_turn', 'refusal', 'model_context_window_exceeded'))


def _clock_metadata(info):
    info = time.get_clock_info('perf_counter') if info is None else info
    get = info.get if isinstance(info, dict) else lambda key, default=None: getattr(info, key, default)
    resolution = get('resolution')
    valid = type(resolution) in (float, int) and math.isfinite(resolution) and resolution > 0
    implementation = get('implementation')
    known = ('QueryPerformanceCounter()', 'clock_gettime(CLOCK_MONOTONIC)', 'mach_absolute_time()')
    return {'name': 'perf_counter', 'implementation': implementation if implementation in known else 'unknown',
            'resolution_ns': max(1, round(resolution * 1_000_000_000)) if valid else None,
            'monotonic': get('monotonic') is True, 'adjustable': get('adjustable') is True}


class StreamObservation:
    """Caller timestamps are absolute perf_counter_ns; snapshots are relative."""

    def __init__(self, protocol, request_id, request_sha256, started_ns, clock_info=None):
        if protocol not in ('openai', 'anthropic'):
            raise ValueError('unsupported_protocol')
        if type(started_ns) is not int or started_ns < 0:
            raise ValueError('invalid_timestamp')
        if not isinstance(request_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', request_id):
            raise ValueError('invalid_request_id')
        if not isinstance(request_sha256, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', request_sha256):
            raise ValueError('invalid_request_sha256')
        self.clock = _clock_metadata(clock_info)
        self.last_ns = started_ns
        self.protocol = protocol
        self.request_id = request_id
        self.request_sha256 = request_sha256
        self.started_ns = started_ns
        self.timing = dict.fromkeys(('headers', 'first_body', 'first_text', 'first_reasoning', 'first_tool', 'first_arguments', 'first_semantic', 'terminal', 'eof', 'end'))
        self.counts = dict.fromkeys(('text_chars', 'reasoning_chars', 'tool_argument_chars', 'empty_thinking_blocks', 'events'), 0)
        self.chunks = []
        self.errors = []
        self.finish_reason = None
        self.buffer = bytearray()
        self.offset = 0
        self.lines = []
        self.skip_lf = False
        self.blocks = {}
        self.started_message = False

    def _error(self, code):
        if code not in self.errors:
            self.errors.append(code)

    def _timestamp(self, at_ns):
        if type(at_ns) is not int or at_ns < self.last_ns:
            self._error('invalid_timestamp')
            return None
        if self.timing['end'] is not None:
            self._error('invalid_lifecycle')
            return None
        self.last_ns = at_ns
        return at_ns - self.started_ns

    def headers(self, at_ns):
        elapsed = self._timestamp(at_ns)
        if elapsed is None:
            return
        if self.timing['headers'] is not None or self.timing['first_body'] is not None or self.timing['eof'] is not None:
            self._error('invalid_lifecycle')
            return
        self.timing['headers'] = elapsed

    def feed(self, data, at_ns):
        elapsed = self._timestamp(at_ns)
        if elapsed is None:
            return
        if self.timing['eof'] is not None:
            self._error('invalid_lifecycle')
            return
        if not isinstance(data, bytes):
            self._error('invalid_body')
            return
        if not data:
            return
        if self.timing['first_body'] is None:
            self.timing['first_body'] = elapsed
        self.chunks.append(dict(start_offset=self.offset, end_offset=self.offset + len(data), elapsed_ns=elapsed))
        self.offset += len(data)
        for byte in data:
            if self.skip_lf and byte == 10:
                self.skip_lf = False
                continue
            self.skip_lf = byte == 13
            if byte in (10, 13):
                line, self.buffer = bytes(self.buffer), bytearray()
                if not line:
                    if self.lines:
                        payload = b'\n'.join(self.lines)
                        self.lines = []
                        self._event(payload, elapsed)
                elif line.startswith(b'data:'):
                    value = line[5:]
                    self.lines.append(value[1:] if value.startswith(b' ') else value)
            else:
                self.buffer.append(byte)

    def _semantic(self, category, value, elapsed):
        if value is not None and not isinstance(value, str):
            self._error('unsupported_payload')
            return
        if value:
            key, count = {'text': ('first_text', 'text_chars'), 'reasoning': ('first_reasoning', 'reasoning_chars'), 'arguments': ('first_arguments', 'tool_argument_chars')}[category]
            self.counts[count] += len(value)
            self._mark(key, elapsed)
            self._mark('first_semantic', elapsed)

    def _mark(self, key, elapsed):
        if self.timing[key] is None:
            self.timing[key] = elapsed

    def _event(self, payload, elapsed):
        self.counts['events'] += 1
        if payload == b'[DONE]' and self.protocol == 'openai':
            self._terminal(elapsed)
            return
        try:
            obj = json.loads(payload.decode('utf-8'))
        except UnicodeDecodeError:
            self._error('invalid_utf8')
            return
        except (ValueError, RecursionError):
            self._error('invalid_json')
            return
        if not isinstance(obj, dict):
            self._error('unsupported_event')
            return
        if 'error' in obj or obj.get('type') == 'error':
            self._error('provider_error')
            return
        if self.timing['terminal'] is not None:
            self._error('event_after_terminal')
            return
        try:
            if self.protocol == 'anthropic':
                self._anthropic(obj, elapsed)
            else:
                self._openai(obj, elapsed)
        except (KeyError, TypeError, AttributeError, ValueError, RecursionError):
            self._error('unsupported_payload')

    def _terminal(self, elapsed):
        if self.timing['terminal'] is not None:
            self._error('duplicate_terminal')
        self._mark('terminal', elapsed)

    def _reason(self, value):
        reason = value if isinstance(value, str) and value in FINISH_REASONS else 'unknown'
        if self.finish_reason is not None and self.finish_reason != reason:
            self._error('conflicting_finish_reason')
        self.finish_reason = reason

    def _openai(self, obj, elapsed):
        choices = obj.get('choices')
        if not isinstance(choices, list):
            self._error('unsupported_event')
            return
        if len(choices) > 1:
            self._error('ambiguous_choices')
            return
        for choice in choices:
            index = choice.get('index', 0)
            if type(index) is not int or index != 0:
                self._error('ambiguous_choices')
                return
            delta = choice.get('delta', {})
            if 'function_call' in delta:
                self._error('unsupported_payload')
            self._semantic('text', delta.get('content'), elapsed)
            for key in ('reasoning_content', 'reasoning', 'reasoning_text'):
                if delta.get(key):
                    self._semantic('reasoning', delta[key], elapsed)
                    break
            for tool in delta.get('tool_calls', []):
                self._tool(elapsed)
                self._semantic('arguments', tool.get('function', {}).get('arguments'), elapsed)
            if choice.get('finish_reason') is not None:
                self._reason(choice['finish_reason'])

    def _tool(self, elapsed):
        self._mark('first_tool', elapsed)
        self._mark('first_semantic', elapsed)

    def _anthropic(self, obj, elapsed):
        kind = obj.get('type')
        if kind == 'message_start':
            # Repeated startup messages are metadata, not replayed output.
            if not self.started_message:
                self.started_message = True
                for index, block in enumerate(obj.get('message', {}).get('content', [])):
                    self._block_start(index, block, elapsed)
        elif kind == 'content_block_start':
            self._block_start(obj['index'], obj['content_block'], elapsed)
        elif kind == 'content_block_delta':
            index, delta = obj['index'], obj['delta']
            kind = delta.get('type')
            expected = {'text_delta': 'text', 'thinking_delta': 'thinking', 'signature_delta': 'thinking', 'input_json_delta': 'tool_use'}.get(kind)
            if index not in self.blocks or expected != self.blocks[index][0]:
                self._error('invalid_block')
                return
            if kind == 'text_delta':
                self._semantic('text', delta.get('text'), elapsed)
            elif kind == 'thinking_delta':
                value = delta.get('thinking')
                self._semantic('reasoning', value, elapsed)
                if value:
                    self.blocks[index] = ('thinking', True)
            elif kind == 'input_json_delta':
                self._semantic('arguments', delta.get('partial_json'), elapsed)
        elif kind == 'content_block_stop':
            block = self.blocks.pop(obj['index'], None)
            if block is None:
                self._error('invalid_block')
            if block == ('thinking', False):
                self.counts['empty_thinking_blocks'] += 1
        elif kind == 'message_delta':
            if obj.get('delta', {}).get('stop_reason') is not None:
                self._reason(obj['delta']['stop_reason'])
        elif kind == 'message_stop':
            if self.blocks:
                self._error('unclosed_blocks')
            self._terminal(elapsed)
        elif kind != 'ping':
            self._error('unsupported_event')

    def _block_start(self, index, block, elapsed):
        kind = block.get('type')
        if type(index) is not int or index < 0 or index in self.blocks:
            self._error('invalid_block')
            return
        if kind not in ('text', 'thinking', 'tool_use', 'redacted_thinking'):
            self._error('unsupported_payload')
            return
        self.blocks[index] = (kind, bool(block.get('thinking')))
        if kind == 'thinking':
            self._semantic('reasoning', block.get('thinking'), elapsed)
        elif kind == 'text':
            self._semantic('text', block.get('text'), elapsed)
        elif kind == 'tool_use':
            self._tool(elapsed)
            if block.get('input'):
                self._semantic('arguments', json.dumps(block['input'], ensure_ascii=False, separators=(',', ':')), elapsed)

    def eof(self, at_ns):
        elapsed = self._timestamp(at_ns)
        if elapsed is None:
            return
        if self.timing['eof'] is not None:
            self._error('invalid_lifecycle')
            return
        self.timing['eof'] = elapsed
        if self.buffer or self.lines:
            self._error('truncated_event')
        self.buffer, self.lines = bytearray(), []
        if self.timing['terminal'] is None:
            self._error('missing_terminal')

    def finish(self, at_ns, error=None):
        elapsed = self._timestamp(at_ns)
        if elapsed is None:
            return
        self.timing['end'] = elapsed
        if error is not None:
            self._error('request_error')
        if self.timing['eof'] is None:
            self._error('missing_eof')
        if self.timing['terminal'] is None:
            self._error('missing_terminal')
        self.buffer, self.lines = bytearray(), []

    def snapshot(self):
        return {'schema_version': 1, 'protocol': self.protocol, 'request_id': self.request_id,
                'request_sha256': self.request_sha256, 'boundary': 'upstream_http_body',
                'clock': self.clock.copy(), 'timing_ns': self.timing.copy(), 'chunks': [c.copy() for c in self.chunks],
                'counts': self.counts.copy(), 'errors': self.errors.copy(), 'finish_reason': self.finish_reason,
                'complete': all(self.timing[k] is not None for k in ('terminal', 'eof', 'end')) and not self.errors}
