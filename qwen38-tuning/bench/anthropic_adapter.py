"""Loopback-only Anthropic Messages adapter for a llama.cpp OpenAI endpoint."""
from __future__ import annotations

import asyncio
import json
import hmac
import hashlib
import time
import uuid
import sys
import threading
from pathlib import Path
from urllib.parse import urlsplit

from aiohttp import ClientSession, ClientTimeout, web

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "qwen38-tuning" / "serving" / "exl3"))
import anthropic_compat as _compat


class LlamaStreamTranslator(_compat.StreamTranslator):
    """Coalesce llama tool fragments and retain usage arriving after finish_reason.

    EXL3 supplies complete calls; llama supplies index-keyed argument deltas.
    Thinking/text still stream immediately. Tool calls close only at upstream EOF.
    """
    def __init__(self, model):
        super().__init__(model)
        self.calls = {}
        self.final_reason = None
        self.final_usage = {}
        self.final_timings = {}
        self.failed = False

    def feed(self, chunk):
        if chunk.get('error'):
            self.failed = True
            return super().feed(chunk)
        self.final_usage.update(chunk.get('usage') or {})
        self.final_timings.update(chunk.get('timings') or {})
        choice = (chunk.get('choices') or [{}])[0]
        delta = dict(choice.get('delta') or {})
        for call in delta.pop('tool_calls', []) or []:
            index = call.get('index', 0)
            target = self.calls.setdefault(index, {'index':index, 'type':'function',
                'function':{'name':'', 'arguments':''}})
            if call.get('id'):
                target['id'] = call['id']
            function = call.get('function') or {}
            for key in ('name', 'arguments'):
                target['function'][key] += function.get(key) or ''
        if choice.get('finish_reason') is not None:
            self.final_reason = choice['finish_reason']
        return super().feed(dict(chunk, choices=[dict(choice, delta=delta, finish_reason=None)]))

    def finish(self):
        if self.failed:
            return []
        if self.final_reason is None:
            raise ValueError('upstream ended without finish_reason')
        delta = {'tool_calls':[self.calls[key] for key in sorted(self.calls)]} if self.calls else {}
        return super().feed({'choices':[{'delta':delta, 'finish_reason':self.final_reason}],
                             'usage':self.final_usage, 'timings':self.final_timings})


async def observed_lines(chunks):
    """Split one observed body iterator into SSE lines, retaining the remainder."""
    pending = b''
    async for chunk in chunks:
        pending += chunk
        while b'\n' in pending:
            line, pending = pending.split(b'\n', 1)
            yield line + b'\n'
    if pending:
        yield pending


class AdapterServer:
    def __init__(self, upstream_url: str, listen_host: str = "127.0.0.1", listen_port: int = 0,
                 request_transform=None, *, observation_directory=None,
                 client_api_key=None, upstream_api_key=None) -> None:
        parsed = urlsplit(upstream_url)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.port is None or parsed.path not in ("", "/")
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("upstream must be explicit loopback HTTP")
        if listen_host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("adapter must bind loopback")
        for key in (client_api_key, upstream_api_key):
            if key is not None and (not isinstance(key, str) or not key or not key.isascii()
                                    or any(char.isspace() or ord(char) < 33 or ord(char) > 126 for char in key)):
                raise ValueError('nonce must be a nonempty printable ASCII string')
        self.upstream_url = upstream_url.rstrip("/")
        self.request_transform = request_transform
        self._client_api_key = client_api_key
        self._upstream_headers = {'X-Api-Key': upstream_api_key} if upstream_api_key is not None else {}
        self.observation_directory = Path(observation_directory) if observation_directory is not None else None
        self.observations = []
        self.observation_errors = []
        self.listen_host, self.port = listen_host, listen_port
        self._thread = None
        self._loop = None
        self._runner = None
        self._ready = threading.Event()
        self._error = None

    def start(self) -> "AdapterServer":
        if self._thread is not None:
            raise RuntimeError("adapter already started")
        self._thread = threading.Thread(target=self._serve, name="anthropic-adapter", daemon=True)
        self._thread.start()
        if not self._ready.wait(5):
            raise RuntimeError("adapter startup timeout")
        if self._error:
            raise RuntimeError("adapter startup failed") from self._error
        return self

    def stop(self) -> None:
        if self._loop is None:
            return
        future = asyncio.run_coroutine_threadsafe(self._runner.cleanup(), self._loop)
        future.result(timeout=5)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError("adapter thread did not stop")
        self._loop = None

    def _serve(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._start_app())
            self._ready.set()
            loop.run_forever()
        except BaseException as error:
            self._error = error
            self._ready.set()
        finally:
            loop.close()

    async def _start_app(self) -> None:
        @web.middleware
        async def authenticate(request, handler):
            if self._client_api_key is not None:
                keys = request.headers.getall('X-Api-Key', [])
                supplied = keys[0] if len(keys) == 1 else ''
                if not hmac.compare_digest(supplied.encode(), self._client_api_key.encode()):
                    return web.json_response({'error': 'unauthorized'}, status=401)
            return await handler(request)

        app = web.Application(middlewares=[authenticate])
        app.router.add_get("/health", self._proxy_json)
        app.router.add_get("/props", self._proxy_json)
        app.router.add_post("/v1/messages", self._messages)
        app.router.add_post("/v1/messages/count_tokens", self._count_tokens)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.listen_host, self.port)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]

    async def _proxy_json(self, request: web.Request) -> web.Response:
        async with ClientSession(trust_env=False, headers=self._upstream_headers, timeout=ClientTimeout(total=10)) as session:
            async with session.get(self.upstream_url + request.path, allow_redirects=False) as response:
                payload = await response.read()
                return web.Response(status=response.status, body=payload,
                                    content_type="application/json")

    async def _count_tokens(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
            text = json.dumps(body.get("messages") or [], ensure_ascii=False)
            return web.json_response({"input_tokens": max(1, len(text) // 4)})
        except (ValueError, TypeError):
            return web.json_response({"type": "error", "error": {"type": "invalid_request_error", "message": "invalid JSON"}}, status=400)

    def _persist_observation(self, observation, request_id, request_sha256, content_sha256):
        try:
            record = observation.snapshot()
            record.update(request_id=request_id, request_sha256=request_sha256,
                          request_content_sha256=content_sha256,
                          request_content_hash_algorithm='sha256-canonical-json-v1')
            self.observation_directory.mkdir(parents=True, exist_ok=True)
            directory = self.observation_directory / request_id
            directory.mkdir(exist_ok=False)
            with (directory / 'timing.json').open('x', encoding='utf-8') as handle:
                json.dump(record, handle, allow_nan=False)
            self.observations.append(record)
        except Exception:
            self.observation_errors.append({'request_id': request_id,
                                            'error': 'observation_persistence_failed'})
            raise RuntimeError('observation_persistence_failed') from None

    async def _messages(self, request: web.Request) -> web.StreamResponse | web.Response:
        try:
            body = await request.json()
            openai_body = _compat.anthropic_to_openai(body)
            if self.request_transform is not None:
                openai_body = self.request_transform(openai_body)
            request_bytes = json.dumps(openai_body, allow_nan=False).encode('utf-8')
        except (ValueError, TypeError) as error:
            return web.json_response({"type": "error", "error": {"type": "invalid_request_error", "message": str(error)}}, status=400)
        observation = None
        request_id = uuid.uuid4().hex
        request_sha256 = hashlib.sha256(request_bytes).hexdigest()
        content_sha256 = None
        if self.observation_directory is not None:
            from stream_observation import StreamObservation
            canonical = json.dumps(openai_body, sort_keys=True, ensure_ascii=False,
                                   separators=(',', ':'), allow_nan=False).encode('utf-8')
            content_sha256 = hashlib.sha256(canonical).hexdigest()
        response = None
        eof_ns = None
        failure = None
        async with ClientSession(trust_env=False, headers=self._upstream_headers,
                                 timeout=ClientTimeout(total=None)) as session:
            try:
                started_ns = time.perf_counter_ns()
                if self.observation_directory is not None:
                    observation = StreamObservation('openai', request_id, request_sha256, started_ns)
                async with session.post(self.upstream_url + "/v1/chat/completions",
                                        data=request_bytes, headers={'Content-Type': 'application/json'},
                                        allow_redirects=False) as upstream:
                    headers_ns = time.perf_counter_ns()
                    if observation is not None:
                        observation.headers(headers_ns)

                    async def chunks():
                        nonlocal eof_ns
                        async for chunk in upstream.content.iter_any():
                            read_ns = time.perf_counter_ns()
                            if observation is not None:
                                observation.feed(chunk, read_ns)
                            yield chunk
                        eof_ns = time.perf_counter_ns()
                        if observation is not None:
                            observation.eof(eof_ns)

                    body_chunks = chunks()
                    if upstream.status != 200 or not openai_body.get('stream'):
                        payload_bytes = b''.join([chunk async for chunk in body_chunks])
                        if 300 <= upstream.status < 400:
                            failure = 'upstream_redirect'
                            response = web.json_response({'error': failure}, status=502)
                        else:
                            payload = json.loads(payload_bytes)
                            if upstream.status != 200:
                                failure = 'upstream_http_error'
                                if openai_body.get('stream'):
                                    response = web.StreamResponse(headers={'Content-Type': 'text/event-stream',
                                                                           'Cache-Control': 'no-cache'})
                                    await response.prepare(request)
                                    translator = LlamaStreamTranslator(model=openai_body.get('model'))
                                    for event, value in translator.start() + translator.feed({'error': payload.get('error', payload)}):
                                        await response.write(_compat.sse(event, value))
                                else:
                                    response = web.json_response(payload, status=upstream.status)
                            else:
                                response = web.json_response(_compat.openai_to_anthropic(payload))
                    else:
                        response = web.StreamResponse(headers={"Content-Type": "text/event-stream",
                                                               "Cache-Control": "no-cache"})
                        await response.prepare(request)
                        translator = LlamaStreamTranslator(model=openai_body.get('model'))
                        for event, data in translator.start():
                            await response.write(_compat.sse(event, data))
                        lines = observed_lines(body_chunks)
                        async for event, data in _compat.pump(lines, translator, ping_s=5):
                            await response.write(_compat.sse(event, data))
                        # Resume the SAME iterator, including bytes buffered after DONE.
                        async for _ in lines:
                            pass
                        for event, data in translator.finish():
                            await response.write(_compat.sse(event, data))
            except BaseException:
                failure = 'adapter_request_failed'
                raise
            finally:
                if observation is not None:
                    observation.finish(eof_ns if failure is None and eof_ns is not None
                                       else time.perf_counter_ns(), error=failure)
                    self._persist_observation(observation, request_id, request_sha256, content_sha256)
        if isinstance(response, web.StreamResponse) and not isinstance(response, web.Response):
            await response.write_eof()
        return response
