import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import anyio
import httpx
import pytest

import gateway
import call_policy
from gateway import create_app


CHAT_PATH = "/v1/chat/completions"
pytestmark = pytest.mark.anyio
MESSAGES = [
    {"role": "system", "content": "Reply in Chinese."},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好，有什么可以帮你？"},
    {"role": "user", "content": "请介绍自己"},
]


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def upstream_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-upstream-key")


@pytest.fixture(autouse=True)
def isolated_policy(monkeypatch, tmp_path):
    monkeypatch.setenv("GATEWAY_USAGE_DB", str(tmp_path / "usage.sqlite3"))
    monkeypatch.setenv("GATEWAY_PRO_RPM", "60")
    monkeypatch.setenv("GATEWAY_FLASH_RPM", "60")

    async def skip_wait(_):
        pass

    monkeypatch.setattr(gateway, "retry_backoff", skip_wait)


@asynccontextmanager
async def gateway_client(upstream, *, on_body=None):
    app = create_app(transport=httpx.MockTransport(upstream))

    async def observed_app(scope, receive, send):
        async def observe_send(message):
            await send(message)
            if on_body is not None and message["type"] == "http.response.body":
                on_body(message.get("body", b""))

        await app(scope, receive, observe_send)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=observed_app),
            base_url="http://gateway.test",
        ) as client:
            yield client


async def test_same_endpoint_routes_two_models(upstream_key):
    calls = []

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        assert request.method == "POST"
        assert request.url.host == "api.deepseek.com"
        body = json.loads(request.content)
        assert body["stream"] is False
        if request.url.path == "/responses":
            assert request.headers["Authorization"] == "Bearer test-upstream-key"
            assert "x-api-key" not in request.headers
            assert body["model"] == "deepseek-v4-pro"
            assert body["input"] == MESSAGES
            assert "messages" not in body
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "output": [
                        {"type": "reasoning", "content": []},
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": "你好，"},
                                {"type": "output_text", "text": "我是 Pro。"},
                            ],
                        },
                    ],
                },
            )

        assert request.url.path == "/anthropic/v1/messages"
        assert request.headers["x-api-key"] == "test-upstream-key"
        assert request.headers["anthropic-version"] == "2023-06-01"
        assert "Authorization" not in request.headers
        assert body["model"] == "deepseek-flash"
        assert body["system"] == MESSAGES[0]["content"]
        assert body["messages"] == MESSAGES[1:]
        assert "input" not in body
        return httpx.Response(
            200,
            json={
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "你好，我是 Flash。"}],
            },
        )

    async with gateway_client(upstream) as client:
        for model, text in [("pro", "你好，我是 Pro。"), ("flash", "你好，我是 Flash。")]:
            response = await client.post(
                CHAT_PATH,
                headers={"Authorization": "Bearer caller-key"},
                json={"model": model, "messages": MESSAGES},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["object"] == "chat.completion"
            assert data["model"] == model
            assert data["choices"][0]["message"] == {"role": "assistant", "content": text}

    assert calls == ["/responses", "/anthropic/v1/messages"]


@pytest.mark.parametrize(
    ("model", "extra", "status", "error_code"),
    [
        ("unknown", {}, 404, "model_not_found"),
        ("pro", {"messages": []}, 422, "invalid_request"),
        ("flash", {"stream": "invalid"}, 422, "invalid_request"),
    ],
)
async def test_request_errors_do_not_call_upstream(upstream_key, model, extra, status, error_code):
    calls = []

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(500)

    async with gateway_client(upstream) as client:
        response = await client.post(CHAT_PATH, json={"model": model, "messages": MESSAGES, **extra})

    assert response.status_code == status
    assert response.json()["error"]["code"] == error_code
    assert calls == []


@pytest.mark.parametrize(
    ("status", "body", "error_code"),
    [
        (503, {"error": {"message": "internal provider detail"}}, "upstream_error"),
        (200, {"status": "completed", "output": []}, "invalid_upstream_response"),
        (200, {"status": "incomplete", "output": []}, "upstream_incomplete"),
    ],
)
async def test_upstream_failures_use_gateway_error(upstream_key, status, body, error_code):
    def upstream(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)

    async with gateway_client(upstream) as client:
        response = await client.post(CHAT_PATH, json={"model": "pro", "messages": MESSAGES})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == error_code
    assert "internal provider detail" not in response.text


async def test_missing_upstream_key_does_not_call_upstream(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    def upstream(_: httpx.Request) -> httpx.Response:
        pytest.fail("A missing API key must prevent the upstream request")

    async with gateway_client(upstream) as client:
        response = await client.post(CHAT_PATH, json={"model": "flash", "messages": MESSAGES})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "upstream_not_configured"


class MockSSE(httpx.AsyncByteStream):
    def __init__(self, frames):
        self.frames = frames
        self.closed = False

    async def __aiter__(self):
        async for frame in self.frames:
            yield frame

    async def aclose(self):
        self.closed = True


def upstream_frame(event_type, **fields):
    data = json.dumps({"type": event_type, **fields}, ensure_ascii=False)
    return f"event: {event_type}\r\ndata: {data}\r\n\r\n".encode()


@pytest.mark.parametrize("model", ["pro", "flash"])
async def test_stream_forwards_text_before_upstream_ends(upstream_key, model, monkeypatch, caplog):
    first_sent = anyio.Event()
    clock = [10.0]
    monkeypatch.setattr(gateway, "time", SimpleNamespace(time=lambda: 100, perf_counter=lambda: clock[0]))
    caplog.set_level("INFO", logger=gateway.logger.name)

    def frame(text):
        if model == "pro":
            return upstream_frame("response.output_text.delta", delta=text)
        return upstream_frame("content_block_delta", delta={"type": "text_delta", "text": text})

    async def frames():
        clock[0] = 10.1
        yield b": heartbeat\r\n\r\n"
        yield frame("")
        clock[0] = 10.25
        first = frame("你")
        split = first.index("你".encode()) + 1
        yield first[:split]
        yield first[split:]
        with anyio.fail_after(2):
            await first_sent.wait()
        yield frame("好")
        if model == "pro":
            yield upstream_frame("response.completed", response={"usage": {
                "input_tokens": 12, "output_tokens": 3, "input_tokens_details": {"cached_tokens": 2},
                "output_tokens_details": {"reasoning_tokens": 1},
            }})
        else:
            yield upstream_frame("message_start", message={"usage": {"input_tokens": 10}})
            yield upstream_frame("message_delta", delta={"stop_reason": "end_turn"}, usage={
                "input_tokens": 10, "output_tokens": 3, "cache_read_input_tokens": 2,
                "cache_creation_input_tokens": 0,
            })
            yield upstream_frame("message_stop")

    stream = MockSSE(frames())

    def upstream(request):
        assert request.headers["accept"] == "text/event-stream"
        assert json.loads(request.content)["stream"] is True
        expected_path = "/responses" if model == "pro" else "/anthropic/v1/messages"
        assert request.url.path == expected_path
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=stream)

    def on_body(body):
        if '"content": "你"'.encode() in body:
            first_sent.set()

    async with gateway_client(upstream, on_body=on_body) as client:
        response = await client.post(CHAT_PATH, json={"model": model, "messages": MESSAGES, "stream": True})
        record = (await client.get(f"/v1/usage/{response.headers['x-request-id']}")).json()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames_sent = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
    assert frames_sent[-1] == "[DONE]"
    chunks = [json.loads(data) for data in frames_sent[:-1]]
    assert [chunk["choices"][0]["delta"] for chunk in chunks] == [
        {"role": "assistant"}, {"content": "你"}, {"content": "好"}, {},
    ]
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    request_id = response.headers["x-request-id"]
    assert {chunk["id"] for chunk in chunks} == {request_id}
    assert [record.message for record in caplog.records if record.name == gateway.logger.name] == [
        f"model={model} request_id={request_id} first_token_ms=250.000"
    ]
    assert stream.closed
    assert record["status"] == "success"
    assert record["first_token_ms"] == 250
    assert record["latency_ms"] >= record["first_token_ms"]
    assert record["usage"]["input_tokens"] == 12
    assert record["usage"]["output_tokens"] == 3
    assert record["usage"]["cached_input_tokens"] == 2
    assert record["usage"]["reasoning_tokens"] == (1 if model == "pro" else None)
    assert record["estimated_cost_usd"] == pytest.approx(0.000025168 if model == "pro" else 0.000006612)


@pytest.mark.parametrize("model", ["pro", "flash"])
@pytest.mark.parametrize("failure", ["eof", "disconnect", "failed_event"])
async def test_interrupted_stream_reports_error_without_success(upstream_key, model, failure):
    async def frames():
        if model == "pro":
            yield upstream_frame("response.output_text.delta", delta="Partial")
        else:
            yield upstream_frame("content_block_delta", delta={"type": "text_delta", "text": "Partial"})
        if failure == "disconnect":
            raise httpx.ReadError("upstream disconnected")
        if failure == "failed_event":
            if model == "pro":
                yield upstream_frame("response.failed")
            else:
                yield b'data: {"error":{"message":"generation timed out"}}\n\n'

    stream = MockSSE(frames())
    calls = []

    def upstream(request):
        calls.append(request)
        return httpx.Response(200, stream=stream)

    async with gateway_client(upstream) as client:
        response = await client.post(CHAT_PATH, json={"model": model, "messages": MESSAGES, "stream": True})
        record = (await client.get(f"/v1/usage/{response.headers['x-request-id']}")).json()

    assert response.status_code == 200
    assert '"content": "Partial"' in response.text
    assert "[DONE]" not in response.text
    assert '"finish_reason": "stop"' not in response.text
    terminal = json.loads(response.text.strip().splitlines()[-1].removeprefix("data: "))
    expected_code = "upstream_incomplete" if failure == "eof" else "upstream_error"
    assert terminal["error"]["code"] == expected_code
    assert len(calls) == 1
    assert stream.closed
    assert record["status"] == "error"
    assert record["error_code"] == expected_code
    assert record["retries"] == 0


async def test_stream_failure_before_text_has_no_first_token_log(upstream_key, caplog):
    caplog.set_level("INFO", logger=gateway.logger.name)

    def upstream(_):
        return httpx.Response(503)

    async with gateway_client(upstream) as client:
        response = await client.post(CHAT_PATH, json={"model": "pro", "messages": MESSAGES, "stream": True})

    assert response.status_code == 200
    frames = response.text.strip().splitlines()
    assert len(frames) == 1
    assert json.loads(frames[0].removeprefix("data: "))["error"]["code"] == "upstream_error"
    assert all(record.name != gateway.logger.name for record in caplog.records)


SCHEMA_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer",
        "schema": {
            "type": "object", "properties": {"answer": {"type": "string"}},
            "required": ["answer"], "additionalProperties": False,
        },
    },
}


@pytest.mark.parametrize("model", ["pro", "flash"])
@pytest.mark.parametrize(
    ("option", "text", "error"),
    [
        (SCHEMA_FORMAT, '{"answer":"ok"}', None),
        (SCHEMA_FORMAT, '{"answer":123}', "schema_validation_failed"),
        ({"type": "json_object"}, '{"answer":NaN}', "invalid_json_output"),
        ({"type": "json_object"}, '["ok"]', "invalid_json_output"),
    ],
)
async def test_output_format_is_translated_and_validated(upstream_key, model, option, text, error):
    def upstream(request):
        body = json.loads(request.content)
        if model == "pro":
            assert body["text"]["format"]["type"] == option["type"]
            if option == SCHEMA_FORMAT:
                assert body["text"]["format"]["schema"] == option["json_schema"]["schema"]
            return httpx.Response(200, json={"status": "completed", "output": [
                {"type": "message", "content": [{"type": "output_text", "text": text}]},
            ]})
        assert "output_config" not in body
        assert "Return only valid JSON" in body["system"] or "Return only a valid JSON" in body["system"]
        return httpx.Response(200, json={"stop_reason": "end_turn", "content": [
            {"type": "text", "text": text},
        ]})

    async with gateway_client(upstream) as client:
        response = await client.post(CHAT_PATH, json={
            "model": model, "messages": MESSAGES, "response_format": option,
        })
    if error:
        assert response.status_code == 502
        assert response.json()["error"]["code"] == error
    else:
        assert response.status_code == 200
        assert json.loads(response.json()["choices"][0]["message"]["content"]) == {"answer": "ok"}


@pytest.mark.parametrize("model", ["pro", "flash"])
async def test_prompt_versions_render_into_upstream_messages(upstream_key, model):
    rendered = []

    def upstream(request):
        body = json.loads(request.content)
        assert "gateway_prompt" not in body
        if model == "pro":
            rendered.append(body["input"][0]["content"])
            return httpx.Response(200, json={"status": "completed", "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "ok"}]},
            ]})
        prompt, original_system = body["system"].split("\n\n")
        rendered.append(prompt)
        assert original_system == MESSAGES[0]["content"]
        return httpx.Response(200, json={"stop_reason": "end_turn", "content": [
            {"type": "text", "text": "ok"},
        ]})

    async with gateway_client(upstream) as client:
        for version in (1, 2):
            response = await client.post(CHAT_PATH, json={
                "model": model, "messages": MESSAGES,
                "gateway_prompt": {"id": "reviewer", "version": version, "variables": {"language": "中文"}},
            })
            assert response.status_code == 200
    assert rendered == [
        "Review the following request in 中文. Give a short answer.",
        "Review the following request in 中文. Include one concrete example.",
    ]


@pytest.mark.parametrize(("extra", "status", "code"), [
    ({"gateway_prompt": {"id": "reviewer", "version": 99}}, 404, "prompt_not_found"),
    ({"gateway_prompt": {"id": "reviewer", "version": 1}}, 422, "missing_prompt_variable"),
    ({"stream": True, "response_format": {"type": "json_object"}}, 422, "unsupported_combination"),
    ({"response_format": {"type": "json_schema", "json_schema": {
        "name": "test", "schema": {"type": "unknown"},
    }}}, 422, "invalid_response_schema"),
    ({"response_format": {"type": "json_schema", "json_schema": {
        "name": "test", "schema": {"$ref": "https://example.test/schema"},
    }}}, 422, "invalid_response_schema"),
])
async def test_invalid_output_options_stop_before_upstream(upstream_key, extra, status, code):
    def upstream(_):
        pytest.fail("Invalid options must not reach upstream")

    async with gateway_client(upstream) as client:
        response = await client.post(CHAT_PATH, json={"model": "pro", "messages": MESSAGES, **extra})
    assert response.status_code == status
    assert response.json()["error"]["code"] == code


@pytest.mark.parametrize("stream", [False, True])
async def test_exponential_retry_stops_at_three_retries(upstream_key, monkeypatch, stream):
    calls = []
    delays = []

    async def fake_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(call_policy, "asyncio", SimpleNamespace(sleep=fake_sleep))
    monkeypatch.setattr(gateway, "retry_backoff", call_policy.retry_backoff)

    def upstream(request):
        calls.append(request)
        return httpx.Response(503)

    async with gateway_client(upstream) as client:
        response = await client.post(CHAT_PATH, json={"model": "pro", "messages": MESSAGES, "stream": stream})
        record = (await client.get(f"/v1/usage/{response.headers['x-request-id']}")).json()
    assert len(calls) == 4
    assert delays == [0.25, 0.5, 1.0]
    assert record["retries"] == 3
    assert record["status"] == "error"
    assert record["first_token_ms"] is None
    assert record["usage"]["input_tokens"] is None
    assert response.status_code == (200 if stream else 502)


async def test_transient_error_retries_and_records_normal_usage(upstream_key):
    calls = 0

    def upstream(_):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429)
        return httpx.Response(200, json={
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 5, "cache_read_input_tokens": 2, "output_tokens": 3},
        })

    async with gateway_client(upstream) as client:
        response = await client.post(CHAT_PATH, json={"model": "flash", "messages": MESSAGES})
        record = (await client.get(f"/v1/usage/{response.headers['x-request-id']}")).json()
    assert response.status_code == 200
    assert calls == 2
    assert record["retries"] == 1
    assert record["usage"]["input_tokens"] == 7
    assert record["usage"]["cached_input_tokens"] == 2
    assert record["latency_ms"] >= 0
    assert record["first_token_ms"] is None


async def test_nonretryable_upstream_failure_makes_one_attempt(upstream_key):
    calls = []

    def upstream(request):
        calls.append(request)
        return httpx.Response(401)

    async with gateway_client(upstream) as client:
        response = await client.post(CHAT_PATH, json={"model": "pro", "messages": MESSAGES})
    assert response.status_code == 502
    assert len(calls) == 1


async def test_rate_limits_are_independent(upstream_key, monkeypatch):
    monkeypatch.setenv("GATEWAY_PRO_RPM", "1")
    monkeypatch.setenv("GATEWAY_FLASH_RPM", "1")
    calls = []

    def upstream(request):
        calls.append(request.url.path)
        if request.url.path == "/responses":
            return httpx.Response(200, json={"status": "completed", "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "ok"}]},
            ]})
        return httpx.Response(200, json={"stop_reason": "end_turn", "content": [
            {"type": "text", "text": "ok"},
        ]})

    async with gateway_client(upstream) as client:
        for model in ("pro", "flash"):
            first = await client.post(CHAT_PATH, json={"model": model, "messages": MESSAGES})
            assert first.status_code == 200
            blocked = await client.post(CHAT_PATH, json={"model": model, "messages": MESSAGES, "stream": True})
            assert blocked.status_code == 429
            assert blocked.json()["error"]["code"] == "rate_limit_exceeded"
            record = (await client.get(f"/v1/usage/{blocked.headers['x-request-id']}")).json()
            assert record["error_code"] == "rate_limit_exceeded"
    assert calls == ["/responses", "/anthropic/v1/messages"]
