import json
import logging
import os
import sqlite3
import time
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager
from pathlib import Path
from uuid import uuid4

import anyio
import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from adapters import (
    DEEPSEEK_BASE_URL,
    AnthropicMessagesAdapter,
    GatewayError,
    Message,
    ModelAdapter,
    ResponsesAdapter,
)
from call_policy import MAX_RETRIES, CallRecord, CallStore, ModelRateLimiter, TokenUsage, retry_backoff
from output_options import (
    JsonObjectFormat,
    JsonSchemaFormat,
    PromptReference,
    load_templates,
    output_constraint,
    validate_output,
)


logger = logging.getLogger("uvicorn.error").getChild("gateway")


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=1)
    stream: bool = False
    response_format: JsonObjectFormat | JsonSchemaFormat | None = Field(default=None, discriminator="type")
    gateway_prompt: PromptReference | None = None


def error_body(status_code: int, code: str, message: str) -> dict[str, dict[str, str]]:
    error_type = "invalid_request_error" if status_code < 500 else "server_error"
    if status_code == 429:
        error_type = "rate_limit_error"
    return {"error": {"type": error_type, "code": code, "message": message}}


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=error_body(status_code, code, message))


def sse_data(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def record_call(store: CallStore, call: CallRecord) -> None:
    data = call.encoded(time.perf_counter())
    with anyio.CancelScope(shield=True):
        try:
            await anyio.to_thread.run_sync(store.record, call.request_id, data)
        except sqlite3.Error:
            logger.exception("Could not persist call record request_id=%s", call.request_id)


async def stream_completion(
    adapter: ModelAdapter, payload: ChatRequest, call: CallRecord, store: CallStore
) -> AsyncIterator[str]:
    chunk = {
        "id": call.request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": payload.model,
    }
    error_frame = None
    try:
        while True:
            try:
                async with aclosing(adapter.stream(payload.messages)) as deltas:
                    async for text in deltas:
                        if isinstance(text, TokenUsage):
                            call.usage = text
                            continue
                        if not text:
                            continue
                        if call.first_token_ms is None:
                            call.first_token_ms = round((time.perf_counter() - call.started) * 1000, 3)
                            logger.info(
                                "model=%s request_id=%s first_token_ms=%.3f",
                                payload.model, call.request_id, call.first_token_ms,
                            )
                            yield sse_data({
                                **chunk,
                                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                            })
                        yield sse_data({
                            **chunk,
                            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                        })
                break
            except GatewayError as exc:
                if call.first_token_ms is not None or not exc.retryable or call.retries >= MAX_RETRIES:
                    raise
                call.retries += 1
                await retry_backoff(call.retries)
        if call.first_token_ms is None:
            raise GatewayError(502, "invalid_upstream_response", "The upstream returned no reply text.")
        call.status = "success"
    except GatewayError as error:
        call.status, call.error_code = "error", error.code
        error_frame = sse_data(error_body(error.status_code, error.code, str(error)))
    finally:
        await record_call(store, call)

    if error_frame is not None:
        yield error_frame
        return

    yield sse_data({
        **chunk,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    })
    yield "data: [DONE]\n\n"


def create_app(*, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        store = CallStore(Path(os.getenv("GATEWAY_USAGE_DB", "gateway-usage.sqlite3")))
        await anyio.to_thread.run_sync(store.initialize)
        app.state.call_store = store
        app.state.limiter = ModelRateLimiter({
            model: int(os.getenv(f"GATEWAY_{model.upper()}_RPM", "60"))
            for model in ("pro", "flash")
        })
        async with httpx.AsyncClient(
            base_url=DEEPSEEK_BASE_URL,
            timeout=30.0,
            transport=transport,
        ) as client:
            app.state.upstream_configured = bool(api_key)
            app.state.templates = load_templates()
            app.state.adapters = {
                "pro": ResponsesAdapter(client, api_key),
                "flash": AnthropicMessagesAdapter(client, api_key),
            }
            yield

    app = FastAPI(title="Course LLM Gateway", lifespan=lifespan)

    @app.exception_handler(GatewayError)
    async def handle_gateway_error(_: Request, error: GatewayError) -> JSONResponse:
        return error_response(error.status_code, error.code, str(error))

    @app.exception_handler(RequestValidationError)
    async def handle_invalid_request(_: Request, error: RequestValidationError) -> JSONResponse:
        fields = [f"{'.'.join(map(str, item['loc']))}: {item['msg']}" for item in error.errors()]
        return error_response(422, "invalid_request", "; ".join(fields))

    @app.get("/v1/usage/{request_id}")
    async def get_usage(request_id: str, request: Request):
        data = await anyio.to_thread.run_sync(request.app.state.call_store.get, request_id)
        if data is None:
            raise GatewayError(404, "request_not_found", "No usage record was found for this request.")
        return data

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: ChatRequest, request: Request):
        adapter: ModelAdapter | None = request.app.state.adapters.get(payload.model)
        if adapter is None:
            raise GatewayError(404, "model_not_found", "Choose model 'pro' or 'flash'.")
        if not request.app.state.upstream_configured:
            raise GatewayError(503, "upstream_not_configured", "Set DEEPSEEK_API_KEY on the gateway.")

        call = CallRecord(f"chatcmpl-{uuid4().hex}", payload.model, payload.stream, time.perf_counter())
        if payload.gateway_prompt is not None:
            call.prompt_id = payload.gateway_prompt.id
            call.prompt_version = payload.gateway_prompt.version
        store = request.app.state.call_store
        headers = {"X-Request-ID": call.request_id}
        owns_record = True
        try:
            output_format = validator = None
            if payload.response_format is not None:
                if payload.stream:
                    raise GatewayError(422, "unsupported_combination", "Use response_format with stream=false.")
                output_format, validator = output_constraint(payload.response_format)
            if payload.gateway_prompt is not None:
                prompt = payload.gateway_prompt.render(request.app.state.templates)
                payload.messages = [prompt, *payload.messages]
            if not request.app.state.limiter.admit(payload.model):
                raise GatewayError(429, "rate_limit_exceeded", "The requested model rate limit was reached.")
            if payload.stream:
                response = StreamingResponse(
                    stream_completion(adapter, payload, call, store),
                    media_type="text/event-stream",
                    headers={**headers, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
                owns_record = False
                return response

            while True:
                try:
                    result = await adapter.complete(payload.messages, output_format=output_format)
                    break
                except GatewayError as exc:
                    if not exc.retryable or call.retries >= MAX_RETRIES:
                        raise
                    call.retries += 1
                    await retry_backoff(call.retries)
            call.usage = result.usage
            if output_format is not None:
                validate_output(result.text, validator)
            call.status = "success"
        except GatewayError as error:
            call.status, call.error_code = "error", error.code
            response = error_response(error.status_code, error.code, str(error))
            response.headers.update(headers)
            return response
        finally:
            if owns_record:
                await record_call(store, call)

        return JSONResponse({
            "id": call.request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": "stop",
                }
            ],
        }, headers=headers)

    return app


app = create_app()
