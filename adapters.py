import json
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict

from call_policy import TokenUsage


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MAX_OUTPUT_TOKENS = 1024


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class ModelResult:
    text: str
    usage: TokenUsage = TokenUsage()


class ModelAdapter(Protocol):
    async def complete(
        self, messages: list[Message], *, output_format: dict[str, Any] | None = None
    ) -> ModelResult: ...

    def stream(self, messages: list[Message]) -> AsyncGenerator[str | TokenUsage, None]: ...


class GatewayError(Exception):
    def __init__(self, status_code: int, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.retryable = retryable


def upstream_failure(exc: httpx.HTTPError) -> GatewayError:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 429:
            return GatewayError(429, "upstream_rate_limited", "The provider rate limit was reached.", retryable=True)
        return GatewayError(502, "upstream_error", "The upstream request failed.", retryable=status in (
            408, 500, 502, 503, 504, 529,
        ))
    if isinstance(exc, httpx.TimeoutException):
        return GatewayError(504, "upstream_timeout", "The upstream request timed out.", retryable=True)
    retryable = isinstance(exc, (httpx.NetworkError, httpx.RemoteProtocolError))
    return GatewayError(502, "upstream_error", "The upstream request failed.", retryable=retryable)


async def post_json(
    client: httpx.AsyncClient,
    path: str,
    headers: dict[str, str],
    body: dict[str, Any],
) -> dict[str, Any]:
    try:
        response = await client.post(path, headers=headers, json=body)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise upstream_failure(exc) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise GatewayError(502, "invalid_upstream_response", "Expected a JSON response.") from exc
    if not isinstance(payload, dict):
        raise GatewayError(502, "invalid_upstream_response", "Expected a JSON object.")
    return payload


async def stream_events(
    client: httpx.AsyncClient,
    path: str,
    headers: dict[str, str],
    body: dict[str, Any],
) -> AsyncGenerator[dict[str, Any], None]:
    try:
        async with client.stream(
            "POST", path, headers={**headers, "Accept": "text/event-stream"}, json=body
        ) as response:
            response.raise_for_status()
            data_lines: list[str] = []
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    data_lines.append(line[5:].removeprefix(" "))
                elif not line and data_lines:
                    try:
                        event = json.loads("\n".join(data_lines))
                    except ValueError as exc:
                        raise GatewayError(502, "invalid_upstream_response", "Invalid SSE data.") from exc
                    data_lines.clear()
                    if not isinstance(event, dict):
                        raise GatewayError(502, "invalid_upstream_response", "Expected an SSE JSON object.")
                    if event.get("type") == "error" or isinstance(event.get("error"), dict):
                        raise GatewayError(502, "upstream_error", "The upstream stream reported an error.")
                    yield event
    except httpx.HTTPError as exc:
        raise upstream_failure(exc) from exc


def stream_text(value: Any) -> str:
    if not isinstance(value, str):
        raise GatewayError(502, "invalid_upstream_response", "Invalid upstream text delta.")
    return value


def text_blocks(blocks: Any, block_type: str) -> list[str]:
    if not isinstance(blocks, list):
        raise GatewayError(502, "invalid_upstream_response", "Invalid upstream content blocks.")
    parts = []
    for block in blocks:
        if not isinstance(block, dict):
            raise GatewayError(502, "invalid_upstream_response", "Invalid upstream content block.")
        if block.get("type") == block_type:
            text = block.get("text")
            if not isinstance(text, str):
                raise GatewayError(502, "invalid_upstream_response", "Invalid upstream text block.")
            parts.append(text)
    return parts


def token_count(payload: Any, key: str) -> int | None:
    value = payload.get(key) if isinstance(payload, dict) else None
    return value if type(value) is int and value >= 0 else None


def responses_usage(usage: Any) -> TokenUsage:
    usage = usage if isinstance(usage, dict) else {}
    return TokenUsage(
        input_tokens=token_count(usage, "input_tokens"),
        output_tokens=token_count(usage, "output_tokens"),
        cached_input_tokens=token_count(usage.get("input_tokens_details"), "cached_tokens"),
        reasoning_tokens=token_count(usage.get("output_tokens_details"), "reasoning_tokens"),
    )


def messages_usage(usage: Any) -> TokenUsage:
    uncached = token_count(usage, "input_tokens")
    read = token_count(usage, "cache_read_input_tokens")
    created = token_count(usage, "cache_creation_input_tokens")
    return TokenUsage(
        input_tokens=None if uncached is None else uncached + (read or 0) + (created or 0),
        output_tokens=token_count(usage, "output_tokens"),
        cached_input_tokens=read,
        cache_creation_input_tokens=created,
        reasoning_tokens=token_count(usage, "reasoning_tokens"),
    )


def model_result(parts: list[str], usage: TokenUsage) -> ModelResult:
    text = "".join(parts)
    if not text.strip():
        raise GatewayError(502, "invalid_upstream_response", "The upstream returned no reply text.")
    return ModelResult(text=text, usage=usage)


class ResponsesAdapter:
    path = "/responses"

    def __init__(self, client: httpx.AsyncClient, api_key: str) -> None:
        self.client = client
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def request_body(self, messages: list[Message], *, stream: bool) -> dict[str, Any]:
        return {
            "model": "deepseek-v4-pro",
            "input": [message.model_dump() for message in messages],
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "reasoning": {"effort": "none"},
            "stream": stream,
        }

    async def complete(
        self, messages: list[Message], *, output_format: dict[str, Any] | None = None
    ) -> ModelResult:
        body = self.request_body(messages, stream=False)
        if output_format is not None:
            body["text"] = {"format": output_format}
            body["instructions"] = "Return only valid JSON matching the requested format."
        payload = await post_json(self.client, self.path, self.headers, body)
        if payload.get("status") != "completed":
            raise GatewayError(502, "upstream_incomplete", "The upstream response did not complete.")

        output = payload.get("output")
        if not isinstance(output, list):
            raise GatewayError(502, "invalid_upstream_response", "Invalid Responses output.")
        parts = []
        for item in output:
            if not isinstance(item, dict):
                raise GatewayError(502, "invalid_upstream_response", "Invalid Responses output item.")
            if item.get("type") == "message":
                parts.extend(text_blocks(item.get("content"), "output_text"))
        return model_result(parts, responses_usage(payload.get("usage")))

    async def stream(self, messages: list[Message]) -> AsyncGenerator[str | TokenUsage, None]:
        events = stream_events(self.client, self.path, self.headers, self.request_body(messages, stream=True))
        async with aclosing(events):
            async for event in events:
                event_type = event.get("type")
                if event_type == "response.output_text.delta":
                    yield stream_text(event.get("delta"))
                elif event_type in ("response.completed", "response.failed", "response.incomplete"):
                    response = event.get("response")
                    if isinstance(response, dict) and isinstance(response.get("usage"), dict):
                        yield responses_usage(response["usage"])
                    if event_type == "response.completed":
                        return
                    if event_type == "response.failed":
                        raise GatewayError(502, "upstream_error", "The upstream response failed.")
                    break
        raise GatewayError(502, "upstream_incomplete", "The upstream response did not complete.")


class AnthropicMessagesAdapter:
    path = "/anthropic/v1/messages"

    def __init__(self, client: httpx.AsyncClient, api_key: str) -> None:
        self.client = client
        self.headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}

    def request_body(self, messages: list[Message], *, stream: bool) -> dict[str, Any]:
        body = {
            "model": "deepseek-flash",
            "messages": [message.model_dump() for message in messages if message.role != "system"],
            "max_tokens": MAX_OUTPUT_TOKENS,
            "thinking": {"type": "disabled"},
            "stream": stream,
        }
        system_messages = [message.content for message in messages if message.role == "system"]
        if system_messages:
            body["system"] = "\n\n".join(system_messages)
        return body

    async def complete(
        self, messages: list[Message], *, output_format: dict[str, Any] | None = None
    ) -> ModelResult:
        body = self.request_body(messages, stream=False)
        if output_format is not None:
            instruction = "Return only a valid JSON object without Markdown or surrounding text."
            if output_format["type"] == "json_schema":
                schema = json.dumps(output_format["schema"], ensure_ascii=False)
                instruction = f"Return only valid JSON matching this schema, without Markdown: {schema}"
            body["system"] = "\n\n".join(filter(None, (body.get("system"), instruction)))
        payload = await post_json(self.client, self.path, self.headers, body)
        if payload.get("stop_reason") not in ("end_turn", "stop_sequence"):
            raise GatewayError(502, "upstream_incomplete", "The upstream response did not complete.")
        return model_result(text_blocks(payload.get("content"), "text"), messages_usage(payload.get("usage")))

    async def stream(self, messages: list[Message]) -> AsyncGenerator[str | TokenUsage, None]:
        events = stream_events(self.client, self.path, self.headers, self.request_body(messages, stream=True))
        stop_reason = None
        async with aclosing(events):
            async for event in events:
                event_type = event.get("type")
                if event_type == "content_block_start":
                    block = event.get("content_block")
                    if isinstance(block, dict) and block.get("type") == "text":
                        yield stream_text(block.get("text"))
                elif event_type == "content_block_delta":
                    delta = event.get("delta")
                    if isinstance(delta, dict) and delta.get("type") == "text_delta":
                        yield stream_text(delta.get("text"))
                elif event_type == "message_delta":
                    delta = event.get("delta")
                    if isinstance(delta, dict) and delta.get("stop_reason") is not None:
                        stop_reason = delta["stop_reason"]
                    if isinstance(event.get("usage"), dict):
                        yield messages_usage(event["usage"])
                elif event_type == "message_stop":
                    if stop_reason in ("end_turn", "stop_sequence"):
                        return
                    break
        raise GatewayError(502, "upstream_incomplete", "The upstream response did not complete.")
