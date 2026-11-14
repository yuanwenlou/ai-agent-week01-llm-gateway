import asyncio
import json
import sqlite3
import time
from collections import deque
from contextlib import closing
from dataclasses import asdict, dataclass, field
from pathlib import Path


# USD per million tokens at peak rates, dated 2026-11-14.
PRICES = {"pro": (1.32, 0.044, 3.96), "flash": (0.30, 0.006, 1.20)}
MAX_RETRIES = 3


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    reasoning_tokens: int | None = None

    def estimated_cost(self, model: str) -> float | None:
        if None in (self.input_tokens, self.output_tokens, self.cached_input_tokens):
            return None
        assert self.input_tokens is not None and self.cached_input_tokens is not None
        assert self.output_tokens is not None
        if self.cached_input_tokens > self.input_tokens:
            return None
        input_rate, cache_rate, output_rate = PRICES[model]
        return round((
            (self.input_tokens - self.cached_input_tokens) * input_rate
            + self.cached_input_tokens * cache_rate
            + self.output_tokens * output_rate
        ) / 1_000_000, 9)


@dataclass
class CallRecord:
    request_id: str
    model: str
    stream: bool
    started: float
    prompt_id: str | None = None
    prompt_version: int | None = None
    first_token_ms: float | None = None
    retries: int = 0
    status: str = "cancelled"
    error_code: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)

    def encoded(self, finished: float) -> str:
        data = asdict(self)
        data.pop("started")
        data["latency_ms"] = round((finished - self.started) * 1000, 3)
        data["estimated_cost_usd"] = self.usage.estimated_cost(self.model) if self.model in PRICES else None
        return json.dumps(data)


async def retry_backoff(retries: int) -> None:
    await asyncio.sleep(0.25 * 2 ** (retries - 1))


class ModelRateLimiter:
    def __init__(self, limits: dict[str, int]) -> None:
        self.limits = limits
        self.windows: dict[str, deque[float]] = {model: deque() for model in limits}

    def admit(self, model: str) -> bool:
        now = time.monotonic()
        window = self.windows[model]
        # This method has no await, so another request cannot interleave the check and append.
        while window and window[0] <= now - 60:
            window.popleft()
        if len(window) >= self.limits[model]:
            return False
        window.append(now)
        return True


class CallStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS calls (request_id TEXT PRIMARY KEY, data TEXT NOT NULL)")

    def record(self, request_id: str, data: str) -> None:
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("INSERT INTO calls (request_id, data) VALUES (?, ?)", (request_id, data))

    def get(self, request_id: str) -> dict | None:
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute("SELECT data FROM calls WHERE request_id = ?", (request_id,)).fetchone()
        return json.loads(row[0]) if row else None
