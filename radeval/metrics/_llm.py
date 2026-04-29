"""Shared LLM API utilities: cost tracking, sync/async calls with retry, concurrency.

This module centralizes all LLM provider interactions (OpenAI, Gemini) so that
metrics don't duplicate client management, retry logic, or cost tracking.
"""
from __future__ import annotations

import asyncio
import logging
import random
import threading
from typing import Any, Callable, List, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Pricing (per 1M tokens: input, output)
# Update via: python scripts/update_pricing.py
# ---------------------------------------------------------------------------

PRICING_PER_1M: dict[str, tuple[float, float]] = {
    # OpenAI (https://openai.com/api/pricing/)
    "gpt-4o-mini":  (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o":       (2.50, 10.00),
    "gpt-5.2":      (2.00, 8.00),
    # Google Gemini (https://ai.google.dev/gemini-api/docs/pricing)
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro":   (1.25, 10.00),
}

# ---------------------------------------------------------------------------
# Cost tracker
# ---------------------------------------------------------------------------


class CostTracker:
    """Thread-safe accumulator for API token usage and dollar cost."""

    def __init__(self, model: str):
        self._lock = threading.Lock()
        self.input_tokens = 0
        self.output_tokens = 0
        per_m = PRICING_PER_1M.get(model, (0.15, 0.60))
        self._input_rate = per_m[0] / 1_000_000
        self._output_rate = per_m[1] / 1_000_000
        self.model = model

    def add(self, input_tok: int, output_tok: int):
        with self._lock:
            self.input_tokens += input_tok
            self.output_tokens += output_tok

    @property
    def cost(self) -> float:
        return (self.input_tokens * self._input_rate
                + self.output_tokens * self._output_rate)

    def reset(self):
        with self._lock:
            self.input_tokens = 0
            self.output_tokens = 0


# ---------------------------------------------------------------------------
# Retry constants
# ---------------------------------------------------------------------------

MAX_RETRIES = 6
RETRY_MIN_DELAY = 1.0
RETRY_MAX_DELAY = 60.0

# ---------------------------------------------------------------------------
# OpenAI helpers
# ---------------------------------------------------------------------------


def _track_openai_usage(resp, cost_tracker: Optional[CostTracker]):
    if cost_tracker and resp.usage:
        cost_tracker.add(resp.usage.prompt_tokens, resp.usage.completion_tokens)


def call_openai(
    client,
    model: str,
    messages: list[dict],
    temperature: float = 0.0,
    cost_tracker: Optional[CostTracker] = None,
    **kwargs,
) -> str:
    """Sync OpenAI chat completion with retry and cost tracking.

    Uses tenacity with random exponential backoff per OpenAI cookbook recommendation.
    """
    import openai
    from tenacity import (
        retry, retry_if_exception_type,
        stop_after_attempt, wait_random_exponential,
    )

    @retry(
        wait=wait_random_exponential(min=RETRY_MIN_DELAY, max=RETRY_MAX_DELAY),
        stop=stop_after_attempt(MAX_RETRIES),
        retry=retry_if_exception_type((
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
        )),
        reraise=True,
    )
    def _call():
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, **kwargs)
        _track_openai_usage(resp, cost_tracker)
        return resp.choices[0].message.content or ""

    return _call()


async def call_openai_async(
    client,
    model: str,
    messages: list[dict],
    temperature: float = 0.0,
    cost_tracker: Optional[CostTracker] = None,
    **kwargs,
) -> str:
    """Async OpenAI chat completion with exponential backoff + jitter."""
    import openai

    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.chat.completions.create(
                model=model, messages=messages, temperature=temperature,
                **kwargs)
            _track_openai_usage(resp, cost_tracker)
            return resp.choices[0].message.content or ""
        except (openai.RateLimitError, openai.APITimeoutError,
                openai.APIConnectionError) as e:
            if attempt == MAX_RETRIES - 1:
                raise
            delay = min(RETRY_MIN_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
            delay *= 0.5 + random.random()
            logger.warning(
                "OpenAI async call failed (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1, MAX_RETRIES, delay, e)
            await asyncio.sleep(delay)
        except Exception as e:
            err_str = str(e).lower()
            is_retryable = any(s in err_str for s in
                               ("429", "500", "502", "503", "overloaded"))
            if not is_retryable or attempt == MAX_RETRIES - 1:
                raise
            delay = min(RETRY_MIN_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
            delay *= 0.5 + random.random()
            logger.warning(
                "OpenAI async call failed (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1, MAX_RETRIES, delay, e)
            await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------


def _track_gemini_usage(response, cost_tracker: Optional[CostTracker]):
    """Extract token counts from Gemini response metadata."""
    if not cost_tracker:
        return
    try:
        usage = response.usage_metadata
        if usage:
            cost_tracker.add(
                getattr(usage, "prompt_token_count", 0) or 0,
                getattr(usage, "candidates_token_count", 0) or 0,
            )
    except Exception:
        pass


def call_gemini(
    client,
    model: str,
    contents: str,
    config=None,
    cost_tracker: Optional[CostTracker] = None,
):
    """Sync Gemini call with retry and cost tracking.

    Returns the raw response object (caller handles text extraction).
    """
    from tenacity import (
        retry, retry_if_exception_type,
        stop_after_attempt, wait_random_exponential,
    )

    retryable_errors = (Exception,)
    try:
        from google.api_core.exceptions import (
            ResourceExhausted, ServiceUnavailable,
        )
        retryable_errors = (ResourceExhausted, ServiceUnavailable)
    except ImportError:
        pass

    @retry(
        wait=wait_random_exponential(min=RETRY_MIN_DELAY, max=RETRY_MAX_DELAY),
        stop=stop_after_attempt(MAX_RETRIES),
        retry=retry_if_exception_type(retryable_errors),
        reraise=True,
    )
    def _call():
        kwargs = {"model": model, "contents": contents}
        if config is not None:
            kwargs["config"] = config
        resp = client.models.generate_content(**kwargs)
        _track_gemini_usage(resp, cost_tracker)
        return resp

    return _call()


async def call_gemini_async(
    client,
    model: str,
    contents: str,
    config=None,
    cost_tracker: Optional[CostTracker] = None,
):
    """Async Gemini call with exponential backoff + jitter."""
    for attempt in range(MAX_RETRIES):
        try:
            kwargs = {"model": model, "contents": contents}
            if config is not None:
                kwargs["config"] = config
            resp = await client.aio.models.generate_content(**kwargs)
            _track_gemini_usage(resp, cost_tracker)
            return resp
        except Exception as e:
            err_str = str(e).lower()
            is_retryable = any(s in err_str for s in
                               ("429", "resource", "exhausted",
                                "unavailable", "500", "503"))
            if not is_retryable or attempt == MAX_RETRIES - 1:
                raise
            delay = min(RETRY_MIN_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
            delay *= 0.5 + random.random()
            logger.warning(
                "Gemini async call failed (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1, MAX_RETRIES, delay, e)
            await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# Concurrency helper
# ---------------------------------------------------------------------------


def run_concurrent(
    coro_fn: Callable[..., Any],
    items: list,
    max_concurrent: int = 50,
    on_item_done: Optional[Callable] = None,
) -> list:
    """Run an async coroutine over items with semaphore-based concurrency.

    Args:
        coro_fn: async callable that takes a single item and returns a result.
        items: list of items to process.
        max_concurrent: max number of concurrent tasks.
        on_item_done: optional callback invoked after each item completes.

    Returns:
        List of results in the same order as items.
    """
    async def _run():
        sem = asyncio.Semaphore(max_concurrent)

        async def _wrapped(item):
            async with sem:
                result = await coro_fn(item)
                if on_item_done:
                    on_item_done()
                return result

        tasks = [_wrapped(item) for item in items]
        return list(await asyncio.gather(*tasks))

    return asyncio.run(_run())
