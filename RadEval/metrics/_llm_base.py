"""Base class for LLM-backed evaluation metrics.

Provides unified provider validation, API key resolution, cost tracking,
sync/async chat completion, and concurrent execution so that individual
metrics only implement prompt building + response parsing + aggregation.
"""
from __future__ import annotations

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional

from ._llm import (
    CostTracker,
    call_gemini,
    call_gemini_async,
    call_openai,
    call_openai_async,
)

logger = logging.getLogger(__name__)


class LLMMetricBase(ABC):
    """Abstract base for metrics that call an LLM API (OpenAI / Gemini / local HF).

    Subclasses **must** define:
      - ``SUPPORTED_PROVIDERS`` class variable (e.g. ``{"openai", "gemini"}``)
      - ``_build_request``
      - ``_parse_response``
      - ``_aggregate``

    The base class provides:
      - provider / key validation
      - ``CostTracker`` for API-based providers
      - sync ``_chat_completion`` and async ``_chat_completion_async``
      - ``_evaluate_one`` / ``_evaluate_one_async`` with retry
      - ``__call__`` with semaphore-based concurrency for API providers
    """

    SUPPORTED_PROVIDERS: ClassVar[set[str]] = set()

    def __init__(
        self,
        provider: str,
        model_name: str,
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        max_concurrent: int = 50,
        **kwargs,
    ):
        # --- provider validation ---
        if provider not in self.SUPPORTED_PROVIDERS:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support provider "
                f"'{provider}'. Supported: {sorted(self.SUPPORTED_PROVIDERS)}"
            )

        self.provider = provider
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self.cost_tracker: Optional[CostTracker] = None

        # --- key resolution + client setup ---
        if provider == "openai":
            self._resolved_openai_key = (
                openai_api_key
                or os.environ.get("OPENAI_API_KEY")
            )
            if not self._resolved_openai_key:
                raise EnvironmentError(
                    "OpenAI API key required. Pass openai_api_key= or set "
                    "the OPENAI_API_KEY environment variable."
                )
            from openai import AsyncOpenAI, OpenAI
            self._openai_client = OpenAI(api_key=self._resolved_openai_key)
            self._openai_async_client = AsyncOpenAI(
                api_key=self._resolved_openai_key,
            )
            self.cost_tracker = CostTracker(model_name)

        elif provider == "gemini":
            self._resolved_gemini_key = (
                gemini_api_key
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
            )
            if not self._resolved_gemini_key:
                raise EnvironmentError(
                    "Gemini API key required. Pass gemini_api_key= or set "
                    "GEMINI_API_KEY / GOOGLE_API_KEY environment variable."
                )
            from google import genai
            self._gemini_client = genai.Client(
                api_key=self._resolved_gemini_key,
            )
            self.cost_tracker = CostTracker(model_name)

        # provider == "hf" or "local" → no key resolution, no cost tracker

    # ------------------------------------------------------------------
    # Abstract interface – subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_request(self, ref: str, hyp: str, **kwargs) -> dict[str, Any]:
        """Build the LLM request payload for a single sample.

        Must return a dict with at least ``"messages"`` (list[dict]) for
        OpenAI or ``"contents"`` (str) for Gemini.  May include extra
        provider-specific keys (``response_format``, ``config``, etc.).
        """

    @abstractmethod
    def _parse_response(self, raw: str) -> dict:
        """Parse the raw LLM text into a metric-specific result dict."""

    @abstractmethod
    def _aggregate(
        self, results: list[dict], refs: list[str], hyps: list[str],
    ) -> tuple:
        """Aggregate per-sample result dicts into the final return value."""

    # ------------------------------------------------------------------
    # Chat completion – sync & async, dispatching by provider
    # ------------------------------------------------------------------

    def _chat_completion(self, request: dict[str, Any]) -> str:
        """Synchronous LLM call dispatched by provider."""
        if self.provider == "openai":
            messages = request["messages"]
            extra = {k: v for k, v in request.items() if k != "messages"}
            return call_openai(
                self._openai_client,
                self.model_name,
                messages,
                cost_tracker=self.cost_tracker,
                **extra,
            )
        elif self.provider == "gemini":
            contents = request["contents"]
            config = request.get("config")
            resp = call_gemini(
                self._gemini_client,
                self.model_name,
                contents,
                config=config,
                cost_tracker=self.cost_tracker,
            )
            return self._extract_gemini_text(resp)
        else:
            raise NotImplementedError(
                f"Sync chat completion not implemented for provider "
                f"'{self.provider}'."
            )

    async def _chat_completion_async(self, request: dict[str, Any]) -> str:
        """Asynchronous LLM call dispatched by provider."""
        if self.provider == "openai":
            messages = request["messages"]
            extra = {k: v for k, v in request.items() if k != "messages"}
            return await call_openai_async(
                self._openai_async_client,
                self.model_name,
                messages,
                cost_tracker=self.cost_tracker,
                **extra,
            )
        elif self.provider == "gemini":
            contents = request["contents"]
            config = request.get("config")
            resp = await call_gemini_async(
                self._gemini_client,
                self.model_name,
                contents,
                config=config,
                cost_tracker=self.cost_tracker,
            )
            return self._extract_gemini_text(resp)
        else:
            raise NotImplementedError(
                f"Async chat completion not implemented for provider "
                f"'{self.provider}'."
            )

    @staticmethod
    def _extract_gemini_text(response) -> str:
        """Extract text from a Gemini response object."""
        try:
            return response.text
        except Exception:
            pass
        try:
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    return candidate.content.parts[0].text
        except Exception:
            pass
        raise ValueError("Could not extract text from Gemini response.")

    # ------------------------------------------------------------------
    # Evaluate one sample (build → call → parse), with retry
    # ------------------------------------------------------------------

    def _evaluate_one(
        self, ref: str, hyp: str, max_retries: int = 2, **kwargs,
    ) -> dict:
        """Evaluate one (ref, hyp) pair synchronously with retry."""
        request = self._build_request(ref, hyp, **kwargs)
        last_error: Exception | None = None

        for attempt in range(1 + max_retries):
            try:
                raw = self._chat_completion(request)
                return self._parse_response(raw)
            except Exception as e:
                last_error = e
                logger.warning(
                    "%s: attempt %d/%d failed: %s",
                    self.__class__.__name__, attempt + 1, 1 + max_retries, e,
                )

        raise RuntimeError(
            f"{self.__class__.__name__}: all {1 + max_retries} attempts "
            f"failed. Last error: {last_error}"
        )

    async def _evaluate_one_async(
        self, ref: str, hyp: str, max_retries: int = 2, **kwargs,
    ) -> dict:
        """Evaluate one (ref, hyp) pair asynchronously with retry."""
        request = self._build_request(ref, hyp, **kwargs)
        last_error: Exception | None = None

        for attempt in range(1 + max_retries):
            try:
                raw = await self._chat_completion_async(request)
                return self._parse_response(raw)
            except Exception as e:
                last_error = e
                logger.warning(
                    "%s: async attempt %d/%d failed: %s",
                    self.__class__.__name__, attempt + 1, 1 + max_retries, e,
                )

        raise RuntimeError(
            f"{self.__class__.__name__}: all {1 + max_retries} async "
            f"attempts failed. Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # __call__: concurrent dispatch + aggregation
    # ------------------------------------------------------------------

    def __call__(
        self,
        refs: list[str],
        hyps: list[str],
        on_sample_done=None,
        **kwargs,
    ) -> tuple:
        if not isinstance(refs, list) or not isinstance(hyps, list):
            raise TypeError("refs and hyps must be lists")
        if len(refs) != len(hyps):
            raise ValueError("refs and hyps must have the same length")

        if self.provider in ("openai", "gemini"):
            results = self._run_concurrent(refs, hyps, on_sample_done, **kwargs)
        else:
            results = []
            for ref, hyp in zip(refs, hyps):
                results.append(self._evaluate_one(ref, hyp, **kwargs))
                if on_sample_done:
                    on_sample_done()

        return self._aggregate(results, refs, hyps)

    def _run_concurrent(
        self,
        refs: list[str],
        hyps: list[str],
        on_sample_done=None,
        **kwargs,
    ) -> list[dict]:
        """Run evaluations concurrently via asyncio with a semaphore."""
        sem = asyncio.Semaphore(self.max_concurrent)

        async def _sem_eval(ref, hyp):
            async with sem:
                result = await self._evaluate_one_async(ref, hyp, **kwargs)
                if on_sample_done:
                    on_sample_done()
                return result

        async def _gather():
            tasks = [_sem_eval(r, h) for r, h in zip(refs, hyps)]
            return list(await asyncio.gather(*tasks))

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                return pool.submit(lambda: asyncio.run(_gather())).result()
        return asyncio.run(_gather())
