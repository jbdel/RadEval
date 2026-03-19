"""RadFact-CT: LLM-based factual evaluation for CT radiology reports.

Ported from microsoft/RadFact (MIT License). Implements the core pipeline:
  1. Report to Phrases — split narrative text into atomic finding phrases
  2. Negative Filtering (optional) — remove phrases about absent/normal findings
  3. Bidirectional NLI — for each phrase, check entailment against the other report
  4. Scoring — logical precision/recall/F1

Supports async concurrent evaluation with cost tracking.
Requires an OpenAI-compatible API key.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from openai import AsyncOpenAI, OpenAI

from .._llm import (
    CostTracker, call_openai, call_openai_async,
)

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts" / "ct"


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------

def _build_messages(
    system: str,
    few_shot_pairs: list[tuple[str, str]],
    query: str,
) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": system}]
    for human, ai in few_shot_pairs:
        msgs.append({"role": "user", "content": human})
        msgs.append({"role": "assistant", "content": ai})
    msgs.append({"role": "user", "content": query})
    return msgs


# ---------------------------------------------------------------------------
# Step 1: Report to Phrases
# ---------------------------------------------------------------------------

def _load_report_to_phrases_prompts() -> tuple[str, list[tuple[str, str]]]:
    system = (_PROMPTS_DIR / "report_to_phrases_system.txt").read_text()
    examples = json.loads((_PROMPTS_DIR / "report_to_phrases_examples.json").read_text())

    schema_hint = (
        "\n\nPlease respond ONLY with a JSON object in this exact format:\n"
        '{"sentence_list": [{"orig": "...", "new": ["...", "..."]}, ...]}'
    )
    system += schema_hint

    pairs = []
    for ex in examples:
        human = ex["findings_text"]
        ai = json.dumps({"sentence_list": ex["parsed_report"]["sentence_list"]})
        pairs.append((human, ai))
    return system, pairs


def _parse_phrases_from_response(text: str) -> list[str]:
    """Extract the flat list of atomic phrases from the LLM JSON response."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
        else:
            logger.warning("Failed to parse report-to-phrases response")
            return []

    phrases = []
    for sentence in data.get("sentence_list", []):
        for p in sentence.get("new", []):
            if p.strip():
                phrases.append(p.strip())
    return phrases


def report_to_phrases(
    client, model: str, text: str, temperature: float,
    cost_tracker: Optional[CostTracker] = None,
) -> list[str]:
    system, pairs = _load_report_to_phrases_prompts()
    msgs = _build_messages(system, pairs, text)
    raw = call_openai(client, model, msgs, temperature, cost_tracker)
    return _parse_phrases_from_response(raw)


async def report_to_phrases_async(
    client, model: str, text: str, temperature: float,
    cost_tracker: Optional[CostTracker] = None,
) -> list[str]:
    system, pairs = _load_report_to_phrases_prompts()
    msgs = _build_messages(system, pairs, text)
    raw = await call_openai_async(client, model, msgs, temperature, cost_tracker)
    return _parse_phrases_from_response(raw)


# ---------------------------------------------------------------------------
# Step 1.5: Negative Filtering (optional, RadFact+ mode)
# ---------------------------------------------------------------------------

def _load_negative_filtering_prompts() -> tuple[str, list[tuple[str, str]]]:
    system = (_PROMPTS_DIR / "negative_filtering_system.txt").read_text()
    examples = json.loads((_PROMPTS_DIR / "negative_filtering_examples.json").read_text())

    schema_hint = (
        '\n\nPlease respond ONLY with a JSON object: {"phrases": ["...", ...]}'
    )
    system += schema_hint

    pairs = []
    for ex in examples:
        human = json.dumps(ex["input"])
        ai = json.dumps(ex["output"])
        pairs.append((human, ai))
    return system, pairs


def _parse_negatives_response(raw: str, original_phrases: list[str]) -> list[str]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(raw[start:end])
        else:
            logger.warning("Failed to parse negative filtering response")
            return original_phrases
    return data.get("phrases", original_phrases)


def filter_negatives(
    client, model: str, phrases: list[str], temperature: float,
    cost_tracker: Optional[CostTracker] = None,
) -> list[str]:
    if not phrases:
        return []
    system, pairs = _load_negative_filtering_prompts()
    msgs = _build_messages(system, pairs, json.dumps(phrases))
    raw = call_openai(client, model, msgs, temperature, cost_tracker)
    return _parse_negatives_response(raw, phrases)


async def filter_negatives_async(
    client, model: str, phrases: list[str], temperature: float,
    cost_tracker: Optional[CostTracker] = None,
) -> list[str]:
    if not phrases:
        return []
    system, pairs = _load_negative_filtering_prompts()
    msgs = _build_messages(system, pairs, json.dumps(phrases))
    raw = await call_openai_async(client, model, msgs, temperature, cost_tracker)
    return _parse_negatives_response(raw, phrases)


# ---------------------------------------------------------------------------
# Step 2: Bidirectional NLI (single-phrase entailment verification)
# ---------------------------------------------------------------------------

def _load_nli_prompts() -> tuple[str, list[tuple[str, str]]]:
    system = (_PROMPTS_DIR / "nli_system.txt").read_text()
    examples_raw = json.loads((_PROMPTS_DIR / "nli_examples.json").read_text())

    schema_hint = (
        "\n\nFor each hypothesis, respond ONLY with a YAML object:\n"
        "phrase: <the hypothesis phrase verbatim>\n"
        "status: entailment  # or not_entailment\n"
        "evidence:\n"
        "- <supporting phrase from reference, if entailed>\n"
    )
    system += schema_hint

    pairs = []
    for ex in examples_raw:
        inp = ex["input"]
        out = ex["output"]
        for direction, ref_key, hyp_key in [
            ("A", "phrases_B", "phrases_A_evidenced"),
            ("B", "phrases_A", "phrases_B_evidenced"),
        ]:
            ref_phrases = inp[ref_key.replace("_evidenced", "")]
            for ev_phrase in out[hyp_key]:
                status = ev_phrase["status"]
                if status in ("contradiction", "neutral"):
                    status = "not_entailment"
                query = yaml.dump(
                    {"reference": ref_phrases, "hypothesis": ev_phrase["phrase"]},
                    sort_keys=False,
                )
                answer = yaml.dump(
                    {
                        "phrase": ev_phrase["phrase"],
                        "status": status,
                        "evidence": ev_phrase["evidence"] if status == "entailment" else [],
                    },
                    sort_keys=False,
                )
                pairs.append((query, answer))
    return system, pairs


def _parse_nli_response(raw: str, hypothesis: str) -> dict:
    try:
        result = yaml.safe_load(raw)
        if not isinstance(result, dict):
            raise ValueError()
    except Exception:
        logger.warning(f"Failed to parse NLI response for: {hypothesis}")
        return {"phrase": hypothesis, "status": "not_entailment", "evidence": []}

    status = result.get("status", "not_entailment")
    if status not in ("entailment", "not_entailment"):
        status = "not_entailment"

    return {
        "phrase": hypothesis,
        "status": status,
        "evidence": result.get("evidence", []) if status == "entailment" else [],
    }


def _classify_single_phrase(
    client, model: str,
    reference_phrases: list[str], hypothesis: str,
    system: str, pairs: list[tuple[str, str]],
    temperature: float,
    cost_tracker: Optional[CostTracker] = None,
) -> dict:
    query = yaml.dump(
        {"reference": reference_phrases, "hypothesis": hypothesis},
        sort_keys=False,
    )
    msgs = _build_messages(system, pairs, query)
    raw = call_openai(client, model, msgs, temperature, cost_tracker)
    return _parse_nli_response(raw, hypothesis)


async def _classify_single_phrase_async(
    client, model: str,
    reference_phrases: list[str], hypothesis: str,
    system: str, pairs: list[tuple[str, str]],
    temperature: float,
    cost_tracker: Optional[CostTracker] = None,
) -> dict:
    query = yaml.dump(
        {"reference": reference_phrases, "hypothesis": hypothesis},
        sort_keys=False,
    )
    msgs = _build_messages(system, pairs, query)
    raw = await call_openai_async(client, model, msgs, temperature, cost_tracker)
    return _parse_nli_response(raw, hypothesis)


def bidirectional_nli(
    client, model: str,
    candidate_phrases: list[str],
    reference_phrases: list[str],
    temperature: float,
    on_phrase_done=None,
    cost_tracker: Optional[CostTracker] = None,
) -> tuple[list[dict], list[dict]]:
    """Run entailment verification in both directions."""
    system, pairs = _load_nli_prompts()

    candidate_evidenced = []
    for phrase in candidate_phrases:
        result = _classify_single_phrase(
            client, model, reference_phrases, phrase, system, pairs,
            temperature, cost_tracker)
        candidate_evidenced.append(result)
        if on_phrase_done:
            on_phrase_done()

    reference_evidenced = []
    for phrase in reference_phrases:
        result = _classify_single_phrase(
            client, model, candidate_phrases, phrase, system, pairs,
            temperature, cost_tracker)
        reference_evidenced.append(result)
        if on_phrase_done:
            on_phrase_done()

    return candidate_evidenced, reference_evidenced


async def bidirectional_nli_async(
    client, model: str,
    candidate_phrases: list[str],
    reference_phrases: list[str],
    temperature: float,
    cost_tracker: Optional[CostTracker] = None,
) -> tuple[list[dict], list[dict]]:
    """Run entailment verification in both directions, all phrases concurrently."""
    system, pairs = _load_nli_prompts()

    cand_tasks = [
        _classify_single_phrase_async(
            client, model, reference_phrases, phrase, system, pairs,
            temperature, cost_tracker)
        for phrase in candidate_phrases
    ]
    ref_tasks = [
        _classify_single_phrase_async(
            client, model, candidate_phrases, phrase, system, pairs,
            temperature, cost_tracker)
        for phrase in reference_phrases
    ]

    all_results = await asyncio.gather(*cand_tasks, *ref_tasks)
    n_cand = len(candidate_phrases)
    candidate_evidenced = list(all_results[:n_cand])
    reference_evidenced = list(all_results[n_cand:])
    return candidate_evidenced, reference_evidenced


# ---------------------------------------------------------------------------
# Step 3: Scoring (pure Python, no LLM)
# ---------------------------------------------------------------------------

def _entailed_fraction(evidenced: list[dict]) -> float:
    if not evidenced:
        return float("nan")
    n_entailed = sum(1 for e in evidenced if e["status"] == "entailment")
    return n_entailed / len(evidenced)


def compute_radfact_scores(
    candidate_evidenced: list[dict],
    reference_evidenced: list[dict],
) -> dict[str, float]:
    precision = _entailed_fraction(candidate_evidenced)
    recall = _entailed_fraction(reference_evidenced)
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {
        "logical_precision": precision,
        "logical_recall": recall,
        "logical_f1": f1,
    }


# ---------------------------------------------------------------------------
# Main metric class
# ---------------------------------------------------------------------------

class RadFactCT:
    """RadFact-CT: LLM-based factual evaluation for CT radiology reports.

    Implements both RadFact +/- (default) and RadFact + (filter_negatives=True).
    Supports async concurrent evaluation via max_concurrent parameter.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        filter_negatives: bool = False,
        max_concurrent: int = 50,
    ):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError(
                "No API key provided. Set OPENAI_API_KEY or pass api_key=."
            )
        self.client = OpenAI(api_key=key)
        self._api_key = key
        self.model_name = model_name
        self.temperature = temperature
        self.filter_negatives = filter_negatives
        self.max_concurrent = max_concurrent
        self.cost_tracker = CostTracker(model_name)

    def _process_report(self, text: str) -> list[str]:
        phrases = report_to_phrases(
            self.client, self.model_name, text, self.temperature,
            self.cost_tracker)
        if self.filter_negatives and phrases:
            phrases = filter_negatives(
                self.client, self.model_name, phrases, self.temperature,
                self.cost_tracker)
        return phrases

    async def _process_report_async(self, aclient, text: str) -> list[str]:
        phrases = await report_to_phrases_async(
            aclient, self.model_name, text, self.temperature,
            self.cost_tracker)
        if self.filter_negatives and phrases:
            phrases = await filter_negatives_async(
                aclient, self.model_name, phrases, self.temperature,
                self.cost_tracker)
        return phrases

    def score_pair(
        self, hyp: str, ref: str, on_phrase_done=None,
    ) -> dict[str, Any]:
        hyp_phrases = self._process_report(hyp)
        ref_phrases = self._process_report(ref)

        if not hyp_phrases and not ref_phrases:
            return {
                "logical_precision": float("nan"),
                "logical_recall": float("nan"),
                "logical_f1": float("nan"),
                "hyp_phrases": [], "ref_phrases": [],
                "hyp_evidenced": [], "ref_evidenced": [],
            }

        hyp_evidenced, ref_evidenced = bidirectional_nli(
            self.client, self.model_name,
            hyp_phrases, ref_phrases,
            self.temperature,
            on_phrase_done=on_phrase_done,
            cost_tracker=self.cost_tracker,
        )

        scores = compute_radfact_scores(hyp_evidenced, ref_evidenced)
        scores["hyp_phrases"] = hyp_phrases
        scores["ref_phrases"] = ref_phrases
        scores["hyp_evidenced"] = hyp_evidenced
        scores["ref_evidenced"] = ref_evidenced
        return scores

    async def score_pair_async(
        self, aclient, hyp: str, ref: str,
    ) -> dict[str, Any]:
        hyp_phrases, ref_phrases = await asyncio.gather(
            self._process_report_async(aclient, hyp),
            self._process_report_async(aclient, ref),
        )

        if not hyp_phrases and not ref_phrases:
            return {
                "logical_precision": float("nan"),
                "logical_recall": float("nan"),
                "logical_f1": float("nan"),
                "hyp_phrases": [], "ref_phrases": [],
                "hyp_evidenced": [], "ref_evidenced": [],
            }

        hyp_evidenced, ref_evidenced = await bidirectional_nli_async(
            aclient, self.model_name,
            hyp_phrases, ref_phrases,
            self.temperature,
            cost_tracker=self.cost_tracker,
        )

        scores = compute_radfact_scores(hyp_evidenced, ref_evidenced)
        scores["hyp_phrases"] = hyp_phrases
        scores["ref_phrases"] = ref_phrases
        scores["hyp_evidenced"] = hyp_evidenced
        scores["ref_evidenced"] = ref_evidenced
        return scores

    def __call__(
        self, hyps: List[str], refs: List[str], on_sample_done=None,
    ) -> Tuple[dict, list]:
        return self.forward(hyps=hyps, refs=refs, on_sample_done=on_sample_done)

    async def forward_async(
        self, hyps: List[str], refs: List[str], on_sample_done=None,
    ) -> Tuple[dict, list]:
        sem = asyncio.Semaphore(self.max_concurrent)
        aclient = AsyncOpenAI(api_key=self._api_key)

        async def score_one(hyp, ref):
            async with sem:
                result = await self.score_pair_async(aclient, hyp, ref)
                if on_sample_done:
                    on_sample_done()
                return result

        try:
            tasks = [score_one(h, r) for h, r in zip(hyps, refs)]
            per_sample = list(await asyncio.gather(*tasks))
        finally:
            await aclient.close()
        return self._aggregate(per_sample)

    def forward(
        self, hyps: List[str], refs: List[str], on_sample_done=None,
    ) -> Tuple[dict, list]:
        if not isinstance(hyps, list) or not isinstance(refs, list):
            raise TypeError("hyps and refs must be lists")
        if len(hyps) != len(refs):
            raise ValueError("hyps and refs must have the same length")
        if len(hyps) == 0:
            return {"logical_precision": 0, "logical_recall": 0, "logical_f1": 0}, []

        self.cost_tracker.reset()

        if self.max_concurrent > 1:
            return asyncio.run(
                self.forward_async(hyps, refs, on_sample_done=on_sample_done))

        per_sample = []
        for hyp, ref in zip(hyps, refs):
            result = self.score_pair(hyp, ref)
            per_sample.append(result)
            if on_sample_done:
                on_sample_done()
        return self._aggregate(per_sample)

    @staticmethod
    def _aggregate(per_sample: list[dict]) -> Tuple[dict, list]:
        import numpy as np

        precisions = [r["logical_precision"] for r in per_sample]
        recalls = [r["logical_recall"] for r in per_sample]

        mean_p = float(np.nanmean(precisions)) * 100
        mean_r = float(np.nanmean(recalls)) * 100
        if mean_p + mean_r > 0:
            mean_f1 = 2 * mean_p * mean_r / (mean_p + mean_r)
        else:
            mean_f1 = 0.0

        aggregate = {
            "logical_precision": round(mean_p, 2),
            "logical_recall": round(mean_r, 2),
            "logical_f1": round(mean_f1, 2),
        }
        return aggregate, per_sample
