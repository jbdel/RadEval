"""TRL-compatible reward function wrappers for RadEval metrics.

TRL is not a required dependency — install it via ``pip install RadEval[rl]``.
Only the reward-function interface convention is adopted here; none of this
module imports ``trl`` at module load time.

Public surface:

- :func:`make_reward_fn` — wrap a RadEval metric as a callable suitable for
  ``trl.GRPOTrainer`` (and other TRL trainers that consume a reward-function
  callable, e.g. RLOO; not separately validated in this module's tests).
- :func:`validate_rewards` — validate a list of reward values, raising on
  NaN/Inf while passing ``None`` entries through unchanged (TRL uses
  ``None`` to skip samples in multi-task routing).
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Optional


def validate_rewards(values: list, metric_name: str) -> list:
    """Normalize and validate a list of per-sample reward values.

    * ``None`` entries are preserved unchanged — TRL uses ``None`` to skip
      samples in multi-task routing; coercing them would break that.
    * Other entries are reduced to a scalar via ``.item()`` when available
      (handles 0-D and 1-element 1-D torch tensors uniformly), then cast
      through ``float()``. Catches ``numpy.nan`` / numpy scalars / Python
      floats the same way.
    * Raises ``ValueError`` naming the metric and sample index on NaN/Inf.

    Returns a new list (same length) suitable for returning from a TRL
    reward function.
    """
    normalized: list = []
    for i, v in enumerate(values):
        if v is None:
            normalized.append(None)
            continue
        scalar = v.item() if hasattr(v, "item") else v
        f = float(scalar)
        if math.isnan(f) or math.isinf(f):
            raise ValueError(
                f"{metric_name}: non-finite reward at sample index {i} "
                f"(got {f!r}). Fix the upstream scorer or the input."
            )
        normalized.append(f)
    return normalized


def _last_assistant_content(messages: list[dict]) -> str:
    """Extract ``content`` from the last assistant message in ``messages``.

    Best-effort for the common OpenAI-style layout. May raise ``KeyError``
    or ``IndexError`` if the layout differs; callers wrap and translate
    those into a ``TypeError`` with upstream-preprocess guidance.
    """
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            return msg["content"]
    # Fall through: no assistant turn found — use last message's content.
    return messages[-1]["content"]


def _as_strings(completions: list) -> list[str]:
    """Coerce TRL completions to ``list[str]``.

    Accepts:

    * ``list[str]`` — returned as-is.
    * ``list[list[dict]]`` (OpenAI-style message layout) — extracts
      ``content`` from the last assistant message per conversation.
      Fires a ``UserWarning`` so users know the heuristic is in play.
      This is filterable via the standard ``warnings`` module
      (e.g. ``warnings.simplefilter("once")``).

    Raises:
        TypeError: for any other shape, and for ``list[dict]`` entries
            that lack the expected ``role``/``content`` keys. The
            recommended fix is to preprocess completion text upstream in
            the dataset/collator pipeline (not via ``score_transform``,
            which operates on scores).
    """
    if not completions:
        return []
    first = completions[0]
    if isinstance(first, str):
        return list(completions)
    if isinstance(first, list) and first and isinstance(first[0], dict):
        warnings.warn(
            "make_reward_fn: extracting 'content' from the last assistant "
            "message in list[dict] completions (best-effort heuristic). "
            "If your dataset uses a different layout, preprocess the "
            "completion text upstream in your dataset/collator pipeline.",
            UserWarning,
            stacklevel=2,
        )
        try:
            return [_last_assistant_content(msgs) for msgs in completions]
        except (KeyError, IndexError) as exc:
            raise TypeError(
                "make_reward_fn: list[dict] completions lack expected "
                "'role'/'content' keys. Preprocess your completions "
                "upstream in the dataset/collator pipeline."
            ) from exc
    raise TypeError(
        f"make_reward_fn: unsupported completions shape "
        f"(got {type(first).__name__}); expected list[str] or "
        f"list[list[dict]]. Preprocess your completions upstream in "
        f"the dataset/collator pipeline."
    )


def make_reward_fn(
    metric: str,
    reference_column: str = "ground_truth",
    key: Optional[str] = None,
    score_transform: Optional[Callable[[float], float]] = None,
    **metric_kwargs: Any,
):
    """Wrap a RadEval metric as a TRL-compatible reward function.

    Args:
        metric: Metric name registered in RadEval (e.g. ``"bleu"``,
            ``"bertscore"``, ``"f1chexbert"``, ``"radgraph"``, ``"radcliq"``).
        reference_column: Name of the dataset column carrying reference
            reports. TRL forwards all non-``prompt`` dataset columns to
            the reward function as kwargs; by convention we default to
            ``"ground_truth"``.
        key: Which per-sample output key to return when the metric's
            ``compute(..., per_sample=True)`` result has more than one
            key (e.g. F1CheXbert → ``"f1chexbert_sample_acc_5"`` or
            ``"f1chexbert_sample_acc_all"``; RadGraph → ``"radgraph_simple"``,
            ``"radgraph_partial"``, ``"radgraph_complete"``). Required for
            any metric that produces multi-key per-sample output; the
            first call raises ``ValueError`` listing valid keys if it is
            omitted.
        score_transform: Optional per-sample transform (e.g.
            ``lambda x: max(0.0, min(1.0, (x - 0.5) * 2))`` to clip and
            rescale). Applied only to non-``None`` entries.
        **metric_kwargs: Forwarded to the metric adapter's ``__init__``.

    Returns:
        Callable with signature ``reward_fn(completions, **kwargs) -> list[float|None]``.
        Absorbs additional TRL kwargs (``prompts``, ``completion_ids``,
        ``trainer_state``, ``log_extra``, ``log_metric``, ``environments``,
        and extra dataset columns) via ``**kwargs`` without error.

    Example:
        >>> from RadEval.rewards import make_reward_fn
        >>> reward = make_reward_fn("bleu")
        >>> reward(completions=["hello world"], ground_truth=["hello world"])  # doctest: +SKIP
        [1.0]
    """
    from .metrics._registry import get_metric_class

    cls = get_metric_class(metric)

    if getattr(cls, "is_api_based", False):
        warnings.warn(
            f"'{metric}' makes API calls per sample — very slow and expensive "
            f"as an RL reward. Consider fast local metrics: bleu, rouge, "
            f"bertscore, radeval_bertscore, f1chexbert, f1radbert_ct.",
            UserWarning, stacklevel=2,
        )

    scorer = cls(**metric_kwargs)

    def reward_fn(completions, **kwargs):
        if reference_column not in kwargs:
            raise KeyError(
                f"make_reward_fn: expected reference column "
                f"{reference_column!r} in reward-fn kwargs (got "
                f"{sorted(kwargs)!r}). Either pass references via that "
                f"column or set reference_column=<your column name>."
            )
        refs = kwargs[reference_column]
        hyps = _as_strings(completions)
        result = scorer.compute(refs=refs, hyps=hyps, per_sample=True)

        # Pick the right per-sample key. Some adapters return multi-key
        # per-sample dicts whose canonical default is ambiguous
        # (e.g. F1CheXbert: sample_acc_5 vs sample_acc_all) — require
        # an explicit `key=` on first call so we never silently miscompute.
        if len(result) == 1:
            (only_key,) = result.keys()
            chosen = only_key if key is None else key
        elif key is None:
            raise ValueError(
                f"Metric '{metric}' returns multiple per-sample keys: "
                f"{sorted(result.keys())}. "
                f"Pass key=<one of these> to make_reward_fn()."
            )
        else:
            chosen = key

        if chosen not in result:
            raise KeyError(
                f"Metric '{metric}' key={chosen!r} not in per-sample "
                f"output keys {sorted(result.keys())}."
            )

        values = result[chosen]
        if score_transform is not None:
            values = [
                None if v is None else score_transform(v) for v in values
            ]
        return validate_rewards(values, metric_name=metric)

    return reward_fn
