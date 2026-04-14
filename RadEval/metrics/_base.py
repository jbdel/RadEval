"""Base class for all RadEval metrics."""
from abc import ABC, abstractmethod
from typing import Any


class MetricBase(ABC):
    """Base class for all RadEval metrics.

    Simple metrics (~12 of 16): override _compute_raw() — the default
    compute() handles output-mode branching automatically.

    Complex metrics (~4: f1chexbert, f1radbert_ct, srrbert, radgraph):
    override compute() directly because per_sample mode uses different
    output keys than default mode.
    """

    name: str
    display_name: str
    is_api_based: bool = False

    @abstractmethod
    def metric_keys(self, detailed: bool = False) -> list[str]:
        """Return output keys this metric produces in the given mode."""
        ...

    def compute(self, refs, hyps, per_sample=False, detailed=False,
                on_progress=None) -> dict[str, Any]:
        """Public entry point.

        Default: calls _compute_raw(), then _format_output().
        Override this for metrics with mode-dependent output keys.
        """
        raw = self._compute_raw(refs, hyps, on_progress=on_progress)
        return self._format_output(raw, per_sample=per_sample, detailed=detailed)

    def _compute_raw(self, refs, hyps, on_progress=None) -> dict[str, Any]:
        """Override this for simple metrics.

        Return: {"bleu": {"aggregate": 0.36, "per_sample": [0.85, ...],
                          "detailed": {"bleu_1": 0.55}}}

        "detailed" key is optional. "per_sample" is required.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _compute_raw() or "
            f"override compute() directly."
        )

    def _format_output(self, raw, per_sample, detailed):
        """Standard output formatting — picks the right slice from raw data."""
        out = {}
        for key, data in raw.items():
            if per_sample:
                out[key] = data["per_sample"]
            elif detailed:
                agg = data["aggregate"]
                out[key] = round(agg, 4) if isinstance(agg, float) else agg
                for dk, dv in data.get("detailed", {}).items():
                    out[dk] = round(dv, 4) if isinstance(dv, float) else dv
            else:
                agg = data["aggregate"]
                out[key] = round(agg, 4) if isinstance(agg, float) else agg
        return out
