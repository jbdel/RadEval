from .._base import MetricBase


class HopprCrimsonCTMetric(MetricBase):
    name = "hoppr_crimson_ct"
    display_name = "HopprCrimsonCT"
    is_api_based = True

    def __init__(self, provider="openai", model_name=None, openai_api_key=None,
                 gemini_api_key=None, batch_size=1, max_concurrent=50,
                 cache_dir=None, **kwargs):
        # cache_dir is injected by RadEval.__init__ for is_api_based metrics
        # but HopprCrimsonCT.__init__ does not accept it; drop it here rather
        # than widen the underlying class signature.
        from .hoppr_crimson_ct import CRIMSON_CT
        if CRIMSON_CT is None:
            raise ImportError(
                "HopprCrimsonCT failed to import — missing dependency. "
                "See radeval/metrics/hoppr_crimson_ct/__init__.py.")
        self._scorer = CRIMSON_CT(
            provider=provider, model_name=model_name,
            openai_api_key=openai_api_key, gemini_api_key=gemini_api_key,
            batch_size=batch_size, max_concurrent=max_concurrent,
        )

    @property
    def cost_tracker(self):
        return getattr(self._scorer, "cost_tracker", None)

    def metric_keys(self, detailed=False):
        return ["hoppr_crimson_ct"] + (["hoppr_crimson_ct_std"] if detailed else [])

    def _compute_raw(self, refs, hyps, on_progress=None):
        mean, std, samples, _ = self._scorer(
            refs, hyps, on_sample_done=on_progress)
        return {"hoppr_crimson_ct": {
            "aggregate": mean,
            "per_sample": samples,
            "detailed": {"hoppr_crimson_ct_std": std},
        }}
