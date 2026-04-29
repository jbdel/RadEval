from .._base import MetricBase


class CrimsonMetric(MetricBase):
    name = "crimson"
    display_name = "CRIMSON"
    is_api_based = True

    def __init__(self, provider="hf", model_name=None, openai_api_key=None,
                 gemini_api_key=None, batch_size=1, max_concurrent=50,
                 cache_dir=None):
        from .crimson import CRIMSON
        self._scorer = CRIMSON(
            provider=provider, model_name=model_name,
            openai_api_key=openai_api_key, gemini_api_key=gemini_api_key,
            batch_size=batch_size, max_concurrent=max_concurrent,
            cache_dir=cache_dir,
        )

    @property
    def cost_tracker(self):
        return getattr(self._scorer, "cost_tracker", None)

    def metric_keys(self, detailed=False):
        return ["crimson"] + (["crimson_std"] if detailed else [])

    def _compute_raw(self, refs, hyps, on_progress=None):
        mean, std, samples, _ = self._scorer(
            refs, hyps, on_sample_done=on_progress)
        return {"crimson": {
            "aggregate": mean,
            "per_sample": samples,
            "detailed": {"crimson_std": std},
        }}
