from .._base import MetricBase


class GreenMetric(MetricBase):
    name = "green"
    display_name = "GREEN"

    def __init__(self):
        from .green import GREEN
        self._scorer = GREEN("StanfordAIMI/GREEN-radllama2-7b", output_dir=".")

    def metric_keys(self, detailed=False):
        return ["green"] + (["green_std"] if detailed else [])

    def _compute_raw(self, refs, hyps, on_progress=None):
        mean, std, sample_scores, _ = self._scorer(refs, hyps)
        return {"green": {
            "aggregate": mean,
            "per_sample": sample_scores,
            "detailed": {"green_std": std},
        }}


class MammoGreenMetric(MetricBase):
    name = "mammo_green"
    display_name = "MammoGREEN"
    is_api_based = True

    def __init__(self, model_name="gpt-4o-mini", openai_api_key=None,
                 gemini_api_key=None, max_concurrent=50):
        from .mammo_green import MammoGREEN
        self._scorer = MammoGREEN(
            model_name=model_name,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            max_concurrent=max_concurrent,
            output_dir=".",
        )

    @property
    def cost_tracker(self):
        return getattr(self._scorer, 'cost_tracker', None)

    def metric_keys(self, detailed=False):
        return ["mammo_green"] + (["mammo_green_std"] if detailed else [])

    def _compute_raw(self, refs, hyps, on_progress=None):
        mean, std, sample_scores, _ = self._scorer(
            refs, hyps, on_sample_done=on_progress)
        return {"mammo_green": {
            "aggregate": mean,
            "per_sample": sample_scores,
            "detailed": {"mammo_green_std": std},
        }}
