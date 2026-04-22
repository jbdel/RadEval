from .._base import MetricBase


class RadCliQMetric(MetricBase):
    name = "radcliq"
    display_name = "RadCliQ-v1"

    def __init__(self):
        from .radcliq import CompositeMetric
        self._scorer = CompositeMetric()

    def metric_keys(self, detailed=False):
        return ["radcliq_v1"]

    def _compute_raw(self, refs, hyps, on_progress=None):
        mean_scores, detail_scores = self._scorer.predict(refs, hyps)
        return {"radcliq_v1": {
            "aggregate": mean_scores,
            "per_sample": detail_scores.tolist(),
        }}
