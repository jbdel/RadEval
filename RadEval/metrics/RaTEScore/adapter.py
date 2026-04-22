from .._base import MetricBase


class RaTEScoreMetric(MetricBase):
    name = "ratescore"
    display_name = "RaTEScore"

    def __init__(self):
        from .score import RaTEScore
        self._scorer = RaTEScore()

    def metric_keys(self, detailed=False):
        return ["ratescore"]

    def _compute_raw(self, refs, hyps, on_progress=None):
        rate_scores, _, _ = self._scorer.compute_score(
            candidate_list=hyps, reference_list=refs,
            on_sample_done=on_progress)
        return {"ratescore": {
            "aggregate": sum(rate_scores) / len(rate_scores),
            "per_sample": rate_scores,
        }}
