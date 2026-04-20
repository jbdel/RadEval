from .._base import MetricBase


class RougeMetric(MetricBase):
    name = "rouge"
    display_name = "ROUGE"

    def __init__(self):
        from rouge_score import rouge_scorer
        self._rouge_types = ["rouge1", "rouge2", "rougeL"]
        self._scorer = rouge_scorer.RougeScorer(
            self._rouge_types, use_stemmer=True)

    def metric_keys(self, detailed=False):
        return ["rouge1", "rouge2", "rougeL"]

    def _compute_raw(self, refs, hyps, on_progress=None):
        raw_scores = [
            self._scorer.score(ref, hyp) for ref, hyp in zip(refs, hyps)
        ]
        result = {}
        for rt in self._rouge_types:
            f1s = [s[rt].fmeasure for s in raw_scores]
            result[rt] = {
                "aggregate": sum(f1s) / len(f1s),
                "per_sample": f1s,
            }
        return result
