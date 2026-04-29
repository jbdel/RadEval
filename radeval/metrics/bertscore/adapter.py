from .._base import MetricBase


class BertScoreMetric(MetricBase):
    name = "bertscore"
    display_name = "BERTScore"

    def __init__(self):
        from .bertscore import BertScore
        self._scorer = BertScore(
            model_type='distilbert-base-uncased', num_layers=5)

    def metric_keys(self, detailed=False):
        return ["bertscore"]

    def _compute_raw(self, refs, hyps, on_progress=None):
        mean_f1, sample_scores = self._scorer(
            refs, hyps, on_batch_done=on_progress)
        return {"bertscore": {
            "aggregate": mean_f1,
            "per_sample": sample_scores,
        }}
