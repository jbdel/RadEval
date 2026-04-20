from .._base import MetricBase


class RadEvalBertScoreMetric(MetricBase):
    name = "radeval_bertscore"
    display_name = "RadEval-BERTScore"

    def __init__(self):
        from .radevalbertscore import RadEvalBERTScorer
        self._scorer = RadEvalBERTScorer(
            model_type="IAMJB/RadEvalModernBERT",
            num_layers=22,
            use_fast_tokenizer=True,
            rescale_with_baseline=False)

    def metric_keys(self, detailed=False):
        return ["radeval_bertscore"]

    def _compute_raw(self, refs, hyps, on_progress=None):
        mean_f1, f1_tensor = self._scorer.score(
            refs=refs, hyps=hyps, on_batch_done=on_progress)
        return {"radeval_bertscore": {
            "aggregate": mean_f1,
            "per_sample": f1_tensor.tolist(),
        }}
