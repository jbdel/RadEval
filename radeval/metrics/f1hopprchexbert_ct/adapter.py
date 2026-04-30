from .._base import MetricBase


class F1HopprCheXbertCTMetric(MetricBase):
    name = "f1hopprchexbert_ct"
    display_name = "F1HopprCheXbertCT"

    def __init__(self, **kwargs):
        from .f1hopprchexbert_ct import HopprF1CheXbertCT
        if HopprF1CheXbertCT is None:
            raise ImportError(
                "HopprF1CheXbertCT failed to import — missing dependency or "
                "checkpoint. See radeval/metrics/f1hopprchexbert_ct/__init__.py.")
        self._scorer = HopprF1CheXbertCT()

    def metric_keys(self, detailed=False):
        return [
            "f1hopprchexbert_ct_accuracy", "f1hopprchexbert_ct_micro_f1",
            "f1hopprchexbert_ct_macro_f1", "f1hopprchexbert_ct_weighted_f1",
        ]

    def compute(self, refs, hyps, per_sample=False, detailed=False,
                on_progress=None):
        """Override: per_sample mode returns different keys than default."""
        accuracy, sample_acc, report = self._scorer(
            hyps, refs, on_batch_done=on_progress)

        if per_sample:
            return {
                "f1hopprchexbert_ct_sample_acc": (
                    sample_acc.tolist() if hasattr(sample_acc, 'tolist')
                    else list(sample_acc)),
            }
        elif detailed:
            labels = {k: v["f1-score"] for k, v in list(report.items())[:-4]}
            return {
                "f1hopprchexbert_ct_accuracy": round(accuracy, 4),
                "f1hopprchexbert_ct_micro_f1": round(report["micro avg"]["f1-score"], 4),
                "f1hopprchexbert_ct_macro_f1": round(report["macro avg"]["f1-score"], 4),
                "f1hopprchexbert_ct_weighted_f1": round(report["weighted avg"]["f1-score"], 4),
                "f1hopprchexbert_ct_label_scores_f1": labels,
            }
        else:
            return {
                "f1hopprchexbert_ct_accuracy": round(accuracy, 4),
                "f1hopprchexbert_ct_micro_f1": round(report["micro avg"]["f1-score"], 4),
                "f1hopprchexbert_ct_macro_f1": round(report["macro avg"]["f1-score"], 4),
                "f1hopprchexbert_ct_weighted_f1": round(report["weighted avg"]["f1-score"], 4),
            }
