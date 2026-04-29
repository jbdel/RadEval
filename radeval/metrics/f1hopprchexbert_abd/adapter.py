from .._base import MetricBase


class F1HopprCheXbertAbdMetric(MetricBase):
    name = "f1hopprchexbert_abd"
    display_name = "F1HopprCheXbertAbd"

    def __init__(self, **kwargs):
        from .f1hopprchexbert_abd import HopprF1CheXbertAbd
        if HopprF1CheXbertAbd is None:
            raise ImportError(
                "HopprF1CheXbertAbd failed to import — missing dependency or "
                "checkpoint. See RadEval/metrics/f1hopprchexbert_abd/__init__.py.")
        self._scorer = HopprF1CheXbertAbd()

    def metric_keys(self, detailed=False):
        return [
            "f1hopprchexbert_abd_accuracy", "f1hopprchexbert_abd_micro_f1",
            "f1hopprchexbert_abd_macro_f1", "f1hopprchexbert_abd_weighted_f1",
        ]

    def compute(self, refs, hyps, per_sample=False, detailed=False,
                on_progress=None):
        """Override: per_sample mode returns different keys than default."""
        accuracy, sample_acc, report = self._scorer(
            hyps, refs, on_batch_done=on_progress)

        if per_sample:
            return {
                "f1hopprchexbert_abd_sample_acc": (
                    sample_acc.tolist() if hasattr(sample_acc, 'tolist')
                    else list(sample_acc)),
            }
        elif detailed:
            labels = {k: v["f1-score"] for k, v in list(report.items())[:-4]}
            return {
                "f1hopprchexbert_abd_accuracy": round(accuracy, 4),
                "f1hopprchexbert_abd_micro_f1": round(report["micro avg"]["f1-score"], 4),
                "f1hopprchexbert_abd_macro_f1": round(report["macro avg"]["f1-score"], 4),
                "f1hopprchexbert_abd_weighted_f1": round(report["weighted avg"]["f1-score"], 4),
                "f1hopprchexbert_abd_label_scores_f1": labels,
            }
        else:
            return {
                "f1hopprchexbert_abd_accuracy": round(accuracy, 4),
                "f1hopprchexbert_abd_micro_f1": round(report["micro avg"]["f1-score"], 4),
                "f1hopprchexbert_abd_macro_f1": round(report["macro avg"]["f1-score"], 4),
                "f1hopprchexbert_abd_weighted_f1": round(report["weighted avg"]["f1-score"], 4),
            }
