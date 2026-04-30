from .._base import MetricBase


class F1HopprCheXbertMetric(MetricBase):
    name = "f1hopprchexbert"
    display_name = "F1HopprCheXbert"

    def __init__(self, **kwargs):
        from .f1hopprchexbert import HopprF1CheXbert
        if HopprF1CheXbert is None:
            raise ImportError(
                "HopprF1CheXbert failed to import — missing dependency or "
                "checkpoint. See radeval/metrics/f1hopprchexbert/__init__.py.")
        self._scorer = HopprF1CheXbert()

    def metric_keys(self, detailed=False):
        return [
            "f1hopprchexbert_5_micro_f1", "f1hopprchexbert_all_micro_f1",
            "f1hopprchexbert_5_macro_f1", "f1hopprchexbert_all_macro_f1",
            "f1hopprchexbert_5_weighted_f1", "f1hopprchexbert_all_weighted_f1",
        ]

    def compute(self, refs, hyps, per_sample=False, detailed=False,
                on_progress=None):
        """Override: per_sample mode returns different keys than default."""
        _, _, cr_all, cr_5, sample_acc_full, sample_acc_5 = \
            self._scorer.forward(hyps, refs, on_batch_done=on_progress)

        if per_sample:
            return {
                "f1hopprchexbert_sample_acc_5": (
                    sample_acc_5.tolist() if hasattr(sample_acc_5, 'tolist')
                    else list(sample_acc_5)),
                "f1hopprchexbert_sample_acc_all": (
                    sample_acc_full.tolist() if hasattr(sample_acc_full, 'tolist')
                    else list(sample_acc_full)),
            }
        elif detailed:
            labels_5 = {k: v["f1-score"] for k, v in list(cr_5.items())[:-4]}
            labels_all = {k: v["f1-score"] for k, v in list(cr_all.items())[:-4]}
            return {
                "f1hopprchexbert_5_micro_f1": cr_5["micro avg"]["f1-score"],
                "f1hopprchexbert_all_micro_f1": cr_all["micro avg"]["f1-score"],
                "f1hopprchexbert_5_macro_f1": cr_5["macro avg"]["f1-score"],
                "f1hopprchexbert_all_macro_f1": cr_all["macro avg"]["f1-score"],
                "f1hopprchexbert_5_weighted_f1": cr_5["weighted avg"]["f1-score"],
                "f1hopprchexbert_all_weighted_f1": cr_all["weighted avg"]["f1-score"],
                "f1hopprchexbert_label_scores_f1": {
                    "f1hopprchexbert_5": labels_5,
                    "f1hopprchexbert_all": labels_all},
            }
        else:
            return {
                "f1hopprchexbert_5_micro_f1": round(cr_5["micro avg"]["f1-score"], 4),
                "f1hopprchexbert_all_micro_f1": round(cr_all["micro avg"]["f1-score"], 4),
                "f1hopprchexbert_5_macro_f1": round(cr_5["macro avg"]["f1-score"], 4),
                "f1hopprchexbert_all_macro_f1": round(cr_all["macro avg"]["f1-score"], 4),
                "f1hopprchexbert_5_weighted_f1": round(cr_5["weighted avg"]["f1-score"], 4),
                "f1hopprchexbert_all_weighted_f1": round(cr_all["weighted avg"]["f1-score"], 4),
            }
