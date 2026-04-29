from .._base import MetricBase


class F1CheXbertMetric(MetricBase):
    name = "f1chexbert"
    display_name = "F1CheXbert"

    def __init__(self):
        from .f1chexbert import F1CheXbert
        self._scorer = F1CheXbert()

    def metric_keys(self, detailed=False):
        return [
            "f1chexbert_5_micro_f1", "f1chexbert_all_micro_f1",
            "f1chexbert_5_macro_f1", "f1chexbert_all_macro_f1",
            "f1chexbert_5_weighted_f1", "f1chexbert_all_weighted_f1",
        ]

    def compute(self, refs, hyps, per_sample=False, detailed=False,
                on_progress=None):
        """Override: per_sample mode returns different keys than default."""
        _, _, cr_all, cr_5, sample_acc_full, sample_acc_5 = \
            self._scorer.forward(hyps, refs, on_batch_done=on_progress)

        if per_sample:
            return {
                "f1chexbert_sample_acc_5": (
                    sample_acc_5.tolist() if hasattr(sample_acc_5, 'tolist')
                    else list(sample_acc_5)),
                "f1chexbert_sample_acc_all": (
                    sample_acc_full.tolist() if hasattr(sample_acc_full, 'tolist')
                    else list(sample_acc_full)),
            }
        elif detailed:
            labels_5 = {k: v["f1-score"] for k, v in list(cr_5.items())[:-4]}
            labels_all = {k: v["f1-score"] for k, v in list(cr_all.items())[:-4]}
            return {
                "f1chexbert_5_micro_f1": cr_5["micro avg"]["f1-score"],
                "f1chexbert_all_micro_f1": cr_all["micro avg"]["f1-score"],
                "f1chexbert_5_macro_f1": cr_5["macro avg"]["f1-score"],
                "f1chexbert_all_macro_f1": cr_all["macro avg"]["f1-score"],
                "f1chexbert_5_weighted_f1": cr_5["weighted avg"]["f1-score"],
                "f1chexbert_all_weighted_f1": cr_all["weighted avg"]["f1-score"],
                "f1chexbert_label_scores_f1": {
                    "f1chexbert_5": labels_5, "f1chexbert_all": labels_all},
            }
        else:
            return {
                "f1chexbert_5_micro_f1": round(cr_5["micro avg"]["f1-score"], 4),
                "f1chexbert_all_micro_f1": round(cr_all["micro avg"]["f1-score"], 4),
                "f1chexbert_5_macro_f1": round(cr_5["macro avg"]["f1-score"], 4),
                "f1chexbert_all_macro_f1": round(cr_all["macro avg"]["f1-score"], 4),
                "f1chexbert_5_weighted_f1": round(cr_5["weighted avg"]["f1-score"], 4),
                "f1chexbert_all_weighted_f1": round(cr_all["weighted avg"]["f1-score"], 4),
            }
