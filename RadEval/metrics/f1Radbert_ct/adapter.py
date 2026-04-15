from .._base import MetricBase


class F1RadbertCTMetric(MetricBase):
    name = "f1radbert_ct"
    display_name = "F1RadBERT-CT"

    def __init__(self):
        from .f1Radbert_ct import F1RadbertCT
        self._scorer = F1RadbertCT(
            model_id="IAMJB/RadBERT-CT", threshold=0.5, batch_size=16)

    def metric_keys(self, detailed=False):
        return [
            "f1radbert_ct_accuracy", "f1radbert_ct_micro_f1",
            "f1radbert_ct_macro_f1", "f1radbert_ct_weighted_f1",
        ]

    def compute(self, refs, hyps, per_sample=False, detailed=False,
                on_progress=None):
        """Override: per_sample mode returns different keys than default."""
        accuracy, sample_acc, report = self._scorer(
            hyps, refs, on_batch_done=on_progress)

        if per_sample:
            return {
                "f1radbert_ct_sample_acc": (
                    sample_acc.tolist() if hasattr(sample_acc, 'tolist')
                    else list(sample_acc)),
            }
        elif detailed:
            labels = {k: v["f1-score"] for k, v in list(report.items())[:-4]}
            return {
                "f1radbert_ct_accuracy": round(accuracy, 4),
                "f1radbert_ct_micro_f1": round(report["micro avg"]["f1-score"], 4),
                "f1radbert_ct_macro_f1": round(report["macro avg"]["f1-score"], 4),
                "f1radbert_ct_weighted_f1": round(report["weighted avg"]["f1-score"], 4),
                "f1radbert_ct_label_scores_f1": labels,
            }
        else:
            return {
                "f1radbert_ct_accuracy": round(accuracy, 4),
                "f1radbert_ct_micro_f1": round(report["micro avg"]["f1-score"], 4),
                "f1radbert_ct_macro_f1": round(report["macro avg"]["f1-score"], 4),
                "f1radbert_ct_weighted_f1": round(report["weighted avg"]["f1-score"], 4),
            }
