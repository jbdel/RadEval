from .._base import MetricBase


class SRRBertMetric(MetricBase):
    name = "srrbert"
    display_name = "SRR-BERT"

    def __init__(self):
        import nltk
        nltk.download('punkt_tab', quiet=True)
        from .srr_bert import SRRBert
        self._scorer = SRRBert(model_type="leaves_with_statuses")

    def metric_keys(self, detailed=False):
        keys = ["srrbert_weighted_f1", "srrbert_weighted_precision",
                "srrbert_weighted_recall"]
        if detailed:
            keys.append("srrbert_label_scores")
        return keys

    def _compute_raw(self, refs, hyps, on_progress=None):
        classification_dict, sample_precision, sample_recall, sample_f1 = \
            self._scorer.evaluate(refs, hyps, on_sample_done=on_progress)

        # build detailed label scores
        label_names = [
            label for label, idx in sorted(
                self._scorer.mapping.items(), key=lambda x: x[1])
        ]
        label_scores = {}
        for label in label_names:
            if label in classification_dict:
                f1 = classification_dict[label]["f1-score"]
                support = classification_dict[label]["support"]
                if f1 > 0 or support > 0:
                    label_scores[label] = {
                        "f1-score": f1,
                        "precision": classification_dict[label]["precision"],
                        "recall": classification_dict[label]["recall"],
                        "support": support,
                    }

        weighted = classification_dict["weighted avg"]
        return {
            "srrbert_weighted_f1": {
                "aggregate": weighted["f1-score"],
                "per_sample": sample_f1.tolist(),
            },
            "srrbert_weighted_precision": {
                "aggregate": weighted["precision"],
                "per_sample": sample_precision.tolist(),
            },
            "srrbert_weighted_recall": {
                "aggregate": weighted["recall"],
                "per_sample": sample_recall.tolist(),
            },
            "srrbert_label_scores": {
                "aggregate": label_scores,
                "per_sample": label_scores,  # same in per_sample mode
                "detailed": {},
            },
        }

    def _format_output(self, raw, per_sample, detailed):
        """Custom formatting: label_scores only appears in detailed mode."""
        out = {}
        for key, data in raw.items():
            if key == "srrbert_label_scores":
                if detailed:
                    out[key] = data["aggregate"]
                continue
            if per_sample:
                out[key] = data["per_sample"]
            else:
                agg = data["aggregate"]
                out[key] = round(agg, 4) if isinstance(agg, float) else agg
        return out
