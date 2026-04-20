from .._base import MetricBase


class RadFactCTMetric(MetricBase):
    name = "radfact_ct"
    display_name = "RadFact-CT"
    is_api_based = True

    def __init__(self, model_name="gpt-4o-mini", openai_api_key=None,
                 filter_negatives=False, max_concurrent=50):
        from .radfact_ct import RadFactCT
        self._scorer = RadFactCT(
            model_name=model_name,
            openai_api_key=openai_api_key,
            filter_negatives=filter_negatives,
            max_concurrent=max_concurrent,
        )

    @property
    def cost_tracker(self):
        return getattr(self._scorer, "cost_tracker", None)

    def metric_keys(self, detailed=False):
        return ["radfact_ct_precision", "radfact_ct_recall", "radfact_ct_f1"]

    def _compute_raw(self, refs, hyps, on_progress=None):
        agg, per_sample = self._scorer(hyps, refs, on_sample_done=on_progress)
        return {
            "radfact_ct_precision": {
                "aggregate": agg["logical_precision"],
                "per_sample": [s["logical_precision"] for s in per_sample],
            },
            "radfact_ct_recall": {
                "aggregate": agg["logical_recall"],
                "per_sample": [s["logical_recall"] for s in per_sample],
            },
            "radfact_ct_f1": {
                "aggregate": agg["logical_f1"],
                "per_sample": [s["logical_f1"] for s in per_sample],
            },
        }
