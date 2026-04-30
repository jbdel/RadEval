from .._base import MetricBase


class TemporalF1Metric(MetricBase):
    name = "temporal"
    display_name = "Temporal F1"

    def __init__(self):
        import stanza
        stanza.download('en', package='radiology',
                        processors={'ner': 'radiology'})
        from .f1temporal import F1Temporal
        self._F1Temporal = F1Temporal

    def metric_keys(self, detailed=False):
        return ["temporal_f1"]

    def _compute_raw(self, refs, hyps, on_progress=None):
        result = self._F1Temporal(
            predictions=hyps, references=refs,
            on_sample_done=on_progress)
        return {"temporal_f1": {
            "aggregate": result["f1"],
            "per_sample": result["sample_scores"],
        }}
