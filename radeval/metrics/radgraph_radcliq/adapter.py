from .._base import MetricBase


class RadGraphRadCliQMetric(MetricBase):
    name = "radgraph_radcliq"
    display_name = "RadGraph-RadCliQ"

    def __init__(self):
        from .radgraph_radcliq import RadGraphRadCliQ
        self._scorer = RadGraphRadCliQ()

    def metric_keys(self, detailed=False):
        return ["radgraph_radcliq"]

    def _compute_raw(self, refs, hyps, on_progress=None):
        mean_rg, sample_rg = self._scorer(
            hyps, refs, on_sample_done=on_progress)
        return {"radgraph_radcliq": {
            "aggregate": mean_rg,
            "per_sample": sample_rg,
        }}
