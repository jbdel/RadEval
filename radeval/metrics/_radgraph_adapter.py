from ._base import MetricBase


class RadGraphMetric(MetricBase):
    name = "radgraph"
    display_name = "RadGraph"

    def __init__(self):
        from .radgraph import F1RadGraph
        self._scorer = F1RadGraph(reward_level="all", model_type="radgraph-xl")

    def metric_keys(self, detailed=False):
        return ["radgraph_simple", "radgraph_partial", "radgraph_complete"]

    def compute(self, refs, hyps, per_sample=False, detailed=False,
                on_progress=None):
        """Override: detailed mode does not round (matches original behavior)."""
        result = self._scorer(refs=refs, hyps=hyps)
        f1_scores = result[0]
        per_level = result[1]

        if per_sample:
            return {
                "radgraph_simple": list(per_level[0]),
                "radgraph_partial": list(per_level[1]),
                "radgraph_complete": list(per_level[2]),
            }
        elif detailed:
            return {
                "radgraph_simple": f1_scores[0],
                "radgraph_partial": f1_scores[1],
                "radgraph_complete": f1_scores[2],
            }
        else:
            return {
                "radgraph_simple": round(f1_scores[0], 4),
                "radgraph_partial": round(f1_scores[1], 4),
                "radgraph_complete": round(f1_scores[2], 4),
            }
