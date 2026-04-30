from .._base import MetricBase


class BleuMetric(MetricBase):
    name = "bleu"
    display_name = "BLEU"

    def __init__(self):
        from .bleu import Bleu
        self._scorer = Bleu(n=4)
        self._scorer_1 = Bleu(n=1)
        self._scorer_2 = Bleu(n=2)
        self._scorer_3 = Bleu(n=3)

    def metric_keys(self, detailed=False):
        return ["bleu"] + (["bleu_1", "bleu_2", "bleu_3"] if detailed else [])

    def _compute_raw(self, refs, hyps, on_progress=None):
        score, samples = self._scorer(refs, hyps)
        return {"bleu": {
            "aggregate": score,
            "per_sample": samples,
            "detailed": {
                "bleu_1": self._scorer_1(refs, hyps)[0],
                "bleu_2": self._scorer_2(refs, hyps)[0],
                "bleu_3": self._scorer_3(refs, hyps)[0],
            },
        }}
