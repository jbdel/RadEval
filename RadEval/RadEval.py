import json
import os
import warnings
import logging


class RadEval:
    def __init__(self,
                 metrics: dict[str, dict] = None,
                 openai_api_key=None,
                 gemini_api_key=None,
                 per_sample=False,
                 detailed=False,
                 show_progress=True,
                 cache_dir=None):
        """
        Args:
            metrics: {"bleu": {}, "crimson": {"provider": "openai"}, ...}
            openai_api_key: shared key for LLM metrics (CRIMSON, RadFact-CT, MammoGREEN)
            gemini_api_key: shared key for Gemini-backed metrics
            per_sample: return list[float] per metric instead of aggregate
            detailed: return extra keys (label breakdowns, std, etc.)
            show_progress: show Rich progress bars
            cache_dir: shared cache dir for model downloads
        """
        from .metrics._registry import get_metric_class

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        warnings.filterwarnings("ignore")
        logging.getLogger("RadEval").setLevel(logging.ERROR)

        self.per_sample = per_sample
        self.detailed = detailed
        self.show_progress = show_progress

        self._active_metrics = []
        self.metric_keys = []

        for name, opts in (metrics or {}).items():
            kwargs = dict(opts)

            cls = get_metric_class(name)
            if getattr(cls, 'is_api_based', False):
                if openai_api_key:
                    kwargs.setdefault("openai_api_key", openai_api_key)
                if gemini_api_key:
                    kwargs.setdefault("gemini_api_key", gemini_api_key)
                if cache_dir:
                    kwargs.setdefault("cache_dir", cache_dir)

            metric = cls(**kwargs)
            self._active_metrics.append(metric)
            self.metric_keys.extend(metric.metric_keys(detailed=detailed))

    def __call__(self, refs, hyps):
        if not (isinstance(hyps, list) and isinstance(refs, list)):
            raise TypeError("hyps and refs must be of type list")
        if len(hyps) != len(refs):
            raise ValueError("hyps and refs lists don't have the same size")
        if len(refs) == 0:
            return {}
        return self.compute_scores(refs=refs, hyps=hyps)

    def _make_progress(self):
        from rich.progress import (
            Progress, SpinnerColumn, TextColumn, BarColumn,
            MofNCompleteColumn, TimeElapsedColumn,
        )
        return Progress(
            SpinnerColumn(), TextColumn("[bold]{task.description}"),
            BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
            disable=not self.show_progress,
        )

    def compute_scores(self, refs, hyps):
        """Run each metric sequentially, merge results."""
        scores = {}
        n = len(refs)

        with self._make_progress() as progress:
            metric_task = progress.add_task("Starting...",
                                            total=len(self._active_metrics))
            for metric in self._active_metrics:
                progress.update(metric_task,
                                description=f"Computing {metric.display_name}")

                sub_task = progress.add_task(
                    f"  [dim]{metric.display_name}", total=n)

                result = metric.compute(
                    refs, hyps,
                    per_sample=self.per_sample,
                    detailed=self.detailed,
                    on_progress=lambda _st=sub_task: progress.advance(_st),
                )

                progress.update(sub_task, completed=n)
                progress.remove_task(sub_task)
                scores.update(result)
                progress.advance(metric_task)

        return scores

    @classmethod
    def from_config(cls, path):
        """Load from a YAML or JSON config file."""
        import yaml
        from pathlib import Path
        raw = Path(path).read_text()
        config = json.loads(raw) if path.endswith(".json") else yaml.safe_load(raw)
        return cls(
            metrics=config.get("metrics", {}),
            per_sample=config.get("output", {}).get("mode") == "per_sample",
            detailed=config.get("output", {}).get("mode") == "detailed",
        )


def main():
    evaluator = RadEval(metrics={"bleu": {}, "rouge": {}})
    refs = ["No acute cardiopulmonary process."]
    hyps = ["No acute cardiopulmonary process."]
    results = evaluator(refs=refs, hyps=hyps)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
