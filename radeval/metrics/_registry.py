"""Lazy metric registry — maps metric names to adapter classes.

Nothing is imported until get_metric_class() is called for a specific name.
This preserves lazy loading: `import radeval` loads zero metric dependencies.
"""

METRIC_REGISTRY: dict[str, tuple[str, str]] = {
    "bleu":              ("radeval.metrics.bleu.adapter",              "BleuMetric"),
    "rouge":             ("radeval.metrics.rouge.adapter",             "RougeMetric"),
    "bertscore":         ("radeval.metrics.bertscore.adapter",         "BertScoreMetric"),
    "radeval_bertscore": ("radeval.metrics.radevalbertscore.adapter",  "RadEvalBertScoreMetric"),
    "f1chexbert":        ("radeval.metrics.f1chexbert.adapter",        "F1CheXbertMetric"),
    "f1radbert_ct":      ("radeval.metrics.f1Radbert_ct.adapter",      "F1RadbertCTMetric"),
    "radgraph":          ("radeval.metrics._radgraph_adapter",         "RadGraphMetric"),
    "ratescore":         ("radeval.metrics.RaTEScore.adapter",         "RaTEScoreMetric"),
    "radgraph_radcliq":  ("radeval.metrics.radgraph_radcliq.adapter",  "RadGraphRadCliQMetric"),
    "radcliq":           ("radeval.metrics.RadCliQv1.adapter",         "RadCliQMetric"),
    "srrbert":           ("radeval.metrics.SRRBert.adapter",           "SRRBertMetric"),
    "temporal":          ("radeval.metrics.f1temporal.adapter",         "TemporalF1Metric"),
    "green":             ("radeval.metrics.green_score.adapter",        "GreenMetric"),
    "mammo_green":       ("radeval.metrics.green_score.adapter",        "MammoGreenMetric"),
    "crimson":           ("radeval.metrics.crimson.adapter",            "CrimsonMetric"),
    "radfact_ct":        ("radeval.metrics.radfact_ct.adapter",         "RadFactCTMetric"),
    # --- PRIVATE METRICS (stripped by scripts/publish_public.py) ---
    "f1hopprchexbert":     ("radeval.metrics.f1hopprchexbert.adapter",     "F1HopprCheXbertMetric"),
    "f1hopprchexbert_ct":  ("radeval.metrics.f1hopprchexbert_ct.adapter",  "F1HopprCheXbertCTMetric"),
    "f1hopprchexbert_msk": ("radeval.metrics.f1hopprchexbert_msk.adapter", "F1HopprCheXbertMSKMetric"),
    "f1hopprchexbert_abd": ("radeval.metrics.f1hopprchexbert_abd.adapter", "F1HopprCheXbertAbdMetric"),
    "hoppr_crimson_ct":    ("radeval.metrics.hoppr_crimson_ct.adapter",    "HopprCrimsonCTMetric"),
    "nodule_eval":         ("radeval.metrics.nodule_eval.adapter",         "NoduleEvalMetric"),
    # --- END PRIVATE METRICS ---
}


def get_metric_class(name: str) -> type:
    """Lazily import and return a metric class."""
    if name not in METRIC_REGISTRY:
        raise ValueError(
            f"Unknown metric '{name}'. Available: {sorted(METRIC_REGISTRY)}")
    module_path, class_name = METRIC_REGISTRY[name]
    from importlib import import_module
    return getattr(import_module(module_path), class_name)
