"""Lazy metric registry — maps metric names to adapter classes.

Nothing is imported until get_metric_class() is called for a specific name.
This preserves lazy loading: importing RadEval loads zero metric dependencies.
"""

METRIC_REGISTRY: dict[str, tuple[str, str]] = {
    "bleu":              ("RadEval.metrics.bleu.adapter",              "BleuMetric"),
    "rouge":             ("RadEval.metrics.rouge.adapter",             "RougeMetric"),
    "bertscore":         ("RadEval.metrics.bertscore.adapter",         "BertScoreMetric"),
    "radeval_bertscore": ("RadEval.metrics.radevalbertscore.adapter",  "RadEvalBertScoreMetric"),
    "f1chexbert":        ("RadEval.metrics.f1chexbert.adapter",        "F1CheXbertMetric"),
    "f1radbert_ct":      ("RadEval.metrics.f1Radbert_ct.adapter",      "F1RadbertCTMetric"),
    "radgraph":          ("RadEval.metrics._radgraph_adapter",         "RadGraphMetric"),
    "ratescore":         ("RadEval.metrics.RaTEScore.adapter",         "RaTEScoreMetric"),
    "radgraph_radcliq":  ("RadEval.metrics.radgraph_radcliq.adapter",  "RadGraphRadCliQMetric"),
    "radcliq":           ("RadEval.metrics.RadCliQv1.adapter",         "RadCliQMetric"),
    "srrbert":           ("RadEval.metrics.SRRBert.adapter",           "SRRBertMetric"),
    "temporal":          ("RadEval.metrics.f1temporal.adapter",         "TemporalF1Metric"),
    "green":             ("RadEval.metrics.green_score.adapter",        "GreenMetric"),
    "mammo_green":       ("RadEval.metrics.green_score.adapter",        "MammoGreenMetric"),
    "crimson":           ("RadEval.metrics.crimson.adapter",            "CrimsonMetric"),
    "radfact_ct":        ("RadEval.metrics.radfact_ct.adapter",         "RadFactCTMetric"),
    # --- PRIVATE METRICS (stripped by scripts/publish_public.py) ---
    "f1hopprchexbert":     ("RadEval.metrics.f1hopprchexbert.adapter",     "F1HopprCheXbertMetric"),
    "f1hopprchexbert_ct":  ("RadEval.metrics.f1hopprchexbert_ct.adapter",  "F1HopprCheXbertCTMetric"),
    "f1hopprchexbert_msk": ("RadEval.metrics.f1hopprchexbert_msk.adapter", "F1HopprCheXbertMSKMetric"),
    "f1hopprchexbert_abd": ("RadEval.metrics.f1hopprchexbert_abd.adapter", "F1HopprCheXbertAbdMetric"),
    "hoppr_crimson_ct":    ("RadEval.metrics.hoppr_crimson_ct.adapter",    "HopprCrimsonCTMetric"),
    "nodule_eval":         ("RadEval.metrics.nodule_eval.adapter",         "NoduleEvalMetric"),
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
