from .._base import MetricBase


class HopprModifierCthSv001Metric(MetricBase):
    name = "hoppr_modifier_cth_sv001"
    display_name = "Hoppr Modifier CTH (sv001)"
    is_api_based = True

    def __init__(self, model_name="gemini-3.1-flash-lite", provider="gemini",
                 openai_api_key=None, gemini_api_key=None, temperature=0.0,
                 max_concurrent=50, cache_dir=None):
        from .hoppr_modifier_cth_sv001 import HopprModifierCthSv001

        if HopprModifierCthSv001 is None:
            raise ImportError(
                "hoppr_modifier_cth_sv001 is unavailable. It requires "
                "'google-genai' (pip install google-genai) for the Gemini "
                "provider, or 'openai' for the OpenAI provider."
            )
        self._scorer = HopprModifierCthSv001(
            model_name=model_name, provider=provider,
            openai_api_key=openai_api_key, gemini_api_key=gemini_api_key,
            temperature=temperature, max_concurrent=max_concurrent,
        )

    @property
    def cost_tracker(self):
        return getattr(self._scorer, "cost_tracker", None)

    def metric_keys(self, detailed=False):
        keys = ["hoppr_modifier_cth_sv001"]
        if detailed:
            keys += [
                "hoppr_modifier_cth_sv001_std",
                "hoppr_modifier_cth_sv001_total_severity_errors",
                "hoppr_modifier_cth_sv001_total_localization_errors",
                "hoppr_modifier_cth_sv001_total_laterality_errors",
                "hoppr_modifier_cth_sv001_total_region_errors",
                "hoppr_modifier_cth_sv001_total_sinus_errors",
            ]
        return keys

    def _compute_raw(self, refs, hyps, on_progress=None):
        mean, std, samples, _ = self._scorer(
            refs, hyps, on_sample_done=on_progress)
        totals = self._scorer.error_totals()
        return {"hoppr_modifier_cth_sv001": {
            "aggregate": mean,
            "per_sample": samples,
            "detailed": {
                "hoppr_modifier_cth_sv001_std": std,
                "hoppr_modifier_cth_sv001_total_severity_errors":
                    totals.get("severity_errors", 0),
                "hoppr_modifier_cth_sv001_total_localization_errors":
                    totals.get("localization_errors", 0),
                "hoppr_modifier_cth_sv001_total_laterality_errors":
                    totals.get("laterality_errors", 0),
                "hoppr_modifier_cth_sv001_total_region_errors":
                    totals.get("region_errors", 0),
                "hoppr_modifier_cth_sv001_total_sinus_errors":
                    totals.get("sinus_errors", 0),
            },
        }}
