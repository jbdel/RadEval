"""NoduleEval: nodule-focused LLM-as-judge metric for CT report evaluation.

Evaluates the PULMONARY NODULES: section of clean_findings using a judge LLM
(OpenAI or Gemini) with deterministic Python scoring on top of LLM-parsed JSON.

Metrics emitted:
    - nodule_eval_detection_{precision, recall, f1}
    - nodule_eval_size_accuracy        (tolerance-based)
    - nodule_eval_size_exact_match     (strict equality)
    - nodule_eval_size_mae_mm          (mean absolute error in mm)
    - nodule_eval_size_mape            (mean absolute percentage error)
    - nodule_eval_{type, location, noun, uncertainty}_accuracy
    - nodule_eval_composite            (CRIMSON-style [-1, 1])

See /fss/jb/.cursor/plans/ for the design plan.
"""

from .nodule_eval import NoduleEvalScore

__all__ = ["NoduleEvalScore"]
