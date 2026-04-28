# Plan Revision Changelog — Round 04

## Input
- previous plan: round_03/revised.md
- critic feedback: round_04/feedback_oai.md (revise_minor, **confidence medium**), round_04/feedback_gem.md (**approve**)
- current revision: round_04/revised.md → finals into round_05/revised.md

## Accepted Critiques (Convergent)
- **Both critics flag the `_HEURISTIC_LOGGED` global + `logging.info` pattern.** Gemini: "minor anti-pattern, unpredictable under multiprocessing; use `warnings.warn`." OpenAI: "limited value if users don't enable INFO logs; put in docs + error messages instead." Convergent. Round 5 switches to `warnings.warn(..., category=UserWarning)`, which: (a) is Pythonic and filterable, (b) fires by default (visible to the user), (c) doesn't need a module-level global (Python dedupes identical warning sites when `simplefilter("once")` is used, and `warnings.warn` stacklevel handles the dedup correctly).
- **OpenAI: tighten "tested with GRPO" to "GRPO smoke-tested."** Agree. Round 5 applies this everywhere.
- **OpenAI: explicitly separate the F1CheXbert bugfix from the general `key=` API hardening.** Agree. Round 5 restructures §3 item 1 into two bullets: "(a) bugfix: F1CheXbert works; (b) general hardening: required `key=` on any metric with >1 per-sample key" with a short rationale that (b) is the minimal generic solution and costs ~3 LOC extra.
- **OpenAI: acknowledge quickstart model choice.** Round 5 adds an explicit justification: Qwen2.5-0.5B is chosen because a random tiny model won't produce meaningful reward signal for the user-facing demo; the integration test uses a tiny random model.

## Accepted Critiques (Gemini only — low-severity polish)
- **Fragile dict-key extraction** (`content` not guaranteed). Round 5's conversational helper wraps key access in try/except and on failure raises the standard `TypeError` with the upstream-preprocess guidance.
- **`validate_rewards` tensor handling.** Round 5 notes: use `float(v.item() if hasattr(v, "item") else v)` to handle single-element tensors. ~2 extra LOC.

## Rejected Critiques
- **OpenAI: "broad multi-key API hardening is broader than the evidence."** Rejected. The generic required-`key=` check is ~3 LOC of extra logic over the F1CheXbert-specific fix and provides correct behavior for every other multi-key metric should one appear. Shipping F1CheXbert-specific logic now would mean re-touching `rewards.py` the first time another metric exposes the same shape. The generic handling is the smallest maintainable solution.
- **OpenAI: "conversational heuristic needs TRL-version validation before shipping."** Partial rejection. The heuristic already fails loud on any shape it doesn't recognize (`TypeError`), and the added dict-key safety closes the remaining hole. We're not over-claiming — docs say "best-effort for OpenAI-style layout; unrecognized layouts raise." Validating against every TRL release before shipping is cost-disproportionate for a defensive adapter.

## Uncertain / Deferred
- Whether to use `warnings.simplefilter("once")` explicitly or rely on the user's default `warnings` configuration. Decision: let `warnings.warn` fire every call; it is the user's prerogative to filter. Simpler surface; no module state.

## Major Plan Changes (Round 05)
- Replace `_HEURISTIC_LOGGED` + `logging.info` with `warnings.warn(..., UserWarning)`.
- Wrap `_last_assistant_content` key access in try/except; fall through to `TypeError`.
- Cast `.item()` when available in `validate_rewards`.
- Restructure §3(1) to separate F1CheXbert bugfix from general `key=` hardening, with a one-line justification.
- Change "tested with GRPO" → "GRPO smoke-tested" in docs/announcement/pitch.
- Add explicit justification for Qwen2.5-0.5B as the user-facing quickstart model.

## Net Effect
- scope: unchanged
- confidence: increased (both critics converged on the same two fixes; Gemini already approves)

## Stopping rule
After round 5 the plan will have gone through two rounds where Gemini approves and OpenAI reports only low/medium-severity polish items. Per the skill rules, "after 2+ rounds, if only low-severity `revise_minor` issues remain, Claude may declare the materiality threshold met and stop early with explicit justification." Round 5 is the intended final round.
