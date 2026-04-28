# Plan Revision Changelog — Round 02

## Input
- previous plan: round_01/revised.md
- critic feedback: round_02/feedback_oai.md (revise_minor), round_02/feedback_gem.md (approve)
- current revision: round_02/revised.md (written before critics ran) → updates flow into round_03/revised.md

## Accepted Critiques
- **Both critics: drop `clip_range`.** Overlapping flag from both. `score_transform` can express any clipping a user needs (`lambda x: max(lo, min(hi, x))`). Removing `clip_range` tightens the API and eliminates ambiguity about operation order. Round 3 drops it.
- **OpenAI: soften "by construction" compatibility claim.** Round 3 changes docs/announcement to: "tested with GRPO; designed to work with other TRL trainers that consume a reward-fn callable (e.g., RLOO) — not separately validated."
- **OpenAI: "last assistant message" extraction rule asserted without verification.** Round 3 documents this as a best-effort heuristic and notes that users who hit edge cases can pass a custom `completions` preprocessor via `score_transform` or open an issue. Also adds an explicit test case.
- **OpenAI: runtime target for quickstart is speculative.** Round 3 moves the "under 10 minutes on a laptop GPU" claim from the plan into a validation gate ("measure before publishing the time in docs"). Docs/announcement will not publish a specific minute count until benchmarked.
- **OpenAI: "16+ metrics usable as a reward function" announcement language is broader than evidence supports.** Round 3 narrows the pitch: "many RadEval metrics can be wrapped as TRL reward functions; lightweight metrics (BLEU, ROUGE, BERTScore, RadGraph, F1CheXbert) are the primary supported path; API-based metrics are warned against for RL."
- **Gemini: VRAM warning for duplicate heavy-metric loads.** Round 3 adds a callout in the docs' "combining metrics" section: "two reward functions that share a heavy underlying metric will load the model twice (no cache in v2.2.0) — plan VRAM accordingly, or combine via a single reward fn that computes multiple keys."
- **Gemini: reassure users about batching.** Round 3 documents that RadEval processes completions as a batch, matching TRL.

## Rejected Critiques
- None. Both critics' points are actionable and low-risk.

## Uncertain / Deferred
- Whether to validate conversational handling against a real TRL run as part of the PR, or leave as a manual check during dogfooding. Current plan: include in the unit test fixture (synthetic message-dict form); do not require a live TRL conversational test in CI.

## Major Plan Changes (Round 03)
- **Drop `clip_range`.** Users use `score_transform` for clipping.
- **Soften compatibility language** to tested-scope only.
- **Remove specific runtime claims** from publishable surfaces until benchmarked.
- **Narrow announcement claims** to "many metrics"/"lightweight-primary" framing.
- **Add VRAM callout** to docs.
- **Add explicit "best-effort conversational adapter" note** + test.

## Net Effect
- scope: reduced (dropped `clip_range`)
- confidence: increased (no remaining API expansion beyond the bugfix; compatibility claims match verified scope)
