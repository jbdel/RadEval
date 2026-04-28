# Plan Revision Changelog — Round 01

## Input
- previous plan: plan/03.md (original, with user-added §9 and §10 critiques)
- critic feedback: round_01/feedback_oai.md, round_01/feedback_gem.md
- current revision: round_01/revised.md

## Accepted Critiques
- **OpenAI: scope estimate understated for a "~300-line PR".** Agree. Round 2 will cut the scorer cache entirely from v2.2.0 and defer to a follow-up PR. This trims both critic concerns (overengineering + memory/DDP surprises) in one move.
- **OpenAI: default multi-key selection is ambiguous.** Agree. Round 2 will remove implicit default selection and require explicit `key=` for metrics with mode-dependent keys, with a clear error naming the valid keys when the user omits it. Safer than guessing.
- **OpenAI: integration test too narrow to support broad compatibility claims.** Agree. Round 2 will soften docs/announcement language to "tested with GRPO; expected to work with other TRL trainers that accept a reward-function callable (e.g. RLOO)" and add one focused unit test for the conversational-format path.
- **OpenAI: `.metric_keys` attribute on the callable has no concrete consumer.** Agree. Drop it from the plan.
- **OpenAI / Gemini: `make_reward_fn("radcliq")` may be too slow for online RL.** Agree. Round 2 will position it as "recommended for final-fine-tune / eval reward; profile before using for online RL; see speed table."
- **Gemini: conversational-completion handling must be defensive — check `isinstance(completions[0], str)` before extracting `content`.** Agree. Round 2 will specify this behavior explicitly.
- **Gemini: broad adapter audit expands scope without evidence.** Agree with OpenAI's parallel point. Round 2 will fix F1CheXbert (confirmed) and document `key=` as the canonical disambiguator for any other metric users encounter. No repo-wide audit.

## Rejected Critiques
- **Gemini: "raise hard TypeError on unhashable kwargs" in scorer cache.** Rejected — but only because we are deferring the cache entirely (see above). If the cache lands in a later PR, the cleanest approach is Gemini's own second suggestion: use `functools.lru_cache` on the internal factory and let Python raise natively.
- **Gemini: bounded LRU cache (max 2-3 models).** Rejected for the same reason — feature deferred.
- **Gemini: DDP/Accelerate cache behavior concern.** Rejected for the same reason. When the cache ships, we will explicitly document "per-process cache; each DDP rank loads its own scorer."

## Uncertain / Deferred
- Whether to ship a `validate_rewards()` NaN/Inf helper in v2.2.0 or defer. Current plan: ship it (~15 LOC, called inside `make_reward_fn`; fails loud on NaN/Inf with metric name + sample index). Defers easily if size is a problem in review.
- Exact TRL version floor for `[rl]` extras — to be decided during implementation when we verify kwargs against a specific TRL release.

## Major Plan Changes (Round 02)
- **Scorer cache removed from v2.2.0 scope.** Moved to "deferred to follow-up" section. Keeps PR small and sidesteps the DDP / unhashable-kwargs / memory-bound concerns entirely.
- **`key=` is required when metric has multiple per-sample keys.** No implicit default selection. `make_reward_fn("f1chexbert")` without a `key=` raises with a message naming the valid keys.
- **Conversational handling is defensive**: `if isinstance(completions[0], str): ... else: extract content`.
- **`radcliq` positioned as "evaluation / final-tune reward, not primary online-training reward"** pending benchmark.
- **`.metric_keys` attribute dropped.**
- **Integration test claims softened** in docs + announcement.

## Net Effect
- scope: reduced (scorer cache cut)
- confidence: increased (fewer failure modes; smaller attack surface for reviewers)
