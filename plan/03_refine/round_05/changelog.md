# Plan Revision Changelog — Round 05 (final)

## Input
- previous plan: round_04/revised.md
- critic feedback: round_05/feedback_oai.md (revise_minor, high confidence), round_05/feedback_gem.md (revise_minor, high confidence)
- current revision: round_05/revised.md (final)

## Accepted Critiques
- **Gemini (high severity): `validate_rewards` breaks TRL's `None` sample-skip.** Sharp catch — the round-4 sketch would crash if a scorer returned `None`. Round 5 patches `validate_rewards` to short-circuit on `None`, adds a corresponding unit test, and documents the intent in the §3(1) bullet.
- **OpenAI: inconsistency between `logging.info` (stale) and `UserWarning` (current) in the plan text.** Correct — the risks section §7 still had the old `logging.info` phrasing. Round 5 cleans it up. Everything now consistently says `UserWarning` fired per invocation via `warnings.warn` with the tradeoff spelled out.
- **OpenAI: "generic `key=` handling" justification language should be "minimal guardrail," not "future-proofing."** The round-4 §3(1) bullet already positioned this as "~3 LOC over the F1CheXbert-only patch" — reasonable. Round 5 adds one sentence clarifying that the generic check is a *minimal guard*, not an attempt to solve cases that don't exist. No behavioral change.

## Rejected Critiques
- **Gemini: "drop the conversational heuristic entirely."** Rejected. OpenAI round 4 explicitly endorsed keeping it defensive; Gemini round 5 now recommends dropping it. With the two critics disagreeing, I weight toward keeping the current design for three reasons: (a) the OpenAI-style message layout *is* the common case TRL emits for conversational datasets, so the heuristic pays off for most users; (b) the `UserWarning` and `TypeError` paths make every failure mode loud; (c) removing the heuristic would force even standard users to write a wrapping lambda. Rejection documented in §9. Also added an explicit §7 note naming this as a tradeoff we accepted.
- **OpenAI: "make benchmark publication a docs follow-up rather than a release gate."** Partial rejection — the release gate is *run the quickstart end-to-end once before tagging*, which is a sanity check, not a benchmarking project. Keeping it. The *docs numbers* are already framed as follow-up fill-ins ("publish measured runtime only after benchmarking"), consistent with the critic.
- **OpenAI: "narrow the generic `key=` to only adapters whose `per_sample=True` result is a dict of lists."** Implicitly rejected — every RadEval adapter's `compute(..., per_sample=True)` output is a dict (mapping key → list), by `MetricBase` contract. The generic check is already scoped to that shape. Round 5 does not change behavior.

## Uncertain / Deferred
- None remaining that are actionable inside the plan scope.

## Stopping Rationale

This is round 5 of 5 — **max iterations reached**. Both critics return `revise_minor` rather than `approve`, but with high confidence and only one actually-new high-severity issue (the `None` pass-through), which round 5 fixes. The remaining OpenAI items reduce to:

1. "Tighten 'tested with GRPO' language" — already applied everywhere ("GRPO smoke-tested").
2. "Reduce confidence of the generic `key=` justification" — applied (added "minimal guardrail" language).
3. "Narrow claims to exactly what is validated" — applied throughout.
4. Remaining critiques are framing-preferences where the plan already reflects the preferred framing.

Remaining Gemini items:
1. `None` pass-through — **fixed.**
2. Drop heuristic — **explicitly rejected with justification in §9 and tradeoff note in §7.**

The plan is internally consistent, the high-severity concern is addressed, and further rounds would be shuffling wording without substantive change. Stopping here is the right call.

## Net Effect
- scope: unchanged
- confidence: increased (the one truly-new concern was material; it's now fixed)
