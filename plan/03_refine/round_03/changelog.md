# Plan Revision Changelog — Round 03

## Input
- previous plan: round_02/revised.md
- critic feedback: round_03/feedback_oai.md (revise_minor), round_03/feedback_gem.md (**approve**)
- current revision: round_03/revised.md → updates flow into round_04/revised.md

## Accepted Critiques
- **OpenAI: `score_transform` is the wrong mitigation for conversational formatting.** Correct — `score_transform` operates on scores, not completion text. Round 4 fixes the doc language to "preprocess completions upstream in your dataset pipeline" and says the heuristic is scoped to the common OpenAI-style message layout.
- **OpenAI: add integration test for forwarded kwargs / extra dataset columns.** Agree. Round 4 adds a narrow test that passes an unused extra dataset column and asserts the reward fn returns correctly without error.
- **OpenAI: tighten compatibility to a specific verified TRL version/range.** Agree. Round 4 specifies "tested against TRL vX.Y.Z; may work with nearby versions but not separately validated."
- **OpenAI: PR scope language** — separate core code delta from total PR footprint. Round 4 uses clearer framing: "~150 LOC of core code change; the PR also includes docs, tests, and one example."
- **OpenAI: docs breadth.** Mostly agree. Round 4 tightens `docs/trl_rewards.md` structure: GRPO is the primary tested path (detailed); RLOO / PPO / DPO-curation / VLM are short subsections with "not separately tested" markers.
- **Gemini: tiny random model for CI integration test.** Strong agree. Round 4 specifies `trl-internal-testing/tiny-random-LlamaForCausalLM` (or similar) for `tests/test_trl_integration.py` so it can run on a CPU CI runner. `Qwen/Qwen2.5-0.5B` stays in the user-facing quickstart only.
- **Gemini: `validate_rewards` must handle `np.nan` / tensor NaN.** Agree. Round 4 specifies using `math.isnan(float(v))` + `math.isinf(float(v))` after an explicit `float()` cast, which handles numpy scalars, torch scalars, and Python floats uniformly.
- **Gemini: add first-batch info log when conversational heuristic fires.** Agree (low cost). Round 4 adds a one-time `logging.info` on first use: "Extracting 'content' from last assistant message; preprocess your dataset upstream if this isn't what you want."

## Rejected Critiques
- **OpenAI: announcement positioning is overengineered.** Partial rejection. Section 8 is intentional — the user's original plan file explicitly requested polished announcement-worthy framing. Not removing §8, but round 4 tightens the pitch paragraph to match the narrower docs framing and the now-known CI/runtime constraints. Acceptable compromise.
- **Gemini: "drop the conversational heuristic entirely and force `score_transform`."** Rejected — `score_transform` operates on scores (this is the same confusion OpenAI called out). The right alternative is to raise `TypeError` on unrecognized `list[dict]` shapes, which round 3 already does; the last-assistant heuristic handles the common case and fails loud on unusual ones. Keep with the added first-use info log (Gemini's fallback suggestion).

## Uncertain / Deferred
- Whether to add `logging.info` or a `UserWarning` for the heuristic firing. Current decision: `logging.info` (quiet by default; visible when users enable INFO logging). Promote to `UserWarning` if we see support questions.

## Major Plan Changes (Round 04)
- Fix conversational-formatting doc language: "preprocess upstream in dataset pipeline" not "via `score_transform`".
- Add extra-dataset-column test to `tests/test_trl_integration.py`.
- Pin one verified TRL version/range for compatibility claims.
- Swap to tiny random model in `tests/test_trl_integration.py`; keep `Qwen/Qwen2.5-0.5B` only in `examples/trl_grpo_quickstart.py`.
- Specify `validate_rewards` casts to `float()` before `isnan`/`isinf` checks.
- Add `logging.info` first-use notice when the conversational heuristic is applied.
- Tighten `docs/trl_rewards.md` structure: GRPO primary & tested; everything else short subsections tagged "not separately tested."
- Reframe PR scope language to split core-code LOC from total-PR footprint.

## Net Effect
- scope: unchanged (all round-4 changes are clarifications/additions to existing items, not new features)
- confidence: increased (Gemini already approves; remaining OpenAI items are actionable)
