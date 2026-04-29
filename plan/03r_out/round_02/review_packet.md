# Best Practices

# Best Practices

## Design
- Prefer the simplest solution that satisfies current requirements.
- YAGNI: do not add infrastructure, abstractions, or configuration for hypothetical future needs.
- KISS: clear and direct beats clever and general.
- Reuse existing code, patterns, and libraries before adding new ones.
- Delete or simplify before adding.
- Keep changes small, local, and reversible.
- Prefer explicit logic over indirection.
- Avoid premature abstraction.
- Prefer configuration changes over code changes when practical.

## Code Quality
- Keep diffs small and readable.
- Match existing local conventions.
- Fix and consolidate rather than duplicate.
- Extract utility functions only when they clearly improve clarity or eliminate repetition.
- Replace magic numbers and opaque strings with named constants when helpful.
- Fail fast with clear errors.

## Planning
- Separate facts, assumptions, hypotheses, and design decisions explicitly.
- For each major component, ask: what simpler alternative was considered, and why is it insufficient?
- Do not introduce new subsystems unless necessary for the current task.
- Be skeptical of architectural expansion.
- Prefer narrow, high-leverage next steps.

## Scientific / Experimental Reasoning
- Distinguish observation from interpretation.
- Do not infer causality without appropriate evidence.
- Identify confounders, missing controls, and alternative explanations.
- Prefer the minimum experiment that reduces uncertainty.
- Calibrate confidence; do not overclaim.

## Testing
- Default to TDD when practical.
- Write minimal, meaningful tests.
- Treat failing tests as design feedback.
- Test behavior that matters; avoid noisy or decorative tests.

## Safety
- Never modify outside the active worktree or intended scope.
- Avoid destructive actions without explicit need.
- Explain tooling, dependency, or test changes when they materially affect the system.

## Collaboration
- Be concise and factual.
- Critics should challenge reasoning, scope, and assumptions rather than rewrite the whole solution.

---

# Critic Instructions

# External Critic Prompt

You are an external critic reviewing an implementation report. The report describes work that has already been completed: code written, commits made, and decisions taken. Your job is to evaluate the quality and completeness of the implementation, not to propose alternative designs from scratch.

## Your role
- Identify gaps between the original plan and what was implemented.
- Identify correctness risks, edge cases, or failure modes in the implementation.
- Identify unnecessary complexity or overengineering that was introduced.
- Flag testing gaps — areas where the implementation lacks adequate coverage.
- Surface remaining uncertainties or weaknesses the author has identified and assess whether their severity is correctly calibrated.
- Challenge claims of completeness when evidence is insufficient.

## Your non-role
- Do not propose major redesigns of already-committed work.
- Do not invent new requirements or scope beyond what the plan specified.
- Do not assume repository facts not present in the provided materials.
- Do not act as the final decision-maker.

## Review criteria
1. Does the implementation faithfully execute the plan?
2. Are there deviations from the plan, and are they justified?
3. Are the reported challenges and uncertainties accurately characterized?
4. What risks or failure modes are not addressed?
5. What testing is missing or insufficient?
6. Are there simpler approaches that were overlooked for specific components?

## Response format

Return concise structured markdown with these sections only:

```
# Verdict
approve | revise_minor | revise_major | block

# Critical Issues
- severity: high | medium | low
- issue: [description]
- why it matters: [explanation]
- recommended fix: [suggestion]

# Implementation Gaps
- component: [name]
- gap: [what's missing or incomplete]
- risk: [what could go wrong]

# Testing Gaps
- area: [what lacks coverage]
- recommended test: [suggestion]

# Uncertainty Assessment
- [agree/disagree with author's characterization of uncertainties, with reasoning]

# Recommended Revisions
- [specific, actionable revision]

# Confidence
low | medium | high
```

---

# Candidate Plan

# Implementation Report — RadEval 2.2.0 TRL reward hardening (round 1 revision)

## Plan Reference

`plan/03r.md` — "RadEval × TRL: Reinforcement-Learning Support Proposal",
five rounds of external review.

## Summary

Implemented the v2.2.0 TRL reward hardening as a sequence of 10 commits
on branch `trl`. Commits 1-9 shipped the initial implementation (rewrote
`RadEval/rewards.py`, added `[rl]` extras, runnable GRPO quickstart, unit
and integration tests, docs rewrite, README update, changelog). Commit 10
addressed round 1 review feedback by adding a conversational GRPO
integration test that exercises TRL 1.3.0's *actual* `list[list[dict]]`
payload, adding an import-surface test for the user-facing quickstart,
re-running the full repo test suite to confirm no regressions from the
`__init__.py` change, and documenting the benign NCCL bind warning.

All verification ran against `/nfs/cluster/miniconda3/envs/radeval-t5`
(transformers 5.6.2, torch 2.9.1, trl 1.3.0 — the RadEval 2.1+
known-good stack).

## Commits

```
9cbf447  chore(setup): bump to 2.2.0, add [rl] extras for trl
123a5bc  test(fixtures): add synthetic_reports.json for RL examples
aafd4b1  feat(rewards): harden make_reward_fn for TRL GRPO use
4ed6bb7  test(rewards): unit tests for TRL-compat reward surface
108d1ff  test(rewards): TRL integration smoke test on tiny random model
9e615a6  feat(examples): runnable GRPO quickstart replaces stub demo
2172cf9  docs(trl_rewards): rewrite for v2.2.0 reward-provider framing
f4e507c  docs(readme): update RL section for v2.2.0 quickstart
3920925  docs(changelog): 260428b v2.2.0 TRL reward hardening
424a0a3  test(rewards): conversational GRPO integration test + quickstart
         surface (round-1 review feedback)
```

## Deviations from Plan

- **Commit 2 (.gitignore edit).** `tests/**/*.json` was globally
  ignored as "generated snapshots." Added a targeted exception
  (`!tests/fixtures/synthetic_reports.json`) so the hand-authored
  fixture can be tracked.
- **TRL GRPOConfig API drift.** TRL 1.3.0 removed `max_prompt_length`.
  Dropped it from the sketch; kept `max_completion_length`.
- **Changelog filename `260428b.md`.** Same-day convention — `260428.md`
  already records the transformers-v5 changelog.
- **TRL installed into `radeval-t5`.** `trl==1.3.0` added to
  `/nfs/cluster/miniconda3/envs/radeval-t5` (the RadEval 2.1+
  known-good env). No other env or system config touched.
- **`build_config()` extracted from quickstart's `main()`.** Done
  in commit 10 so the new quickstart-surface test can exercise
  the config path without downloading Qwen2.5-0.5B.

## Implementation Challenges

- **F1CheXbert multi-key bug was broader.** The plan identified
  F1CheXbert specifically; during testing, ROUGE also triggered the
  required-`key=` path (`rouge1`/`rouge2`/`rougeL`). This is the
  generic hardening working as intended — `test_rewards.py` adds
  `test_rouge_round_trip` with `key="rouge1"` and a negative test
  `test_rouge_without_key_raises`.
- **Clipping-idiom test initially used wrong BLEU value assumptions.**
  Fixed by picking bounds that clearly exercise both clamp branches
  (5/10 for the low-clamp, -10/-5 for the high-clamp).
- **TRL API drift.** `max_prompt_length` removed in TRL 1.3.0; caught
  on first `test_trl_integration.py` run. Dropped.
- **Conversational integration required a real TRL run** (round 1
  review). Added `test_grpo_smoke_conversational`: drives
  `GRPOTrainer` with `{prompt: [messages], ground_truth}`-format
  dataset, captures the concrete shape TRL passes to the reward fn,
  and asserts the `UserWarning` fires when the shape is `list[dict]`.
  Confirmed TRL 1.3.0 passes `list[list[dict]]` with `role`/`content`
  keys — exactly the shape our heuristic was designed for.

## Remaining Uncertainties

- **TRL API drift across minor 1.x releases.** Pinned `<2` in extras.
  Integration tests cover 1.3.0 specifically. Mitigation: the new
  `test_quickstart_config_surface` test ensures the user-facing
  quickstart's exact config-kwarg surface regression-fails on drift,
  not just the integration test's.
- **`Qwen/Qwen2.5-0.5B` first-run download** (~1 GB). Standard HF
  behavior; not mitigated in this PR.
- **Paper-`radcliq` per-sample runtime** not benchmarked inside a
  real GRPO loop. Documentation explicitly positions it as
  eval / final-tune, not primary online-training reward.

## Known Weaknesses

- **No scorer caching** (deferred per plan). Users who build two
  reward fns for the same metric pay for duplicate model loads.
  Documented via a VRAM callout in `docs/trl_rewards.md`.
- **VLM is docs-only** (deferred per plan). No committed VLM script;
  users wire up `Qwen2.5-VL` following TRL's own `grpo_vlm.py` plus
  our docs snippet.
- **Integration tests are narrow.** 1-step GRPO on tiny random model
  verifies wiring (including the conversational path), not long-run
  algorithmic stability. Docs language matches this scope.
- **No `CompositeReward` helper.** TRL's native `reward_funcs=[...]`
  + `reward_weights` covers the weighted-sum case.

## Testing Status

**Tested:**

- `tests/test_rewards.py`: 26 passed. Covers round-trip on BLEU /
  ROUGE / BERTScore; F1CheXbert multi-key error and success paths;
  ROUGE multi-key error path; invalid-key `KeyError`; conversational
  `list[list[dict]]` extraction + `UserWarning`; missing `role`/
  `content` keys → `TypeError`; unknown shape → `TypeError`;
  `score_transform` composition and clipping idiom;
  `validate_rewards` passes floats through, preserves `None`, raises
  on NaN/Inf, handles `numpy.nan` / numpy scalars / 0-D torch /
  1-element 1-D torch; TRL kwargs absorption + extra dataset columns;
  missing/custom `reference_column`; API-metric `UserWarning`; empty
  completions.
- `tests/test_trl_integration.py`: **3 passed**.
  - `test_quickstart_config_surface`: imports the user-facing
    quickstart, constructs its `GRPOConfig` and dataset — catches
    future TRL API drift against the quickstart's exact surface
    without a Qwen2.5-0.5B download.
  - `test_grpo_smoke_with_make_reward_fn`: 1-step GRPO on
    `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5` with prompt-only
    dataset.
  - `test_grpo_smoke_conversational`: 1-step GRPO with a conversational
    `messages`-format dataset. Captures the concrete shape TRL passes
    to the reward fn (`list[list[dict]]` with `role`/`content` keys
    confirmed empirically on TRL 1.3.0) and asserts the
    `UserWarning` fires.
- `tests/test_e2e.py`: 14 passed (pre-existing reward tests pruned to
  `test_rewards.py`).
- **Full repo test suite** (round-1 feedback):
  `pytest tests/ -m "not integration"` → **191 passed, 3 skipped,
  17 deselected** in 633s. No regressions from the
  `RadEval/__init__.py` / `rewards.py` changes.
- `examples/trl_grpo_quickstart.py`: dogfood run, 5 steps in ~13s
  on available GPU, reward fn invoked each step.
- `scripts/publish_public.py --message "dry run"`: 0 unexpected
  leaks.

**Not tested:**

- Full VLM flow end-to-end (docs-only per plan).
- Full RLOO flow (docs-only per plan).
- `radcliq` as an online reward (documented as eval/final-tune).
- Quickstart on a true laptop GPU (plan's discipline: no speculative
  published runtime numbers).

## Round 1 feedback resolution

| Critique | Disposition | Resolution |
|---|---|---|
| Conversational handler only validated against synthetic dicts (both) | Accepted | Added `test_grpo_smoke_conversational`; captured TRL 1.3.0's actual `list[list[dict]]` output and verified our heuristic path. |
| Full 187-test suite skipped (Gemini) | Accepted | Re-ran full suite: 191 passed, 3 skipped, 17 deselected. Zero regressions. |
| Quickstart not CI-tested (OpenAI) | Accepted | Extracted `build_config()` + added `test_quickstart_config_surface`. |
| NCCL stderr warning (OpenAI, low) | Accepted | Documented in new "Operational notes" section of `docs/trl_rewards.md`. |
