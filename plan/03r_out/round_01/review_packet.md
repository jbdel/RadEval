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

# Implementation Report — RadEval 2.2.0 TRL reward hardening

## Plan Reference

`plan/03r.md` — "RadEval × TRL: Reinforcement-Learning Support Proposal",
five rounds of external review.

## Summary

Implemented the v2.2.0 TRL reward hardening as a sequence of 9 small
commits on branch `trl`: rewrote `RadEval/rewards.py` (required `key=`
for multi-key metrics, defensive conversational completions handler,
`validate_rewards` helper with `None` pass-through and NaN/Inf raise),
added `[rl]` extras, replaced the stub `examples/trl_reward_demo.py`
with a runnable GRPO quickstart, added unit tests (`test_rewards.py`,
26 tests) and a TRL integration smoke test (`test_trl_integration.py`,
1-step GRPO on a tiny random model), rewrote `docs/trl_rewards.md`,
updated `README.md`, and wrote a changelog. All verification ran against
`/nfs/cluster/miniconda3/envs/radeval-t5` (transformers 5.6.2, torch
2.9.1, trl 1.3.0 — the RadEval 2.1+ known-good stack).

## Commits

```
9cbf447  chore(setup): bump to 2.2.0, add [rl] extras for trl
         setup.py

123a5bc  test(fixtures): add synthetic_reports.json for RL examples
         tests/fixtures/synthetic_reports.json, .gitignore

aafd4b1  feat(rewards): harden make_reward_fn for TRL GRPO use
         RadEval/rewards.py (+179 lines over prior 50-LOC version),
         RadEval/__init__.py (exports validate_rewards)

4ed6bb7  test(rewards): unit tests for TRL-compat reward surface
         tests/test_rewards.py (new, 269 lines),
         tests/test_e2e.py (prunes 3 legacy reward tests)

108d1ff  test(rewards): TRL integration smoke test on tiny random model
         tests/test_trl_integration.py (new, 76 lines)

9e615a6  feat(examples): runnable GRPO quickstart replaces stub demo
         examples/trl_grpo_quickstart.py (new),
         examples/trl_reward_demo.py (deleted stub)

2172cf9  docs(trl_rewards): rewrite for v2.2.0 reward-provider framing
         docs/trl_rewards.md (rewritten, ~80% diff)

f4e507c  docs(readme): update RL section for v2.2.0 quickstart
         README.md (version badge, RL section, known-good stack note)

3920925  docs(changelog): 260428b v2.2.0 TRL reward hardening
         docs/changelog/260428b.md (new)
```

## Deviations from Plan

- **Commit 2 scope expanded by one file.** `.gitignore` previously
  ignored `tests/**/*.json` as "generated snapshots." Added a targeted
  exception (`!tests/fixtures/synthetic_reports.json`) so the
  hand-authored fixture can be tracked. Mentioned in the fixture
  commit message.
- **TRL GRPOConfig API drift.** The original plan sketch used
  `max_prompt_length` and `max_completion_length`; TRL 1.3.0 has
  removed `max_prompt_length`. Kept `max_completion_length=16`
  (integration test) and `=32` (quickstart), dropped the prompt cap.
  Tests and dogfood runs confirm TRL defaults are fine.
- **Changelog filename `260428b.md`.** The existing `docs/changelog/`
  convention is two-digit-year + month + day, and `260428.md`
  already records the transformers-v5 changelog from the same date.
  Appended a `b` suffix per the pattern other repos use when two
  changelogs land the same day.
- **Installed TRL into the existing env rather than creating a new one.**
  `trl==1.3.0` added to `/nfs/cluster/miniconda3/envs/radeval-t5`
  (the same env used for the transformers-v5 changelog's Gate 1 run).
  No other env or system config touched.

## Implementation Challenges

- **F1CheXbert multi-key bug was broader than confirmed.** The plan
  identified F1CheXbert as the only verified case. During
  `test_rewards.py` execution, ROUGE also triggered the required-`key=`
  path (`rouge1` / `rouge2` / `rougeL`). This is the generic hardening
  doing its job — the test suite updated to pass `key="rouge1"` and
  added a `test_rouge_without_key_raises` test to lock in the contract.
  No bug fix beyond what was already in `make_reward_fn`; documented.
- **Clipping-idiom test initially used wrong BLEU value assumptions.**
  The first draft assumed BLEU on "totally different" text would clip
  to 0.1; BLEU per-sample values are actually ~1e-11 for non-matching
  pairs. Rewrote the test to use clearly out-of-range bounds
  (`max(5.0, min(10.0, x))` forces the low-clamp; negative range
  forces the high-clamp) so both clamp branches are exercised.
- **TRL API drift.** Discovered during `test_trl_integration.py`
  first run — `max_prompt_length` no longer exists. Removed it.
- **NCCL bind warnings on stderr.** The integration test and
  quickstart both emit a harmless `NCCL WARN Call to bind failed:
  Address already in use` to stderr from the GPU environment. It
  doesn't fail the run (`train_runtime` logged, all steps complete,
  assertions pass), but the exit code from shell-redirected runs
  can come back as 1 from the stderr stream alone. In pytest the
  tests pass cleanly. Documented only in this report; not surfaced
  to users.

## Remaining Uncertainties

- **TRL API drift risk.** Pinned `trl>=1.3.0,<2` in extras. Integration
  test verifies 1-step behavior against 1.3.0 specifically. Minor
  kwarg renames in future 1.x releases could break the quickstart
  file before the integration test catches them (the integration test
  doesn't set `max_completion_length=16` the same way the quickstart
  does). Mitigation: both files use `**kwargs`-friendly config paths.
- **`Qwen/Qwen2.5-0.5B` download-on-first-run.** The user-facing
  quickstart downloads ~1 GB on first invocation. No cached mirror is
  part of this PR. Users with no outbound network will see the
  download fail; this matches the behavior of every other HF-model
  consumer in the repo.
- **Paper-`radcliq` per-sample runtime** not benchmarked inside a real
  GRPO loop. Documentation explicitly positions it as eval / final-tune,
  not primary online-training reward.
- **Conversational heuristic validity across TRL versions.** The
  `list[list[dict]]` extraction is tested against a synthetic message
  dict; we haven't verified what TRL 1.3.0 actually passes when the
  dataset uses the conversational format vs. prompt-only. The
  `UserWarning` + loud `TypeError` path keeps surprises visible.

## Known Weaknesses

- **No scorer caching** (deferred per plan). Users who build two
  reward fns for the same metric pay for duplicate model loads.
  Documented via a VRAM callout in `docs/trl_rewards.md`.
- **VLM is docs-only** (deferred per plan). No committed VLM script
  means users who want to wire up `Qwen2.5-VL` follow TRL's own
  `grpo_vlm.py` plus our docs snippet — one extra hop compared to a
  copy-pasteable script.
- **Integration test is narrow.** 1-step GRPO on a tiny random model
  verifies wiring, not algorithmic behavior, long-run stability, or
  VLM / RLOO / non-conversational conversational-datasets. Docs and
  announcement language match this scope.
- **No `CompositeReward` helper.** Users who want a single callable
  with per-component diagnostics need to write it themselves. TRL's
  native `reward_funcs=[...]` + `reward_weights` covers the
  weighted-sum case with per-function logging.
- **Quickstart isn't CI-tested.** `tests/test_trl_integration.py`
  runs a much smaller model. The plan's §6 validation gate (dogfood
  the quickstart before tagging) was satisfied on a real GPU but
  isn't re-run automatically.

## Testing Status

**Tested:**

- `tests/test_rewards.py`: 26 passed. Covers round-trip on BLEU /
  ROUGE / BERTScore; F1CheXbert multi-key error and success paths;
  ROUGE multi-key error path; invalid-key KeyError; conversational
  `list[list[dict]]` extraction + `UserWarning`; missing `role`/
  `content` keys → `TypeError`; unknown shape → `TypeError`;
  `score_transform` composition and clipping idiom;
  `validate_rewards` passes floats through, preserves `None`, raises
  on NaN/Inf, handles `numpy.nan` / numpy scalars / 0-D torch /
  1-element 1-D torch; TRL kwargs absorption + extra dataset columns;
  missing/custom `reference_column`; API-metric `UserWarning`; empty
  completions.
- `tests/test_trl_integration.py`: 1 passed. 1-step GRPO on
  `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5`, reward-fn
  invocation count asserted.
- `tests/test_e2e.py`: 14 passed (down from 17 after pruning the
  3 legacy reward tests now subsumed by `test_rewards.py`).
- `examples/trl_grpo_quickstart.py`: dogfood run, 5 steps in ~13s
  on available GPU, reward fn invoked each step, reward trajectory
  printed.
- `scripts/publish_public.py --message "dry run"`: runs cleanly;
  changelog stripped, registry stripped, 0 unexpected leaks.

**Not tested:**

- Full repo test suite (187 tests from the transformers-v5 changelog).
  Not re-run in this PR because no metric adapter, registry entry,
  or core RadEval evaluator code was touched. Only `rewards.py`,
  one `test_e2e.py` prune, docs, fixtures, examples, and setup.py
  changes — none of which affect the 16-metric evaluation paths.
- VLM flow end-to-end. Docs-only per plan.
- RLOO flow. Docs-only per plan.
- `radcliq` as an online reward. Documented as eval/final-tune only.
- The quickstart on a true laptop GPU. Ran on available cluster GPU
  (~13s for 5 steps); laptop runtime not measured per plan's
  "no speculative published numbers" discipline.
