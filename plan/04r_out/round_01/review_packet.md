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

# Implementation Report — RadEval RL benchmarks page

## Plan Reference

`plan/04r.md` — "RadEval RL benchmarks page" (5 rounds of external review).

## Summary

Implemented the benchmarks page as 3 sequential commits on branch
`trl`: (1) `scripts/bench_rewards.py` + static fixtures
(`docs/benchmarks/fixtures/{speed_workload,divergence_examples}.json`)
+ 10 pure-logic unit tests; (2) canonical JSON snapshot
`docs/benchmarks/trl_rewards_260429.json` produced by running the
script end-to-end on `radeval-t5` / A100, plus the write-up
`docs/trl_rewards_benchmarks.md`; (3) one-line pointers from
`README.md` and `docs/trl_rewards.md` to the new page.

All 12 RL-eligible public metrics loaded and scored successfully —
zero skipped rows in the speed table. Speed range: 0.09 ms/sample
(bleu, CPU) to 160.62 ms/sample (radcliq, ~1.2 GB VRAM). The
divergence gallery's headline row (negation flip) confirms the
plan's hypothesis: BERTScore scores the clinically-wrong rollout
at 0.893 — nearly the exact-match ceiling of 1.0 — while clinical
metrics correctly drop (F1CheXbert 0.80, RadGraph 0.50, RadCliQ
rises to 11.06).

## Commits

```
d96258f  chore(scripts): add bench_rewards.py + fixtures + logic tests
          scripts/bench_rewards.py (new, 383 LOC)
          tests/test_bench_rewards_logic.py (new, 111 LOC)
          docs/benchmarks/fixtures/speed_workload.json (new)
          docs/benchmarks/fixtures/divergence_examples.json (new)

89d6b8c  docs(benchmarks): add trl_rewards_260429 snapshot + write-up
          docs/benchmarks/trl_rewards_260429.json (new, ~220 lines)
          docs/trl_rewards_benchmarks.md (new, ~200 lines)

bb75c80  docs(readme,trl_rewards): point to benchmarks page
          README.md (2 edits: RL section sentence + docs table row)
          docs/trl_rewards.md (expanded "Choosing a metric" section)
```

## Deviations from Plan

- **BERTScore on row-1 paraphrase came in at 0.22, not "reasonably
  high."** The plan predicted BERTScore would agree with clinical
  metrics on the paraphrase row. Measured value is much lower than
  expected — the BERTScore variant RadEval exposes appears to run
  without baseline rescaling (raw scores), so short-string /
  low-overlap pairs get compressed. The headline row-2 negation
  flip (BERTScore = 0.893) still works: the argument is "BERTScore
  gives a negation-flipped rollout a near-ceiling score," which is
  true regardless of how the paraphrase row behaves. The doc
  acknowledges this in row 1 prose ("BERTScore here scores only
  0.22 because it's vocabulary-sensitive; it doesn't recover the
  clinical paraphrase as cleanly as readers might expect"). No
  narrative collapse.

- **F1CheXbert's "multi-label accuracy" interpretation.** I
  discovered during the run that F1CheXbert's `per_sample` mode
  returns *label-vector accuracy across 14 CheXpert labels*, not
  binary per-finding F1. A negation flip that only changes one of
  14 labels scores 0.80 (not 0.0 as I initially implied in earlier
  plan drafts). The final doc numbers and narrative are consistent
  with what the metric actually does — the negation-flip F1CheXbert
  drop from 1.0 (exact match) to 0.80 is still distinctly lower
  than BERTScore 0.893 and correctly signals the label change.

- **Fixture migration.** Plan used `tests/fixtures/synthetic_reports.json`
  for E1 workload; implementation uses a fresh
  `docs/benchmarks/fixtures/speed_workload.json` with hand-picked
  (ref, hyp) pairs. The plan's later revisions had already moved
  this direction; implementation matches the final plan spec.

- **No `radeval_bertscore` in gallery columns.** Plan said to
  exclude it. Implemented as planned.

## Implementation Challenges

- **BERTScore paraphrase score.** Row 1 BERTScore of 0.22 looked
  wrong initially — expected ~0.7+. Verified it's consistent with
  how the vendored BERTScore adapter runs (no rescale against
  baseline), so it's a real measurement, not a bug. Adjusted doc
  narrative for row 1 to acknowledge this honestly.

- **RadCliQ floor behavior.** Exact-match row 5 scores RadCliQ = 9.37,
  which is a high number in absolute terms but the *lowest* of any
  row — consistent with RadCliQ being a standardized distance
  (lower = better, but not bounded at 0 because of the linear
  regression composite design). Doc's column header calls this out
  (↓ = better) and the takeaways bullets use phrasing that's
  consistent ("RadCliQ rises to 11.06" for row 2's worse outcome).

- **Bash cwd permission warnings.** Running the script printed a
  non-fatal `/bin/bash: line 1: /tmp/claude-<hash>-cwd: Permission
  denied` to stderr after the snapshot was written. This is an
  environment quirk unrelated to our code; the snapshot lands
  correctly before the warning prints. No remediation needed.

## Remaining Uncertainties

- **Reproducibility on other hardware.** All numbers pinned to A100
  SXM4-80GB. On consumer GPUs (e.g. 24 GB) the `radcliq` 1175 MB
  VRAM delta is small enough to fit, but queue / allocator behavior
  may differ. Documented.
- **Row 1 BERTScore value will likely differ on other transformers
  versions.** The vendored BERTScore's tokenizer handling is
  version-specific. Not guarded.
- **`torch.cuda.max_memory_allocated` contamination** from earlier
  in-process metrics is still a risk despite teardown; the doc
  labels VRAM as approximate.

## Known Weaknesses

- **No CI coverage of the measurement path.** Intentional — measured
  numbers shouldn't be regression-gated. Only the pure-logic
  helpers are tested.
- **No validation that the gallery rows produce the predicted
  direction on other RadEval versions.** The fixture is static;
  future adapter changes could shift numbers without surfacing
  anywhere automated.
- **Gallery has 8 rows, not 8-10.** Within the plan's stated range
  but on the low end.
- **BLEU+BERTScore+clinical comparison is entirely single-sample
  per-row scoring.** Real GRPO scores a batch; the gallery numbers
  are what a batch-of-1 reward call would produce. Note for any
  reader trying to reproduce inside a training loop.
- **Reproduction snippet in the doc assumes `pip install RadEval[rl]`
  gives you the benchmarked env.** In practice, users may have
  version drift relative to the pinned stack. The doc snapshot's
  env block records exact versions for reference.

## Testing Status

**Tested:**

- `tests/test_bench_rewards_logic.py`: 10 passed in ~12 s. Covers
  skip/speed/divergence record shapes, `validate_key` (single-key OK
  + drift, multi-key OK + drift), structural assertions on
  `METRIC_PLAN` (radcliq last) and `GALLERY_METRICS` (BLEU +
  BERTScore + 3 clinical).
- `scripts/bench_rewards.py --help`: CLI parses correctly.
- **End-to-end dogfood**: script ran twice (warmup + canonical).
  All 12 RL-eligible metrics succeeded. Divergence gallery produced
  meaningful separation on the headline row. Canonical run completed
  in ~3 minutes with HF cache warm.
- JSON schema of the canonical snapshot: verified by loading and
  inspecting in the same shell before committing.

**Not tested:**

- Measurement-path CI (intentional per plan).
- Reproduction on a non-A100 machine.
- Script behavior with HF cache cold (user will hit this on first
  run but dogfood was with pre-populated cache from prior RadEval
  work).
- Skip-path behavior (no metric failed during the canonical run,
  so the `skip_record` path is exercised only in unit tests).
