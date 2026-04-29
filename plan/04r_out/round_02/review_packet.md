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

# Implementation Report — RadEval RL benchmarks page (round 1 revision)

## Plan Reference

`plan/04r.md` — "RadEval RL benchmarks page" (5 rounds of external review).

## Summary

Implemented the benchmarks page as **4 commits** on branch `trl`.
Commits 1-3 shipped the initial implementation; commit 4 addresses
round-1 review feedback by adding a `--dry-run` flag to
`bench_rewards.py`, two new integration-style tests that validate
the JSON snapshot structure and skip-path accounting, tightened
VRAM wording in the doc (explicit about metric-order-sensitivity
and PyTorch's allocator caching), and added the F1CheXbert
"14-label vector accuracy" documentation note.

All 12 RL-eligible public metrics loaded and scored successfully
in the canonical run — zero skipped rows. Speed range: 0.09
ms/sample (bleu, CPU) to 160.62 ms/sample (radcliq, ~1.2 GB VRAM).
Divergence headline confirmed: BERTScore = 0.893 on the
negation-flip rollout (near the exact-match ceiling of 1.0),
clinical metrics correctly drop. 38 tests pass across both
`test_bench_rewards_logic.py` (12) and `test_rewards.py` (26).

## Commits

```
d96258f  chore(scripts): add bench_rewards.py + fixtures + logic tests
          scripts/bench_rewards.py (new, ~380 LOC)
          tests/test_bench_rewards_logic.py (new, 10 tests)
          docs/benchmarks/fixtures/speed_workload.json (new)
          docs/benchmarks/fixtures/divergence_examples.json (new)

89d6b8c  docs(benchmarks): add trl_rewards_260429 snapshot + write-up
          docs/benchmarks/trl_rewards_260429.json (new, canonical)
          docs/trl_rewards_benchmarks.md (new)

bb75c80  docs(readme,trl_rewards): point to benchmarks page
          README.md (RL section + docs table row)
          docs/trl_rewards.md (Choosing a metric section expanded)

446b5b7  test(bench): add --dry-run flag + structural snapshot tests
          scripts/bench_rewards.py (dry-run branch + helpers)
          tests/test_bench_rewards_logic.py (+2 integration tests)
          docs/trl_rewards_benchmarks.md (VRAM + F1CheXbert notes)
```

## Deviations from Plan

- **BERTScore on row-1 paraphrase came in at 0.22, not "reasonably
  high."** The doc acknowledges this honestly in row-1 prose. The
  headline row-2 negation flip (BERTScore = 0.893) still works.
- **F1CheXbert's per_sample returns 14-label vector accuracy**, not
  binary per-finding F1 — discovered during dogfood. Numbers are
  self-consistent; the doc now documents this behavior explicitly
  (commit `446b5b7`).
- **Fixture migration**: plan used `tests/fixtures/synthetic_reports.json`
  for E1 in some drafts; implementation uses
  `docs/benchmarks/fixtures/speed_workload.json` per the final plan.
- **No `radeval_bertscore` in gallery columns** — planned exclusion.

## Implementation Challenges

- **BERTScore paraphrase score.** Verified not a bug; the vendored
  BERTScore adapter runs without baseline rescaling. Doc adjusted.
- **RadCliQ floor behavior.** Exact-match row scores 9.37, not 0.
  Doc column header (↓ = better) and takeaways handle this.
- **Bash cwd permission warnings.** Environment quirk; did not
  affect this run — the snapshot writes before the warning prints.

## Remaining Uncertainties

- **Reproducibility on other hardware.** Pinned to A100-SXM4-80GB.
  Documented in the doc's snapshot header.
- **Row 1 BERTScore will drift on other transformers versions** due
  to tokenizer-default sensitivity. Not guarded.
- **`torch.cuda.max_memory_allocated` contamination.** Doc now
  explicitly labels this and tells readers who need trustworthy
  absolute numbers to isolate the metric in its own Python process.

## Known Weaknesses

- **No CI coverage of the canonical numerical snapshot.** Intentional.
  Structural integrity of the script is now CI-tested via the
  `--dry-run` path.
- **Gallery has 8 rows, plan said 8-10.** Within range, on the low
  end.
- **Single-sample per-row scoring in the gallery.** Real GRPO sees
  a batch; the gallery numbers are what a batch-of-1 reward call
  would emit. Reader's reproduction path uses the same fixture so
  they can see this directly.

## Testing Status

**Tested:**

- `tests/test_bench_rewards_logic.py`: **12 passed** (10 pure-logic +
  2 dry-run integration). Covers skip/speed/divergence record shapes,
  `validate_key` drift detection, `METRIC_PLAN` structure (radcliq
  last), `GALLERY_METRICS` coverage, full JSON schema structure via
  `--dry-run`, and skip-path accounting via monkeypatched mid-run
  failure.
- `tests/test_rewards.py`: **26 passed** (unchanged by this PR).
- `scripts/bench_rewards.py --help`: CLI parses.
- `scripts/bench_rewards.py --dry-run --output <tmp>`: completes in
  <1 s, emits a structurally valid snapshot.
- **End-to-end dogfood**: canonical run completed with HF cache warm
  in ~3 minutes. All 12 RL-eligible metrics succeeded. Divergence
  gallery produced meaningful separation on the headline row.

**Not tested:**

- Measurement-path CI (intentional per plan).
- Reproduction on a non-A100 machine.
- Script behavior with HF cache cold (user will see first-run
  download times; dogfood was with pre-populated cache).

## Round 1 feedback resolution

| Critique | Disposition | Resolution |
|---|---|---|
| Structural / integration test gap (both critics) | Accepted | `--dry-run` flag + 2 integration tests (`test_dry_run_produces_expected_snapshot_structure`, `test_dry_run_skip_accounting_path`). |
| VRAM contamination under-qualified (OpenAI) | Accepted | Doc VRAM paragraph rewritten to explicitly flag metric-order-sensitivity and PyTorch allocator caching. |
| VRAM contamination (Gemini, low) | Accepted | Same as above. Teardown was already correct; the doc now communicates it. |
| F1CheXbert 14-label behavior (Gemini) | Accepted | Doc "Picking a reward" F1CheXbert bullet now documents 14-label vector-accuracy semantics. |
| Skip-path not exercised integration-style (OpenAI) | Accepted | `test_dry_run_skip_accounting_path` monkeypatches one metric to fail, asserts the skip lands and other metrics still run. |
| Stderr warning dismissed too broadly (OpenAI, low) | Partial | Softened report language to "did not affect this run"; doc not changed (not user-facing). |
