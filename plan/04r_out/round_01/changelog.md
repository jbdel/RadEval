# Review Loop Changelog — Round 01

## Input
- previous document: plan/04r_out/implementation_report.md
- critic feedback: round_01/feedback_oai.md (revise_minor), round_01/feedback_gem.md (**approve**)
- current revision: round_01/revised.md

## Accepted Critiques

### Convergent: both critics want a structural integration test
- **OpenAI medium: no automated structural validation of the canonical JSON / doc consistency.** **Gemini: add `--dry-run` flag + structural CI test.** Both addressed in commit `446b5b7`:
  - `scripts/bench_rewards.py --dry-run`: bypasses model loading / scoring; emits placeholder numbers; exercises fixture loading, `METRIC_PLAN` iteration, divergence row assembly, and JSON writing.
  - `tests/test_bench_rewards_logic.py::test_dry_run_produces_expected_snapshot_structure`: validates every top-level key, workload shape, complete `METRIC_PLAN` ordering in speed rows, full `GALLERY_METRICS` coverage in divergence scores.
  - `test_dry_run_skip_accounting_path`: monkeypatches one metric to a skip record, asserts the skip lands in the JSON correctly and other metrics still appear. Addresses OpenAI's "skip-path not exercised in integration."

### VRAM wording (both critics)
- **OpenAI medium: VRAM prominently presented but under-qualified.** **Gemini low: allocator-caching contamination.** Addressed in commit `446b5b7`: `docs/trl_rewards_benchmarks.md` VRAM paragraph now explicitly flags metric-order-sensitivity, names PyTorch's caching allocator as the contamination source, and tells readers who need trustworthy absolute numbers to isolate the metric in its own process.

### F1CheXbert 14-label documentation (Gemini)
- **Gemini: document that F1CheXbert `per_sample` returns 14-label vector accuracy.** Addressed — the "Picking a reward" F1CheXbert bullet now explicitly says "per-sample agreement rate across 14 CheXpert labels (compressed multi-label accuracy), not per-finding F1; a single-label flip drops the score by ~1/14 at most." Explains the row-7 severity-flip 1.00 ("mild" and "severe" edema map to the same CheXpert label).

## Partially Accepted

### Stderr warning note (OpenAI low)
- OpenAI: "stated dismissively." Not changing the doc (the warning is not user-facing — it only appears in the bash wrapper when running the script interactively) but softened the implementation-report language in the revised report: "did not affect this run" rather than "no remediation needed in general."

## Rejected Critiques
- None. Everything actionable was applied in commit `446b5b7`.

## Additional commits from the review loop
- `446b5b7` — `test(bench): add --dry-run flag + structural snapshot tests`

## Net Effect
- scope: slightly expanded (+1 flag, +2 tests, +2 doc paragraphs)
- confidence: increased — both critics' main concerns addressed with one commit; all 12 structural + 2 new integration tests pass; full RL test suite (38 tests) green.
