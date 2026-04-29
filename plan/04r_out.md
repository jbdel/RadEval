# Implementation Report — RadEval RL benchmarks page (round 2 revision, final)

## Plan Reference

`plan/04r.md` — "RadEval RL benchmarks page" (5 rounds of external review).

## Summary

Implemented the benchmarks page as **4 commits** on branch `trl`
(commits 1-3 initial implementation, commit 4 applied round-1
review feedback). Round 2 critics: **Gemini approves** with one
sanity-check (snapshot already includes `transformers`/`torch`
versions — verified). **OpenAI revise_minor** with
wording-calibration items around separating "structurally tested"
from "numerically validated" claims; this revision tightens those
framings.

The canonical numerical snapshot
(`docs/benchmarks/trl_rewards_260429.json`) was produced by **one
manual end-to-end run** on A100/radeval-t5 and is **not
regression-tested**. The `--dry-run` test path added in commit 4
validates orchestration/schema only, not metric correctness. The
doc's headline narrative rests on a single measured run
interpreted in light of the RadCliQ paper and RadEval paper;
future users rerunning will get different numbers and should treat
the published page as a snapshot, not a contract.

## Commits

```
d96258f  chore(scripts): add bench_rewards.py + fixtures + logic tests
89d6b8c  docs(benchmarks): add trl_rewards_260429 snapshot + write-up
bb75c80  docs(readme,trl_rewards): point to benchmarks page
446b5b7  test(bench): add --dry-run flag + structural snapshot tests
```

## Scope of validation (calibrated)

- **Structurally tested in CI** (via `--dry-run`): JSON schema
  top-level keys, `METRIC_PLAN` ordering (`radcliq` last),
  `GALLERY_METRICS` coverage, skip-record accounting under a
  mocked failure. 12 tests in `test_bench_rewards_logic.py`.
  Does **not** exercise model loading, adapter behavior, per-sample
  output semantics, or actual scoring correctness.
- **Numerically validated by one manual canonical run**: all 12
  RL-eligible metrics loaded and returned scores for the fixture
  workload; zero skipped rows. Measurements recorded in
  `docs/benchmarks/trl_rewards_260429.json`.
- **Interpretation anchored in external literature**: RadCliQ
  paper for radiologist-preference correlation claims; RadEval
  paper for population-level comparison context.
- **Not validated**: reproduction on hardware other than
  A100-SXM4-80GB; reproduction on `transformers` versions other
  than 5.6.2 (BERTScore tokenizer handling is version-sensitive;
  both versions are recorded in the snapshot's `env` block).

## Deviations from Plan

- **BERTScore row-1 paraphrase = 0.22, lower than anticipated.**
  Not independently validated as "not a bug" — observed in one run
  on the vendored BERTScore adapter (no baseline rescaling) and
  documented honestly in row-1 doc prose. A user seeing drastically
  different numbers on a different BERTScore configuration could
  reasonably reach different conclusions about that row. Headline
  row-2 narrative (BERTScore = 0.893 on negation flip) is more
  robust because it depends only on "BERTScore assigns high reward
  when most tokens overlap," which is its definitional behavior.
- **F1CheXbert per_sample = 14-label vector accuracy**, documented
  as such. Semantics verified by reading
  `RadEval/metrics/_chexbert_base.py` and checking the single-pair
  negation-flip output (0.80 on rows where one label flips matches
  the 13/14 agreement formula). Not independently cross-validated.
- **Static `docs/benchmarks/fixtures/speed_workload.json`** per the
  final plan.
- **`radeval_bertscore` excluded from gallery** per the final plan.

## Implementation Challenges

- **BERTScore paraphrase score.** Treated as measured data, not a
  bug; doc row-1 prose honestly notes vocabulary-sensitivity.
- **RadCliQ floor.** Exact-match row 5 scores 9.37 (lowest-but-not-0).
  Documented.
- **Bash cwd permission warnings.** Observed environment-specific
  warning in the interactive bash wrapper; **did not affect this
  run's snapshot output**. Broader impact on other environments is
  unknown.

## Remaining Uncertainties

- **Reproducibility on other hardware.** Pinned to A100-SXM4-80GB;
  snapshot `env` block records this plus torch/transformers/python/
  cuda versions.
- **BERTScore / F1CheXbert interpretation.** Documented from
  observed outputs and source inspection, not from an independent
  validation suite.
- **`torch.cuda.max_memory_allocated` contamination.** Doc
  explicitly flags VRAM as approximate / metric-order-sensitive
  and recommends per-metric subprocess isolation for absolute
  numbers.
- **Unknown** whether the bash stderr warning affects other
  environments' exit codes or partial output.

## Known Weaknesses

- **No CI coverage of canonical numeric values.** Intentional.
- **No non-dry-run smoke path in CI.** `--dry-run` exercises
  orchestration/schema only. A user with a broken adapter would
  only discover it during their manual benchmark run, not in CI.
  Acceptable for a one-shot documentation tool.
- **Gallery has 8 rows, low end of the 8-10 planned range.**
- **Single-sample per-row scoring in the gallery.** Matches real
  GRPO batch-of-1 reward semantics; noted in the reproduction
  path.

## Testing Status

**Tested:**

- `tests/test_bench_rewards_logic.py`: **12 passed**.
  - 10 pure-logic tests: record shapes, key validation, structural
    assertions.
  - 2 integration tests via `--dry-run`: full JSON schema +
    skip-path accounting via monkeypatched failure.
  - **Does not exercise real model loading / metric correctness.**
- `tests/test_rewards.py`: **26 passed** (unchanged).
- Full RL test suite combined: **38 passed.**
- `scripts/bench_rewards.py --help` and `--dry-run --output <tmp>`:
  both work.
- **One manual end-to-end canonical run** on
  `/nfs/cluster/miniconda3/envs/radeval-t5` / A100. All 12
  RL-eligible metrics completed. Duration: ~3 minutes with HF
  cache warm.

**Not tested:**

- Measurement-path CI (intentional; numbers aren't regression-gated).
- Non-dry-run smoke execution in any automated context.
- Reproduction on non-A100 machines or non-5.6.2 transformers.
- Real-metric failure (only monkeypatched failures exercised).

## Round 1 + Round 2 feedback resolution

| Round | Critique | Disposition | Resolution |
|---|---|---|---|
| 1 | Structural / integration test gap (both) | Accepted | `--dry-run` + 2 integration tests, commit `446b5b7`. |
| 1 | VRAM under-qualified (both) | Accepted | Doc VRAM paragraph rewritten (commit `446b5b7`). |
| 1 | F1CheXbert 14-label doc note (Gemini) | Accepted | Doc bullet (commit `446b5b7`). |
| 1 | Skip-path not integration-tested (OpenAI) | Accepted | Monkeypatched skip test (commit `446b5b7`). |
| 1 | Stderr warning dismissed broadly (OpenAI) | Partial | Softened report wording. |
| 2 | Snapshot already has versions? (Gemini) | Accepted (no change needed) | Verified: `env` block already contains `transformers` + `torch` + `python` + `cuda` + `gpu`. |
| 2 | "Structurally tested" vs "numerically validated" conflation (OpenAI) | Accepted | This revision adds an explicit "Scope of validation" section separating the two. |
| 2 | `--dry-run` doesn't exercise real path (OpenAI) | Accepted as framing | "Scope of validation" states `--dry-run` tests orchestration/schema only. |
| 2 | BERTScore/F1CheXbert claims lack independent validation (OpenAI) | Accepted | Deviations + Remaining Uncertainties now honestly note these are "observed in one run," not independently validated. |
| 2 | Stderr warning claim still too broad (OpenAI, low) | Accepted | "Did not affect this run's snapshot output" + "broader impact on other environments is unknown." |

## Stopping rationale

After round 2: **Gemini approves** with high confidence; OpenAI
went `revise_minor` with only wordsmithing-level items, all applied
in this revision. Per the skill's materiality-threshold rule
("after 2+ rounds, if only low-severity `revise_minor` issues
remain, Claude may declare materiality threshold met and stop early
with explicit justification"), stopping here is appropriate — the
remaining OpenAI concerns are framing preferences on an
implementation report whose underlying artifacts are sound,
tested at the structural level, and explicitly calibrated as a
dated snapshot rather than a validated contract.
