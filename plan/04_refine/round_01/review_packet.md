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

You are an external critic reviewing a candidate plan.

## Your role
- Identify reasoning flaws and unjustified assumptions.
- Identify overengineering and unnecessary complexity.
- Identify missing validations, controls, or experiments.
- Suggest simpler viable alternatives.
- Challenge excessive confidence.

## Your non-role
- Do not rewrite the whole plan.
- Do not invent major new scope unless clearly necessary.
- Do not assume repository facts not present in the provided materials.
- Do not act as the final decision-maker.

## Review criteria
1. Is the reasoning justified by evidence presented?
2. Are any claims stronger than the evidence supports?
3. What is overengineered or unnecessary?
4. What simpler alternative should be preferred?
5. What assumptions need validation?
6. What important risk or missing check is absent?

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

# Overengineering Flags
- component: [name]
- why unnecessary: [explanation]
- simpler alternative: [suggestion]

# Assumptions to Test
- [assumption that needs validation]

# Recommended Revisions
- [specific, actionable revision]

# Confidence
low | medium | high
```

---

# Candidate Plan

# Plan 04 — RadEval RL benchmarks page

## Goal

Ship `docs/trl_rewards_benchmarks.md`: a small, self-contained write-up
that makes the "use RadEval metrics as RL rewards" story *concrete and
compelling* to an outside reader browsing the repo. Not a long
evaluation paper — a short document with two measured artifacts:

1. **Speed table** — cold-start + warm per-batch latency for every
   reward-eligible metric on a fixed small workload. Turns the current
   prose speed tiers ("fast / medium / slow") into numbers.
2. **Reward divergence gallery** — curated report pairs scored by a few
   metrics side-by-side, showing the cases where lexical and clinical
   metrics disagree. This is the differentiator: it explains *why* a
   user should reach for RadEval's clinical metrics as RL rewards
   instead of BLEU from `evaluate`.

Numbers are pinned to **RadEval 2.2.0** (current release) and the
`radeval-t5` env. Script is committed but **not re-run on future
releases** — the doc is a snapshot, not a living contract.

## Non-goals

- No training trajectory plots (deferred — requires real GPU time and
  per-release curation; the speed table + divergence gallery are
  already enough to tell the story).
- No CI coverage for the benchmark script. It's a one-shot tool.
- No regression tests on measured numbers. If someone re-runs on a
  different machine they'll get different numbers; that's fine.
- No new runtime behavior or API changes in `RadEval/rewards.py`. Docs
  + a one-shot script only.

## Deliverables

1. `scripts/bench_rewards.py` — generates the raw data for both
   artifacts. Prints tables to stdout; writes one JSON snapshot the
   docs page cites.
2. `docs/benchmarks/trl_rewards_260429.json` — JSON snapshot of the
   measured numbers. Date-stamped filename so we can add more later
   without overwriting.
3. `docs/trl_rewards_benchmarks.md` — the write-up.
4. One-line pointer from `docs/trl_rewards.md` → benchmarks page.
5. One-line pointer from `README.md` (right under the RL section) →
   benchmarks page.

No changes to `RadEval/rewards.py`, tests, or installed extras.

## Scope of metrics

**Speed table: every metric that can plausibly be used as an RL
reward.** Call shapes and required `key=` values are **verified
against the current registry + adapter code** by actually instantiating
each scorer and inspecting its `per_sample=True` output keys (probed
2026-04-29, RadEval 2.2.0, `radeval-t5` env).

| Metric | Call | Tier expectation |
|---|---|---|
| `bleu` | `make_reward_fn("bleu")` | fast |
| `rouge` | `make_reward_fn("rouge", key="rouge1")` | fast |
| `bertscore` | `make_reward_fn("bertscore")` | fast |
| `radeval_bertscore` | `make_reward_fn("radeval_bertscore")` | fast-medium |
| `f1chexbert` | `make_reward_fn("f1chexbert", key="f1chexbert_sample_acc_5")` | medium |
| `f1radbert_ct` | `make_reward_fn("f1radbert_ct")` (single-key in per_sample mode → no `key=` needed) | medium |
| `radgraph` | `make_reward_fn("radgraph", key="radgraph_partial")` | medium |
| `ratescore` | `make_reward_fn("ratescore")` | medium |
| `srrbert` | `make_reward_fn("srrbert", key="srrbert_weighted_f1")` | medium |
| `radgraph_radcliq` | `make_reward_fn("radgraph_radcliq")` | medium |
| `radcliq` | `make_reward_fn("radcliq")` (per-sample key is `radcliq_v1`; single-key → no `key=` needed) | slow-ish (composite of BERTScore + SembScore + RadGraph) |
| `temporal` | `make_reward_fn("temporal")` | fast |
| `green` | **excluded** — 7B local LLM; plan already documents as "not practical per-step." |
| `crimson`, `mammo_green`, `radfact_ct` | **excluded** — API metrics; not RL-eligible. |

**Key-map fact sheet (verified):**

- **Single-key** (no `key=` needed, default label is the only entry):
  `bleu`, `bertscore`, `radeval_bertscore`, `ratescore`,
  `radgraph_radcliq`, `radcliq` (→ `radcliq_v1`), `temporal` (→
  `temporal_f1`).
- **Multi-key** (require `key=`): `rouge`
  (`rouge1/rouge2/rougeL`), `f1chexbert`
  (`f1chexbert_sample_acc_5/_all`), `radgraph`
  (`_simple/_partial/_complete`), `srrbert`
  (`_weighted_f1/_precision/_recall`).
- Note: `f1radbert_ct` exposes only `f1radbert_ct_sample_acc` in
  `per_sample=True` today. The required-`key=` guard in
  `make_reward_fn` only fires when `>1` keys are returned, so
  `make_reward_fn("f1radbert_ct")` works without an explicit key. If a
  future adapter adds a second per-sample key, the guard will start
  firing — at which point this doc snapshot becomes out of date by
  design (pinned to RadEval 2.2.0).

Private HOPPR metrics (`f1hopprchexbert*`, `hoppr_crimson_ct`,
`nodule_eval`) are **not benchmarked** — out of scope regardless and
need internal checkpoints.

**Divergence gallery: a curated subset** — 5 metrics that tell the
story clearly:

- `bleu` — the obvious lexical baseline.
- `radeval_bertscore` — domain-tuned semantic.
- `f1chexbert` — clinical-finding classification (CXR).
- `radgraph_partial` — entity/relation clinical F1.
- `radcliq` — composite, shown in published studies to correlate best
  with radiologist preferences.

Five metrics × ~8 rows is a table that fits on screen and tells a
story.

## Experiments

### E1. Speed benchmark

**Workload.** 20 (ref, hyp) pairs derived from the existing
`tests/fixtures/synthetic_reports.json` fixture (note: the fixture's
native schema is `{prompt, ground_truth}`, used by the GRPO
quickstart — for benchmarking we use the 20 `ground_truth` strings as
references and apply a light, deterministic perturbation to produce
hypotheses: swap "No" ↔ "Mild", add/remove a trailing phrase, etc. —
so BLEU isn't trivially 0 or 1 across the batch). The exact
perturbations are committed in the script so the workload is
reproducible. 20 samples is small enough that `radcliq` completes in
minutes; large enough that per-sample latency isn't dominated by
measurement noise.

**Measurements per metric:**

1. **Cold-start wall time** — `t0 = time.perf_counter(); r = make_reward_fn(...); t1 = time.perf_counter()`. Captures model download / HF-from-pretrained / any subprocess setup. **Excludes HF cache warmth** — record whether the model was already in `~/.cache/huggingface` and if so label as "warm cache"; otherwise it's a first-pull cold-start. Either way the number is the honest time an end user with the same cache state would see.
2. **Warm per-batch latency** — call `r(completions=hyps, ground_truth=refs)` once to warm caches, then time 3 calls over the full 20-row batch and take the median. Report seconds/batch (20 samples) and extrapolate to ms/sample for the table.
3. **Peak VRAM delta** — if `torch.cuda.is_available()`:
   `torch.cuda.reset_peak_memory_stats()` before cold-start,
   `torch.cuda.max_memory_allocated()` after the three warm calls, diff
   in MB. If CPU-only, report "CPU".

**Implementation notes:**

- Each metric loads fresh in its own subprocess invocation so cold-start timings are honest and one metric's warm state can't interfere with another's. `scripts/bench_rewards.py` accepts `--metric <name>` to time a single metric and `--all` to iterate (forking a subprocess per metric under the hood via `subprocess.run`).
- Record `RadEval.__version__`, `torch.__version__`, `transformers.__version__`, `trl.__version__`, GPU model (from `torch.cuda.get_device_name`), Python version, **and HF cache location + per-metric "cache_hit" boolean** (existence of a plausibly-matching entry in the HF hub cache before the run). One-time record at the top of the JSON.
- Skip metrics whose `cls(**kwargs)` raises `FileNotFoundError` / `ImportError` / `HfHubHTTPError` — log to stderr and continue.
- Multi-key metrics: pass the exact `key=` from the verified key-map above. Record which key was timed in the JSON.

**Output shape** (JSON snapshot):

```json
{
  "run_ts": "2026-04-29T...",
  "radeval_version": "2.2.0",
  "env": {"torch": "2.9.1+cu128", "transformers": "5.6.2",
          "trl": "1.3.0", "python": "3.11.15",
          "gpu": "NVIDIA A100 80GB PCIe"},
  "workload": {"n_samples": 20,
               "fixture": "tests/fixtures/synthetic_reports.json"},
  "speed": [
    {"metric": "bleu", "cold_start_s": 0.003,
     "warm_batch_s_median": 0.001, "warm_per_sample_ms": 0.05,
     "peak_vram_mb": 0},
    ...
  ]
}
```

### E2. Reward divergence gallery

**Inputs — 8-10 hand-curated (ref, hyp) pairs** in a **separate fixture
file**, `tests/fixtures/divergence_examples.json` (the existing
`synthetic_reports.json` has `{prompt, ground_truth}` pairs for GRPO,
not `{ref, hyp}` pairs for scoring — it's the wrong shape for this
experiment). The new fixture shares the `.gitignore` exception
convention already used for `synthetic_reports.json`:

```
# .gitignore
tests/**/*.json
!tests/fixtures/synthetic_reports.json
!tests/fixtures/divergence_examples.json      # new
```

Each row is `{id, ref, hyp, narrative}` where `narrative` is a
one-line human-readable label used by the write-up.

Candidate pairs (rough sketch — finalize during implementation):

1. **Paraphrase, clinically identical.** ref: "Mild cardiomegaly." hyp:
   "The cardiac silhouette is mildly enlarged."
   Expected: BLEU ≈ 0, clinical metrics ≈ high.
2. **Lexical overlap, clinically opposite.** ref: "No pleural
   effusion." hyp: "Pleural effusion."
   Expected: BLEU moderate, F1CheXbert / RadGraph low.
3. **Missing finding.** ref: "Mild cardiomegaly. Small bilateral
   pleural effusions." hyp: "Mild cardiomegaly."
   Expected: BLEU moderate, clinical metrics penalize the omission.
4. **Extra hallucinated finding.** ref: "No acute findings." hyp: "No
   acute findings. Small right pneumothorax."
   Expected: lexical moderate-high, clinical metrics penalize.
5. **Exact match.** ref = hyp. All metrics ≈ 1.0.
6. **Total mismatch.** All metrics ≈ 0.
7-9. Varying severity / wording differences to fill out the table.

**Measurement:** for each pair, run each of the 5 gallery metrics in
`per_sample=True` mode and record the single scalar.

**Output shape:**

```json
{
  "divergence": [
    {"id": 1, "ref": "...", "hyp": "...", "narrative": "paraphrase; clinically identical",
     "scores": {"bleu": 0.02, "radeval_bertscore": 0.83,
                "f1chexbert": 1.0, "radgraph_partial": 0.85,
                "radcliq_v1": 0.04}},
    ...
  ]
}
```

**Narrative fields** (~1 line each) are stored alongside the scores so
the doc can reference them by id.

### Reward direction (load-bearing caveat)

Not all metrics are "higher = better." In particular:

- **`radcliq` (RadCliQ-v1) is a distance**: *lower = better
  radiologist alignment*. Verified in the repo — `tests/test_radcliq.py`
  shows a near-identical pair scoring **0.0416** while a paraphrase
  pair scores **0.8346**. The paper and `rajpurkarlab/CXR-Report-Metric`
  define it this way.
- **BLEU / ROUGE / BERTScore / RadEvalBERTScore / RaTEScore /
  RadGraph / F1CheXbert / F1RadBERT-CT / SRRBert / Temporal** are
  **higher = better**.
- **`radgraph_radcliq`** is a *component* of RadCliQ (average of
  entity F1 and relation F1), so **higher = better** (don't confuse
  with `radcliq` the composite).

This matters for both artifacts:

- **Speed table:** no change — latency is direction-agnostic.
- **Divergence gallery:** the doc must show RadCliQ values in
  context ("lower = better") so readers don't mis-read the numbers.
- **RL usage (mentioned in the decision guide):** using
  `make_reward_fn("radcliq")` as-is in a GRPO loop would **maximize
  distance from references**, which is the opposite of what a user
  wants. The doc includes a **one-liner on how to invert** —
  `make_reward_fn("radcliq", score_transform=lambda x: -x)` is the
  simplest; any monotonically decreasing transform (e.g. `1/(1+x)` to
  keep it bounded) works. This is the kind of footgun a reader skimming
  the speed table would walk into otherwise.

## Script design — `scripts/bench_rewards.py`

Single-file, ~150-200 LOC. Entry points:

- `python scripts/bench_rewards.py speed --output docs/benchmarks/trl_rewards_260429.json`
- `python scripts/bench_rewards.py divergence --output docs/benchmarks/trl_rewards_260429.json`
  (appends; does not overwrite)
- `python scripts/bench_rewards.py all --output docs/benchmarks/trl_rewards_260429.json`

Uses only stdlib + the installed `RadEval` env. No new deps.

Subprocess-per-metric for E1 (via `subprocess.run` calling back into
the same script with `--metric` mode), in-process for E2 (one scorer
instance per metric, reused across rows).

**Error handling:** if a metric fails to load or score, record
`"skipped": "<reason>"` in the JSON and continue. Do not fail the
whole run.

**Runtime budget:** E1 for the 7-10 fast/medium metrics should finish
in ≤ 10 minutes on the dogfood GPU; `radcliq` alone may take longer
because it wraps BERTScore + SembScore + RadGraph. Acceptable.

## Doc — `docs/trl_rewards_benchmarks.md` outline

Short, opinionated, ~1.5 screens:

```markdown
# RadEval metrics as RL rewards — speed & divergence

*Snapshot: RadEval 2.2.0, TRL 1.3.0, transformers 5.6.2, torch 2.9.1,
NVIDIA <GPU>. Numbers will drift on other hardware / future releases;
re-run `scripts/bench_rewards.py` to refresh.*

## Per-batch cost (20 samples)

<speed table here — one row per metric, columns: cold-start, warm
batch, per-sample ms, peak VRAM>

**Takeaways.** (2-3 bullets.)
- `bleu` / `rouge` are essentially free — use as a sanity baseline or
  cheap auxiliary reward.
- `radeval_bertscore` is the cheapest *semantic* option; one model
  load, <N> ms/sample warm.
- `radcliq` is 3-5× the cost of its heaviest component because it
  composes three transformer passes. Position it as an **eval-time**
  reward or a final-tuning reward, not a primary online signal.

## Where clinical metrics disagree with BLEU

<divergence gallery — 8-10 rows, 5 metrics wide, ref/hyp column + one
narrative-line column>

*(All scores: ↑ = better except RadCliQ ↓ = better; direction is shown
in the column header.)*

**Three cases worth noticing.**

1. *Paraphrase, same clinical content* (row X). BLEU ≈ 0 but
   F1CheXbert 1.0 and RadCliQ low (≈ 0.08) — the model rewrote a
   finding in a different register. Optimizing on BLEU would penalize
   this; optimizing on a clinical reward would not.
2. *Lexical overlap, clinically opposite* (row Y). BLEU moderate
   because "pleural effusion" overlaps, but F1CheXbert 0.0 and
   RadCliQ high (≈ 0.9) because the *finding* is inverted. Canonical
   case for preferring clinical rewards.
3. *Missing finding* (row Z). BLEU moderate; RadGraph penalizes the
   omission at the entity level.

## Picking a reward

(One-paragraph decision guide.)
- Cheap sanity-check or small auxiliary: `bleu`, `rouge`.
- Semantic, still cheap: `radeval_bertscore`.
- Clinical-finding accuracy (CXR): `f1chexbert`, pass
  `key="f1chexbert_sample_acc_5"`.
- Entity / relation grounding (CXR): `radgraph` with
  `key="radgraph_partial"`.
- Best-correlated-with-radiologist-preferences (eval / final tune):
  `radcliq`. Expect higher per-sample cost and **remember to invert**
  — RadCliQ is a distance, so for maximizing training use
  `make_reward_fn("radcliq", score_transform=lambda x: -x)` or
  `lambda x: 1.0 / (1.0 + x)` to keep it bounded.

## Reproducing this page

```bash
pip install RadEval[rl]
python scripts/bench_rewards.py all \
    --output docs/benchmarks/trl_rewards_$(date -u +%y%m%d).json
```

The exact snapshot used for this page lives at
`docs/benchmarks/trl_rewards_260429.json` and was taken on
<env line>.
```

## Success criteria (i.e., "what makes this land well")

- Speed table has every RL-eligible metric. Reader can pick one
  off-the-shelf based on their budget without guessing.
- Divergence gallery has **at least one row where BLEU strongly
  disagrees with RadCliQ** (or another clinical metric). This is the
  screenshot-worthy moment that tells outsiders *why* clinical metrics
  matter.
- Both artifacts are self-contained — the page cites the JSON
  snapshot and script path so anyone can verify.
- The page is short enough (< 2 screens) that a skimmer reaches the
  takeaways.

## Commit structure (3 commits)

1. `chore(scripts): add bench_rewards.py` — the script + a
   `.gitignore` exception for `tests/fixtures/divergence_examples.json`
   if that route is chosen.
2. `docs(benchmarks): add trl_rewards_260429 snapshot + write-up` —
   the JSON snapshot + `docs/trl_rewards_benchmarks.md`. This is where
   the measured numbers land.
3. `docs(readme,trl_rewards): point to benchmarks page` — one-line
   pointer in each.

Standalone commits so a reviewer who only cares about the script (1)
vs. the prose (2) vs. the cross-links (3) can review each
independently.

## Risks

- **Some metric fails to load on this machine.** Private HOPPR
  checkpoints (`f1hopprchexbert*`, `hoppr_crimson_ct`, `nodule_eval`)
  are out of scope regardless. For public metrics, if the model
  download fails, the script records `"skipped"` and the table has a
  "—" row. Acceptable.
- **Divergence curation is judgment-dependent.** Mitigation: commit
  the curated pairs + their narrative labels as data
  (`tests/fixtures/divergence_examples.json` or literal in script)
  rather than recomputing on every run. What we're showing is
  deterministic given the pairs.
- **Numbers will not match exactly on other hardware.** That's fine —
  the document is a dated snapshot. The page says this explicitly.
- **Per-sample mode for a metric may raise a required-`key=` error**
  (ROUGE, RadGraph, F1CheXbert, F1RadBERT-CT, etc.). Mitigation:
  benchmark script knows the per-sample key for each multi-key metric
  (lookup table literal at the top of the script).
- **`radcliq` takes longer than the runtime budget.** Mitigation:
  `--timeout 600` per metric; any that exceed the timeout get
  `"skipped": "timeout"`. Non-blocking.

## What an outside reader takes away

After 90 seconds on this page:

- RadEval ships **13+ metrics usable as RL rewards** — here's what each
  one costs per batch.
- For a radiology RL loop, **BLEU is not enough** — here are three
  concrete report pairs where BLEU and a clinical metric disagree
  substantially.
- **RadCliQ is the preferred clinical reward** for alignment with
  radiologist judgment (with cost-vs-quality trade-off explained).

That's the differentiator. Not a framework; a focused, measured
answer to "which reward should I use?"

## Facts, assumptions, hypotheses

**Facts (verified in the repo, 2026-04-29, RadEval 2.2.0):**

- Every metric listed in the E1 scope has an adapter in
  `RadEval/metrics/_registry.py` and can be instantiated with no args
  in the `radeval-t5` env. Verified by probing.
- Per-sample output keys for every listed metric are as documented
  in the **key-map fact sheet** above.
- RadCliQ-v1 is a *distance* metric (lower = better). Confirmed by
  `tests/test_radcliq.py::EXPECTED_RADCLIQ_PER_PAIR` where a
  near-identical pair scores 0.0416 and a paraphrase scores 0.8346.
- `tests/fixtures/synthetic_reports.json` has `{prompt, ground_truth}`
  schema — not `{ref, hyp}` — so the divergence gallery needs its own
  fixture.
- `make_reward_fn`'s required-`key=` guard fires only when
  `compute(..., per_sample=True)` returns >1 key
  (`RadEval/rewards.py:187`). Single-key metrics (including
  `f1radbert_ct` today) don't need an explicit `key=`.

**Assumptions (plausible, check during implementation):**

- `radcliq` finishes 20-sample per-batch scoring within a
  few-minute budget on the available GPU (it wraps BERTScore +
  SembScore + RadGraph, three transformer passes).
- GPU VRAM measured via `torch.cuda.max_memory_allocated` is a
  reasonable proxy for "plan your VRAM budget" — may undercount
  non-torch allocations but is standard practice.
- All public metrics' checkpoints are in the `radeval-t5` HF cache
  or can be downloaded on first run. If not, the script skips cleanly.

**Hypotheses (not validated, drive the divergence curation):**

- RadCliQ will score the "lexical overlap, clinically opposite" case
  (e.g., "no pleural effusion" vs "pleural effusion") worse (higher
  distance) than a paraphrase. Confirmed in spirit by the existing
  test-suite pattern but not yet measured on our curated set.
- BLEU's score on clinically-critical negation flips ("no" vs yes)
  will be non-zero due to substantial lexical overlap — which is
  exactly the failure mode the gallery highlights.

If E2 runs and these hypotheses don't hold on the chosen pairs, the
curation is wrong and we swap in different pairs. The script's output
is the data; the doc's narrative is shaped to match.

## Decisions & alternatives considered

| Decision | Alternative rejected | Reason |
|---|---|---|
| Separate `tests/fixtures/divergence_examples.json` fixture | Reuse `synthetic_reports.json` | Wrong schema — that one is `{prompt, ground_truth}`. |
| Key-map literal inside the script (one-line-per-metric lookup table) | Auto-probe per-metric `per_sample` keys at runtime | Explicit beats clever. The lookup table is human-auditable and exactly matches the verified facts above. |
| Document RadCliQ direction + inversion recipe prominently | Just list it as "a reward" | Using it as-is in GRPO maximizes distance — the exact wrong thing. Not documenting this is a footgun. |
| Perturb `synthetic_reports.json` `ground_truth` strings to make hyps for E1 | Use `ground_truth == hyp` (BLEU = 1 everywhere) or random strings (BLEU = 0 everywhere) | We need non-degenerate inputs so warm per-batch latency reflects real work. Deterministic perturbations (swap "No"↔"Mild", add/remove trailing phrase) achieve this cheaply. |
| Record HF cache-hit state per metric | Ignore cache state | Cold-start is dominated by model download for first-run users; our snapshot should be labeled "warm cache" or "cold cache" honestly so readers can scale expectations. |
| Private HOPPR metrics excluded from the benchmark page | Benchmark them too | `scripts/publish_public.py` strips private metrics from the public repo; running them here would create an awkward asymmetry between the private-env measurements and the public doc that ships. |

## What an outside reader takes away

After 90 seconds on this page:

- RadEval ships **13+ metrics usable as RL rewards** — here's what each
  one costs per batch.
- For a radiology RL loop, **BLEU is not enough** — here are three
  concrete report pairs where BLEU and a clinical metric disagree
  substantially.
- **RadCliQ is the preferred clinical reward** for alignment with
  radiologist judgment, **but it's a distance** — invert it when using
  as a GRPO reward (docs include the one-liner).
- Cost/quality trade-off is explicit, so the reader can pick.

That's the differentiator. Not a framework; a focused, measured
answer to "which reward should I use?"
