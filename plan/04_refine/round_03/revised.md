# Plan 04 — RadEval RL benchmarks page

## Goal

Ship `docs/trl_rewards_benchmarks.md`: a small, self-contained write-up
that makes the "use RadEval metrics as RL rewards" story *concrete and
compelling* to an outside reader browsing the repo. Not a long
evaluation paper — a short document with two measured artifacts:

1. **Speed table** — cached-init + warm per-batch latency (+
   approximate peak VRAM) for every in-scope metric on a fixed small
   workload. Turns the current prose speed tiers ("fast / medium /
   slow") into numbers.
2. **Reward divergence gallery** — a hand-picked set of report pairs
   scored by a few metrics side-by-side, illustrating that lexical and
   clinical metrics *can* disagree on plausible inputs. Labelled as
   illustrative, not representative.

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

**Speed table: benchmarked public metrics selected as plausible RL
rewards in this environment.** Inclusion is a **judgment call for
this snapshot**, not a universal claim.

- **Included:** public metrics registered in
  `RadEval/metrics/_registry.py` that, in the author's judgment,
  plausibly run fast enough for per-step use on a single-GPU GRPO
  training loop on the benchmark machine — as verified by the
  measurements in this document.
- **Excluded — explicit, with reasons:**
  - API-based metrics (`crimson`, `mammo_green`, `radfact_ct`) — one
    HTTP call per sample; not RL-eligible at all.
  - Very-large local LLM metrics (`green`, 7B model) — technically
    runnable but cost dominates the training step.
  - Private HOPPR metrics (`f1hopprchexbert*`, `hoppr_crimson_ct`,
    `nodule_eval`) — ship only in the private tree.

A reader on different hardware may reach a different verdict on
"plausible"; this doc aims to give them concrete numbers to base
that on, not to declare a universal tier ranking.

Call shapes and required `key=` values are **verified against the
current registry + adapter code** by actually instantiating each
scorer and inspecting its `per_sample=True` output keys (probed
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

**Workload.** Static 20-row `docs/benchmarks/fixtures/speed_workload.json`
with hand-written `{ref, hyp}` pairs designed to produce
non-degenerate BLEU across the batch (some near-paraphrases, some
partial matches, some mismatches — no perturbation logic inside the
script). Committing the workload as data keeps benchmarking code
free of string-munging and makes the input deterministic and
inspectable. 20 samples is small enough that `radcliq` completes in
minutes; large enough that per-sample latency isn't dominated by
measurement noise.

**Measurements per metric (two columns, clean methodology):**

1. **Cached init time (`cached_init_s`) — primary.** Precondition:
   models pulled into `~/.cache/huggingface` by a one-time
   `bench_rewards.py prewarm` pass run before measurement. Captured
   as `t0 = time.perf_counter(); scorer = cls(); t1 = ...` with warm
   disk cache and no network. Directly comparable across metrics.
2. **Warm per-batch latency (`warm_batch_s_median`)** — call
   `r(completions=hyps, ground_truth=refs)` once to warm caches,
   then time 3 calls over the full 20-row batch and take the median.
   Report seconds/batch (20 samples) and ms/sample.

**Secondary (approximate) column:**

3. **Peak VRAM delta (`peak_vram_mb_approx`)** — if
   `torch.cuda.is_available()`:
   `torch.cuda.reset_peak_memory_stats()` before cached init,
   `torch.cuda.max_memory_allocated()` after the three warm calls,
   diff in MB. **Labeled "approx" in the table.** Known limitations:
   (a) only counts torch allocations, not ONNX / raw CUDA;
   (b) in-process measurement can be contaminated by allocator
   caching; (c) `radcliq`'s number is measured in its own
   subprocess (see below) so that contamination doesn't leak into
   subsequent metrics. VRAM is shown as directional guidance for
   GPU budget planning, not absolute truth.

**First-download time is NOT reported.** It depends on network speed
+ HF mirror behavior and is not controllable; mixing it with
init time produces numbers nobody can compare. The `prewarm` pass is
mandatory.

**Implementation notes:**

- **Default: in-process loop over metrics**, with explicit teardown
  between metrics (`del scorer; gc.collect(); torch.cuda.empty_cache()`).
  The `cached_init_s` measurement is taken at `cls()` time — prior
  state cannot leak into init timing.
- **Predefined rule: `radcliq` is always benchmarked in its own
  subprocess.** No user-visible flag. `radcliq` is the only composite
  metric (BERTScore + SembScore + RadGraph stacked), so its VRAM
  footprint and cleanup behavior differ from everything else —
  isolating it prevents state contamination of subsequent metrics'
  VRAM measurements. Implemented inside the `all` command: the main
  process runs every metric except `radcliq` in-process; `radcliq`
  is spawned via `subprocess.run` as a one-off with its own
  `--metric radcliq` invocation of this same script; the child's
  stdout JSON is merged into the parent snapshot. ~20 LOC.
- **`prewarm` subcommand**: one-time pass that instantiates each
  scorer once so model weights land in the HF cache. Purpose is
  solely to make `cached_init_s` measurements honest.
- **Startup key-map validation** (new): before timing, for each
  metric the script instantiates the scorer with two trivial inputs,
  calls `compute(refs=['a'], hyps=['a'], per_sample=True)`, and
  asserts the configured `key` (from the literal key-map at the top
  of the script) is present in the output. If absent, the metric is
  marked `"skipped": "key-drift:<details>"` — fast, loud failure on
  adapter drift.
- **Environment metadata** at the top of the JSON:
  `RadEval.__version__`, `torch.__version__`,
  `transformers.__version__`, Python version, CUDA version, GPU
  model, HF cache path. **`trl` is NOT recorded** — not in the
  benchmark execution path.
- Skip metrics whose `cls(**kwargs)` raises `FileNotFoundError` /
  `ImportError` / `HfHubHTTPError` — stderr log, continue with
  `"skipped": "<reason>"`.
- **OOM handling**: each metric's scoring call is wrapped in
  `try/except (torch.cuda.OutOfMemoryError, RuntimeError)`; on OOM
  record `"skipped": "OOM"`, clear torch cache, continue.
- Multi-key metrics: pass the exact `key=` from the verified key-map
  above. Record which key was timed in the JSON.

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

**Inputs — 8-10 hand-curated `{id, ref, hyp, narrative}` rows** in
`docs/benchmarks/fixtures/divergence_examples.json`. Colocating
benchmark data under `docs/benchmarks/fixtures/` (not `tests/fixtures/`)
keeps benchmark assets clearly separate from test fixtures — these
are inputs to a documentation script, not pytest data. The
`speed_workload.json` from E1 lives beside it.

```
# .gitignore
tests/**/*.json
!tests/fixtures/synthetic_reports.json
# (no new exceptions needed — docs/benchmarks/fixtures/ is already
#  outside the tests/**/*.json pattern)
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
  distance from references** — the opposite of what a user wants. The
  doc includes the safe one-liner: `make_reward_fn("radcliq",
  score_transform=lambda x: -x)`. We do NOT recommend a bounded
  inversion like `1/(1+x)` — RadCliQ is a linear-regression composite
  and its theoretical range isn't strictly non-negative; division-based
  transforms can asymptote. Negation is universally safe.

## Script design — `scripts/bench_rewards.py`

Single-file, ~200-250 LOC. Two commands plus one hidden internal mode:

- `python scripts/bench_rewards.py prewarm` — instantiate every
  in-scope metric once so HF cache is warm. Run this before
  measuring. Writes nothing to disk other than HF cache.
- `python scripts/bench_rewards.py all --output docs/benchmarks/trl_rewards_260429.json`
  — run the full benchmark (E1 + E2) and write a single JSON
  snapshot.
- `python scripts/bench_rewards.py --metric <name>` — **internal**
  entry point used by `all` to spawn a subprocess for `radcliq`
  (see "predefined rule" above). Documented here for reproducibility
  but not exposed as the primary user-facing workflow.

Uses only stdlib + the installed `RadEval` env. No new deps.

**Default path: in-process loop** — construct each non-`radcliq`
scorer in order, take its `cached_init_s` and VRAM baseline, run
warm-batch timing, tear down (`del`, `gc.collect`,
`torch.cuda.empty_cache()`), move on. When the loop reaches
`radcliq`, `subprocess.run` spawns a fresh invocation with
`--metric radcliq`, captures its JSON via stdout, and merges into
the full snapshot. Single JSON emitted at the end.

**Error handling:**
- Load failure (`FileNotFoundError`, `ImportError`, `HfHubHTTPError`,
  `TimeoutError`) → record `"skipped": "<reason>"`, continue.
- OOM at scoring time
  (`torch.cuda.OutOfMemoryError` / `RuntimeError` with "out of
  memory") → record `"skipped": "OOM"`, clear torch cache, continue.
- Any other unexpected exception → record `"skipped": "<exc type>:
  <msg truncated>"`, continue. No metric failure aborts the whole
  run.

**Runtime budget:** E1 for the fast+medium metrics should finish in
≤ 10 minutes on the dogfood GPU; `radcliq` alone may take longer
because it composes BERTScore + SembScore + RadGraph sequentially.
If `radcliq` blows the budget, fall back to `--isolate radcliq` (see
above); the main path is unaffected.

## Doc — `docs/trl_rewards_benchmarks.md` outline

Short, opinionated, ~1.5 screens:

```markdown
# RadEval metrics as RL rewards — speed & divergence

*Snapshot: RadEval 2.2.0, TRL 1.3.0, transformers 5.6.2, torch 2.9.1,
NVIDIA <GPU>. Numbers will drift on other hardware / future releases;
re-run `scripts/bench_rewards.py` to refresh.*

## Per-batch cost (20 samples)

<speed table here — one row per metric, columns: cached init, warm
batch (s/batch), ms/sample, peak VRAM (approx)>

*`cached init` is `cls()` time after HF models are already on disk.
`peak VRAM` is `torch.cuda.max_memory_allocated` delta — directional
guidance for GPU budget planning, **not absolute truth**: it only
counts torch allocations and in-process measurement can be
contaminated by allocator caching; `radcliq` is measured in its own
subprocess to keep its footprint from leaking into adjacent rows.
First-download time is not reported — it's network-dependent and
not useful for comparison.*

**Takeaways.** (2-3 bullets.)
- `bleu` / `rouge` are essentially free — use as a sanity baseline or
  cheap auxiliary reward.
- `radeval_bertscore` is the cheapest *semantic* option; one model
  load, <N> ms/sample warm.
- `radcliq` is the costliest because it composes three transformer
  passes sequentially (BERTScore + SembScore + RadGraph), so its
  per-sample latency is roughly the sum of those three. Position it
  as an **eval-time** reward or a final-tuning reward, not a
  primary online signal.

## Where clinical metrics disagree with BLEU

<divergence gallery — 8-10 rows, 5 metrics wide, ref/hyp column + one
narrative-line column>

*(All scores: ↑ = better except RadCliQ ↓ = better; direction is
shown in the column header.)*

> **Note on method.** This gallery is hand-picked to illustrate that
> lexical and clinical metrics *can* disagree on plausible clinical
> inputs. It is not a representative sample, not a randomized
> comparison, and not evidence of average metric superiority —
> examples were selected specifically to exhibit disagreement. Treat
> the rows as existence proofs, not prevalence estimates. For
> population-level comparisons, see the RadEval paper and the
> RadCliQ reference studies.

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
  `radcliq`. Expect higher per-sample cost (≈ BERTScore + SembScore +
  RadGraph stacked) and **remember to invert** — RadCliQ is a
  distance, so for maximizing-reward training use
  `make_reward_fn("radcliq", score_transform=lambda x: -x)`. Negation
  is universally safe; we don't recommend bounded inversions like
  `1/(1+x)` because RadCliQ's theoretical range isn't strictly
  non-negative.

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

- Speed table covers every in-scope metric (per the explicit
  inclusion rule). Reader can pick one based on measured cost.
- Divergence gallery includes at least one row where lexical and
  clinical metrics disagree by a large margin — a concrete
  existence proof that metric choice matters.
- Both artifacts are self-contained — the page cites the JSON
  snapshot and script path so anyone can verify.
- The page is short enough (< 2 screens) that a skimmer reaches the
  takeaways.
- Framing is calibrated: gallery is explicitly illustrative (not
  representative), VRAM is explicitly approximate, inclusion is
  explicitly a judgment call for this snapshot.

## Commit structure (3 commits)

1. `chore(scripts): add bench_rewards.py` — the script +
   `docs/benchmarks/fixtures/{speed_workload,divergence_examples}.json`.
   No `.gitignore` edit needed (the fixtures live outside
   `tests/**/*.json`).
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
  the curated pairs + their narrative labels as static data in
  `docs/benchmarks/fixtures/divergence_examples.json`; the doc
  explicitly labels the gallery as illustrative, not representative.
  What we're showing is deterministic given the pairs.
- **Numbers will not match exactly on other hardware.** That's fine —
  the document is a dated snapshot. The page says this explicitly.
- **Per-sample mode for a metric may raise a required-`key=` error**
  (ROUGE, RadGraph, F1CheXbert, SRRBert). Mitigation: benchmark
  script embeds the verified key-map literal at the top.
- **`radcliq` takes longer than the runtime budget.** Mitigation in
  order of escalation: (a) try it in-process first; (b) if it blows
  the budget or perturbs subsequent metrics' VRAM numbers, rerun
  with `--isolate radcliq` in a separate subprocess and merge the
  JSON output. No per-metric hard timeout.
- **In-process state leak between metrics.** Mitigation: explicit
  teardown (`del scorer; gc.collect(); torch.cuda.empty_cache()`)
  plus the `--isolate` escape hatch. If any metric's VRAM measurement
  looks visibly tainted (e.g. massively larger than a clean baseline),
  we rerun with `--isolate`.

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
| `docs/benchmarks/fixtures/{speed_workload,divergence_examples}.json` | Reuse `tests/fixtures/synthetic_reports.json` | Wrong schema (`{prompt, ground_truth}` vs `{ref, hyp}`); also `tests/fixtures/` is for pytest assets, not doc data. |
| Static hand-written `speed_workload.json` with diverse pairs | Perturb `synthetic_reports.json` strings at runtime | KISS — data belongs in a file, not in runtime munging logic. Static is inspectable, reproducible, and keeps the script focused on measurement. |
| Key-map literal inside the script + runtime validation | Auto-probe per-metric keys | Explicit beats clever. Literal is human-auditable; startup validation fails fast on future drift. |
| Document RadCliQ direction + negation inversion prominently | List as "a reward" with no caveat | Using it as-is maximizes distance — the exact wrong thing. Not documenting this is a footgun. |
| Negation (`-x`) only for RadCliQ inversion | Bounded `1/(1+x)` | RadCliQ is a linear-regression composite; its range isn't strictly non-negative. Division-based transforms can asymptote or flip sign. |
| Primary table: `cached_init_s` + `warm_batch_s` only | Include `first_download_s` | First-download is network-dependent and not comparable across runs. Dropping it makes the table honest. |
| `radcliq` always isolated to a subprocess (predefined rule, no flag) | Optional `--isolate <metric>` | "Predefined" is predictable; an ad-hoc flag makes results vary with how the user invokes. RadCliQ is the only composite metric, so the rule is trivial. |
| VRAM kept in the table but labeled `approx` | Drop VRAM entirely | VRAM is directionally useful for GPU budget planning; just be explicit that it's approximate (allocator-dependent, torch-only). |
| No `trl` version in env metadata | Record `trl` | TRL isn't in the benchmark execution path — the benchmark calls `make_reward_fn` directly, not through TRL. |
| Private HOPPR metrics excluded | Benchmark them too | `scripts/publish_public.py` strips private metrics from the public tree; running them here would create an asymmetric doc. |

## What an outside reader takes away

After 90 seconds on this page:

- RadEval exposes a dozen public metrics as TRL-compatible rewards;
  here are measured numbers for each on a fixed workload and hardware.
- On hand-picked examples, lexical and clinical metrics can diverge
  sharply — so "which reward to use" is a real question with
  tradeoffs, not a formality.
- RadCliQ correlates best with radiologist judgment in the published
  literature, has ≈3× the per-sample cost of its heaviest component,
  and is a *distance* metric — the doc shows the one-liner to invert
  it for maximize-reward training.
- Cost/quality trade-off is explicit, so the reader can pick.

A focused, measured answer to "which reward should I use?" — grounded
in a small, transparent set of measurements.
