# Plan Revision Changelog — Round 02

## Input
- previous plan: round_01/revised.md
- critic feedback: round_02/feedback_oai.md (revise_major), round_02/feedback_gem.md (revise_minor)
- current revision: round_02/revised.md (produced before these critics ran) → changes flow into round_03/revised.md

## Accepted Critiques

### Scope language (both)
- **OpenAI high: "every practical" / "usable as RL rewards" overclaims.** Round 3 replaces with "benchmarked public metrics selected as plausible RL rewards in this environment" in every scope-claim location. Inclusion/exclusion explicitly labeled as "judgment calls for this snapshot."

### Methodology simplifications (OpenAI)
- **OpenAI high: drop `first_download_s` entirely.** Round 3 deletes the anecdotal first-download capture. The primary E1 artifact is (a) `cached_init_s` after a `prewarm` pass and (b) warm batch latency. Simpler, honest, fully comparable across metrics.
- **OpenAI medium: VRAM contamination.** Round 3 keeps VRAM but moves it to a secondary column labeled "Peak VRAM (approx)" and adds an explicit caveat: "in-process-measured; contaminated by allocator state; read as directional, not absolute."
- **OpenAI medium: isolation rule underspecified.** Round 3 replaces the ad-hoc `--isolate <metric>` flag with a **predefined rule: `radcliq` is always benchmarked in its own subprocess** (it's the only composite and the only likely VRAM offender). Other metrics always run in-process. Simple, deterministic, zero flag surface.
- **OpenAI medium: key-map drift.** Round 3 adds a startup validation pass: for each metric, call `cls().compute(refs=['a'], hyps=['a'], per_sample=True)` and assert that the configured `key` exists in the result. Any drift fails that metric fast with a clear message.
- **OpenAI: drop `trl` from env metadata.** Round 3 records `torch`/`transformers`/`python`/`cuda`/`gpu`/`RadEval` only. TRL isn't in the benchmark execution path.

### Gallery framing (OpenAI)
- **OpenAI high: selection-on-the-outcome.** Round 3 strips the "differentiator" language and replaces it with a one-paragraph honest framing: "curated examples chosen to illustrate that disagreement *can* happen on plausible clinical inputs; prevalence and average behavior are out of scope for this doc." Success criterion rewritten to match.

### Fixes (Gemini)
- **Gemini medium: contradictory workload spec.** Round 3 deletes the stale "Perturb `synthetic_reports.json`" row from the Decisions table — that was an earlier draft; the static fixture is the committed plan.
- **Gemini medium: `1/(1+x)` unsafe for unbounded RadCliQ.** Round 3 removes the bounded-inversion suggestion. The only recommended inversion is `score_transform=lambda x: -x`, which is safe for any range.

### Overengineering (both agree)
- **Both flagged the `--isolate` CLI.** Round 3 drops the arbitrary-metric `--isolate` flag. The predefined rule ("`radcliq` always runs as a subprocess") is handled internally inside the `all` command — no user-visible flag.

## Rejected Critiques
- **OpenAI medium: drop VRAM from the main table entirely.** Partial rejection. Round 3 *downgrades* VRAM to a secondary "approx" column rather than removing it, because VRAM is exactly the kind of directional info a user planning a GPU budget needs. We keep it but label it approximate and explain why.

## Uncertain / Deferred
- Whether the `radcliq`-in-subprocess split complicates the single-JSON output. Decision: the main process spawns `radcliq` once, reads its JSON from stdout, merges into the full snapshot. ~20 lines. Not exposed as a user knob.

## Major Plan Changes (Round 03)
- Scope: narrow all "every practical" / "usable" / "differentiator" language.
- E1: drop first-download; report only `cached_init_s` + `warm_batch` + `peak_vram_approx`.
- Key-map startup validation added (5 LOC).
- `radcliq` always isolated to a subprocess (predefined rule, no flag).
- Drop arbitrary `--isolate` flag.
- RadCliQ inversion guidance: `-x` only, no `1/(1+x)`.
- Delete contradictory perturbation row from Decisions table.
- Drop `trl` version from env metadata.

## Net Effect
- scope: reduced (dropped `first_download_s`, `--isolate` flag, `1/(1+x)` inversion, `trl` metadata)
- confidence: increased — two strong critics converged on the same simplifications, methodology is now cleaner and less ambiguous.
