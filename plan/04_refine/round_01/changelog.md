# Plan Revision Changelog — Round 01

## Input
- previous plan: plan/04.md (original)
- critic feedback: round_01/feedback_oai.md (revise_minor), round_01/feedback_gem.md (approve)
- current revision: round_01/revised.md

## Accepted Critiques
- **OpenAI medium: "every RL-eligible metric" overclaims (`green` excluded on practicality, not eligibility).** Round 2 narrows language to "every practical public metric for per-step RL rewards" and states the exclusion rule explicitly (API metrics + very-large local LLMs).
- **OpenAI medium: cold-start conflates init with download.** Round 2 splits into a primary `cached_init_s` measurement taken after a `--prewarm` pass (always reported; directly comparable across metrics) and an optional `first_download_s` anecdote captured only when the HF cache was cold. Drops the per-metric `cache_hit` heuristic.
- **OpenAI medium: divergence gallery risks cherry-picking claims.** Round 2 adds a one-line disclaimer to the doc outline: "illustrative curated examples, not a representative sample."
- **OpenAI low: `tests/fixtures/` is for test assets, not benchmark data.** Round 2 moves the new fixture to `docs/benchmarks/fixtures/divergence_examples.json` (colocated with the snapshot). Also moves the E1 workload fixture to the same dir. `.gitignore` exception targets the new path.
- **OpenAI overengineering: subprocess-per-metric.** Round 2 simplifies to an in-process loop per metric with explicit torch.cuda cleanup between metrics. Subprocess isolation is retained as an optional `--isolate <metric>` flag only if a specific metric misbehaves — YAGNI default.
- **OpenAI overengineering: append-mode JSON.** Round 2 ships single `all` mode; no append semantics.
- **OpenAI overengineering: HF cache-hit heuristic.** Round 2 drops it; relies on the clean `cached_init_s` measurement taken after prewarm.
- **Gemini overengineering: dynamic perturbation for E1 workload.** Round 2 uses a **static 20-row `speed_workload.json` fixture** with hand-written (ref, hyp) pairs (non-degenerate BLEU across the batch). No perturbation logic inside the script.
- **Gemini: document RadCliQ cost decomposition.** Round 2 adds to the decision guide: "RadCliQ cost ≈ BERTScore + SembScore + RadGraph because it composes three transformer passes sequentially."
- **Gemini: subprocess OOM handling.** Since subprocess-per-metric is dropped, OOM in the main loop is handled with a try/except around each metric's scoring call that records `"skipped": "OOM"` and clears torch cache before continuing.

## Rejected Critiques
- None from round 1. All points are actionable and low-risk.

## Uncertain / Deferred
- Whether to run `radcliq` inside its own subprocess after all (in case the three-model composite creates VRAM pressure that interferes with subsequent metrics). Decision: try in-process first; fall back to `--isolate radcliq` only if we observe interference during implementation. Keeps the default path simple.

## Major Plan Changes (Round 02)
- "every RL-eligible metric" → "every practical public per-step RL reward."
- Cold-start: split `cached_init_s` (primary, always reported) vs `first_download_s` (anecdotal, only when cache was cold).
- Drop `cache_hit` per-metric boolean.
- Drop subprocess-per-metric by default; keep `--isolate` as an optional fallback.
- Drop append-mode; single `all` command.
- Move `divergence_examples.json` and new `speed_workload.json` under `docs/benchmarks/fixtures/`.
- Static hand-written `speed_workload.json` (no perturbation logic).
- Doc: add "illustrative, not representative" disclaimer on the gallery.
- Doc: note RadCliQ composite cost breakdown in the decision guide.

## Net Effect
- scope: reduced (drops subprocess orchestration, heuristic cache detection, perturbation code)
- confidence: increased (two critics converged on the simplifications; key-map and RadCliQ-direction facts are now verified)
