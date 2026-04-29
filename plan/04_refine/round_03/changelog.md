# Plan Revision Changelog — Round 03

## Input
- previous plan: round_02/revised.md
- critic feedback: round_03/feedback_oai.md (revise_major), round_03/feedback_gem.md (revise_minor)
- current revision: round_03/revised.md → flows into round_04/revised.md

## Accepted Critiques

### Genuine bugs caught (Gemini high)
- **Gemini HIGH: key-map validation before timing pollutes `cached_init_s`.** This is a real bug in the round-3 methodology: doing a validate-keys pass instantiates every model, warming HF disk cache and creating CUDA contexts, which makes the *measured* `cached_init_s` artificially fast. **Round 4 folds key validation into the warm-up call inside the main timing loop.** Scoring the warm-up batch (`compute(refs, hyps, per_sample=True)` on the 20-row workload) yields the key dict, which we then validate; if the configured key is missing, record `"skipped": "key-drift:<details>"` and continue.
- **Gemini overengineering: `radcliq` subprocess isolation is overkill — sort it last instead.** Accepted. Running `radcliq` last in the in-process loop means its VRAM state can't contaminate anything downstream (there's nothing downstream). Drops ~20 LOC of subprocess+merge logic and the internal `--metric` entry point.

### Framing (OpenAI high, partial)
- **OpenAI high: residual broad claims ("every in-scope metric", "13+ metrics usable").** Accepted. Round 4 sweeps the remaining "13+" and "every in-scope" in the executive summary, outside-reader section, and success criteria. Replaces with "a dozen public metrics in this snapshot" / "the metrics benchmarked here."
- **OpenAI medium: placeholder quantitative claims ("≈3× the cost").** Accepted. Round 4 replaces with placeholder-style language: "roughly the sum of its three components — exact factor in the measured table below."
- **OpenAI medium: gallery-to-recommendation linkage.** Accepted. Round 4 explicitly grounds the "RadCliQ correlates best" claim in external literature (RadCliQ paper / RadEval paper), not in the curated gallery. The gallery is purely illustrative of *disagreement possibility*.

### Rigor (OpenAI medium, pragmatic)
- **OpenAI medium: cross-metric contamination beyond `radcliq`.** Partial accept. Rather than isolating every model-backed metric (big scope bump), round 4 adds a **one-time validation gate during implementation**: run one representative fast metric (BLEU or ROUGE) and one representative model-backed metric (BERTScore or F1CheXbert) in both in-process (with `radcliq`-last) and isolated-subprocess modes; if the two measurements agree within ~10% for warm-batch latency and ~50 MB for VRAM, proceed with the in-process plan and note this in the doc. If they diverge more, escalate to subprocess-per-model-backed-metric. Concrete gate, not hand-waving.
- **OpenAI medium: narrow OOM detection.** Accepted. Round 4 specifies: catch only `torch.cuda.OutOfMemoryError` and `RuntimeError` *whose `str()` contains "out of memory"*; any other `RuntimeError` is recorded as a distinct `"skipped": "runtime-error:<msg>"`.
- **OpenAI medium: add minimal script logic tests.** Partial accept. Round 4 adds a tiny `tests/test_bench_rewards_logic.py` covering only (a) the skip-record shape, (b) JSON-merge output schema, (c) the key-map lookup function. No measurement tests, no full integration — just defensive tests on the pure-logic pieces. ~30 LOC.

### Also (Gemini assumption-checks)
- **Gemini: CUDA context baseline.** Accepted. Round 4 changes VRAM measurement: capture `torch.cuda.memory_allocated()` *immediately before* `cls()` for that specific metric (per-metric baseline), not a global zero baseline. This accommodates the CUDA-context residual footprint.
- **Gemini: negation sign-sensitive RL checks.** Accepted as a note. Round 4 mentions in the docs "note: TRL GRPO handles negative rewards fine, but if you're using a different trainer that asserts `>= 0`, add a positive offset." One line.

## Rejected Critiques
- **OpenAI: isolate every model-backed metric in its own subprocess.** Rejected in favor of the measurement-gate compromise above. Isolating every metric doubles the script complexity for little benefit if the gate confirms in-process measurement is good enough. Keeps the script in its ~200 LOC envelope.
- **OpenAI overengineering: drop date-stamped snapshot naming.** Rejected — keeping `trl_rewards_260429.json` date-stamped is 0 LOC of extra logic and costs nothing. If we ever re-snapshot we want the old file preserved.
- **OpenAI overengineering: three-commit structure adds review overhead.** Rejected. The commit split is standard separation of concerns (script / snapshot+doc / cross-links) and aids review. No downside.

## Uncertain / Deferred
- Whether the in-process-with-radcliq-last measurement gate actually passes. If it fails during implementation, we escalate per the plan (subprocess per model-backed metric). Recorded in Risks.

## Major Plan Changes (Round 04)
- Key-map validation moves into the warm-up call inside the main loop (fixes Gemini high).
- `radcliq` subprocess replaced by "sort `radcliq` last in the in-process loop" (drops ~20 LOC).
- Add an implementation-time measurement gate comparing in-process vs isolated on one fast + one model-backed metric.
- Replace residual "every in-scope" / "13+ metrics" language with "metrics benchmarked here."
- Replace quantitative placeholder claims in the doc outline with "exact factor in the measured table below."
- Narrow OOM detection criteria.
- Add ~30-LOC `tests/test_bench_rewards_logic.py` for pure-logic pieces.
- Per-metric VRAM baseline (capture just before `cls()`), not global zero.
- Doc note on negative rewards + hypothetical trainer assertions.

## Net Effect
- scope: reduced (dropped subprocess merge path; simpler sort-last rule)
- confidence: increased — caught a real bug (Gemini), framed the OpenAI comparability concern as a concrete implementation gate rather than unbounded scope growth.

## Stopping posture

Round 4 is intended to be final. OpenAI has gone `revise_major` twice in a row, but the second wave of critiques was primarily about wording calibration and a real-but-addressable comparability concern. With: (a) the Gemini high-severity bug fixed, (b) comparability bounded by the measurement gate, (c) quantitative claims replaced with placeholders, and (d) residual "every" language swept — the plan is as tight as it's going to get without turning into a research paper. If round-4 critics still push revise_major on framing, we'll stop at round 5 per max-iterations and accept that at some point this is just a plan for a doc page, not a defended methodology paper.
