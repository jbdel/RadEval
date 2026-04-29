[Back to README](../README.md) · [Back to RL rewards](./trl_rewards.md)

# RadEval metrics as RL rewards — speed & divergence

This page is for readers asking **"I'm about to wire up a RadEval metric
as a GRPO reward. What is the computational cost, and how much does metric choice
matter?"** Two artifacts answer those questions:

1. A **speed table** — measured per-sample cost for every public RadEval
   metric, from sub-millisecond BLEU to seconds-per-sample for 7B-local-LLM
   and API-backed metrics.
2. A **reward-divergence gallery** — same rollouts, scored by five
   metrics, showing how reward choice changes what the GRPO policy
   learns.

## Per-batch cost (20 samples)

| Metric | `key=` | Cached init | Per-sample | Peak VRAM |
|---|---|---:|---:|---:|
| `bleu` | — | 0.00 s | 0.09 ms | CPU |
| `rouge` | `rouge1` | 0.05 s | 0.16 ms | CPU |
| `bertscore` | — | 7.65 s | 0.64 ms | 260 MB |
| `radeval_bertscore` | — | 0.95 s | 1.14 ms | 606 MB |
| `f1chexbert` | `f1chexbert_sample_acc_5` | 3.26 s | 1.75 ms | 434 MB |
| `f1radbert_ct` | `f1radbert_ct_sample_acc` | 0.76 s | 2.03 ms | 485 MB |
| `srrbert` | `srrbert_weighted_f1` | 0.64 s | 4.97 ms | 420 MB |
| `temporal` | — | 0.42 s | 17.33 ms | 191 MB |
| `ratescore` | — | 2.20 s | 52.98 ms | 1130 MB |
| `radgraph` | `radgraph_partial` | 3.87 s | 157.79 ms | 890 MB |
| `radgraph_radcliq` | — | 3.74 s | 160.36 ms | 890 MB |
| `radcliq` | — | 4.45 s | 161.13 ms | 1175 MB |
| `green` | — | 0.00 s † | **2 210 ms** | MP ‡ |
| `crimson` | — | 1.00 s | **379 ms** (API) | CPU |
| `mammo_green` | — | 0.01 s | **288 ms** (API) | CPU |
| `radfact_ct` | `radfact_ct_f1` | 0.01 s | **238 ms** (API) | CPU |

Column definitions:
- **Cached init:** `scorer = cls()` wall time after HF models are on
  disk. Network download time is not included.
- **Per-sample:** median per-sample latency across 3 warm-batch scoring
  calls on a 20-sample workload, after a warm-up call. API metrics show
  "(API)" because they're dominated by network round-trip, not compute.
- **Peak VRAM:** `torch.cuda.max_memory_allocated` delta in the parent
  process. CPU-only metrics show "CPU."

† `green` does lazy model loading on first compute call; most of its
cost shows up per-sample, not in init.

‡ `green` runs its 7B model via a multiprocessing pool, so the parent
process sees no VRAM delta; actual usage is ~14 GB in a worker.

### Takeaways

- **The actionable takeaway is relative ordering, not absolute
  numbers.** Absolute latencies vary with hardware, report length,
  and batch size (see "Cost depends on…" below), but the ordering
  BLEU < BERTScore < F1CheXbert < RadGraph < RadCliQ < GREEN ≪ API
  metrics holds up across reasonable GPU setups.
- **BLEU / ROUGE are essentially free** (sub-0.2 ms/sample, CPU): use
  as a sanity baseline or a cheap auxiliary reward.
- **BERTScore is the cheapest semantic option** at 0.64 ms/sample.
- **F1CheXbert / F1RadBERT-CT / SRRBert** cluster in the 2–5 ms range:
  affordable per-step rewards for CXR/CT workflows.
- **RadGraph and its derivatives** (including RadCliQ) jump to
  ~160 ms/sample; 100× the F1CheXbert cost. Tractable but significant
  for large-batch training.
  - RadCliQ, the costliest metric in this tier at ~161 ms/sample,
    correlates best with radiologist judgment in published studies
    ([RadCliQ paper](https://www.cell.com/patterns/fulltext/S2666-3899(23)00157-5)).
- **GREEN is ~2.2 seconds per sample**, ~15,000× BLEU. Label-accurate
  via a 7B radiology LLM, but **not a per-step RL reward**; treat it
  as an evaluation-only metric.
- **API-backed metrics (CRIMSON / MammoGREEN / RadFact-CT) are
  200–400 ms per sample**, dominated by network round-trip. **Not
  practical for online RL** at any useful training throughput; the
  `make_reward_fn` call emits a `UserWarning` when you wrap them.

### Cost depends on several factors

These numbers are **one snapshot for one machine and one workload**.
What they're measuring varies with:

- **Report length:** most metrics tokenize to `max_length=512`.
  Longer reports cost more; shorter reports cost less. The 20-sample
  fixture's reports average ~8–15 tokens each, so **this table is
  closer to a lower bound than a realistic production workload**.
- **Effective batch size:** each metric has its own internal batching
  (BERTScore 64, SRRBert 4, CheXbert-family variable). Larger real
  batches amortize overhead and bring per-sample cost down; a
  20-sample batch is small enough that fixed-cost overhead still
  shows.
- **Hardware class:** measured on A100-SXM4-80GB. Consumer GPUs
  (3090/4090, 24 GB) should see roughly similar per-sample latency on
  compute-bound metrics (RadGraph / RadCliQ), but may pay more on
  memory-bandwidth-bound rows.
- **Padding effects:** single-batch latency is dominated by the
  longest report in the batch. Truncation / bucketing changes the
  shape of the curve.

## What reward would this rollout receive?

Same setup you're imagining: a GRPO trainer samples a candidate
completion for each prompt, the reward function scores each sample,
the score becomes the training signal. Below: 5 curated `(reference,
candidate rollout)` pairs scored by five metrics (BLEU, BERTScore,
RadGraph, RadCliQ, CRIMSON). Each row answers "which reward would
push the policy toward or away from this rollout?"

Reward direction: ↑ = higher is better reward signal, ↓ = lower is
better (RadCliQ only). For RL training with RadCliQ, see the
inversion recipe in "Considerations" below.

Row 1 is an **exact-match sanity row**; it fixes the ceiling for each
metric so the rest of the table is read relative to it (BLEU /
BERTScore / RadGraph / CRIMSON all max at 1.0; RadCliQ's floor is
~9.37 since it's a distance, lower = better).

| # | Reference | Candidate rollout | BLEU↑ | BERTScore↑ | RadGraph↑ | RadCliQ↓ | CRIMSON↑ |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | No acute cardiopulmonary process. | No acute cardiopulmonary process. | 1.00 | 1.00 | 1.00 | 9.37 | 1.00 |
| 2 | Mild cardiomegaly. | The cardiac silhouette is mildly enlarged. | 0.00 | 0.215 | 0.00 | 10.67 | 1.00 |
| 3 | **No pleural effusion.** | **Pleural effusion.** | 0.00 | **0.893** | 0.50 | 11.06 | **−0.333** |
| 4 | Mild pulmonary edema. | Severe pulmonary edema. | 0.00 | 0.955 | 0.67 | 10.20 | 0.50 |
| 5 | Stable bibasilar atelectasis without consolidation. | New right lower lobe consolidation. | 0.00 | 0.355 | 0.00 | 11.12 | −0.333 |

> **Note on method.** This gallery is a hand-picked set of illustrative
> examples, not a representative sample. We selected columns (from a
> run that scored 7 metrics) and rows (from a run that scored 8
> rollouts) to keep the table readable while preserving the sharpest
> contrasts. F1CheXbert was dropped because its per-sample output has
> narrow dynamic range (three unique values across the set); GREEN was
> dropped because its signal on this gallery closely tracked CRIMSON's
> at roughly double the compute cost. Full 7-column × 8-row data in
> [`docs/benchmarks/trl_rewards_260429.json`](./benchmarks/trl_rewards_260429.json);
> the selection rule is documented inline in
> [`examples/bench_rewards.py`](../examples/bench_rewards.py) next to
> `GALLERY_METRICS`.

### Three rollouts worth zooming in on

The headline is **row 3**. If you skim only one row, make it that one.

1. **Row 2: clinically correct paraphrase.** "Mild cardiomegaly." →
   "The cardiac silhouette is mildly enlarged." Same finding, totally
   different wording. BLEU scores this at zero and RadGraph at zero
   (entity tokens don't match); a BLEU- or RadGraph-reward GRPO run
   would push the policy *away* from the paraphrase. BERTScore 0.22
   (vocabulary-sensitive; doesn't recover the clinical paraphrase as
   cleanly as readers might expect). CRIMSON scores 1.0 — the only
   metric that correctly rewards the clinically-equivalent rewording.
   **Same rollout, opposite gradient across rewards.**

2. **Row 3: flipped negation (headline).** "No pleural effusion." →
   "Pleural effusion." Three of four tokens overlap, and BERTScore's
   embeddings are nearly identical between them, so **BERTScore
   scores 0.893, only slightly below the exact-match ceiling of 1.0.**
   Under a BERTScore reward, GRPO would **partially reward flipping a
   negation**, corrupting the clinical meaning while keeping the text
   "looking right." Clinical metrics flag the flip: RadGraph drops to
   0.50, RadCliQ rises to 11.06 (higher distance = worse), and
   **CRIMSON returns −0.333** — a signed negative in CRIMSON's
   defined range (−1, 1], meaning the report has more weighted
   errors than correct findings
   ([CRIMSON paper, §3.3](https://arxiv.org/pdf/2603.06183)). For
   this specific rollout, −0.333 is exactly what CRIMSON's scoring
   function emits for one actionable-non-urgent hallucinated finding
   against a normal reference. **Even the strongest drop-in NLP
   reward is actively dangerous here.**

3. **Row 4: severity flip.** "Mild pulmonary edema." → "Severe
   pulmonary edema." BERTScore 0.955 (one token differs). RadGraph
   0.67 and CRIMSON 0.5 both partially penalize but don't bottom
   out — severity is real but the finding category (edema) is
   correct. RadCliQ registers the change at 10.20 (midway between
   exact-match 9.37 and full mismatch ~11). For severity-sensitive
   RL, lexical and semantic rewards aren't enough; clinical metrics
   and CRIMSON catch the gradient.

### Considerations when picking a reward

Tradeoffs observed above, in ascending order of per-sample cost:

- **Cheap sanity baseline or auxiliary signal:** `bleu` or `rouge`.
  Sub-ms, CPU. Do not use as the sole reward; row 2 shows why.
- **Semantic default, still cheap:** `bertscore` (0.64 ms/sample).
  Better than BLEU at paraphrase, but **row 3 shows it over-rewards
  negation flips**. If you use it as the sole reward, expect the
  policy to learn that inverting negations is fine.
- **Clinical-finding accuracy (CXR):** `f1chexbert` with
  `key="f1chexbert_sample_acc_5"` at 1.75 ms/sample. Returns per-
  sample agreement rate across 14 CheXpert labels (compressed
  multi-label accuracy), not per-finding F1; a single-label flip
  drops the score by ~1/14 ≈ 0.07 at most. Catches label flips but
  not severity changes. Good primary reward for label-centric tasks;
  dropped from the gallery because its narrow dynamic range is less
  illustrative than RadGraph on the same rollouts.
- **Entity / relation grounding (CXR):** `radgraph` with
  `key="radgraph_partial"` at ~158 ms/sample. Catches omissions,
  hallucinations, and severity changes. Heavier but richer than
  F1CheXbert.
- **Best correlation with radiologist preferences (eval or
  final-tune):** `radcliq` at ~161 ms/sample. A distance, so
  **lower = better**. For RL, invert with
  `score_transform=lambda x: -x`:
  ```python
  from RadEval.rewards import make_reward_fn
  reward = make_reward_fn("radcliq", score_transform=lambda x: -x)
  ```
  Negation is universally safe; bounded inversions like `1/(1+x)`
  are unsafe because RadCliQ's range isn't strictly non-negative.
- **LLM judge with signed output:** `crimson` via OpenAI
  (~380 ms/sample, API-bound). Range **(−1, 1]** per the
  [CRIMSON paper, §3.3](https://arxiv.org/pdf/2603.06183):
  `1` = no errors; `0` = no better than submitting a normal
  template; `<0` = more weighted errors than correct findings,
  asymptotically approaching `−1` as hallucinations accumulate.
  The signed negative is a first-class semantic feature, not an
  artifact; rows 3 and 5 in the gallery both land at −0.333,
  corresponding to one actionable non-urgent false finding
  against a normal reference. The sharpest "this is wrong"
  signal in the gallery. Eval-only regardless: a `UserWarning`
  fires when you wrap CRIMSON as a reward, and the network-bound
  per-sample cost makes it impractical for online RL.
- **Other LLM-backed / eval-only:** `green` (2.2 s/sample, 7B
  local LLM; redundant with CRIMSON on this gallery),
  `mammo_green` and `radfact_ct` (hundreds of ms each, network-
  bound). Fine for final-run evaluation.

---

## Methodology & caveats

Everything above is a snapshot; everything below is how the snapshot
was produced.

**Validated stack.** RadEval 2.2.0, `trl==1.3.0`, `transformers==5.6.2`,
`torch==2.9.1+cu128`, Python 3.11, NVIDIA A100-SXM4-80GB, CUDA 12.8.
Full `env` block recorded in the snapshot file.

**Workload.** `docs/benchmarks/fixtures/speed_workload.json`: 20
hand-written (reference, candidate) pairs in chest-X-ray-report style.
Small on purpose: large enough that per-sample latency isn't dominated
by measurement noise, small enough that the slowest metrics (GREEN,
API-backed) complete in reasonable time.

**Measurement loop.**
- Single in-process Python process; one metric at a time.
- **Cached init** is measured after HF weights are already on disk;
  the canonical snapshot is produced by running the script twice (first
  run warms the HF cache; second run is what we publish). This
  excludes network download time, which isn't comparable across
  runs.
- **Warm-batch latency** is the median of 3 scoring calls after one
  warm-up call. The warm-up call also validates that the configured
  `key=` exists in the output (fails loud on adapter drift).
- **Teardown between metrics** is `del scorer; gc.collect();
  torch.cuda.empty_cache()`. Not sufficient to guarantee identical
  baselines (see VRAM caveat below), but honest best-effort.

**VRAM numbers are approximate** (`peak_vram_mb_approx` in the JSON).
`torch.cuda.max_memory_allocated` only counts torch allocations; it
misses ONNX / raw CUDA buffers and is biased by PyTorch's caching
allocator. **Read VRAM as directional guidance for GPU budgeting, not
as a metric-vs-metric comparison.** For a trustworthy absolute number
for one metric, run that metric in a fresh Python process.

**`radcliq` runs last in the loop** so its composite footprint (it
wraps BERTScore + SembScore + RadGraph) can't leak into adjacent
rows. `green` / API metrics follow `radcliq`.

**Integration tests** (`tests/test_bench_rewards_logic.py`) cover the
orchestration layer (JSON schema, `METRIC_PLAN` ordering, skip
accounting) via a `--dry-run` flag that bypasses model loading. They
do **not** exercise real model loading or metric correctness; the
numerical snapshot is produced and validated by a manual run, not by
CI.

**Reproducing this page.**

```bash
pip install RadEval[rl]

# 1. Warm the HF cache (ignore the output).
python examples/bench_rewards.py --output /tmp/warmup.json

# 2. Generate the canonical snapshot.
python examples/bench_rewards.py \
    --output docs/benchmarks/trl_rewards_$(date -u +%y%m%d).json
```

API metrics (`crimson`, `mammo_green`, `radfact_ct`) require
`OPENAI_API_KEY` (and optionally `GEMINI_API_KEY`) set in the
environment; the script skips them cleanly with `"skipped":
"no-api-key:..."` if unset. Expect ~$0.01-0.05 in API charges per
run against `gpt-4o-mini`.

The exact snapshot used for this page is
[`docs/benchmarks/trl_rewards_260429.json`](./benchmarks/trl_rewards_260429.json).

## Pointers

- [docs/trl_rewards.md](./trl_rewards.md): the full reward-callable
  contract, `make_reward_fn` API, known limitations, VLM pointer.
- [RadCliQ paper](https://www.cell.com/patterns/fulltext/S2666-3899(23)00157-5):
  correlation-with-radiologist-preferences study that motivates
  RadCliQ as the clinical gold standard despite its cost.
- [RadEval paper](https://aclanthology.org/2025.emnlp-demos.40/):
  population-level metric comparisons.
