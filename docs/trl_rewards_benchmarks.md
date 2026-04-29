[Back to README](../README.md) · [Back to RL rewards](./trl_rewards.md)

# RadEval metrics as RL rewards — speed & divergence

*Snapshot: RadEval 2.2.0, TRL 1.3.0, transformers 5.6.2, torch
2.9.1+cu128, Python 3.11, NVIDIA A100-SXM4-80GB. Numbers drift on
different hardware and future releases — the snapshot file
(`docs/benchmarks/trl_rewards_260429.json`) is pinned. To reproduce
on your own machine, see "Reproducing this page" at the bottom.*

This page is for readers asking "I'm about to wire up a RadEval
metric as a GRPO reward. Which one, at what cost, and does the
choice actually matter?"

## Per-batch cost (20 samples)

Workload: `docs/benchmarks/fixtures/speed_workload.json` — 20
hand-written (reference, candidate) pairs in chest-X-ray-report
style.

| Metric | `key=` | Cached init | Warm batch (20) | Per-sample | Peak VRAM (approx) |
|---|---|---:|---:|---:|---:|
| `bleu` | — | 0.00 s | 0.002 s | 0.09 ms | CPU |
| `rouge` | `rouge1` | 0.05 s | 0.003 s | 0.15 ms | CPU |
| `bertscore` | — | 5.89 s | 0.013 s | 0.65 ms | 260 MB |
| `radeval_bertscore` | — | 1.08 s | 0.023 s | 1.15 ms | 606 MB |
| `f1chexbert` | `f1chexbert_sample_acc_5` | 3.30 s | 0.034 s | 1.71 ms | 434 MB |
| `f1radbert_ct` | `f1radbert_ct_sample_acc` | 0.86 s | 0.042 s | 2.12 ms | 485 MB |
| `srrbert` | `srrbert_weighted_f1` | 0.91 s | 0.103 s | 5.13 ms | 420 MB |
| `temporal` | — | 0.62 s | 0.331 s | 16.57 ms | 191 MB |
| `ratescore` | — | 2.75 s | 1.084 s | 54.22 ms | 1130 MB |
| `radgraph` | `radgraph_partial` | 4.35 s | 3.169 s | 158.47 ms | 890 MB |
| `radgraph_radcliq` | — | 4.38 s | 3.183 s | 159.14 ms | 890 MB |
| `radcliq` | — | 4.88 s | 3.212 s | 160.62 ms | 1175 MB |

*`cached init` is `cls()` wall time after HF models are already on
disk — i.e., what the second invocation of the script sees. Peak
VRAM is `torch.cuda.max_memory_allocated` delta and is
**approximate and metric-order-sensitive** — read it as
**directional** guidance for GPU budgeting, not a metric-vs-metric
comparison: it only counts torch allocations, and in-process
measurement is biased by PyTorch's caching allocator (earlier
metrics' allocations can inflate later readings despite the
explicit `del`, `gc.collect()`, `torch.cuda.empty_cache()` teardown
the script performs between every metric). All metrics run
in-process with that teardown; `radcliq` is ordered last so its
composite footprint can't leak into adjacent rows. If you need a
trustworthy absolute number for one metric, isolate that metric in
its own Python process. First-download time is not reported
(network-dependent).*

**Takeaways:**

- **BLEU / ROUGE are essentially free** (sub-0.2 ms/sample, CPU).
  Useful as a sanity baseline or a cheap auxiliary reward.
- **BERTScore is the cheapest semantic option** at 0.65 ms/sample
  with a 5.9 s one-time load. Under a ms RL budget, it's the
  obvious default.
- **F1CheXbert, F1RadBERT-CT, SRRBert** cluster in the 1–5 ms
  range — affordable per-step rewards for CXR/CT workflows.
- **RadGraph (and its derivatives)** jump to ~160 ms/sample — 100×
  the F1CheXbert cost. Still tractable for small-batch GRPO; may
  bottleneck large-batch training.
- **RadCliQ is the costliest row** at ~161 ms/sample, ~1.2 GB VRAM.
  It composes BERTScore + SembScore + RadGraph sequentially, so
  its per-sample latency is roughly (BERTScore 0.65 ms + SembScore
  ~few ms + RadGraph 158 ms) = dominated by RadGraph. Position it
  as an **eval-time reward** or a **final-fine-tune reward**, not a
  primary online signal for large-scale training. See the
  [RadCliQ paper](https://www.cell.com/patterns/fulltext/S2666-3899(23)00157-5)
  for correlation-with-radiologists numbers that motivate using it
  at all despite the cost.

## What reward would this rollout receive?

Same setup readers are imagining: a GRPO trainer samples a
candidate completion for each prompt, the reward function scores
each sample, the score becomes the training signal. Below: 8
curated `(reference, candidate rollout)` pairs scored by five
metrics (lexical → semantic → clinical). The question each row
answers is "which reward would push the policy toward or away from
this rollout?"

Column headers show the reward direction: ↑ = higher is better
reward signal, ↓ = lower is better (RadCliQ only). For RL
training with RadCliQ, see the inversion recipe below.

| # | Reference | Candidate rollout | BLEU↑ | BERTScore↑ | F1CheXbert↑ | RadGraph↑ | RadCliQ↓ |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | Mild cardiomegaly. | The cardiac silhouette is mildly enlarged. | 0.00 | 0.215 | 1.00 | 0.00 | 10.67 |
| 2 | **No pleural effusion.** | **Pleural effusion.** | 0.00 | **0.893** | 0.80 | 0.50 | 11.06 |
| 3 | Mild cardiomegaly. Small bilateral pleural effusions. | Mild cardiomegaly. | 0.00 | 0.718 | 0.80 | 0.50 | 10.52 |
| 4 | No acute findings. | No acute findings. Small right pneumothorax. | 0.00 | 0.604 | 1.00 | 0.57 | 10.43 |
| 5 | No acute cardiopulmonary process. | No acute cardiopulmonary process. | 1.00 | 1.00 | 1.00 | 1.00 | 9.37 |
| 6 | Left lower lobe consolidation consistent with pneumonia. | Right upper lobe mass. | 0.00 | 0.579 | 0.80 | 0.22 | 10.93 |
| 7 | Mild pulmonary edema. | Severe pulmonary edema. | 0.00 | 0.955 | 1.00 | 0.67 | 10.20 |
| 8 | Stable bibasilar atelectasis without consolidation. | New right lower lobe consolidation. | 0.00 | 0.355 | 0.60 | 0.00 | 11.12 |

*(Metric values above are rewards, not evaluation scores. Exact-match
row 5 sets the reward ceiling for each metric: BLEU and BERTScore at
1.0, F1CheXbert/RadGraph at 1.0, RadCliQ at ~9.37 (its floor — lower
is better). Row 2 is the headline case — see below.)*

> **Note on method.** This gallery is hand-picked. It is not a
> representative sample, not a randomized comparison, and not
> evidence of average metric superiority — examples were selected
> specifically to show consequential disagreement on plausible
> rollouts. Treat the rows as existence proofs ("this *can* happen"),
> not prevalence estimates. For population-level comparisons, see
> the [RadEval paper](https://aclanthology.org/2025.emnlp-demos.40/)
> and the RadCliQ reference studies.

### Three rollouts worth zooming in on

The headline is **row 2**. If you skim only one row, make it that one.

1. **Row 1 — clinically correct paraphrase.** "Mild cardiomegaly." →
   "The cardiac silhouette is mildly enlarged." Same finding, totally
   different wording. BLEU scores this at zero — a BLEU-reward GRPO
   run would push the policy *away* from the paraphrase. F1CheXbert
   scores 1.0 (both reports map to the same CheXpert label) — a
   clinical-reward run would push the policy *toward* it. **Same
   rollout, opposite gradient under the two rewards.** (BERTScore
   here scores only 0.22 because it's vocabulary-sensitive; it
   doesn't recover the clinical paraphrase as cleanly as readers
   might expect.)

2. **Row 2 — headline: flipped negation.** "No pleural effusion." →
   "Pleural effusion." Three of four tokens overlap, and BERTScore's
   embeddings are nearly identical between them — so **BERTScore
   scores 0.893, only slightly below the exact-match ceiling of 1.0.**
   Under a BERTScore reward, GRPO would **partially reward flipping
   a negation** — corrupting the clinical meaning while keeping the
   text "looking right." F1CheXbert drops to 0.80 (13/14 labels still
   match; only the effusion label flips), RadGraph drops to 0.50, and
   RadCliQ rises to 11.06 (higher distance = worse). Clinical metrics
   flag the flip; BERTScore does not. **Even the strongest drop-in
   NLP reward is actively dangerous here.**

3. **Row 7 — severity flip.** "Mild pulmonary edema." → "Severe
   pulmonary edema." BERTScore 0.955 (one token differs), F1CheXbert
   1.00 (edema is still edema; severity isn't a separate label),
   RadGraph 0.67, RadCliQ 10.20. Here *even F1CheXbert is fooled* —
   a label-classifier reward can't distinguish mild from severe.
   RadGraph and RadCliQ register the change; F1CheXbert doesn't.
   For severity-sensitive RL, the cheaper clinical metrics aren't
   enough.

### Picking a reward

A compressed decision guide. Speed numbers from the table above.

- **Cheap sanity baseline or auxiliary signal**: `bleu` or `rouge`.
  Sub-ms, CPU. Do not use as the sole reward — row 1 above shows
  why.
- **Semantic default, still cheap**: `bertscore` (0.65 ms/sample).
  Better than BLEU at paraphrase, but **row 2 shows it over-rewards
  negation flips**. If you use it as the sole reward, expect the
  policy to learn that inverting negations is fine.
- **Clinical-finding accuracy (CXR)**: `f1chexbert` with
  `key="f1chexbert_sample_acc_5"`. 1.71 ms/sample. **What this
  metric actually returns**: per-sample *agreement rate across
  14 CheXpert labels* (a single-number-per-sample compressed
  multi-label accuracy), not a per-finding F1. A single-label flip
  between ref and hyp drops the score by ~1/14 ≈ 0.07 at most.
  Catches label flips (row 2: 0.80 vs 1.00); does NOT catch
  severity changes (row 7: stays at 1.00 because "mild" and
  "severe" edema map to the same CheXpert label). Good primary
  reward for label-centric tasks; not sensitive to severity.
- **Entity / relation grounding (CXR)**: `radgraph` with
  `key="radgraph_partial"`. ~158 ms/sample. Catches
  omissions (row 3: 0.50), hallucinations (row 4: 0.57), severity
  changes (row 7: 0.67). Heavier but richer.
- **Best-correlated-with-radiologist-preferences (eval or
  final-tune)**: `radcliq`. ~160 ms/sample. A distance, so
  **lower = better**. For RL, invert with
  `score_transform=lambda x: -x`:
  ```python
  from RadEval.rewards import make_reward_fn
  reward = make_reward_fn("radcliq", score_transform=lambda x: -x)
  ```
  Negation is universally safe; we do NOT recommend bounded
  inversions like `1/(1+x)` because RadCliQ's theoretical range
  isn't strictly non-negative.

## Reproducing this page

```bash
pip install RadEval[rl]

# 1. Warm the HF cache (ignore the output).
python scripts/bench_rewards.py --output /tmp/warmup.json

# 2. Generate the canonical snapshot.
python scripts/bench_rewards.py \
    --output docs/benchmarks/trl_rewards_$(date -u +%y%m%d).json
```

The snapshot used for this page is
[`docs/benchmarks/trl_rewards_260429.json`](./benchmarks/trl_rewards_260429.json)
and was generated on the env recorded at the top of that file.

## Pointers

- [docs/trl_rewards.md](./trl_rewards.md) — the full reward-callable
  contract, `make_reward_fn` API, known limitations, VLM pointer.
- [RadCliQ paper](https://www.cell.com/patterns/fulltext/S2666-3899(23)00157-5)
  — correlation-with-radiologist-preferences study that motivates
  RadCliQ as the clinical gold standard despite its cost.
- [RadEval paper](https://aclanthology.org/2025.emnlp-demos.40/) —
  population-level metric comparisons.
