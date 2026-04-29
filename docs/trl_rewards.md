[Back to README](../README.md)

# RadEval as a TRL-compatible reward provider

RadEval's 16+ radiology metrics work as drop-in reward functions for
Hugging Face [TRL](https://github.com/huggingface/trl) — GRPO is the
flagship, tested path, and any TRL trainer that consumes a reward-function
callable (e.g. [RLOO](https://huggingface.co/docs/trl/main/en/rloo_trainer))
uses the same interface.

## Install

```bash
pip install RadEval[rl]          # installs trl>=1.3.0,<2
```

`datasets` is already a core dependency.

**Validated stack** (smoke-tested in CI): TRL 1.3.0, transformers 5.6.2,
torch 2.9.1, Python 3.11. The `trl>=1.3.0,<2` pin is a compatibility
ceiling, not a validated-range claim — other 1.x versions are expected
to work but are not separately tested in this release. For
strictest reproducibility, pin to `trl==1.3.0`.

## Primary path (tested): GRPO with a single RadEval reward

Matches [`examples/trl_grpo_quickstart.py`](../examples/trl_grpo_quickstart.py).

```python
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from RadEval.rewards import make_reward_fn

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

dataset = Dataset.from_list([
    {"prompt": "Generate a chest x-ray report: normal.",
     "ground_truth": "No acute cardiopulmonary abnormality."},
    # ...
])

reward = make_reward_fn("bleu")

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[reward],
    args=GRPOConfig(
        output_dir="out",
        max_steps=5,
        per_device_train_batch_size=2,
        num_generations=2,
    ),
    train_dataset=dataset,
)
trainer.train()
```

Run the quickstart end-to-end:

```bash
python examples/trl_grpo_quickstart.py
```

## Reward callable contract

`make_reward_fn(metric, ...)` returns:

```python
reward_fn(completions, **kwargs) -> list[float | None]
```

- **Batch semantics.** The full completion list is processed per call;
  `make_reward_fn` passes completions as `hyps` and the
  `ground_truth` column as `refs` to the underlying scorer, all in a
  single scoring pass. Matches TRL's batch behavior.
- **`**kwargs` absorption.** TRL forwards every dataset column (except
  `prompt`) as a kwarg, plus its own bookkeeping kwargs (`prompts`,
  `completion_ids`, `trainer_state`, `log_extra`, `log_metric`,
  `environments`). `make_reward_fn` reads the configured
  `reference_column` and ignores everything else.
- **Required `key=` for multi-key metrics.** Some metrics return
  multiple per-sample outputs (e.g. F1CheXbert →
  `f1chexbert_sample_acc_5` / `f1chexbert_sample_acc_all`; RadGraph →
  `radgraph_simple` / `radgraph_partial` / `radgraph_complete`; ROUGE →
  `rouge1` / `rouge2` / `rougeL`). Pass `key=` explicitly:

  ```python
  reward = make_reward_fn("f1chexbert", key="f1chexbert_sample_acc_5")
  reward = make_reward_fn("radgraph",   key="radgraph_partial")
  ```

  If `key=` is omitted on a multi-key metric, the first call raises
  `ValueError` listing the valid keys.
- **`None` pass-through.** TRL uses `None` entries in a reward list to
  skip samples in multi-task routing. `make_reward_fn` preserves this:
  if the underlying scorer emits `None` for a sample, the reward fn
  returns `None` for that sample. `validate_rewards()` (used internally)
  also preserves `None`.
- **NaN / Inf raise.** A non-finite scalar reward raises `ValueError`
  naming the metric and the offending sample index. Prefer loud failure
  over silent zero-rewards that mask tokenizer / edge-case bugs.

## Choosing a metric

Qualitative tiering:

| Speed | Metrics | Notes |
|---|---|---|
| Fast (recommended) | `bleu`, `rouge`, `bertscore`, `radeval_bertscore` | No or tiny GPU footprint; milliseconds per batch. |
| Medium | `f1chexbert`, `f1radbert_ct`, `srrbert`, `ratescore`, `temporal`, `radgraph` | GPU transformer inference; seconds per batch. |
| Slow (local LLM) | `green` | 7B local model inference per sample — treat as **unusable per-step** without a dedicated GPU. |
| Very slow (API) | `crimson`, `mammo_green`, `radfact_ct` | One API call per sample — **not practical for online RL**. A `UserWarning` fires when you wrap them. |

For **measured** per-sample cost across every reward-eligible metric
plus a divergence gallery showing how reward choice changes the GRPO
training signal, see
[docs/trl_rewards_benchmarks.md](./trl_rewards_benchmarks.md).

Paper-`radcliq` is available as `make_reward_fn("radcliq")` and is
recommended for **evaluation / final-tune reward**. Benchmarked
per-sample cost is ~160 ms/sample (composite of BERTScore + SembScore
+ RadGraph) — see the benchmarks page for the measured numbers and a
recommendation on when it's practical as an online reward. RadCliQ is
a **distance** (lower = better), so for RL training use
`make_reward_fn("radcliq", score_transform=lambda x: -x)`.

## Combining metrics (native TRL)

TRL sums `reward_funcs` with optional per-function weights in
`GRPOConfig`:

```python
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[
        make_reward_fn("bertscore"),
        make_reward_fn("radgraph", key="radgraph_partial"),
    ],
    args=GRPOConfig(output_dir="out", reward_weights=[0.6, 0.4]),
    train_dataset=dataset,
)
```

TRL auto-logs `reward/bertscore/mean` / `/std` per function — no extra
RadEval abstraction is needed.

> **VRAM note.** Two reward functions that share a heavy underlying
> metric (e.g. two keys from RadGraph, or two BERTScore variants) will
> load the underlying model twice — scorer caching is not yet shipped
> in v2.2.0. Plan VRAM accordingly, or use a single reward function and
> have it compute multiple keys.

## Operational notes

- **NCCL bind warning on stderr.** On multi-GPU hosts, TRL /
  accelerate may print `NCCL WARN Call to bind failed: Address already
  in use` to stderr at startup. This is a benign environment artifact
  — training proceeds normally. If your wrapper script treats non-empty
  stderr as failure, redirect stderr or filter that line.

## Known limitations

- **Conversational completion format is heuristic-based.** TRL may pass
  completions as `list[str]` or as `list[list[dict]]` (OpenAI-style
  message format) depending on the dataset shape, tokenizer template,
  and TRL version. For the message-list case we extract `content` from
  the last assistant turn and fire a `UserWarning`. Unrecognized
  shapes and missing `role`/`content` keys raise `TypeError`. If your
  dataset uses a different message layout, preprocess completion text
  upstream in your dataset/collator pipeline (not via `score_transform`,
  which operates on scores).

- **`radcliq` per-sample cost.** Paper-RadCliQ composes three
  transformer-based sub-metrics (BERTScore, semantic-similarity,
  RadGraph). It's accurate for final-tune / evaluation, but may
  bottleneck per-step GRPO — benchmark before adopting as primary.

- **Integration tests are narrow.** The shipped smoke tests run one
  step of GRPO on a tiny random model (see
  `tests/test_trl_integration.py`) across a prompt-only and a
  conversational dataset. They verify `make_reward_fn` is a drop-in
  reward callable and that the `list[list[dict]]` heuristic matches
  TRL 1.3.0's actual payload shape — **not** full algorithmic
  behavior, long-run stability, cross-trainer compatibility, or
  schema variations outside the standard OpenAI-style
  `{role, content}` message dict.
  `tests/test_trl_integration.py::test_quickstart_config_surface`
  exercises the quickstart's config-construction surface against the
  pinned TRL version but is a partial regression guard — it does not
  run the full trainer end-to-end.

## Adjacent / untested uses (guidance-only)

These paths use the **same** reward-function signature and are expected
to work by construction, but **none are validated in this release** —
the snippets below are orientation, not verified recipes:

- **[RLOO](https://huggingface.co/docs/trl/main/en/rloo_trainer)** —
  REINFORCE-style online RL; same `reward_funcs=[...]` surface.
- **PPO** — [`trl.experimental.ppo.PPOTrainer`](https://huggingface.co/docs/trl/main/en/ppo_trainer)
  requires a full reward model (`nn.Module`), not a reward-function
  callable, so RadEval metrics don't plug in directly. Use GRPO or
  RLOO for metric-as-reward workflows.
- **VLM GRPO** — TRL's
  [`examples/scripts/grpo_vlm.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/grpo_vlm.py)
  works unchanged with RadEval rewards: the reward function never sees
  the image; only the decoded text completion. Wire it up identically:

  ```python
  from RadEval.rewards import make_reward_fn
  reward = make_reward_fn("bertscore")
  # Plug into TRL's canonical VLM GRPO example.
  ```

- **Preference pair curation for DPO / ORPO / KTO.** RadEval metrics
  can rank paired completions to synthesize preference data, but that's
  a different workflow from "metric as online reward" and isn't covered
  here.

## Useful TRL links

- [GRPO trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [Custom reward function section](https://huggingface.co/docs/trl/main/en/grpo_trainer#using-a-custom-reward-function)
- [VLM training](https://huggingface.co/docs/trl/main/en/grpo_trainer#vision-language-model-vlm-training)
- [Dataset formats](https://huggingface.co/docs/trl/main/en/dataset_formats)
- [RLOO](https://huggingface.co/docs/trl/main/en/rloo_trainer)
