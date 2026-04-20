[Back to README](../README.md)

# Using RadEval Metrics as RL Rewards

RadEval metrics can be used as reward functions for reinforcement learning
with HuggingFace's [TRL](https://github.com/huggingface/trl) library.

## Quick Start

```python
from RadEval.rewards import make_reward_fn
from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[make_reward_fn("bertscore")],
    train_dataset=dataset,   # must have a "ground_truth" column
)
```

## How It Works

`make_reward_fn()` wraps any RadEval metric as a TRL-compatible callable:

1. TRL calls `reward_fn(completions=["..."], ground_truth=["..."])`
2. The wrapper passes `completions` as hypotheses and `ground_truth` as references
3. RadEval computes per-sample scores
4. Scores are returned as `list[float]`

## Recommended Metrics for RL

RL training calls the reward function thousands of times per epoch. Use
**fast, local metrics** -- not API-based ones.

| Speed | Metrics | Notes |
|-------|---------|-------|
| Fast (recommended) | `bleu`, `rouge`, `bertscore`, `radeval_bertscore` | Milliseconds per batch |
| Medium | `f1chexbert`, `f1radbert_ct`, `srrbert`, `ratescore`, `temporal` | Requires GPU, seconds per batch |
| Slow (local LLM) | `green` | 7B model inference, use with caution |
| Very slow (API) | `crimson`, `mammo_green`, `radfact_ct` | API calls per sample -- not practical for RL |

A runtime warning is emitted if you use an API-based metric.

## Score Transform

RL algorithms are sensitive to reward scale. Use `score_transform` to
normalize or shift scores:

```python
# BERTScore is ~[0, 1] -- center around 0 for PPO/GRPO
reward = make_reward_fn("bertscore", score_transform=lambda x: (x - 0.5) * 2)

# BLEU is [0, 1] -- scale up for stronger signal
reward = make_reward_fn("bleu", score_transform=lambda x: x * 10)
```

## Multiple Reward Functions

TRL supports multiple rewards. Combine metrics for a richer signal:

```python
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[
        make_reward_fn("bertscore"),
        make_reward_fn("f1chexbert"),
    ],
    train_dataset=dataset,
)
```

## Dataset Requirements

Your dataset must include a column with reference reports. By default,
`make_reward_fn` looks for `"ground_truth"`. To use a different column:

```python
reward = make_reward_fn("bertscore", reference_column="reference_report")
```

## Demo

See `examples/trl_reward_demo.py` for a runnable demo that works without
TRL installed.
