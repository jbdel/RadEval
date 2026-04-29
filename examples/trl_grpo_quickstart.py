"""Runnable GRPO quickstart: TRL + RadEval BLEU reward on a small LLM.

Run with the `radeval-t5` env (Python 3.11, transformers 5.6.2, torch
2.9.1, trl 1.3.0):

    pip install radeval[rl]
    python examples/trl_grpo_quickstart.py

BLEU is deliberate for the first-run experience: no extra model download
in the reward path, near-instant scoring. To swap in a clinical signal,
replace `make_reward_fn("bleu")` with any of:

    make_reward_fn("bertscore")                                # heavier, richer
    make_reward_fn("f1chexbert", key="f1chexbert_sample_acc_5")
    make_reward_fn("radgraph", key="radgraph_partial")
    make_reward_fn("radcliq")                                  # eval/final-tune

See docs/trl_rewards.md for the full speed/signal trade-off table.

Hardware note: `Qwen/Qwen2.5-0.5B` at `max_completion_length=32` fits
in ~2 GB VRAM; the CI integration test in `tests/test_trl_integration.py`
uses a tiny random model instead.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from radeval.rewards import make_reward_fn


def load_dataset() -> Dataset:
    fixture = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "synthetic_reports.json"
    rows = json.loads(fixture.read_text())
    return Dataset.from_list(rows)


def build_config(output_dir: str = "out/grpo_quickstart") -> GRPOConfig:
    """Build the quickstart's GRPOConfig. Extracted so
    tests/test_trl_integration.py can exercise the exact
    config-kwarg surface against the pinned TRL version."""
    return GRPOConfig(
        output_dir=output_dir,
        max_steps=5,
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=32,
        learning_rate=1e-6,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        use_cpu=not torch.cuda.is_available(),
    )


def main() -> None:
    model_id = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)

    dataset = load_dataset()

    reward = make_reward_fn("bleu")
    # Swap in: make_reward_fn("bertscore") -- heavier but richer signal.
    # See docs/trl_rewards.md.

    config = build_config()

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward],
        args=config,
        train_dataset=dataset,
    )
    trainer.train()

    # Print the reward trajectory so the user can see the loop is live.
    history = getattr(trainer.state, "log_history", [])
    print("\nReward trajectory:")
    for entry in history:
        if "reward" in entry:
            print(f"  step {entry.get('step'):>3}: reward={entry['reward']:.4f}")


if __name__ == "__main__":
    main()
