"""TRL GRPO integration smoke test.

One 1-step GRPO run on a tiny random causal LM to verify that
`make_reward_fn` is a drop-in replacement for a bare reward callable.
Designed to run on CPU / minimal GPU so it can live in a regular test
suite without requiring the full RL stack.

Skips cleanly when `trl` is not installed — install via
`pip install RadEval[rl]`.
"""
import json
from pathlib import Path

import pytest

trl = pytest.importorskip("trl")
pytest.importorskip("datasets")


def test_grpo_smoke_with_make_reward_fn(tmp_path):
    """1-step GRPO run with a make_reward_fn-wrapped BLEU metric."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from RadEval.rewards import make_reward_fn

    fixture_path = (
        Path(__file__).parent / "fixtures" / "synthetic_reports.json"
    )
    rows = json.loads(fixture_path.read_text())
    # Use only a handful so the step runs fast.
    rows = rows[:4]
    dataset = Dataset.from_list(rows)

    model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)

    call_log: list[int] = []
    bleu_reward = make_reward_fn("bleu")

    def reward_fn(completions, **kwargs):
        # Wrap to confirm TRL actually invokes our reward callable.
        call_log.append(len(completions))
        return bleu_reward(completions=completions, **kwargs)

    config = GRPOConfig(
        output_dir=str(tmp_path / "out"),
        max_steps=1,
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=16,
        learning_rate=1e-6,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        use_cpu=not torch.cuda.is_available(),
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=config,
        train_dataset=dataset,
    )
    trainer.train()

    assert trainer.state.global_step == 1, trainer.state.global_step
    assert call_log, "reward_fn was never called by GRPOTrainer"
    # Each invocation should receive a list of completions.
    assert all(n > 0 for n in call_log), call_log
