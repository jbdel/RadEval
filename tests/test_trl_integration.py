"""TRL GRPO integration smoke tests.

Two 1-step GRPO runs on a tiny random causal LM to verify that
`make_reward_fn` is a drop-in replacement for a bare reward callable:

1. `test_grpo_smoke_with_make_reward_fn` — prompt-only dataset (standard
   string completions path).
2. `test_grpo_smoke_conversational` — conversational `messages` dataset
   so we exercise the `list[list[dict]]` branch against TRL's *actual*
   runtime output, not a synthetic fixture.

Designed to run on CPU / minimal GPU so both can live in a regular
test suite without requiring the full RL stack.

Skips cleanly when `trl` is not installed — install via
`pip install RadEval[rl]`.
"""
import json
from pathlib import Path

import pytest

trl = pytest.importorskip("trl")
pytest.importorskip("datasets")


TINY_MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


def test_quickstart_config_surface():
    """Import the user-facing quickstart module and invoke its config
    builder + dataset loader. Catches TRL API-drift breakage in the
    quickstart's GRPOConfig kwargs surface without having to download
    Qwen2.5-0.5B."""
    import importlib.util
    from pathlib import Path

    qs_path = (
        Path(__file__).parent.parent
        / "examples"
        / "trl_grpo_quickstart.py"
    )
    spec = importlib.util.spec_from_file_location("trl_grpo_quickstart", qs_path)
    qs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qs)

    from trl import GRPOConfig

    config = qs.build_config(output_dir="/tmp/quickstart_config_surface_test")
    assert isinstance(config, GRPOConfig)
    assert config.max_steps == 5
    assert config.num_generations == 2

    dataset = qs.load_dataset()
    assert len(dataset) > 0
    assert "prompt" in dataset.column_names
    assert "ground_truth" in dataset.column_names


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

    model_id = TINY_MODEL_ID
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


def test_grpo_smoke_conversational(tmp_path):
    """Drive TRL with a conversational `messages` dataset and confirm
    that `make_reward_fn` handles the real-world `list[list[dict]]`
    completion payload that TRL produces (not a synthetic fixture)."""
    import warnings as _warnings

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from RadEval.rewards import make_reward_fn

    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL_ID)

    # Conversational dataset: `prompt` is a list of messages. TRL's
    # GRPOTrainer will produce completions in whatever shape matches
    # this dataset's prompt format — this is the shape our handler
    # needs to cope with.
    rows = [
        {
            "prompt": [
                {"role": "user", "content": "Say: normal heart size."},
            ],
            "ground_truth": "Normal heart size.",
        },
        {
            "prompt": [
                {"role": "user", "content": "Say: no pleural effusion."},
            ],
            "ground_truth": "No pleural effusion.",
        },
    ]
    dataset = Dataset.from_list(rows)

    observed_shapes: list[str] = []
    bleu_reward = make_reward_fn("bleu")

    def reward_fn(completions, **kwargs):
        # Record the concrete shape TRL passed so the test documents
        # the assumption the handler is built on.
        if completions:
            observed_shapes.append(type(completions[0]).__name__)
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

    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[reward_fn],
            args=config,
            train_dataset=dataset,
        )
        trainer.train()

    assert trainer.state.global_step == 1
    assert observed_shapes, "reward_fn was never called"
    # Either shape is acceptable; what matters is that make_reward_fn
    # handled whatever TRL actually produced without raising.
    assert set(observed_shapes) <= {"str", "list"}, observed_shapes
    # If list[list[dict]] fired, we should have seen the heuristic warning.
    if "list" in observed_shapes:
        assert any(
            issubclass(x.category, UserWarning)
            and "last assistant message" in str(x.message)
            for x in w
        ), "Expected the conversational-heuristic UserWarning to fire"
