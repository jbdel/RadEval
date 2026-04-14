"""Demo: using RadEval metrics as TRL reward functions.

This script shows how to wrap RadEval metrics for use with
HuggingFace TRL's GRPOTrainer. It does NOT require TRL installed --
it just demonstrates the reward function interface.

Usage:
    python examples/trl_reward_demo.py
"""
from RadEval.rewards import make_reward_fn


def main():
    # --- Sample data (simulating what TRL would pass) ---
    completions = [
        "No acute cardiopulmonary process.",
        "Mild cardiomegaly with bilateral pleural effusions.",
        "Normal chest radiograph.",
    ]
    ground_truth = [
        "No acute cardiopulmonary process.",
        "Mild cardiomegaly. Small bilateral pleural effusions.",
        "No acute findings. Normal heart size.",
    ]

    # --- Create reward functions ---
    bleu_reward = make_reward_fn("bleu")
    rouge_reward = make_reward_fn("rouge")

    # With score transform (shift ROUGE-1 to [-1, 1] range)
    rouge_centered = make_reward_fn(
        "rouge",
        score_transform=lambda x: (x - 0.5) * 2,
    )

    # --- Simulate what TRL does: call with completions + dataset columns ---
    bleu_scores = bleu_reward(completions=completions, ground_truth=ground_truth)
    rouge_scores = rouge_reward(completions=completions, ground_truth=ground_truth)
    rouge_centered_scores = rouge_centered(completions=completions,
                                           ground_truth=ground_truth)

    print("BLEU rewards:          ", [round(s, 4) for s in bleu_scores])
    print("ROUGE-1 rewards:       ", [round(s, 4) for s in rouge_scores])
    print("ROUGE-1 (centered):    ", [round(s, 4) for s in rouge_centered_scores])

    # --- How it would look with TRL (requires trl installed) ---
    print("\n# To use with TRL GRPOTrainer:")
    print("""
from RadEval.rewards import make_reward_fn
from trl import GRPOTrainer, GRPOConfig

config = GRPOConfig(output_dir="output", num_train_epochs=1)
trainer = GRPOTrainer(
    model="your-model",
    config=config,
    reward_funcs=[make_reward_fn("bertscore")],
    train_dataset=dataset,   # must have "ground_truth" column
)
trainer.train()
""")


if __name__ == "__main__":
    main()
