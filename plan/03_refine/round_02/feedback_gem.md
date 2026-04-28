# Verdict
approve

# Critical Issues
- severity: medium
- issue: OOM risk from deferred caching on heavy multi-key metrics.
- why it matters: In RL workflows, GPU memory is already heavily consumed by the policy model, reference model, and optimizer states. If a user follows the multi-metric example to extract two keys from the same heavy metric (e.g., `radgraph_partial` and `radgraph_exact`), the lack of caching means the RadGraph model is loaded into VRAM twice. This will almost certainly cause an Out-Of-Memory (OOM) error on standard hardware.
- recommended fix: At minimum, add an explicit warning in the `docs/trl_rewards.md` "Choosing / combining metrics" section about the VRAM duplication when using multiple keys from the same heavy metric. 

# Overengineering Flags
- component: `clip_range=(lo, hi)` argument in `make_reward_fn`.
- why unnecessary: The plan already includes a `score_transform` callable. Adding `clip_range` expands the API surface, requires additional tests, and introduces ambiguity about the order of operations (does it clip then transform, or transform then clip?).
- simpler alternative: Remove `clip_range` entirely. Document how to achieve clipping using the existing `score_transform` (e.g., `score_transform=lambda x: max(0.0, min(1.0, x))`).

# Assumptions to Test
- TRL completion formats: Validate that `completions[0]` is strictly either `str` or `list[dict]` across *both* `GRPOTrainer` and `RLOOTrainer` in the pinned TRL version, as some older/experimental TRL paths occasionally pass token IDs.
- Quickstart memory footprint: Verify that running the `Qwen2.5-0.5B` GRPO quickstart alongside the metric model actually fits within standard laptop GPU VRAM (e.g., 8GB-12GB Mac/Nvidia) without aggressive quantization.

# Recommended Revisions
- Drop the `clip_range` argument to strictly adhere to KISS and YAGNI; rely on `score_transform`.
- Add a VRAM warning to the documentation for users instantiating `make_reward_fn` multiple times for the same underlying heavy metric.
- Explicitly state in the documentation that RadEval processes the `completions` as a batch, matching TRL's batch-passing behavior, to reassure users about throughput.

# Confidence
high
