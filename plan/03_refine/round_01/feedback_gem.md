# Verdict
revise_minor

# Critical Issues
- severity: medium
- issue: Silent cache bypass on unhashable kwargs.
- why it matters: The plan states that unhashable kwargs will "bypass cache with a warning" and load a fresh model. In a memory-constrained GPU environment, silently (or via an easily missed warning) loading a duplicate 1GB+ metric model because a user passed a list instead of a tuple will cause unexpected Out-Of-Memory (OOM) crashes.
- recommended fix: If `cache=True` (the proposed default), raise a hard `TypeError` or `ValueError` when unhashable kwargs are provided, forcing the user to either fix the kwarg type or explicitly set `cache=False`.

- severity: medium
- issue: Assumption about TRL `completions` format.
- why it matters: The plan asserts that "Conversational datasets pass `completions` as `list[list[dict]]`... not strings." While `prompts` are often passed as message dictionaries in TRL, the model's generated `completions` are frequently decoded and passed as `list[str]` depending on the exact TRL version, tokenizer configuration, and whether a custom formatting function is used. Assuming completions are always dicts will cause crashes.
- recommended fix: Ensure the conversational extraction logic in `make_reward_fn` is defensive. It must gracefully handle both `list[str]` and `list[list[dict]]` for the `completions` argument.

# Overengineering Flags
- component: Custom `_cache_key` normalization loop.
- why unnecessary: Writing a custom loop to sort, type-check, and filter kwargs into a tuple re-invents standard Python caching mechanisms and introduces maintenance overhead.
- simpler alternative: Use standard `@functools.lru_cache(maxsize=...)` on an internal factory function. If users pass unhashable arguments (like dicts or lists), Python will natively raise a `TypeError`, which cleanly enforces the requirement that cached metric configurations must be hashable.

# Assumptions to Test
- **TRL `completions` data type:** Validate exactly what `GRPOTrainer` yields for `completions` in v1.3.x across both standard and conversational datasets (string vs. dict).
- **RadCliQ online performance:** Validate if `make_reward_fn("radcliq")` is actually fast enough to be used in an online RL loop. RadCliQ relies on RadGraph, which is notoriously slow. It may bottleneck the GRPO rollout to the point of being unusable for training.
- **DDP / Accelerate behavior:** Validate that the module-level `_SCORER_CACHE` behaves predictably across multiple processes. In distributed training, each GPU process will instantiate its own cache and load its own copy of the model.

# Recommended Revisions
- Make the cache bounded (e.g., max 2-3 models) to prevent memory leaks in interactive environments (like Jupyter notebooks) where users might repeatedly call `make_reward_fn` with slightly different kwargs.
- Update the conversational format handler to check `isinstance(completions[0], str)` before attempting to extract `content` keys.
- If `radcliq` is too slow for per-step training, explicitly document it as an "Evaluation Reward" rather than a "Training Reward" in the quickstart and docs.

# Confidence
high
