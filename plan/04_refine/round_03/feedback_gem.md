# Verdict
revise_minor

# Critical Issues
- severity: high
- issue: Startup key-map validation invalidates `cached_init_s` and VRAM measurements.
- why it matters: The plan states that before timing, the script instantiates each scorer to validate keys. Instantiating models loads weights into RAM/VRAM and populates OS-level page caches and CUDA contexts. Even with `del` and `gc.collect()`, the subsequent `cls()` call in the timing loop will be artificially fast, and residual CUDA contexts may skew the VRAM baseline, completely invalidating the primary measurements.
- recommended fix: Remove the separate startup validation loop. Perform the key validation *inside* the main measurement loop during the warm-up call. If the key is missing, catch the error, record `"skipped": "key-drift"`, and continue to the next metric.

# Overengineering Flags
- component: `radcliq` subprocess isolation
- why unnecessary: Spawning a subprocess, re-invoking the CLI with `--metric radcliq`, and merging JSON outputs adds unnecessary complexity just to prevent VRAM contamination for one metric.
- simpler alternative: Hardcode `radcliq` to run *last* in the in-process loop. If it runs last, its VRAM footprint and cleanup behavior cannot contaminate subsequent metrics because there are none.

# Assumptions to Test
- `torch.cuda.empty_cache()` and `gc.collect()` successfully return VRAM to a clean baseline between metrics. (CUDA context initialization often leaves a permanent ~1GB footprint that cannot be cleared without terminating the process. You may need to record the baseline VRAM *immediately before* each metric rather than assuming it returns to 0).
- `radcliq`'s negation (`-x`) does not cause issues with specific RL algorithms that might expect strictly positive rewards (though GRPO generally handles negative advantages fine, it is worth verifying no internal TRL asserts require `>0`).

# Recommended Revisions
- Move the key-map validation into the warm-up phase of the main timing loop to preserve cold-start memory/timing isolation.
- Remove the subprocess logic for `radcliq` and simply sort the execution order so `radcliq` is evaluated last.
- In the VRAM calculation, ensure you are calculating the delta against the VRAM allocated *just before* `cls()` is called for that specific metric, rather than assuming a global zero baseline.

# Confidence
high
