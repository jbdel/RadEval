# Verdict
approve

# Critical Issues
- severity: low
- issue: CLI design contradiction.
- why it matters: The script design section specifies "One command, no subcommands: `python scripts/bench_rewards.py --output ...`", but the markdown outline's reproduction snippet uses `python scripts/bench_rewards.py all \ ...`. This will cause a CLI parsing error if `all` is not actually configured as an argument.
- recommended fix: Remove `all` from the bash snippet in the documentation outline to match the intended CLI design.

# Overengineering Flags
- component: None. 
- why unnecessary: The plan successfully avoids overengineering by dropping subprocess isolation, keeping fixtures static, and explicitly scoping the script as a one-shot documentation tool.

# Assumptions to Test
- `torch.cuda.empty_cache()` between iterations is sufficient to prevent VRAM fragmentation from causing an OOM when running multiple heavy models sequentially (e.g., `radgraph` followed by `radcliq`). The plan's graceful OOM skipping mitigates this, but it's worth verifying on the target GPU.

# Recommended Revisions
- **Future-proof single-key metrics:** The plan notes that `f1radbert_ct` currently returns one key, so `key=` is omitted, but acknowledges this will break if a second key is added. Since the script is meant to be re-runnable by users ("re-run `scripts/bench_rewards.py` to refresh"), explicitly pass the known key (e.g., `key="f1radbert_ct_sample_acc"`) for *all* metrics where the key is known. This costs nothing and prevents future script breakage.
- **Update the reproduction snippet:** In the `docs/trl_rewards_benchmarks.md` outline, the bash snippet only shows running the script once. Since the methodology relies on a two-run pattern to separate download time from `cached_init_s`, explicitly show the two runs in the bash snippet so users actually follow the correct methodology:
  ```bash
  # 1. Warm up the HF cache (ignore output)
  python scripts/bench_rewards.py
  # 2. Generate the canonical snapshot
  python scripts/bench_rewards.py --output docs/benchmarks/trl_rewards_$(date -u +%y%m%d).json
  ```

# Confidence
high
