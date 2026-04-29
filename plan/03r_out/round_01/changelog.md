# Review Loop Changelog — Round 01

## Input
- previous document: plan/03r_out/implementation_report.md
- critic feedback: round_01/feedback_oai.md (revise_minor), round_01/feedback_gem.md (revise_minor)
- current revision: round_01/revised.md

## Accepted Critiques
- **Both critics: conversational handler only validated against synthetic dicts.** Addressed in commit 424a0a3 by adding `test_grpo_smoke_conversational` which drives `GRPOTrainer` with a real `{prompt: [messages], ground_truth}` dataset and captures the concrete completion shape TRL 1.3.0 passes. Empirically confirmed: TRL 1.3.0 produces `list[list[dict]]` with `role`/`content` keys for conversational datasets — exactly the shape our heuristic was designed for.
- **Gemini: full repo test suite not re-run.** Re-ran `pytest tests/ -m "not integration"` on `/nfs/cluster/miniconda3/envs/radeval-t5`: **191 passed, 3 skipped, 17 deselected**. Zero regressions from the `RadEval/__init__.py` edit that exports `validate_rewards`.
- **OpenAI: quickstart not CI-tested.** Addressed in commit 424a0a3. Extracted `build_config()` from the quickstart's `main()` into a standalone function; added `test_quickstart_config_surface` which imports the quickstart module and exercises `build_config()` + `load_dataset()` against the pinned TRL version. Future TRL API drift against the user-facing surface now surfaces as a CI failure without requiring Qwen2.5-0.5B download.
- **OpenAI (low): NCCL stderr warning.** Documented in a new "Operational notes" section of `docs/trl_rewards.md` explaining it's a benign environment artifact and how to filter if wrapper scripts treat stderr as failure.

## Rejected Critiques
- None. All round-1 feedback was actionable and low-risk.

## Uncertain / Deferred
- None — the two new integration tests (`test_grpo_smoke_conversational` and `test_quickstart_config_surface`) close the main compatibility gaps. Further hardening (e.g., parameterizing the integration test across multiple TRL versions) would be incremental and is appropriate for a follow-up PR if CI matrix is added.

## Net Effect
- scope: unchanged (added 2 integration tests + 1 refactor + 1 doc note, but no new features)
- confidence: increased — the central "TRL compat" claim is now backed by a real-trainer conversational test, full repo test pass, and a quickstart-surface sentinel test.
