# Verdict
revise_minor

# Critical Issues
- severity: medium
- issue: The plan claims a "~300-line PR" and "single focused week" while also including API changes, scorer caching, adapter audit, new extras, example replacement, unit tests, TRL integration smoke test, docs rewrite, README, and changelog.
- why it matters: The scope estimate appears understated relative to the listed work, which risks rushed implementation or incomplete validation.
- recommended fix: Reframe the estimate more conservatively and prioritize a minimum slice for the first PR: harden `make_reward_fn`, fix verified key bugs, add one runnable example, and add minimal tests. Treat cache and broader adapter audit as conditional if they fit cleanly.

- severity: medium
- issue: The default-key selection strategy for multi-key metrics is underspecified and weakly justified.
- why it matters: "Prefer per_sample keys when they exist" may still pick the wrong semantic output, especially given the confirmed F1CheXbert mismatch and possible mode-dependent inconsistencies in other adapters.
- recommended fix: Avoid implicit selection where semantics are ambiguous. Require `key=` for metrics with multiple per-sample outputs unless there is a clearly documented canonical default, and add tests per affected metric.

- severity: medium
- issue: The scorer cache design assumes constructor kwargs are the main determinant of scorer identity and safety.
- why it matters: If scorer behavior depends on hidden global state, environment, device placement, or mutable config objects, cache reuse could be incorrect or produce hard-to-debug memory behavior.
- recommended fix: Narrow the initial cache scope to clearly pure/scalar-configured scorers, document that caching is best-effort, and validate with a small targeted test on at least one heavyweight metric before making it default.

- severity: medium
- issue: The integration validation is too narrow to support some of the stronger compatibility claims.
- why it matters: A 1-step GRPO smoke test on a toy model does not validate conversational completions, dataset-column forwarding, RLOO compatibility, or VLM-related claims.
- recommended fix: Tone down claims to what is actually tested, or add one minimal test for conversational completion extraction. Keep RLOO/VLM as documentation-level compatibility statements, not verified support claims.

- severity: low
- issue: The plan recommends `make_reward_fn("radcliq")` as a canonical clinical choice before validating online-RL practicality.
- why it matters: This risks overclaiming usefulness for training if the metric is too slow or unstable per step.
- recommended fix: Keep `radcliq` available, but present it as supported-not-default until runtime is measured in the intended loop.

# Overengineering Flags
- component: scorer caching
- why unnecessary: It solves a plausible performance issue, but the plan does not show evidence that the first PR needs cross-reward scorer reuse to satisfy current requirements.
- simpler alternative: Ship without cache first, or limit caching to an internal non-default path until a concrete duplicate-load case is demonstrated in tests or user workflow.

- component: broad adapter audit for all mode-dependent keys
- why unnecessary: The only verified bug presented is F1CheXbert. A repo-wide audit may expand scope beyond the evidence.
- simpler alternative: Fix the confirmed metric, add a guard/error for ambiguous multi-key outputs, and audit others only as encountered.

- component: `.metric_keys` attribute on returned callable
- why unnecessary: The plan does not show that TRL consumes this attribute or that it is needed for current functionality.
- simpler alternative: Omit it unless there is a concrete consumer in RadEval docs/tests.

# Assumptions to Test
- TRL version pinned in `[rl]` actually uses the kwargs and completion formats described.
- Conversational completions can be safely reduced by extracting `content` without losing relevant structure for all supported use cases.
- F1CheXbert is the only currently broken metric under `per_sample=True`, or others are at least safely rejected.
- Default scorer caching does not create incorrect reuse or unacceptable memory retention.
- The quickstart model and config run within the advertised time/resource budget.
- `radcliq` is practical enough for online reward use before being positioned as canonical.

# Recommended Revisions
- Reduce the first PR to the smallest validated slice and mark caching/extended audit as optional follow-up if implementation stays small.
- Replace ambiguous default multi-key selection with explicit `key=` for affected metrics unless a canonical default is proven.
- Add one focused test for conversational `list[list[dict]]` completions.
- Soften announcement/docs language from broad TRL compatibility to "tested with GRPO; expected to work with other reward-fn trainers."
- Remove or justify `.metric_keys` on the callable with a concrete use.

# Confidence
high
