# Best Practices

# Best Practices

## Design
- Prefer the simplest solution that satisfies current requirements.
- YAGNI: do not add infrastructure, abstractions, or configuration for hypothetical future needs.
- KISS: clear and direct beats clever and general.
- Reuse existing code, patterns, and libraries before adding new ones.
- Delete or simplify before adding.
- Keep changes small, local, and reversible.
- Prefer explicit logic over indirection.
- Avoid premature abstraction.
- Prefer configuration changes over code changes when practical.

## Code Quality
- Keep diffs small and readable.
- Match existing local conventions.
- Fix and consolidate rather than duplicate.
- Extract utility functions only when they clearly improve clarity or eliminate repetition.
- Replace magic numbers and opaque strings with named constants when helpful.
- Fail fast with clear errors.

## Planning
- Separate facts, assumptions, hypotheses, and design decisions explicitly.
- For each major component, ask: what simpler alternative was considered, and why is it insufficient?
- Do not introduce new subsystems unless necessary for the current task.
- Be skeptical of architectural expansion.
- Prefer narrow, high-leverage next steps.

## Scientific / Experimental Reasoning
- Distinguish observation from interpretation.
- Do not infer causality without appropriate evidence.
- Identify confounders, missing controls, and alternative explanations.
- Prefer the minimum experiment that reduces uncertainty.
- Calibrate confidence; do not overclaim.

## Testing
- Default to TDD when practical.
- Write minimal, meaningful tests.
- Treat failing tests as design feedback.
- Test behavior that matters; avoid noisy or decorative tests.

## Safety
- Never modify outside the active worktree or intended scope.
- Avoid destructive actions without explicit need.
- Explain tooling, dependency, or test changes when they materially affect the system.

## Collaboration
- Be concise and factual.
- Critics should challenge reasoning, scope, and assumptions rather than rewrite the whole solution.

---

# Critic Instructions

# External Critic Prompt

You are an external critic reviewing a candidate plan.

## Your role
- Identify reasoning flaws and unjustified assumptions.
- Identify overengineering and unnecessary complexity.
- Identify missing validations, controls, or experiments.
- Suggest simpler viable alternatives.
- Challenge excessive confidence.

## Your non-role
- Do not rewrite the whole plan.
- Do not invent major new scope unless clearly necessary.
- Do not assume repository facts not present in the provided materials.
- Do not act as the final decision-maker.

## Review criteria
1. Is the reasoning justified by evidence presented?
2. Are any claims stronger than the evidence supports?
3. What is overengineered or unnecessary?
4. What simpler alternative should be preferred?
5. What assumptions need validation?
6. What important risk or missing check is absent?

## Response format

Return concise structured markdown with these sections only:

```
# Verdict
approve | revise_minor | revise_major | block

# Critical Issues
- severity: high | medium | low
- issue: [description]
- why it matters: [explanation]
- recommended fix: [suggestion]

# Overengineering Flags
- component: [name]
- why unnecessary: [explanation]
- simpler alternative: [suggestion]

# Assumptions to Test
- [assumption that needs validation]

# Recommended Revisions
- [specific, actionable revision]

# Confidence
low | medium | high
```

---

# Candidate Plan

# RadEval × TRL: Reinforcement-Learning Support Proposal

*A joint report from three investigating perspectives — TRL expert, HOPPR
`vlm_align` expert, and RadEval expert — on how RadEval should integrate with
Hugging Face TRL for GRPO-style radiology-report RL. Revised after two
rounds of external review.*

---

## 1. Executive summary

**Recommendation: position RadEval as a provider of TRL-compatible *reward
functions*, algorithm-agnostic. Ship a tight v2.2.0 PR that hardens the
existing `make_reward_fn`, closes a confirmed bug, and replaces a stub demo
with a runnable GRPO example. Defer everything else until there is
evidence users need it.**

Framing shift from the original draft: this is not "GRPO support." It is
"TRL-compatible reward support, with GRPO as the flagship example." Any TRL
trainer that consumes a reward-function callable — GRPO, RLOO, and so on —
can consume RadEval rewards unchanged. We test against GRPO; we claim
compatibility with GRPO and with other reward-fn-based TRL trainers
by construction, without asserting verified support for each.

The v2.2.0 PR ships exactly this:

1. **Hardened `make_reward_fn`**
   - Absorbs TRL's reward-fn kwargs (`completion_ids`, `trainer_state`,
     `log_extra`, `log_metric`, `environments`) via `**kwargs`.
   - Handles conversational completions **defensively**: checks
     `isinstance(completions[0], str)` first; only extracts
     `content` from message-dict form when needed.
   - Adds a `key=` argument for metrics with multi-key `per_sample` output.
     **No implicit default selection** — if the metric has multiple keys
     and the user doesn't pass `key=`, we raise with a message listing the
     valid keys (fixes the F1CheXbert bug documented in §2.3).
   - Adds `clip_range=(lo, hi)` for bounded rewards (optional).
   - Calls a tiny `validate_rewards()` helper once per batch to raise on
     NaN/Inf with the metric name and offending sample index. No silent
     defaulting.
   - No `default_on_error` — crashes must raise.
   - No `.metric_keys` attribute on the callable — no current consumer.
2. **`pip install RadEval[rl]` extras** — adds `trl>=X.Y` (floor pinned
   during implementation after verifying kwargs against a specific TRL
   release). `datasets` is already a core dep.
3. **Runnable quickstart** — `examples/trl_grpo_quickstart.py`, BLEU only,
   small LLM, replaces the existing call-shape stub. Must run top-to-bottom
   in under ~10 minutes on a laptop GPU.
4. **Tests** — `tests/test_rewards.py` (unit) and
   `tests/test_trl_integration.py` (`pytest.importorskip("trl")`, 1-step
   GRPO smoke test). Includes a focused unit test for the conversational
   `list[list[dict]]` path.
5. **Rewritten `docs/trl_rewards.md`** — reframes as "TRL-compatible reward
   functions"; GRPO primary; RLOO mentioned as another online fit; PPO
   demoted. VLM support is **documentation-only**: snippet + link to
   `trl/examples/scripts/grpo_vlm.py`.

**Deferred to a follow-up PR** (contingent on demand):

- **Scorer caching.** Removed from v2.2.0 scope in round 02. The memory /
  DDP / unhashable-kwargs concerns (raised by both critics) outweigh the
  benefit for a first PR. Users who build multiple reward functions for the
  same metric pay one extra model load — annoying but not a blocker.
  Revisit with `functools.lru_cache` on an internal factory if a real
  workflow demands it.
- **`CompositeReward` class.** TRL's native `reward_funcs=[...]` +
  `reward_weights=[...]` is the canonical path. YAGNI until requested.
- **Committed VLM example file.** Stays as a docs snippet + link.
- **`log_extra` / `log_metric` forwarding.**
- **Async reward-fn support.**

**Explicitly out of scope forever**: custom GRPO trainer, VLM adapter,
rollout engine, reward-model training, dataset/collator code. These belong
in downstream code (e.g. `vlm_align`) or in TRL itself.

This is a ~150-200 line PR that fixes a confirmed bug, makes a real RL
example work, and positions RadEval cleanly. That is the announcement.

---

## 2. Findings from each expert

### 2.1 TRL expert

**Current TRL GRPO reward API** (v1.3.x):

- TRL calls `reward_fn(prompts=..., completions=..., completion_ids=...,
  trainer_state=..., log_extra=..., log_metric=..., environments=...,
  **dataset_columns)` and expects `list[float]` back (or `None` entries to
  skip samples).
- **Every dataset column except `prompt`** is forwarded as a kwarg.
- **Completions format depends on dataset shape.** Standard prompt-only
  datasets yield `list[str]`. Conversational datasets may yield
  `list[list[dict]]` (OpenAI message form) *or* strings depending on the
  TRL version, tokenizer template, and whether a custom formatting
  function is used. **Our handler must check the type defensively** —
  assuming dicts will crash users on string datasets, and vice versa.
- `async def` reward functions are first-class.
- PPO lives in `trl.experimental.ppo`; GRPO is the idiomatic online-RL
  choice. RLOO is a reasonable alternative that consumes the same reward
  signature.

**Verdict on RadEval's current `make_reward_fn`:** signature is correct;
`**kwargs` already absorbs unknown kwargs. Missing pieces: defensive
conversational-format handling, graceful multi-key metric behavior.

**Composite rewards — TRL already solves this.** Native `reward_funcs=[f1,
f2]` + `reward_weights=[w1, w2]` in `GRPOConfig` sums internally and
auto-logs per-function means. Canonical path.

**VLM — TRL's story is solid now.** Native VLM support in `GRPOTrainer`
across Gemma3, LLaVA-Next, Qwen2-VL/2.5-VL, SmolVLM2. RadEval should not
ship trainer code or VLM collators.

**Recommended URLs to link from RadEval docs:**

- <https://huggingface.co/docs/trl/main/en/grpo_trainer>
- <https://huggingface.co/docs/trl/main/en/grpo_trainer#using-a-custom-reward-function>
- <https://huggingface.co/docs/trl/main/en/grpo_trainer#vision-language-model-vlm-training>
- <https://github.com/huggingface/trl/blob/main/examples/scripts/grpo_vlm.py>
- <https://huggingface.co/docs/trl/main/en/dataset_formats>
- <https://huggingface.co/docs/trl/main/en/rloo_trainer>

### 2.2 `vlm_align` / HOPPR RL expert

**Does `vlm_align` use RadEval?** No. A grep turns up only *anti-references*:
the team chose the standalone `radgraph` pip package over "RadEval
subprocess" to avoid cold-start per batch, and an internal design doc
states that a RadEval-subprocess-based metric adapter is the "wrong
interface for per-sample GRPO rewards. All three metrics must be
implemented locally."

**Central finding:** the team bypassed RadEval because RadEval's reward
surface was corpus/epoch-aggregate, not per-sample online. `make_reward_fn`
has since closed most of that gap — our job now is to finish it.

**Why `vlm_align` has a custom GRPO trainer:**

- HOPPR VLM takes `images_pixels` / `images_masks`, not `pixel_values`.
- Only `text_decoder` is LoRA-adapted; vision encoder + QFormer are frozen.
  TRL's model-class assumptions can't cleanly express this.
- `vision_cache.py` short-circuits the vision encoder across 16 rollouts
  per study.
- Uses `trust_remote_code` with a monkey-patched `_init_vision_encoder`.

All **training-layer** concerns. None belong in RadEval. Modern TRL handles
vanilla HF-registered VLMs fine.

**Reward-function interface in `vlm_align`:** returns
`(scalars, per_component_diagnostics_dict)`. TRL's plain `list[float]` is
less expressive, but `reward_funcs=[...]` + `reward_weights` with
auto-logged per-function means covers the same need for most users. **No
need to port the tuple interface.**

**Generic pieces worth considering for upstream:**

| Piece | Where | Upstream? |
|---|---|---|
| Weighted composite | `rewards/radcliq_reward.py` | **Deferred**; TRL already sums `reward_funcs`. |
| NaN/Inf hard-raise | `rewards/radcliq_reward.py:138-145` | **Yes**, as a ~15-LOC `validate_rewards()` helper inside `make_reward_fn`. |
| Per-metric warn bands | `rewards/radcliq_reward.py:44-49` | **No**. Out-of-range values are usually signal; the standard TRL logging pipeline already exposes the reward trajectory. |
| Group-normalized advantages | `core/advantages.py` | **No** — training-layer. |

**HOPPR-specific, leave behind:** `VLMAdapter`, rollout engine, GRPO loss,
collators, vision cache, the specific `radcliq_v0` weights.

### 2.3 RadEval expert

**What's already in place** (`RadEval/rewards.py`, ~40 LOC): single-metric
`make_reward_fn` with API-metric warning; exported from
`RadEval/__init__.py`. Docs at `docs/trl_rewards.md`. Example
(`examples/trl_reward_demo.py`) never instantiates TRL — it prints a
string. Three tests in `tests/test_e2e.py:106-129` cover BLEU only; nothing
is verified against actual TRL.

**Bugs / gaps** (verified in code):

- **F1CheXbert multi-key bug** — `RadEval/metrics/f1chexbert/adapter.py:12-33`.
  `metric_keys()` advertises `["f1chexbert_5_micro_f1", ...]` but
  `compute(..., per_sample=True)` returns
  `{"f1chexbert_sample_acc_5": [...], "f1chexbert_sample_acc_all": [...]}`.
  `make_reward_fn("f1chexbert")` raises `KeyError` today. **Fix scope:
  F1CheXbert specifically.** Other metrics may have similar issues —
  address as users report them, not via a repo-wide audit.
- **`examples/trl_reward_demo.py` is a stub.**
- **No `tests/test_rewards.py`.**
- **No `[rl]` extras** in `setup.py:75-81`.
- **Speed table in `docs/trl_rewards.md`** omits `radgraph` and under-warns
  on `green` (7B local LLM; unusable per-step without a spare GPU).

**RadEval should ship**: the reward callable, a runnable GRPO example,
speed guidance. Stay out of dataset loading, prompt formatting, chat
templates, trainer config, reward-model training, KL/ref-model, distributed
launchers.

---

## 3. Recommended scope

**Ship in v2.2.0** (single PR, ~150-200 LOC + tests + docs):

1. **`make_reward_fn` hardening**
   - `key=` argument for metrics with multi-key `per_sample` output.
     **Required** (not defaulted) when the metric has >1 per-sample key.
     If the user omits it on such a metric, raise a `ValueError` listing
     the valid keys. No silent guess.
   - Defensive conversational-completion handling:
     `completions[0]` is a string → pass through;
     `completions[0]` is a list of dicts → extract `content` from the last
     assistant message;
     any other shape → raise with the observed shape printed.
   - `clip_range=(lo, hi)` optional.
   - `validate_rewards(values, metric_name)` helper (~15 LOC): raises
     `ValueError` on NaN/Inf, naming the metric and sample index. Called
     once per batch inside `make_reward_fn`. No warn-bands, no silent
     fallback.
2. **`pip install RadEval[rl]` extras**: adds `trl>=X.Y`. Version floor
   pinned during implementation after verifying kwargs against the release
   we test.
3. **`make_reward_fn("radcliq")` works** (already does — RadEval has a
   paper-faithful `radcliq` adapter). Documented as **"evaluation /
   final-tune reward, not a primary online-training reward"** until its
   per-sample cost is measured inside a real GRPO loop. Do not invent a
   branded "RadCliQ-like composite" with our own weights.
4. **Runnable quickstart**: `examples/trl_grpo_quickstart.py`, ~40-60
   lines, small LLM (e.g. `Qwen/Qwen2.5-0.5B`), ~20-row synthetic fixture,
   `GRPOConfig(max_steps=5)`. Uses `make_reward_fn("bleu")` only — cheap,
   no extra model downloads. A single commented line shows the swap to
   `bertscore` or `radcliq`. Deletes `examples/trl_reward_demo.py`.
5. **Tests**:
   - `tests/test_rewards.py`:
     - `make_reward_fn("bleu")`, `rouge`, `bertscore` round-trip.
     - **F1CheXbert:** with correct `key=` → works; without `key=` → raises
       with listed valid keys.
     - Conversational-format path:
       `reward(completions=[[{"role": "assistant", "content": "x"}]], ground_truth=[...])`
       returns correct scores.
     - `clip_range` clamps output.
     - `score_transform` composes with `clip_range`.
     - NaN/Inf in raw scores → raises naming metric + index.
     - `UserWarning` fires for API-based metrics.
   - `tests/test_trl_integration.py`:
     - `pytest.importorskip("trl")`.
     - One `max_steps=1` GRPO run on a toy model.
     - Asserts reward fn was called and `trainer.state.global_step == 1`.
6. **Rewritten `docs/trl_rewards.md`**:
   - Frames RadEval as a TRL-compatible reward provider, trainer-agnostic.
   - GRPO as flagship; RLOO mentioned as another online fit; PPO a
     pointer to `trl.experimental.ppo` with a note that GRPO/RLOO are
     recommended.
   - Documents native `reward_funcs=[...]` + `reward_weights` as the
     canonical composite path.
   - Documents `make_reward_fn("radcliq")` for paper-RadCliQ, with the
     "eval-first" positioning.
   - One-paragraph note: RadEval metrics can also *curate* preference
     pairs for DPO/ORPO/KTO, but that is a different workflow from online
     reward.
   - VLM section: short snippet + link to
     `trl/examples/scripts/grpo_vlm.py`. Key sentence: *"The reward
     interface is unchanged between LLM and VLM — the reward function
     never sees the image."*
   - Speed table updated: adds `radgraph` (medium, GPU), stronger warning
     on `green`, API metrics marked "not practical for RL."
7. **README update**: swap stub reference for quickstart link; update the
   one-paragraph pitch.
8. **Changelog entry** for 2.2.0.

**Deferred to a follow-up PR** (explicit list):

- Scorer caching (`_SCORER_CACHE`, `clear_reward_cache()`).
- `CompositeReward` class.
- Committed VLM example file.
- `log_extra` / `log_metric` forwarding.
- Async rewards.

The deferral rule: add these when a user opens an issue with a concrete
workflow that requires them. Not before.

**Answering the plan's explicit questions:**

| Question | Answer |
|---|---|
| TRL-compatible reward functions? | **Yes** — harden the existing one. |
| Example GRPO/PPO training scripts? | **One GRPO LLM quickstart.** No committed VLM script (docs only). No PPO. |
| Wrappers/adapters around existing metrics? | **No new wrappers.** |
| VLM-specific trainer utilities? | **No.** Documentation only. |
| Documentation only? | **No** — docs + bugfix + hardening + example + tests. |
| Boundary? | RadEval ends at `reward_fn(completions=..., **kwargs) -> list[float]`. |
| LLM vs VLM? | Identical reward interface. |
| Incorporate `vlm_align`'s trainer? | **No.** The NaN/Inf helper is the only thing we port. |
| Concrete examples? | `examples/trl_grpo_quickstart.py`. VLM in docs. |
| API changes? | Additive: new `key=`, `clip_range=`. One behavioral fix: F1CheXbert now works (was broken). |
| Announcement-worthy? | Yes — see §8. |

---

## 4. Proposed API / code structure

```python
from RadEval.rewards import (
    make_reward_fn,          # existing; hardened
    validate_rewards,        # new, small utility
)
```

```python
# Minimal single-metric reward (TRL GRPO or RLOO)
from RadEval.rewards import make_reward_fn
from trl import GRPOTrainer, GRPOConfig

reward = make_reward_fn(
    "f1chexbert",
    key="f1chexbert_sample_acc_5",        # REQUIRED for multi-key metrics
    clip_range=(0.0, 1.0),                # optional
    score_transform=lambda x: (x - 0.5) * 2,
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B",
    reward_funcs=[reward],
    train_dataset=dataset,                # must have "ground_truth" column
    args=GRPOConfig(output_dir="out", max_steps=100),
)
trainer.train()
```

```python
# Multi-metric — idiomatic TRL, no new RadEval abstraction needed
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B",
    reward_funcs=[
        make_reward_fn("bertscore"),
        make_reward_fn("radgraph", key="radgraph_partial"),
    ],
    args=GRPOConfig(output_dir="out", reward_weights=[0.6, 0.4]),
    train_dataset=dataset,
)
```

```python
# Paper-RadCliQ as a single reward — uses the existing radcliq adapter.
# See docs/trl_rewards.md: recommended as an evaluation/final-tune reward
# until per-sample cost is benchmarked for online use.
trainer = GRPOTrainer(
    reward_funcs=[make_reward_fn("radcliq")],
    ...
)
```

**Multi-key-metric error (sketch):**

```python
if len(result) > 1 and key is None:
    raise ValueError(
        f"Metric '{metric}' returns multiple per-sample keys: "
        f"{sorted(result.keys())}. "
        f"Pass key=<one of these> to make_reward_fn()."
    )
```

**Conversational-completion handling (sketch):**

```python
def _as_strings(completions):
    if not completions:
        return []
    first = completions[0]
    if isinstance(first, str):
        return list(completions)
    if isinstance(first, list) and first and isinstance(first[0], dict):
        # OpenAI message form — take last assistant turn
        return [_last_assistant_content(msgs) for msgs in completions]
    raise TypeError(
        f"make_reward_fn: unsupported completions shape "
        f"(got {type(first).__name__}); expected list[str] or "
        f"list[list[dict]]."
    )
```

No scorer cache, no kwarg-normalization loop, no `.metric_keys` attribute.

---

## 5. Example workflows to include

### 5.1 `examples/trl_grpo_quickstart.py` (committed, BLEU, runnable)

- ~40-60 lines.
- Model: `Qwen/Qwen2.5-0.5B`.
- Dataset: 20-row synthetic fixture at
  `tests/fixtures/synthetic_reports.json` (reused by the integration test).
- `make_reward_fn("bleu")`. BLEU is the deliberate choice: zero GPU
  dependencies in the reward path, near-instant, no extra model downloads.
- One commented line: `# Swap in: make_reward_fn("bertscore") — heavier
  but richer signal. See docs/trl_rewards.md.`
- `GRPOConfig(max_steps=5, per_device_train_batch_size=2,
  num_generations=2)`.
- Prints reward trajectory.
- Goal: **copy-paste-and-it-works in under ~10 minutes** on a laptop GPU
  (CPU-tolerant with `max_steps=2`). This is what validates the
  announcement.

### 5.2 VLM — docs only

In `docs/trl_rewards.md`, a short section:

```python
# Identical reward setup — the reward function never sees the image.
from RadEval.rewards import make_reward_fn

reward = make_reward_fn("bertscore")
# Plug into trl's canonical VLM GRPO example:
# https://github.com/huggingface/trl/blob/main/examples/scripts/grpo_vlm.py
```

Plus a paragraph explaining that TRL handles the image-conditioned rollout
end-to-end and our reward interface is unchanged.

### 5.3 Docs walk-through: choosing / combining metrics

In `docs/trl_rewards.md`: a short decision guide.

- Single fast metric → `make_reward_fn("bleu")` or `"rouge"`.
- Single clinical metric → `make_reward_fn("f1chexbert", key="f1chexbert_sample_acc_5")` or `make_reward_fn("radgraph", key="radgraph_partial")`.
- Paper-RadCliQ → `make_reward_fn("radcliq")` *(recommended for eval /
  final-tune; benchmark before using per-step)*.
- Custom mix → `reward_funcs=[...]` + `reward_weights=[...]` in
  `GRPOConfig`. (Shows two-reward example with auto-logging.)
- Why no `CompositeReward` helper: TRL already does this natively. If you
  need per-sample diagnostics beyond what TRL logs, open a GitHub issue
  with the use case.

---

## 6. Implementation plan

Single PR. Realistic estimate: ~1-2 focused days of engineering + a half-day
of dogfooding + a half-day of docs/review. Scope is deliberately small
enough to fit.

1. **Scoping** — `extras_require={"rl": ["trl>=X.Y"]}` in `setup.py`;
   create `tests/fixtures/synthetic_reports.json`.
2. **`rewards.py` rewrite** — `make_reward_fn` gains `key=`, `clip_range=`;
   adds defensive conversational-format handler; wraps each batch through
   `validate_rewards`. Required-`key=` error on multi-key metrics.
   F1CheXbert-specific fix verified.
3. **Tests** — `tests/test_rewards.py` covering everything in §3(5).
   `tests/test_trl_integration.py` with `importorskip("trl")`. Extend
   `tests/conftest.py` fixtures as needed.
4. **Example** — `examples/trl_grpo_quickstart.py` replaces
   `examples/trl_reward_demo.py`. Dogfood it end-to-end on a laptop GPU
   before PR review.
5. **Docs rewrite** — `docs/trl_rewards.md` per §5; README §RL updated;
   short VLM snippet + link.
6. **Changelog** — `docs/changelog/…` for 2.2.0: RL hardening, `[rl]`
   extras, runnable GRPO quickstart, F1CheXbert `per_sample` bugfix,
   `key=` / `clip_range=` / `validate_rewards` additions.

**Validation gate before tagging:** run `examples/trl_grpo_quickstart.py`
end-to-end on a real machine. Confirm reward trajectory is non-degenerate.

---

## 7. Risks / tradeoffs

- **TRL API churn.** Reward-fn kwargs have changed in the last 12 months.
  Mitigation: `**kwargs`-absorb pattern (already in place); pin a known-good
  floor in `[rl]` extras; the integration test catches regressions on
  every TRL upgrade.
- **Defensive conversational handling adds a type-check hop.** Real cost
  is negligible; wrong-shape inputs now raise a clear error instead of a
  cryptic one. Worth it.
- **Required `key=` for multi-key metrics is a friction point.** Users
  will see a `ValueError` the first time. Mitigation: the error message
  lists the valid keys; docs show `key=` in every multi-key example.
  Alternative rejected: a hard-coded canonical default per metric — too
  easy to get wrong and to freeze an unintended convention.
- **No scorer cache.** Users who build two reward fns for the same metric
  load the model twice. Acceptable for v2.2.0. Revisit with
  `functools.lru_cache` on an internal factory in a follow-up if requested.
- **No `default_on_error`.** Intentional. Silent zero-rewards hide
  tokenizer / vocab / edge-case bugs that otherwise surface as an
  immediate crash. Fail loud.
- **`radcliq` may be too slow for online RL.** Documented as "eval /
  final-tune; benchmark before per-step use." Defer hard guidance until
  measured in a real GRPO loop.
- **Integration test is narrow.** A 1-step GRPO smoke test on a toy model
  does not validate RLOO, VLM, or conversational flows end-to-end.
  Mitigation: one focused unit test for conversational completions. Docs
  and announcement language is tuned to match: "tested with GRPO;
  expected to work with other TRL trainers that consume a reward-fn
  callable."
- **VLM-docs-only could go stale.** Lower risk than a committed VLM script
  whose `transformers`/`trl` pins drift. The snippet links to TRL's own
  canonical example and doesn't duplicate it.
- **Deferred features.** Scorer cache, `CompositeReward`, committed VLM
  example, async rewards, `log_extra`/`log_metric` forwarding — all
  explicitly listed in §1 and §3. Each one is easy to add when a real
  workflow justifies it. The cost of deferring is low; the cost of shipping
  pre-emptively is maintenance weight for features with no validated user.

---

## 8. Announcement positioning

**Headline:** *"RadEval 2.2.0: radiology metrics, now TRL-compatible
rewards."*

**One-paragraph pitch (for blog / tweet / README banner):**

> RadEval 2.2.0 makes every one of its 16+ radiology metrics — BLEU,
> ROUGE, BERTScore, RadGraph, F1CheXbert, RadCliQ, and more — usable as a
> reward function for reinforcement learning with Hugging Face TRL. A
> single `make_reward_fn("radgraph", key="radgraph_partial")` call gives
> you a drop-in callable for `GRPOTrainer`, `RLOOTrainer`, or any trainer
> that consumes a reward-function callable. Ships with a runnable GRPO
> quickstart, works identically for LLM and VLM report generation, and
> hardens the existing interface (conversational-completion handling,
> required `key=` on multi-key metrics with the F1CheXbert bug fixed).

**What makes this announcement-worthy without being maintenance debt:**

- Small, well-defined contribution (one module + one example + docs) —
  not a new "RL framework."
- Closes a *real* gap the HOPPR team hit last year, where they had to
  re-implement BERTScore / SembScore / RadGraph locally because RadEval's
  surface was epoch-aggregate.
- Validates via a runnable integration test — "works with TRL GRPO" is
  verified, not asserted. Compatibility with other TRL trainers is
  claimed by construction (same reward-function signature) rather than
  over-claimed as tested.
- Plays the supporting role: TRL is the training framework, Transformers
  is the model framework, RadEval is the metrics framework. No
  empire-building.
- Opens natural paths to future one-paragraph announcements (VLM RL case
  study; preference-pair curation for DPO) without locking us into
  maintaining a trainer.

**Measure of success**: a radiology researcher wanting to GRPO-tune a
report-generation model can `pip install RadEval[rl]`, copy
`examples/trl_grpo_quickstart.py`, swap in their dataset, and be training
in under 15 minutes — with confidence that what they see in their logs is
exactly the metric the paper published.

---

## 9. Design decisions & alternatives considered

| Decision | Alternative rejected | Reason |
|---|---|---|
| Defer scorer caching to follow-up | Ship it in v2.2.0 per earlier draft | Both external critics flagged memory, DDP, and unhashable-kwargs concerns. Simpler to ship without and add `functools.lru_cache`-based version if requested. |
| Require `key=` on multi-key metrics; no implicit default | Auto-pick a `per_sample` key | OpenAI critic flagged implicit selection as ambiguous and possibly wrong per metric. Explicit is safer; error message lists valid keys. |
| Defer `CompositeReward` to follow-up PR | Ship it in v2.2.0 | TRL's native `reward_weights` covers the canonical path; convenience sugar can wait for a concrete request. |
| No `default_on_error` | `default_on_error=0.0` per earlier draft | Silent zero-reward hides real bugs in RL; fail loud. |
| VLM example docs-only, not committed | Committed `examples/trl_grpo_vlm.py` | Committed VLM scripts drift with TRL/transformers pins. Docs link to TRL's canonical VLM example is zero-maintenance. |
| BLEU-only quickstart | BLEU + BERTScore dual-reward | First-run experience must be frictionless. A one-line commented swap shows the upgrade path. |
| Defensive conversational handling (type-check first) | Assume `list[list[dict]]` per earlier draft | Gemini critic flagged that TRL's completions may be strings even with conversational prompts. `isinstance(completions[0], str)` first. |
| Support paper-`radcliq`, not `radcliq_v0` | Bundle a branded composite with HOPPR weights | HOPPR weights are an internal simplification; RadEval's existing `radcliq` adapter is paper-faithful. |
| Framing: "TRL-compatible reward support" | Framing: "GRPO support" | Algorithm-agnostic framing matches the API and future-proofs against TRL RLOO / experimental trainers. |
| Mention RLOO; demote PPO | Support PPO as co-equal | PPO in TRL requires a full reward model (nn.Module), not a reward function; RadEval metrics don't fit. |
| `validate_rewards()` (NaN/Inf raise only) | Full warn-band system | Warn-bands are low-value noise; NaN/Inf is high-value signal. Keep the useful part. |
| Drop `.metric_keys` attribute on the callable | Expose for "TRL logging" | No concrete consumer identified; OpenAI critic flagged as speculative. |
| Narrow adapter audit to F1CheXbert | Repo-wide audit of mode-dependent keys | Only one offender is confirmed; users will surface others if they exist. The required-`key=` path makes those failures loud rather than silent. |

---

## 10. Facts vs assumptions

**Facts** (verified in code):

- `RadEval/rewards.py:38` does `scorer = cls(**metric_kwargs)` per call
  (no caching).
- `RadEval/metrics/f1chexbert/adapter.py:12-33` returns different keys in
  `per_sample=True` mode than those advertised by `metric_keys()`.
- `setup.py:75-81` has only `[api]` extras; no `[rl]`.
- `tests/test_e2e.py:106-129` tests `make_reward_fn` against BLEU only.
- `examples/trl_reward_demo.py` never imports or instantiates TRL.
- `vlm_align` does not import RadEval; it implements rewards locally.

**Assumptions** (to validate during implementation):

- Current TRL (`>=1.3`) passes `completions` as `list[str]` for standard
  prompt-only datasets and *may* pass `list[list[dict]]` for conversational
  (with variance across versions/tokenizers). The defensive handler
  covers both. Integration test pins one specific TRL version.
- Other metrics with multi-key `per_sample` output are rare; the
  required-`key=` error makes any such case surface loudly rather than
  silently miscompute.
- `Qwen/Qwen2.5-0.5B` runs `GRPOConfig(max_steps=5)` in under ~10 minutes
  on a modest laptop GPU. Validate before tagging release.
- `make_reward_fn("radcliq")` is fast enough for *evaluation* reward use.
  Per-step online use requires benchmarking; documented as such.

**Hypotheses** (not validated, not acted on):

- Users will want `CompositeReward` eventually.
- Users will want scorer caching eventually.
- Users will want async reward functions eventually.
- Users will want a committed VLM example eventually.

All four explicitly deferred; we'll act when there is evidence.
