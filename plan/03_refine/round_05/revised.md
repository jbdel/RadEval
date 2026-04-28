# RadEval × TRL: Reinforcement-Learning Support Proposal

*A joint report from three investigating perspectives — TRL expert, HOPPR
`vlm_align` expert, and RadEval expert — on how RadEval should integrate with
Hugging Face TRL for GRPO-style radiology-report RL. Revised through five
rounds of external review.*

---

## 1. Executive summary

**Recommendation: position RadEval as a provider of TRL-compatible *reward
functions*, algorithm-agnostic. Ship a tight v2.2.0 PR that hardens the
existing `make_reward_fn`, closes a confirmed bug, and replaces a stub demo
with a runnable GRPO example. Defer everything else until there is evidence
users need it.**

Framing shift from the original draft: this is not "GRPO support." It is
"TRL-compatible reward support, with GRPO as the flagship example and the
only tested trainer." Other TRL trainers that consume a reward-function
callable (e.g. RLOO) are *designed-to-work* but not separately validated in
this PR.

The v2.2.0 PR ships exactly this:

1. **Hardened `make_reward_fn`**
   - Absorbs TRL's reward-fn kwargs (`completion_ids`, `trainer_state`,
     `log_extra`, `log_metric`, `environments`) via `**kwargs`.
   - Handles conversational completions defensively: checks
     `isinstance(completions[0], str)` first; only extracts `content` from
     message-dict form when needed. The content-extraction rule ("last
     assistant message") is documented as a **best-effort heuristic** for
     the common OpenAI-style message layout. Users whose datasets use a
     different layout should pre-format the completion text upstream in
     their dataset/collator pipeline (not via `score_transform`, which
     operates on scores). On use, fires a `UserWarning` naming the
     heuristic (user-filterable via the standard `warnings` module; no
     module-level state). Unrecognized shapes **or** missing dict keys
     raise `TypeError` with upstream-preprocess guidance.
   - Adds a `key=` argument for metrics with multi-key `per_sample`
     output. **Required** (no implicit default) when the metric has >1
     per-sample key; the user gets a `ValueError` listing the valid
     keys. This both (a) fixes the F1CheXbert `KeyError` bug (§2.3) and
     (b) makes any future metric with the same shape fail loudly. The
     generalized check is ~3 LOC over a F1CheXbert-only patch.
   - Calls a tiny `validate_rewards()` helper once per batch to raise on
     NaN/Inf with metric name and offending sample index. The helper
     first extracts a scalar (`v.item() if hasattr(v, "item") else v`)
     and then casts through `float()`, so it catches `numpy.nan`, 0-D
     torch scalars, 1-element 1-D tensors, and Python floats uniformly.
     No silent defaulting.
   - No `clip_range`, no `default_on_error`, no `.metric_keys` attribute —
     all dropped as YAGNI after review.
2. **`pip install RadEval[rl]` extras** — adds `trl>=X.Y,<Z` (one specific
   verified version/range; compatibility with nearby TRL versions stated as
   "may work but not separately validated"). `datasets` is already a core
   dep.
3. **Runnable quickstart** — `examples/trl_grpo_quickstart.py`, BLEU
   only, `Qwen/Qwen2.5-0.5B` (chosen because a random tiny model won't
   produce a meaningful reward signal for a user-facing demo; the
   integration test uses a tiny random model). Replaces the existing
   call-shape stub. Runtime/resource claims are **validation gates, not
   published numbers**; concrete numbers go into docs only after measuring
   on a reference machine.
4. **Tests** — `tests/test_rewards.py` (unit) and
   `tests/test_trl_integration.py` (`pytest.importorskip("trl")`, 1-step
   GRPO smoke on a tiny random test model — e.g.
   `trl-internal-testing/tiny-random-LlamaForCausalLM` — so the integration
   test can run on a CPU CI runner). Unit tests also cover the
   conversational `list[list[dict]]` path and a narrow test that passes an
   extra (unused) dataset column + representative TRL kwargs through the
   reward callable to confirm the `**kwargs`-absorb contract.
5. **Rewritten `docs/trl_rewards.md`** — reframes as "TRL-compatible reward
   functions"; GRPO primary (tested); RLOO noted as another fit (same
   signature, untested here); PPO demoted. VLM is **documentation-only**:
   snippet + link to `trl/examples/scripts/grpo_vlm.py`. New VRAM callout
   about duplicate model loads when two reward fns share a heavy metric.

**Deferred to a follow-up PR** (contingent on demand):

- Scorer caching. Removed from v2.2.0; multiple critics flagged memory /
  DDP / kwarg-hashing concerns. Revisit with `functools.lru_cache` on an
  internal factory if a real workflow demands it.
- `CompositeReward` class. TRL's native `reward_funcs=[...]` +
  `reward_weights=[...]` is the canonical path.
- Committed VLM example file.
- `log_extra` / `log_metric` forwarding.
- Async reward-fn support.

**Explicitly out of scope forever**: custom GRPO trainer, VLM adapter,
rollout engine, reward-model training, dataset/collator code. These
belong in downstream code (e.g. `vlm_align`) or in TRL itself.

Scope framing: **~150 LOC of core code change in `rewards.py`**, plus a
new example file, two new test files, a rewritten docs page, a README
edit, a changelog entry, and a `setup.py` extras line. The core delta is
small; the full PR footprint is "small + routine supporting changes."

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
  function is used. Defensive type-check in the adapter is required.
- `async def` reward functions are first-class.
- PPO lives in `trl.experimental.ppo`; GRPO is the idiomatic online-RL
  choice. RLOO is a reasonable alternative that consumes the same reward
  signature.

**Verdict on RadEval's current `make_reward_fn`:** signature is correct;
`**kwargs` already absorbs unknown kwargs. Missing: defensive
conversational-format handling, graceful multi-key metric behavior.

**Composite rewards — TRL already solves this.** Native `reward_funcs` +
`reward_weights` in `GRPOConfig` sums internally and auto-logs per-function
means.

**VLM — TRL's story is solid now.** Native VLM support across Gemma3,
LLaVA-Next, Qwen2-VL/2.5-VL, SmolVLM2. RadEval should not ship trainer
code.

**Recommended URLs to link from RadEval docs:**

- <https://huggingface.co/docs/trl/main/en/grpo_trainer>
- <https://huggingface.co/docs/trl/main/en/grpo_trainer#using-a-custom-reward-function>
- <https://huggingface.co/docs/trl/main/en/grpo_trainer#vision-language-model-vlm-training>
- <https://github.com/huggingface/trl/blob/main/examples/scripts/grpo_vlm.py>
- <https://huggingface.co/docs/trl/main/en/dataset_formats>
- <https://huggingface.co/docs/trl/main/en/rloo_trainer>

### 2.2 `vlm_align` / HOPPR RL expert

**Does `vlm_align` use RadEval?** No. A grep turns up only *anti-references*:
the team chose the standalone `radgraph` pip package over a "RadEval
subprocess" to avoid cold-start per batch, and an internal design doc
states that a RadEval-subprocess-based metric adapter is the "wrong
interface for per-sample GRPO rewards. All three metrics must be
implemented locally."

**Central finding:** the team bypassed RadEval because RadEval's reward
surface was corpus/epoch-aggregate, not per-sample online. `make_reward_fn`
has since closed most of that gap — our job now is to finish it.

**Why `vlm_align` has a custom GRPO trainer:**

- HOPPR VLM takes `images_pixels` / `images_masks`, not `pixel_values`.
- Only `text_decoder` is LoRA-adapted; vision encoder + QFormer frozen.
  TRL's model-class assumptions can't cleanly express this.
- `vision_cache.py` short-circuits the vision encoder across 16 rollouts
  per study.
- Uses `trust_remote_code` with a monkey-patched `_init_vision_encoder`.

All **training-layer** concerns. None belong in RadEval.

**Reward-function interface in `vlm_align`:** returns
`(scalars, per_component_diagnostics_dict)`. TRL's plain `list[float]` is
less expressive, but `reward_funcs=[...]` + `reward_weights` with
auto-logged per-function means covers the same need for most users. **No
need to port the tuple interface.**

**Generic pieces worth considering for upstream:**

| Piece | Where | Upstream? |
|---|---|---|
| Weighted composite | `rewards/radcliq_reward.py` | **Deferred**; TRL already sums `reward_funcs`. |
| NaN/Inf hard-raise | `rewards/radcliq_reward.py:138-145` | **Yes**, as a ~15-LOC `validate_rewards()` helper. |
| Per-metric warn bands | `rewards/radcliq_reward.py:44-49` | **No**. TRL's logging already exposes reward trajectories. |
| Group-normalized advantages | `core/advantages.py` | **No** — training-layer. |

**HOPPR-specific, leave behind:** `VLMAdapter`, rollout engine, GRPO loss,
collators, vision cache, the specific `radcliq_v0` weights.

### 2.3 RadEval expert

**What's already in place** (`RadEval/rewards.py`, ~40 LOC): single-metric
`make_reward_fn` with API-metric warning; exported from
`RadEval/__init__.py`. Docs at `docs/trl_rewards.md`. Example
(`examples/trl_reward_demo.py`) never instantiates TRL — it prints a
string. Three tests in `tests/test_e2e.py:106-129` cover BLEU only; nothing
verified against actual TRL.

**Bugs / gaps** (verified in code):

- **F1CheXbert multi-key bug** — `RadEval/metrics/f1chexbert/adapter.py:12-33`.
  `metric_keys()` advertises `["f1chexbert_5_micro_f1", ...]` but
  `compute(..., per_sample=True)` returns
  `{"f1chexbert_sample_acc_5": [...], "f1chexbert_sample_acc_all": [...]}`.
  `make_reward_fn("f1chexbert")` raises `KeyError` today. **Fix scope:
  F1CheXbert specifically.** Required-`key=` behavior makes any similar
  case in other metrics surface loudly.
- **`examples/trl_reward_demo.py` is a stub.**
- **No `tests/test_rewards.py`.**
- **No `[rl]` extras** in `setup.py:75-81`.
- **Speed table in `docs/trl_rewards.md`** omits `radgraph` and under-warns
  on `green` (7B local LLM; unusable per-step without a spare GPU).

**RadEval should ship**: the reward callable, a runnable GRPO example,
speed guidance. Stay out of dataset loading, prompt formatting, chat
templates, trainer config, reward-model training, KL/ref-model,
distributed launchers.

---

## 3. Recommended scope

**Ship in v2.2.0** (single PR; core code change ~150 LOC, plus routine
supporting changes in tests / docs / example / setup.py / README /
changelog):

1. **`make_reward_fn` hardening**
   - **F1CheXbert bugfix (required)**: `make_reward_fn("f1chexbert")`
     currently raises `KeyError` because `metric_keys()[0]` disagrees
     with `per_sample=True` output. Fix: require the caller to pass
     `key=`.
   - **Generic multi-key `key=` behavior (minimal generalization)**: the
     same check is applied to any metric with >1 per-sample key. Raises
     `ValueError` listing valid keys. ~3 LOC over the F1CheXbert-only
     patch and avoids re-touching `rewards.py` for the next such case.
   - **Defensive conversational-completion handling**:
     `completions[0]` is a string → pass through;
     `completions[0]` is `list[dict]` → extract `content` from the last
     assistant message (best-effort heuristic for the common OpenAI-style
     layout); fires a `UserWarning` per call (user-filterable; no
     module-level state); **missing `content` or missing `role` keys →
     fall through to the `TypeError` with upstream-preprocess guidance**;
     any unrecognized shape → same `TypeError`.
   - **`validate_rewards(values, metric_name)` helper (~15 LOC)**:
     **preserves `None` entries unchanged** (TRL uses `None` to skip
     samples in multi-task routing — we must not coerce it). Non-`None`
     values: normalize via `v.item() if hasattr(v, "item") else v`, cast
     through `float()`, raise on NaN/Inf naming metric and sample
     index. Uniform across numpy scalars, 0-D / 1-element-1-D torch
     tensors, and Python floats.
2. **`pip install RadEval[rl]` extras**: `trl>=X.Y,<Z` pinned to one
   verified range.
3. **`make_reward_fn("radcliq")` works** (already does). Documented as
   **"evaluation / final-tune reward; benchmark per-sample cost before
   using for online training"**. No invented "RadCliQ-like composite"
   with our weights.
4. **Runnable quickstart**: `examples/trl_grpo_quickstart.py`, ~40-60
   lines, small LLM (e.g. `Qwen/Qwen2.5-0.5B`), ~20-row synthetic fixture,
   `GRPOConfig(max_steps=5)`. Uses `make_reward_fn("bleu")` only. One
   commented line shows the swap to `bertscore` or `radcliq`. Deletes
   `examples/trl_reward_demo.py`.
5. **Tests**:
   - `tests/test_rewards.py`:
     - `make_reward_fn("bleu")`, `rouge`, `bertscore` round-trip.
     - F1CheXbert: with correct `key=` → works; without → raises with
       valid-keys listed.
     - Conversational-format path:
       `reward(completions=[[{"role": "assistant", "content": "x"}]], ground_truth=[...])`.
     - Unrecognized completion shape (e.g. tensor ids) → raises
       `TypeError`.
     - `score_transform` composes correctly; clipping expressed via
       `score_transform=lambda x: max(lo, min(hi, x))`.
     - NaN/Inf in raw scores → raises naming metric + index.
     - `numpy.nan` / tensor-NaN input → same behavior after the `float()`
       cast in `validate_rewards`.
     - `UserWarning` fires for API-based metrics.
     - **Extra-kwargs absorption**: reward fn is called with an
       unused/extra dataset column and representative TRL kwargs
       (`completion_ids`, `trainer_state=None`, `log_extra`,
       `log_metric`) and returns correctly.
     - **`None` pass-through**: a reward function whose underlying
       scorer returns `None` for some samples (TRL's sample-skip
       convention) produces a list that contains `None` entries, not
       a coerced zero or a raised error.
   - `tests/test_trl_integration.py`:
     - `pytest.importorskip("trl")`.
     - One `max_steps=1` GRPO run on a **tiny random** test model
       (`trl-internal-testing/tiny-random-LlamaForCausalLM` or
       equivalent) so the test runs on a CPU CI runner.
     - Asserts reward fn was called and `trainer.state.global_step == 1`.
6. **Rewritten `docs/trl_rewards.md`** — structure:
   - **Primary path (tested)**: GRPO with a single RadEval reward function.
     Full worked example; matches the quickstart. This is the bulk of the
     doc.
   - **Reward callable contract**: signature, batch semantics (processes
     the full completion list per call, matching TRL), `**kwargs`-absorb
     contract, required `key=` for multi-key metrics.
   - **Composing metrics**: native TRL `reward_funcs=[...]` +
     `reward_weights`; with VRAM callout — *"Two reward functions that
     share a heavy underlying metric (e.g. two keys from RadGraph, or two
     BERTScore variants) will load the underlying model twice — scorer
     caching is not yet shipped. Plan VRAM accordingly."*
   - **Paper-RadCliQ**: `make_reward_fn("radcliq")` with eval-first
     positioning.
   - **Known limitations** — short subsection. Conversational-format
     heuristic documented: "`list[dict]` completions: we extract `content`
     from the last assistant message. If your dataset uses a different
     message layout, preprocess the completion text upstream in your
     dataset/collator pipeline before it reaches the reward function." One
     short note on `radcliq` per-sample cost.
   - **Adjacent / untested uses** — brief. RLOO (same signature; not
     tested here); PPO pointer to `trl.experimental.ppo` with the
     GRPO/RLOO recommendation; VLM snippet + link to
     `trl/examples/scripts/grpo_vlm.py` with *"The reward interface is
     unchanged between LLM and VLM — the reward function never sees the
     image."*; preference-pair curation for DPO/ORPO/KTO as a workflow
     note, not a feature.
   - **Speed table**: adds `radgraph` (medium, GPU); stronger warning on
     `green`; API metrics marked "not practical for RL."
   - **Measured runtime/VRAM numbers** from the quickstart, filled in
     after benchmarking.
7. **README update**: swap stub reference for quickstart link; update the
   pitch per §8.
8. **Changelog entry** for 2.2.0.

**Deferred to a follow-up PR** (explicit):

- Scorer caching.
- `CompositeReward` class.
- Committed VLM example file.
- `log_extra` / `log_metric` forwarding.
- Async rewards.

Add these when a user opens an issue with a concrete workflow.

**Answering the plan's explicit questions:**

| Question | Answer |
|---|---|
| TRL-compatible reward functions? | **Yes** — harden the existing one. |
| Example GRPO/PPO training scripts? | **One GRPO LLM quickstart.** No committed VLM script. No PPO. |
| Wrappers/adapters around existing metrics? | **No new wrappers.** |
| VLM-specific trainer utilities? | **No.** Documentation only. |
| Documentation only? | **No** — docs + bugfix + hardening + example + tests. |
| Boundary? | RadEval ends at `reward_fn(completions=..., **kwargs) -> list[float]`. |
| LLM vs VLM? | Identical reward interface. |
| Incorporate `vlm_align`'s trainer? | **No.** Only the NaN/Inf helper is ported. |
| Concrete examples? | `examples/trl_grpo_quickstart.py`. VLM in docs. |
| API changes? | Additive: new `key=`. One behavioral fix: F1CheXbert now works (was broken). |
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
    key="f1chexbert_sample_acc_5",            # REQUIRED for multi-key metrics
    score_transform=lambda x: max(0.0, min(1.0, (x - 0.5) * 2)),  # clipping via transform
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B",
    reward_funcs=[reward],
    train_dataset=dataset,                    # must have "ground_truth" column
    args=GRPOConfig(output_dir="out", max_steps=100),
)
trainer.train()
```

```python
# Multi-metric — idiomatic TRL, no new RadEval abstraction
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
# Paper-RadCliQ — uses the existing radcliq adapter.
# Recommended as an evaluation / final-tune reward until per-sample cost
# is benchmarked for online use. See docs/trl_rewards.md.
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
import warnings

def _as_strings(completions):
    if not completions:
        return []
    first = completions[0]
    if isinstance(first, str):
        return list(completions)
    if isinstance(first, list) and first and isinstance(first[0], dict):
        warnings.warn(
            "make_reward_fn: extracting 'content' from the last assistant "
            "message in list[dict] completions (best-effort heuristic). "
            "If your dataset uses a different layout, preprocess the "
            "completion text upstream in your dataset/collator pipeline.",
            UserWarning,
            stacklevel=2,
        )
        try:
            return [_last_assistant_content(msgs) for msgs in completions]
        except (KeyError, IndexError) as exc:
            raise TypeError(
                "make_reward_fn: list[dict] completions lack expected "
                "'role'/'content' keys. Preprocess your completions "
                "upstream in the dataset/collator pipeline."
            ) from exc
    raise TypeError(
        f"make_reward_fn: unsupported completions shape "
        f"(got {type(first).__name__}); expected list[str] or "
        f"list[list[dict]]. Preprocess your completions upstream in "
        f"the dataset/collator pipeline."
    )
```

**`validate_rewards` (sketch):**

```python
import math

def validate_rewards(values, metric_name):
    normalized = []
    for i, v in enumerate(values):
        if v is None:
            # TRL uses None to skip a sample in multi-task routing.
            # Passing it through preserves that behavior.
            normalized.append(None)
            continue
        scalar = v.item() if hasattr(v, "item") else v
        f = float(scalar)
        if math.isnan(f) or math.isinf(f):
            raise ValueError(
                f"{metric_name}: non-finite reward at sample index {i} "
                f"(got {f!r}). Fix the upstream scorer or the input."
            )
        normalized.append(f)
    return normalized
```

No `clip_range`, no scorer cache, no kwarg-normalization loop, no
`.metric_keys` attribute.

---

## 5. Example workflows to include

### 5.1 `examples/trl_grpo_quickstart.py` (committed, BLEU, runnable — user-facing)

- ~40-60 lines.
- Model: `Qwen/Qwen2.5-0.5B` (user-facing quickstart only; large enough to
  produce a meaningful reward signal on a laptop GPU).
- Dataset: 20-row synthetic fixture at
  `tests/fixtures/synthetic_reports.json` (reused by the integration test).
- `make_reward_fn("bleu")`. BLEU is deliberate: no GPU dependencies in the
  reward path, near-instant, no model downloads.
- One commented line: `# Swap in: make_reward_fn("bertscore") — heavier
  but richer signal. See docs/trl_rewards.md.`
- `GRPOConfig(max_steps=5, per_device_train_batch_size=2,
  num_generations=2)`.
- Prints reward trajectory.
- Goal: copy-paste-and-it-works. **Runtime / VRAM numbers will be added
  to docs only after benchmarking on a reference machine** — no
  speculative minute counts published.
- **CI** runs the much smaller `tests/test_trl_integration.py` against a
  tiny random test model, not this quickstart.

### 5.2 VLM — docs only

In `docs/trl_rewards.md`:

```python
# Identical reward setup — the reward function never sees the image.
from RadEval.rewards import make_reward_fn

reward = make_reward_fn("bertscore")
# Plug into trl's canonical VLM GRPO example:
# https://github.com/huggingface/trl/blob/main/examples/scripts/grpo_vlm.py
```

Plus a short paragraph: TRL handles the image-conditioned rollout
end-to-end; our reward interface is unchanged.

### 5.3 Docs walk-through: choosing / combining metrics

- Single fast metric → `make_reward_fn("bleu")` or `"rouge"`.
- Single clinical metric → `make_reward_fn("f1chexbert", key="f1chexbert_sample_acc_5")` or `make_reward_fn("radgraph", key="radgraph_partial")`.
- Paper-RadCliQ → `make_reward_fn("radcliq")` *(recommended for eval /
  final-tune; benchmark before using per-step)*.
- Custom mix → `reward_funcs=[...]` + `reward_weights=[...]` in
  `GRPOConfig`. **VRAM note:** if two reward fns share the same heavy
  underlying model (e.g. two keys from RadGraph, or two BERTScore
  variants), it loads twice — scorer caching is not yet shipped. Plan
  VRAM accordingly, or compute both keys from a single reward function.
- Why no `CompositeReward` helper: TRL already does this natively. Open
  a GitHub issue if your use case isn't covered.

---

## 6. Implementation plan

Single PR. Realistic estimate: ~1-2 focused days for the core code
(`rewards.py` + unit tests), + a half-day for the example and integration
test, + a half-day for docs / README / changelog / dogfooding. Core-code
delta is ~150 LOC; overall PR footprint is larger because of the
supporting changes.

1. **Scoping** — `extras_require={"rl": ["trl>=X.Y,<Z"]}` in `setup.py`;
   create `tests/fixtures/synthetic_reports.json`.
2. **`rewards.py` rewrite** — `make_reward_fn` gains required `key=` (for
   multi-key metrics) and defensive conversational-format handling; wraps
   each batch through `validate_rewards`. F1CheXbert-specific fix
   verified.
3. **Tests** — `tests/test_rewards.py` covering §3(5). Integration test
   with `importorskip("trl")`. Extend `tests/conftest.py` fixtures.
4. **Example** — `examples/trl_grpo_quickstart.py` replaces
   `examples/trl_reward_demo.py`. Dogfood end-to-end on a reference
   machine before PR review. Capture actual runtime + VRAM.
5. **Docs rewrite** — `docs/trl_rewards.md` per §5; README §RL updated;
   short VLM snippet + link; VRAM callout; conversational heuristic
   note; batch-contract note. Publish measured runtime only after
   benchmarking.
6. **Changelog** — `docs/changelog/…` for 2.2.0: RL hardening, `[rl]`
   extras, runnable GRPO quickstart, F1CheXbert `per_sample` bugfix,
   required `key=` for multi-key metrics, `validate_rewards` utility.

**Validation gate before tagging:** run
`examples/trl_grpo_quickstart.py` end-to-end on a real machine. Confirm
reward trajectory is non-degenerate. Record actual runtime and VRAM
footprint; update docs with measured numbers.

---

## 7. Risks / tradeoffs

- **TRL API churn.** Reward-fn kwargs have changed in the last 12 months.
  Mitigation: `**kwargs`-absorb pattern (already in place); pin a
  known-good floor in `[rl]` extras; integration test catches regressions
  on each TRL upgrade.
- **Conversational-handler heuristic may mis-score exotic message
  layouts.** Mitigation: documented as best-effort (scoped to the common
  OpenAI-style layout) with clear failure behavior (type-check raises on
  shapes we don't recognize; missing dict keys raise the same
  `TypeError`); unit test covers the OpenAI-style case; a `UserWarning`
  fires on each invocation (user-filterable via the standard `warnings`
  module — we accept the per-batch warning noise as the cost of avoiding
  global state); users on other layouts preprocess the completion text
  upstream in their dataset/collator pipeline (not via `score_transform`,
  which operates on scores). An alternative considered and rejected:
  drop the heuristic entirely and require upstream preprocessing for
  both OpenAI-style and non-OpenAI-style layouts. Keeping the heuristic
  is a small cost that handles the most common case out of the box.
- **Required `key=` for multi-key metrics is a friction point.** Users
  will see a `ValueError` the first time. Mitigation: error message lists
  valid keys; docs show `key=` in every multi-key example. Alternative
  rejected: hard-coded canonical default per metric — too easy to get
  wrong and freeze unintended convention.
- **No scorer cache.** Users who build two reward fns for the same metric
  load the model twice. Documented via the VRAM callout. Revisit with
  `functools.lru_cache` on an internal factory if demanded.
- **No `clip_range`.** Dropped during review. Users clip via
  `score_transform=lambda x: max(lo, min(hi, x))`. Example shown in §4.
- **No `default_on_error`.** Intentional. Silent zero-rewards hide
  tokenizer / vocab / edge-case bugs. Fail loud.
- **`radcliq` may be too slow for online RL.** Documented as "eval /
  final-tune; benchmark before per-step." Defer hard guidance until
  measured.
- **Integration test is narrow** (1-step GRPO, toy model). Mitigation:
  one focused unit test for conversational completions. Docs and
  announcement language match: "GRPO smoke-tested; designed to work with
  other TRL reward-fn trainers (e.g. RLOO) — not separately validated."
- **VLM-docs-only could go stale.** Lower risk than a committed VLM
  script whose `transformers`/`trl` pins drift. Snippet links to TRL's
  own canonical example.
- **Deferred features.** Each one is easy to add when a real workflow
  justifies it. Cost of deferring is low; cost of shipping pre-emptively
  is maintenance weight for features with no validated user.

---

## 8. Announcement positioning

**Headline:** *"RadEval 2.2.0: radiology metrics as TRL-compatible
rewards."*

**One-paragraph pitch (for blog / tweet / README banner):**

> RadEval 2.2.0 makes many of its radiology metrics — BLEU, ROUGE,
> BERTScore, RadGraph, F1CheXbert, RadCliQ — usable as reward functions
> for reinforcement learning with Hugging Face TRL. A single
> `make_reward_fn("radgraph", key="radgraph_partial")` call gives you a
> drop-in callable smoke-tested against `GRPOTrainer`, with the same
> interface designed to work for other TRL trainers that consume a
> reward-function callable (e.g., RLOO) — not separately validated here
> — and unchanged across LLM and VLM report generation. Lightweight metrics (BLEU, ROUGE, BERTScore, RadGraph,
> F1CheXbert) are the primary supported path; API-based metrics
> (CRIMSON, MammoGREEN, RadFact-CT) remain available but are not
> recommended for online RL due to cost. Ships with a runnable GRPO
> quickstart, a long-standing F1CheXbert bugfix, and explicit `key=`
> handling for multi-key metrics.

**What makes this announcement-worthy without being maintenance debt:**

- Small, well-defined contribution (one module + one example + docs) —
  not a new "RL framework."
- Closes a *real* gap the HOPPR team hit last year, where they had to
  re-implement BERTScore / SembScore / RadGraph locally because RadEval's
  surface was epoch-aggregate.
- Validates via a runnable integration test — basic GRPO integration is
  smoke-tested on a toy model, not asserted. Compatibility with other
  TRL trainers is stated as design intent, not claimed as tested.
- Plays the supporting role: TRL is the training framework, Transformers
  is the model framework, RadEval is the metrics framework. No
  empire-building.
- Opens natural paths to future one-paragraph announcements (VLM RL
  case study; preference-pair curation for DPO) without locking us into
  maintaining a trainer.

**Measure of success**: a radiology researcher wanting to GRPO-tune a
report-generation model can `pip install RadEval[rl]`, copy
`examples/trl_grpo_quickstart.py`, swap in their dataset, and be training
with a measured, documented runtime/VRAM budget.

---

## 9. Design decisions & alternatives considered

| Decision | Alternative rejected | Reason |
|---|---|---|
| `validate_rewards` passes `None` through unchanged | Cast every entry via `float()` | Gemini round 5 (high severity): TRL uses `None` entries in a reward list to skip samples during multi-task routing. `float(None)` would crash. Round 5 patches the helper to short-circuit on `None`. |
| Keep conversational heuristic + `UserWarning` per call | Drop the heuristic entirely and require upstream preprocessing for all layouts | Gemini round 5 recommended dropping; OpenAI round 4 wanted it defensive. Compromise: keep the heuristic for the common OpenAI-style layout (most users benefit out of the box), accept per-batch warning noise, document the tradeoff in §7. |
| `warnings.warn(..., UserWarning)` on conversational-heuristic use | Module-level `_HEURISTIC_LOGGED` flag + `logging.info` per round 3 draft | Round 4 critics converged: global state is unpredictable under DDP/multiprocessing; `logging.info` has limited visibility if users don't raise the log level. `warnings.warn` is Pythonic, filterable, visible by default, and stateless. |
| Separate F1CheXbert bugfix from generic `key=` hardening in the plan | Present as one undifferentiated "fix" | OpenAI round 4: reviewers deserve to see that the bugfix is necessary, and that the generic handling is a ~3-LOC minimal generalization (not scope creep). |
| Safe dict-key extraction (try/except → `TypeError`) | Raw `msg["content"]` access | Gemini round 4: if the user's dicts lack `role`/`content`, the current code would throw a cryptic `KeyError`. Fall through to the same `TypeError` as other unsupported shapes for consistent guidance. |
| `validate_rewards` uses `.item()` before `float()` | `float()` alone | Gemini round 4: 1-element 1-D tensors (e.g. `torch.tensor([0.5])`) fail the raw `float()` cast on some PyTorch versions. `.item()` first normalizes shape. |
| "GRPO smoke-tested" everywhere (not "tested with GRPO") | "tested with GRPO" | OpenAI round 4: a 1-step run on a tiny model verifies wiring, not behavior. Honest framing. |
| Explicit justification for Qwen2.5-0.5B in the quickstart | Silent choice | OpenAI round 4: reviewer should see why CI uses a tiny random model while the user-facing quickstart uses a larger real model. |
| Tiny random model in CI integration test; Qwen2.5-0.5B in user-facing quickstart only | Single 0.5B model used for both | Gemini round 3: 0.5B in CI risks OOM / download timeouts on CPU runners. Split: CI-safe tiny model for automated test; meaningful-signal model for user quickstart. |
| "Preprocess upstream in dataset pipeline" (not via `score_transform`) for conversational edge cases | "Use `score_transform`" per round 3 draft | OpenAI round 3: `score_transform` operates on scores, not text. Incorrect guidance. |
| Added extra-kwargs absorption test | Only behavior-test GRPO smoke | OpenAI round 3: central compatibility claim (`**kwargs` absorbs TRL's kwargs + extra dataset columns) wasn't directly tested. One narrow unit test closes that. |
| Pin one verified TRL version range | Open-ended floor | OpenAI round 3: doc claims should match one verified range, not a speculative minimum. |
| Split scope language: "~150 LOC core code" vs "small + supporting changes (docs, tests, example)" | "~150-line PR" | OpenAI round 3: the simple LOC number understates review/maintenance cost of the full PR footprint. |
| Drop `clip_range` | Ship it in v2.2.0 | Round 2 critics both flagged as unjustified API expansion with ambiguous ordering against `score_transform`. KISS: `score_transform=lambda x: max(lo, min(hi, x))` suffices. |
| Soften "by construction" compatibility claim | "Works with all reward-fn trainers by construction" | OpenAI critic: signature similarity is suggestive, not proof. Claim "GRPO smoke-tested; designed to work with others (RLOO); not separately validated." |
| Document conversational heuristic as best-effort | Silent "works for all layouts" | OpenAI critic: "last assistant message" extraction isn't verified across all TRL conversational shapes. Document limitations; raise on unrecognized shapes. |
| Publish runtime only after benchmarking | Publish "<10 min on laptop GPU" claim | OpenAI critic: speculative runtime creates support burden if wrong. Validation gate now, docs-number later. |
| Add VRAM callout for no-cache behavior | Silent about duplicate loads | Gemini critic: two reward fns sharing a heavy metric will OOM on standard hardware without a warning. |
| Narrow announcement language | "16+ metrics usable as a reward function" | OpenAI critic: too broad for API-backed / expensive metrics. Say "lightweight metrics are the primary supported path." |
| Defer scorer caching to follow-up | Ship in v2.2.0 | Round 1 critics: memory, DDP, unhashable-kwargs concerns. Simpler to ship without. |
| Require `key=` on multi-key metrics | Auto-pick a `per_sample` key | OpenAI round 1: implicit selection is ambiguous and possibly wrong per metric. Explicit + error message listing valid keys. |
| Defer `CompositeReward` to follow-up PR | Ship in v2.2.0 | TRL's native `reward_weights` covers the canonical path. |
| No `default_on_error` | `default_on_error=0.0` | User's own critique (§9 of original draft): silent zero-reward hides real bugs. |
| VLM example docs-only | Committed VLM script | Drifts with TRL/transformers pins. Docs link to TRL's canonical VLM example is zero-maintenance. |
| BLEU-only quickstart | BLEU + BERTScore dual-reward | First-run must be frictionless. One-line commented swap shows upgrade path. |
| Defensive conversational handling | Assume one shape | Gemini round 1: TRL returns strings even with conversational prompts sometimes. Check type first. |
| Support paper-`radcliq`, not `radcliq_v0` | Bundle a branded composite | HOPPR weights are internal simplification; RadEval's existing `radcliq` is paper-faithful. |
| Framing: "TRL-compatible reward support" | "GRPO support" | Algorithm-agnostic framing matches the API. |
| Mention RLOO; demote PPO | Support PPO as co-equal | PPO in TRL needs a full reward model (nn.Module). |
| `validate_rewards()` — NaN/Inf raise only | Full warn-band system | Warn-bands are low-value; NaN/Inf is high-value. |
| Drop `.metric_keys` attribute | Expose for TRL logging | OpenAI round 1: no concrete consumer. |
| Narrow adapter audit to F1CheXbert | Repo-wide audit of mode-dependent keys | Only confirmed offender; required-`key=` makes future cases loud. |

---

## 10. Facts vs assumptions

**Facts** (verified in code):

- `RadEval/rewards.py:38` does `scorer = cls(**metric_kwargs)` per call
  (no caching).
- `RadEval/metrics/f1chexbert/adapter.py:12-33` returns different keys in
  `per_sample=True` mode than `metric_keys()` advertises.
- `setup.py:75-81` has only `[api]` extras; no `[rl]`.
- `tests/test_e2e.py:106-129` tests `make_reward_fn` against BLEU only.
- `examples/trl_reward_demo.py` never imports or instantiates TRL.
- `vlm_align` does not import RadEval; it implements rewards locally.

**Assumptions** (to validate during implementation):

- Current TRL (`>=1.3`) passes `completions` as `list[str]` for standard
  prompt-only datasets; conversational shape varies by version/tokenizer.
  The defensive handler covers both. Integration test pins one specific
  TRL version.
- Other metrics with multi-key `per_sample` output are rare; the
  required-`key=` error makes any such case surface loudly rather than
  silently miscompute.
- `Qwen/Qwen2.5-0.5B` with `GRPOConfig(max_steps=5)` fits in modest GPU
  VRAM (to be measured, not assumed published).
- `make_reward_fn("radcliq")` is fast enough for *evaluation* reward use;
  per-step online use requires benchmarking.

**Hypotheses** (not validated, not acted on):

- Users will want `CompositeReward` eventually.
- Users will want scorer caching eventually.
- Users will want async reward functions eventually.
- Users will want a committed VLM example eventually.

All four explicitly deferred; we'll act when there is evidence.
