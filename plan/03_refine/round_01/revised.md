# RadEval × TRL: Reinforcement-Learning Support Proposal

*A joint report from three investigating perspectives — TRL expert, HOPPR
`vlm_align` expert, and RadEval expert — on how RadEval should integrate with
Hugging Face TRL for GRPO-style radiology-report RL. Revised after internal
team review.*

---

## 1. Executive summary

**Recommendation: position RadEval as a provider of TRL-compatible *reward
functions*, algorithm-agnostic and trainer-agnostic. Ship a small, focused PR
that hardens the existing `make_reward_fn`, closes a handful of real bugs,
and replaces a stub demo with a runnable GRPO example. Defer larger
additions until there is evidence users need them.**

Framing shift from the draft: this is not "GRPO support." It is
"TRL-compatible reward support, with GRPO as the flagship example." Any TRL
trainer that accepts `reward_fn(prompts, completions, references, **kwargs)
-> list[float]` — GRPO, RLOO, and the rest — can consume RadEval rewards
unchanged.

Concretely, the v2.2.0 PR ships:

1. **Hardened `make_reward_fn`**
   - Handles TRL's new reward-fn kwargs (`completion_ids`, `trainer_state`,
     `log_extra`, `log_metric`, `environments`) via `**kwargs`.
   - Handles conversational `completions: list[list[dict]]` by extracting
     `content`.
   - Adds `key=` argument to disambiguate multi-key metrics (fixes the
     F1CheXbert latent bug, see §2.3).
   - Adds `clip_range=(lo, hi)` for rewards that need bounded output.
   - **No `default_on_error`** — crashes should raise. (See §7.)
2. **Scorer caching** — small module-level cache so two reward functions
   that share a scorer don't load the same 1 GB model twice. Safely keyed
   (see §4). Opt-out via `cache=False`.
3. **`pip install RadEval[rl]` extras** — adds `trl` (and pins if needed).
   `datasets` is already a core install dep.
4. **Runnable quickstart example** — `examples/trl_grpo_quickstart.py`, BLEU
   only (+ a one-line comment showing how to swap in BERTScore), small LLM,
   runnable in minutes. Replaces the existing call-shape stub.
5. **Tests** — `tests/test_rewards.py` (unit) + `tests/test_trl_integration.py`
   (smoke, `pytest.importorskip("trl")`).
6. **Rewritten `docs/trl_rewards.md`** — reframes as "TRL-compatible reward
   functions"; GRPO primary example, RLOO mentioned as another online
   option, PPO demoted to a pointer. VLM support is **documentation-only**:
   a short snippet + link to TRL's VLM GRPO example, emphasizing that the
   reward interface is unchanged between LLM and VLM.

**Deferred to a follow-up PR** (contingent on demand):

- `CompositeReward` class. TRL's native `reward_funcs=[...]` +
  `reward_weights=[...]` already covers weighted composition with per-function
  auto-logging. `CompositeReward` would be convenience sugar only, and the
  justifying features (NaN/Inf guard, warn-bands) can be exposed as a
  standalone `validate_rewards()` helper — or not at all until someone asks.
- A committed VLM example file. Stays as a docs snippet + link to
  `trl/examples/scripts/grpo_vlm.py`. Keeps us off the hook for tracking
  TRL's VLM API drift.

**Explicitly out of scope forever**: custom GRPO trainer, VLM adapter,
rollout engine, reward-model training, dataset/collator code. These belong
in downstream code (e.g. `vlm_align`) or in TRL itself.

This is a focused ~300-line PR that fixes real bugs, makes a real RL example
work, and positions RadEval cleanly. That is the announcement.

---

## 2. Findings from each expert

### 2.1 TRL expert

**Current TRL GRPO reward API** (v1.3.x):

- TRL calls `reward_fn(prompts=..., completions=..., completion_ids=...,
  trainer_state=..., log_extra=..., log_metric=..., environments=...,
  **dataset_columns)` and expects `list[float]` back (or `None` entries to
  skip samples).
- **Every dataset column except `prompt`** is forwarded as a kwarg.
- **Conversational datasets** pass `completions` as `list[list[dict]]`
  (OpenAI message format), not strings.
- `async def` reward functions are now first-class.
- PPO lives in `trl.experimental.ppo`; GRPO is the idiomatic online-RL
  choice for 2025/2026. RLOO is a reasonable prompt-only alternative and
  consumes the same reward-function signature.

**Verdict on RadEval's current `make_reward_fn`:**

- Signature is correct; `**kwargs` already absorbs unknown kwargs.
- Missing: conversational-format handling.
- Missing: graceful handling of multi-key metrics.
- `log_extra` / `log_metric` forwarding is *nice-to-have*, not required.

**Composite rewards — TRL already solves this.** Native
`reward_funcs=[f1, f2, f3]` + `reward_weights=[w1, w2, w3]` in `GRPOConfig`
sums internally and auto-logs `reward/{func_name}/mean` and `/std`. This is
the canonical path for weighted composite rewards.

**VLM — TRL's story is solid now.** Native VLM support in `GRPOTrainer`
across Gemma3, LLaVA-Next, Qwen2-VL/2.5-VL, SmolVLM2.
`examples/scripts/grpo_vlm.py` is a working template. RadEval should *not*
ship trainer code or VLM collators.

**Recommended URLs to link from RadEval docs:**

- <https://huggingface.co/docs/trl/main/en/grpo_trainer>
- <https://huggingface.co/docs/trl/main/en/grpo_trainer#using-a-custom-reward-function>
- <https://huggingface.co/docs/trl/main/en/grpo_trainer#vision-language-model-vlm-training>
- <https://github.com/huggingface/trl/blob/main/examples/scripts/grpo_vlm.py>
- <https://huggingface.co/docs/trl/main/en/dataset_formats>
- <https://huggingface.co/docs/trl/main/en/rloo_trainer>

### 2.2 `vlm_align` / HOPPR RL expert

**Does `vlm_align` use RadEval?** No. A grep of the codebase turns up only
*anti-references* to RadEval: the team deliberately chose the standalone
`radgraph` pip package over "RadEval subprocess" to avoid cold-start per
batch, and an internal design doc states plainly that
`"hoppr_vlm/metrics/metrics.py` is a HuggingFace Trainer eval adapter that
returns epoch-level aggregates via a RadEval subprocess — wrong interface
for per-sample GRPO rewards. All three metrics must be implemented locally."

**Central finding:** the team bypassed RadEval because RadEval's reward
surface, until recently, was corpus/epoch-aggregate, not per-sample online.
`make_reward_fn` has since closed most of that gap — our job now is to
finish it.

**Why `vlm_align` has a custom GRPO trainer:**

- The HOPPR VLM takes `images_pixels` / `images_masks`, not `pixel_values`.
- Only the `text_decoder` submodule is LoRA-adapted; vision encoder and
  QFormer are frozen. TRL's model-class assumptions can't cleanly express
  this.
- `vision_cache.py` short-circuits the vision encoder across the 16 rollouts
  of the same study — a big speedup that requires a trainer-level seam.
- The model uses `trust_remote_code` with a monkey-patched
  `_init_vision_encoder`.

All **training-layer** concerns. None belong in RadEval. Modern TRL would
handle a vanilla HF-registered VLM fine.

**Reward-function interface in `vlm_align`:** returns
`(scalars, per_component_diagnostics_dict)`. TRL's plain `list[float]` is
less expressive, but TRL's native `reward_funcs=[...]` + `reward_weights`
with auto-logged per-function means covers the same need for 99% of users.
**No need to port the tuple interface.**

**Generic pieces worth considering for upstream:**

| Piece | Where | Upstream? |
|---|---|---|
| Weighted composite w/ renormalization | `rewards/radcliq_reward.py` | **Deferred**; TRL already sums `reward_funcs`. Reconsider if users ask. |
| NaN/Inf hard-raise (names metric + sample) | `rewards/radcliq_reward.py:138-145` | **Yes**, as a lightweight `validate_rewards()` utility called inside `make_reward_fn`. ~15 LOC. |
| Per-metric warn bands | `rewards/radcliq_reward.py:44-49` | **No** for now. Out-of-range scores are usually a real signal worth surfacing via the standard TRL logging pipeline rather than silenced with a warn-band. Reconsider if users report noise. |
| Group-normalized advantages | `core/advantages.py` | **No** — training-layer; belongs in TRL/user code. |

**HOPPR-specific, leave behind:** `VLMAdapter`, rollout engine, GRPO loss,
collators, vision cache, the specific `radcliq_v0` weights.

### 2.3 RadEval expert

**What's already in place:** `RadEval/rewards.py` exports `make_reward_fn`
(~40 LOC, single-metric), with an API-based-metric warning. Exported from
`RadEval/__init__.py`. Docs at `docs/trl_rewards.md`. The example
(`examples/trl_reward_demo.py`) never instantiates TRL — it prints a string
showing what a TRL call *would* look like. Three tests in `tests/test_e2e.py`
(lines 106-129) exercise `make_reward_fn` against BLEU only; nothing is
verified against actual TRL.

**Bugs / gaps** (verified in code):

- **F1CheXbert latent bug** — confirmed in
  `RadEval/metrics/f1chexbert/adapter.py:12-33`. `metric_keys()` returns
  `["f1chexbert_5_micro_f1", ...]`, but `compute(..., per_sample=True)`
  returns `{"f1chexbert_sample_acc_5": [...], "f1chexbert_sample_acc_all":
  [...]}`. `make_reward_fn("f1chexbert")` currently does
  `result[keys()[0]]` → `KeyError`. Other metrics with mode-dependent keys
  may have the same issue; audit needed during implementation.
- **No scorer caching** (`RadEval/rewards.py:38` instantiates fresh). Two
  reward fns for the same metric = 2× model load.
- **`examples/trl_reward_demo.py` is a stub** that prints a string. Worse
  than no demo.
- **No `tests/test_rewards.py`.**
- **No `[rl]` extras** in `setup.py:75-81` (only `[api]`).
- **Speed table in `docs/trl_rewards.md`** omits `radgraph` and under-warns
  on `green` (7B local LLM; unusable per-step without a spare GPU).

**Radeval should ship**: the reward callable + scorer caching + speed
guidance + a real example. Stay out of dataset loading, prompt formatting,
chat templates, trainer config, reward-model training, KL/ref-model,
distributed launchers.

---

## 3. Recommended scope

**Ship in v2.2.0** (single PR):

1. **`make_reward_fn` hardening**
   - `key=` argument for multi-key metrics; default selection prefers
     `per_sample`-mode keys when they exist.
   - Conversational completion-format handling.
   - `clip_range=(lo, hi)` (optional).
   - Light `validate_rewards()` helper: raises on NaN/Inf naming
     `metric_name` and sample index. Called once per batch inside
     `make_reward_fn`. No warn-bands, no silent defaulting.
   - Expose `.metric_keys` attribute on the returned callable so TRL
     logging can name the reward.
   - `cache=True` (default) / `cache=False` flag.
2. **Scorer caching** — module-level `_SCORER_CACHE`, keyed by a
   **normalized-config tuple**: `(metric_name, tuple(sorted(kwargs.items(),
   key=repr)))`. Only known/safe scorer kwargs are allowed through; unknown
   kwargs bypass cache with a warning. Avoids `frozenset` unhashability
   pitfalls. Public `clear_reward_cache()` for escape hatch.
3. **Published RadCliQ as a reward**: `make_reward_fn("radcliq")` works out
   of the box (RadEval already has a paper-faithful `radcliq` adapter).
   Document this as the canonical "clinical composite" choice. Do **not**
   invent a default composite with our own weights.
4. **`pip install RadEval[rl]` extras**: adds `trl>=X.Y` (floor pinned
   based on the kwargs we rely on).
5. **Runnable quickstart**: `examples/trl_grpo_quickstart.py`, ~40-60 lines,
   small LLM (e.g. `Qwen/Qwen2.5-0.5B`), ~20-row synthetic fixture,
   `GRPOConfig(max_steps=5)`. Uses `make_reward_fn("bleu")` only — cheap,
   no GPU strictly required. A one-line commented swap shows how to change
   to `bertscore` or `radcliq`. Deletes `examples/trl_reward_demo.py`.
6. **Tests** — `tests/test_rewards.py` (unit, parametrized over non-API
   metrics; covers multi-key default-key selection; scorer-cache dedup;
   `clip_range`; NaN raise; `UserWarning` for API metrics) and
   `tests/test_trl_integration.py` (smoke, `pytest.importorskip("trl")`:
   1-step GRPO on a toy model; asserts reward fn was called and
   `trainer.state.global_step == 1`).
7. **Rewritten `docs/trl_rewards.md`**:
   - Positions RadEval as a TRL-compatible *reward provider*, trainer-agnostic.
   - GRPO as the flagship example; RLOO mentioned as another online fit;
     PPO acknowledged with a pointer to `trl.experimental.ppo` and a note
     that GRPO/RLOO are recommended.
   - Documents the native TRL `reward_funcs=[...]` + `reward_weights`
     pattern as the canonical way to combine metrics.
   - Documents `make_reward_fn("radcliq")` for the paper-RadCliQ composite.
   - DPO/ORPO/KTO: one-paragraph note that RadEval metrics can help *curate*
     preference pairs but that is a different workflow, not online reward.
   - VLM section: short snippet + link to
     `trl/examples/scripts/grpo_vlm.py`. The key sentence: *"The reward
     interface is unchanged between LLM and VLM — the reward function
     never sees the image."*
   - Updated speed table: adds `radgraph` (medium), stronger warning on
     `green`, marks all API metrics as "not practical for RL."
8. **README update**: swap the stub reference in the RL section for the new
   quickstart; update the one-paragraph pitch.
9. **Changelog entry** for 2.2.0.

**Deferred to a follow-up PR** (explicit):

- `CompositeReward` class.
- A committed VLM example file.
- `log_extra` / `log_metric` forwarding.
- Async reward-fn support.

The deferral rule: add these when a user asks in an issue. Not before.

**Answering the plan's explicit questions:**

| Question | Answer |
|---|---|
| TRL-compatible reward functions? | **Yes** — already exists; harden. |
| Example GRPO/PPO training scripts? | **One GRPO LLM example.** No VLM script (docs only). No PPO. |
| Wrappers/adapters around existing metrics? | **No new wrappers.** |
| VLM-specific trainer utilities? | **No.** Documentation only. |
| Documentation only? | **No** — docs + bugfix + hardening + example + tests. |
| Boundary? | RadEval ends at `reward_fn(completions=..., **kwargs) -> list[float]`. |
| LLM vs VLM? | Identical reward interface. |
| Incorporate `vlm_align`'s trainer? | **No.** The reward-layer NaN guard is the only thing we port, as a small helper. |
| Concrete examples? | `examples/trl_grpo_quickstart.py`. VLM in docs. |
| API changes? | Additive on `make_reward_fn`. One bugfix (multi-key default) behaviorally changes F1CheXbert output. |
| Announcement-worthy? | Yes — see §8. |

---

## 4. Proposed API / code structure

```python
from RadEval.rewards import (
    make_reward_fn,          # existing; hardened
    clear_reward_cache,      # new
    validate_rewards,        # new, small utility
)
```

```python
# Minimal single-metric reward (TRL GRPO or RLOO)
from RadEval.rewards import make_reward_fn
from trl import GRPOTrainer, GRPOConfig

reward = make_reward_fn(
    "f1chexbert",
    key="f1chexbert_sample_acc_5",        # NEW: multi-key disambiguation
    clip_range=(0.0, 1.0),                # NEW, optional
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
# Paper-RadCliQ as a single reward — uses the existing radcliq adapter
trainer = GRPOTrainer(
    reward_funcs=[make_reward_fn("radcliq")],
    ...
)
```

**Scorer-cache key design (to address the `frozenset` unhashability
concern):**

```python
def _cache_key(metric: str, kwargs: dict) -> tuple:
    # Only allow scalar/string/None/tuple kwargs through the cache.
    # Unknown/unhashable kwargs → skip cache (load fresh, warn).
    normalized = []
    for k, v in sorted(kwargs.items()):
        if isinstance(v, (str, int, float, bool, type(None), tuple)):
            normalized.append((k, v))
        else:
            return None  # signal: do not cache
    return (metric, tuple(normalized))
```

Simple, correct, no surprising hash failures. ~15 LOC total.

---

## 5. Example workflows to include

### 5.1 `examples/trl_grpo_quickstart.py` (committed, BLEU, runnable)

- ~40-60 lines.
- Model: `Qwen/Qwen2.5-0.5B`.
- Dataset: 20-row synthetic fixture at
  `tests/fixtures/synthetic_reports.json` (reused by the integration test).
- `make_reward_fn("bleu")`. BLEU is the deliberate choice: zero GPU
  dependencies in the reward path, near-instant, no model downloads.
- A single commented line: `# Swap in: make_reward_fn("bertscore") — heavier
  but richer signal. See docs/trl_rewards.md for the speed/quality table.`
- `GRPOConfig(max_steps=5, per_device_train_batch_size=2,
  num_generations=2)`.
- Prints reward trajectory.
- Goal: **copy-paste-and-it-works in under 10 minutes** on a laptop GPU
  (CPU-tolerant). This is what validates the announcement.

### 5.2 VLM — docs only

In `docs/trl_rewards.md`, a short section titled "Using RadEval rewards
with a VLM":

```python
# Identical reward setup — the reward function never sees the image.
from RadEval.rewards import make_reward_fn

reward = make_reward_fn("bertscore")
# Plug into trl's VLM example:
# https://github.com/huggingface/trl/blob/main/examples/scripts/grpo_vlm.py
```

Plus a paragraph explaining that TRL handles the image-conditioned rollout
end-to-end; our reward interface is unchanged.

### 5.3 Docs walk-through: choosing a metric / combining metrics

In `docs/trl_rewards.md`: a short decision guide.

- Single fast metric → `make_reward_fn("bleu")` or `"rouge"`.
- Single clinical metric → `make_reward_fn("f1chexbert")` or `"radgraph"`.
- Paper-RadCliQ → `make_reward_fn("radcliq")`.
- Custom mix of metrics → `reward_funcs=[...]` + `reward_weights=[...]` in
  `GRPOConfig`. (Show two-reward example with auto-logging.)
- Why no `CompositeReward` helper: TRL already does this natively and logs
  per-function means/stds. If you need per-sample diagnostics beyond that,
  open a GitHub issue describing the use case.

---

## 6. Implementation plan

Single PR, one focused week.

1. **Scoping commit** — `extras_require={"rl": ["trl>=X.Y"]}` in
   `setup.py`; create `tests/fixtures/synthetic_reports.json`.
2. **`rewards.py` rewrite** — `make_reward_fn` gains `key`, `clip_range`,
   conversational-format handling; wraps each batch result through
   `validate_rewards`. Module-level scorer cache with the normalized key
   from §4. `clear_reward_cache()` exported. Multi-key default-selection
   logic (prefer `per_sample` keys when available). Audit all metric
   adapters for mode-dependent keys and either fix or explicitly document
   the right `key=` to pass.
3. **Tests** — `tests/test_rewards.py` + `tests/test_trl_integration.py`.
   Extend `tests/conftest.py` fixtures as needed.
4. **Examples** — `examples/trl_grpo_quickstart.py` replaces
   `examples/trl_reward_demo.py`.
5. **Docs rewrite** — `docs/trl_rewards.md` restructured per §5; README §RL
   updated; short VLM snippet + link.
6. **Changelog entry** — `docs/changelog/…` for 2.2.0: RL hardening,
   `[rl]` extras, runnable GRPO example, F1CheXbert `per_sample` key
   bugfix.

**Rollout:**
- Day 1: scoping, setup.py, fixtures.
- Day 2-3: `rewards.py` hardening + unit tests. Dogfood on a real GRPO run.
- Day 4: integration test + example + docs.
- Day 5: review + changelog + cut 2.2.0.

**Validation gate before tagging:** run `examples/trl_grpo_quickstart.py`
end-to-end on a real machine (laptop GPU). Confirm reward trajectory is
non-degenerate.

---

## 7. Risks / tradeoffs

- **TRL API churn.** Reward-fn kwargs have changed in the last 12 months.
  Mitigation: `**kwargs`-absorb pattern (already in place); pin a known-good
  floor in `[rl]` extras; the integration test catches regressions on each
  TRL upgrade.
- **Scorer-cache memory.** 10 cached metrics × ~1 GB each is real. Mitigation:
  only cache scorers users explicitly build; `cache=False` flag;
  `clear_reward_cache()` escape hatch; document expected GB footprint in
  `docs/trl_rewards.md`.
- **No `default_on_error`.** Intentional. Silent zero-rewards hide
  tokenizer/vocab/edge-case bugs that otherwise surface as an immediate
  crash. RL failures should fail loud. If users push back, reconsider as an
  *opt-in* parameter — never a default.
- **F1CheXbert key fix is behavior-changing.** Current
  `make_reward_fn("f1chexbert")` raises `KeyError`, so "change" here means
  "goes from broken to working." Annotated as bugfix in changelog; no
  SemVer break.
- **Published-RadCliQ in `make_reward_fn`.** The existing `radcliq`
  adapter may be slower per-sample than users expect for online RL.
  Mitigation: document its cost in the speed table; recommend it primarily
  as an *evaluation* reward for final fine-tuning rather than a *training*
  reward for raw exploration.
- **Deferring `CompositeReward` could mean users re-implement it.** Real
  but acceptable. If a user opens an issue with a concrete composite they
  can't build with TRL's native `reward_weights`, we add it in a follow-up
  PR with a clear justification. Until then, YAGNI.
- **VLM-docs-only could become stale.** Lower risk than a committed VLM
  script whose `transformers`/`trl` pins drift. Mitigation: the snippet
  links to TRL's own canonical example and doesn't duplicate it.
- **No async rewards.** Fine until someone has a slow-but-parallelizable
  reward (e.g., a local LLM judge). TRL supports `async def`; users can
  write `async def` reward functions today without any RadEval change.

---

## 8. Announcement positioning

**Headline:** *"RadEval 2.2.0: radiology metrics, now TRL-compatible
rewards."*

**One-paragraph pitch (for blog / tweet / README banner):**

> RadEval 2.2.0 makes every one of its 16+ radiology metrics — BLEU, ROUGE,
> BERTScore, RadGraph, F1CheXbert, RadCliQ, and more — usable as a reward
> function for reinforcement learning with Hugging Face TRL. A single
> `make_reward_fn("radgraph")` call gives you a drop-in callable for
> `GRPOTrainer`, `RLOOTrainer`, or any trainer that consumes a
> reward function. Ships with a runnable GRPO quickstart, works
> identically for LLM and VLM report generation, and hardens the existing
> interface (scorer caching, conversational-completion format, multi-key
> bugfix).

**What makes this announcement-worthy without being maintenance debt:**

- Small, well-defined contribution (one module + one example + docs) — not
  a new "RL framework."
- Closes a *real* gap the HOPPR team hit last year, where they
  re-implemented BERTScore / SembScore / RadGraph locally because RadEval's
  surface was epoch-aggregate.
- Validates via a runnable integration test on a toy model — "works with
  TRL GRPO" is verified, not asserted.
- Plays the supporting role: TRL is the training framework, Transformers is
  the model framework, RadEval is the metrics framework. No empire-building.
- Opens natural paths to future one-paragraph announcements ("VLM RL case
  study", "preference-pair curation for DPO") without locking us into
  maintaining a trainer.

**Measure of success**: a radiology researcher wanting to GRPO-tune a
report-generation model can `pip install RadEval[rl]`, copy
`examples/trl_grpo_quickstart.py`, swap in their dataset, and be training in
under 15 minutes — with confidence that what they see in their logs is
exactly the metric the paper published.

---

## 9. Design decisions & alternatives considered

| Decision | Alternative rejected | Reason |
|---|---|---|
| Defer `CompositeReward` to follow-up PR | Ship it in 2.2.0 per original draft | TRL's native `reward_weights` covers the canonical path; `CompositeReward` is convenience sugar whose justifying features (NaN/warn-bands) can be delivered as a 15-LOC utility. YAGNI until requested. |
| No `default_on_error` | `default_on_error=0.0` per original draft | Silent zero-reward hides real bugs in RL settings where the reward is the only error signal. Fail loud. |
| VLM example docs-only, not committed | Committed `examples/trl_grpo_vlm.py` per original draft | Committed VLM scripts drift with TRL/transformers pins faster than we want to maintain. A docs link to TRL's canonical VLM example is equally useful and zero-maintenance. |
| BLEU-only quickstart | BLEU + BERTScore dual-reward per original draft | First-run experience must be frictionless. BLEU is CPU-instant; BERTScore pulls a model. A one-line commented swap shows the upgrade path. |
| Cache key: sorted-normalized tuple of scalar kwargs | `frozenset(kwargs.items())` per original draft | `frozenset` breaks on unhashable kwargs (e.g., a list option). Normalized tuple key + skip-cache-if-unknown-kwarg is safer. |
| Support paper-`radcliq`, not `radcliq_v0` | Bundle a branded composite with HOPPR weights | The HOPPR `radcliq_v0` weights are an internal simplification of the paper; shipping them under RadEval's name would mislead. RadEval's existing `radcliq` adapter is paper-faithful. |
| Framing: "TRL-compatible reward support" | Framing: "GRPO support" | Algorithm-agnostic framing matches the actual API (reward-fn callable) and future-proofs against TRL evolving RLOO, experimental trainers, etc. |
| Mention RLOO; demote PPO | Support PPO as a co-equal | PPO in TRL requires a full reward model (nn.Module), not a reward function — RadEval metrics don't fit. Not worth supporting. |
| `validate_rewards()` helper (NaN/Inf raise only) | Full warn-band system per original draft | Warn-bands are low-value noise; NaN/Inf is high-value signal. Keep the useful part. |

---

## 10. Facts vs assumptions

**Facts** (verified in code):

- `RadEval/rewards.py:38` does `scorer = cls(**metric_kwargs)` per call — no
  caching.
- `RadEval/metrics/f1chexbert/adapter.py:12-33` returns different keys in
  `per_sample=True` mode than the ones `metric_keys()` advertises.
- `setup.py:75-81` has only `[api]` extras; no `[rl]`.
- `tests/test_e2e.py:106-129` tests `make_reward_fn` against BLEU only.
- `examples/trl_reward_demo.py` never imports or instantiates TRL.
- `vlm_align` does not import RadEval; it implements rewards locally.

**Assumptions** (to validate during implementation):

- Current TRL (`>=1.3`) still passes `completions` as `list[str]` for
  standard datasets and `list[list[dict]]` for conversational — verify in
  the integration test against the exact TRL version we pin.
- All non-F1CheXbert metrics have `metric_keys()[0]` matching their
  `per_sample=True` output. Audit during implementation; fix any other
  offenders or document the right `key=` to pass.
- `Qwen/Qwen2.5-0.5B` runs in `GRPOConfig(max_steps=5)` in under 10 minutes
  on a modest laptop GPU. Validate before tagging release.
- `make_reward_fn("radcliq")` is fast enough to use as an online reward.
  If not, downgrade to "recommended for eval, slow for training."

**Hypotheses** (not validated, not acted on):

- Users will want `CompositeReward` and/or warn-bands eventually.
- Users will want async reward functions.
- Users will want VLM-specific tooling beyond what TRL ships.

All three are explicitly deferred; we'll act when there is evidence.
