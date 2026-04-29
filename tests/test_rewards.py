"""Unit tests for the RadEval TRL-compatible reward surface.

Covers `make_reward_fn` and `validate_rewards` end-to-end *without*
requiring TRL installed. The integration smoke test against an actual
GRPOTrainer lives in `tests/test_trl_integration.py`.
"""
import math
import warnings

import pytest

from radeval.rewards import make_reward_fn, validate_rewards


# ---------- Round-trip tests on lightweight metrics ----------


def test_bleu_round_trip():
    reward = make_reward_fn("bleu")
    out = reward(completions=["hello world"], ground_truth=["hello world"])
    assert isinstance(out, list) and len(out) == 1
    assert isinstance(out[0], float)


def test_rouge_round_trip():
    # ROUGE is multi-key per-sample (rouge1, rouge2, rougeL) — pick one.
    reward = make_reward_fn("rouge", key="rouge1")
    out = reward(
        completions=["no pleural effusion"],
        ground_truth=["no pleural effusions"],
    )
    assert isinstance(out, list) and len(out) == 1


def test_rouge_without_key_raises():
    reward = make_reward_fn("rouge")
    with pytest.raises(ValueError, match="multiple per-sample keys"):
        reward(
            completions=["no pleural effusion"],
            ground_truth=["no pleural effusions"],
        )


def test_bertscore_round_trip():
    reward = make_reward_fn("bertscore")
    out = reward(
        completions=["normal heart size"],
        ground_truth=["no cardiomegaly"],
    )
    assert isinstance(out, list) and len(out) == 1


# ---------- Multi-key metrics (F1CheXbert regression) ----------


def test_f1chexbert_without_key_raises():
    """The F1CheXbert bug: metric_keys()[0] doesn't match per_sample keys."""
    reward = make_reward_fn("f1chexbert")
    with pytest.raises(ValueError, match="multiple per-sample keys"):
        reward(
            completions=["No pleural effusions."],
            ground_truth=["No pleural effusion. Normal heart size."],
        )


def test_f1chexbert_with_key_works():
    reward = make_reward_fn("f1chexbert", key="f1chexbert_sample_acc_5")
    out = reward(
        completions=["No pleural effusions."],
        ground_truth=["No pleural effusion. Normal heart size."],
    )
    assert isinstance(out, list) and len(out) == 1
    assert isinstance(out[0], float)


def test_f1chexbert_invalid_key_raises():
    reward = make_reward_fn("f1chexbert", key="definitely_not_a_key")
    with pytest.raises(KeyError, match="not in per-sample output keys"):
        reward(
            completions=["No pleural effusions."],
            ground_truth=["No pleural effusion. Normal heart size."],
        )


# ---------- Conversational completions ----------


def test_conversational_list_of_dict_extracts_content():
    reward = make_reward_fn("bleu")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = reward(
            completions=[[{"role": "user", "content": "q"},
                          {"role": "assistant", "content": "hello world"}]],
            ground_truth=["hello world"],
        )
    assert len(out) == 1
    assert any(issubclass(x.category, UserWarning) for x in w)
    # Should score the same as passing the extracted string directly.
    direct = reward(completions=["hello world"], ground_truth=["hello world"])
    assert abs(out[0] - direct[0]) < 1e-12


def test_conversational_missing_keys_raises_typeerror():
    reward = make_reward_fn("bleu")
    with pytest.raises(TypeError, match="lack expected 'role'/'content'"):
        reward(
            completions=[[{"foo": "bar"}]],
            ground_truth=["hello world"],
        )


def test_unrecognized_shape_raises_typeerror():
    reward = make_reward_fn("bleu")
    with pytest.raises(TypeError, match="unsupported completions shape"):
        reward(completions=[[1, 2, 3]], ground_truth=["hello world"])


# ---------- score_transform + clipping idiom ----------


def test_score_transform_composes():
    raw = make_reward_fn("bleu")
    scaled = make_reward_fn("bleu", score_transform=lambda x: x * 10)
    refs = ["hello world"]
    assert abs(
        scaled(completions=refs, ground_truth=refs)[0]
        - raw(completions=refs, ground_truth=refs)[0] * 10
    ) < 1e-9


def test_score_transform_clipping_idiom():
    """Plan explicitly recommends `score_transform=lambda x: max(lo, min(hi, x))`
    as the way to clip, instead of a dedicated `clip_range` argument."""
    # Force the score through both clamp branches by picking
    # transforms that span the input range.
    clip_hi = make_reward_fn(
        "bleu",
        score_transform=lambda x: max(5.0, min(10.0, x)),
    )
    # Any BLEU value is < 5.0, so clamp-low kicks in.
    out = clip_hi(completions=["a"], ground_truth=["a"])
    assert out[0] == pytest.approx(5.0)

    clip_lo = make_reward_fn(
        "bleu",
        score_transform=lambda x: max(-10.0, min(-5.0, x)),
    )
    # Any BLEU value is > -5.0, so clamp-high kicks in.
    out = clip_lo(completions=["a"], ground_truth=["a"])
    assert out[0] == pytest.approx(-5.0)


# ---------- validate_rewards ----------


def test_validate_rewards_passes_floats_through():
    assert validate_rewards([0.0, 0.5, 1.0], "m") == [0.0, 0.5, 1.0]


def test_validate_rewards_preserves_none():
    """TRL uses None to skip samples in multi-task routing."""
    assert validate_rewards([None, 0.5, None, 1.0], "m") == [None, 0.5, None, 1.0]


def test_validate_rewards_nan_raises():
    with pytest.raises(ValueError, match="non-finite reward at sample index 0"):
        validate_rewards([float("nan")], "bleu")


def test_validate_rewards_inf_raises():
    with pytest.raises(ValueError, match="non-finite reward at sample index 1"):
        validate_rewards([0.5, float("inf")], "bleu")


def test_validate_rewards_numpy_nan():
    np = pytest.importorskip("numpy")
    with pytest.raises(ValueError, match="non-finite"):
        validate_rewards([np.nan], "m")


def test_validate_rewards_numpy_scalar():
    np = pytest.importorskip("numpy")
    out = validate_rewards([np.float32(0.25), np.float64(0.75)], "m")
    assert out == [pytest.approx(0.25), 0.75]


def test_validate_rewards_torch_0d_tensor():
    torch = pytest.importorskip("torch")
    out = validate_rewards([torch.tensor(0.5)], "m")
    assert out == [0.5]


def test_validate_rewards_torch_1element_1d_tensor():
    """Some metric adapters accidentally return shape [1] tensors.
    .item() normalizes them before the float() cast."""
    torch = pytest.importorskip("torch")
    out = validate_rewards([torch.tensor([0.75])], "m")
    assert out == [0.75]


# ---------- score_transform + None pass-through interaction ----------


def test_score_transform_skips_none():
    """score_transform must not be called on None entries; the reward fn
    preserves the sample-skip convention end-to-end."""
    def boom(_):
        raise AssertionError("score_transform called on None")

    out = validate_rewards(
        [None if v is None else boom(v) for v in [None, None]],
        "m",
    )
    assert out == [None, None]


# ---------- Absorb extra TRL-forwarded kwargs ----------


def test_absorbs_trl_kwargs_and_extra_columns():
    """TRL forwards every dataset column (except `prompt`) as a kwarg, plus
    its own bookkeeping kwargs. The reward fn must ignore unknown kwargs."""
    reward = make_reward_fn("bleu")
    out = reward(
        completions=["hello world"],
        ground_truth=["hello world"],
        # TRL's own kwargs:
        prompts=["a prompt"],
        completion_ids=[[1, 2, 3]],
        trainer_state=None,
        log_extra=lambda *a, **kw: None,
        log_metric=lambda *a, **kw: None,
        environments=None,
        # Unrelated dataset columns forwarded by TRL:
        patient_id="x",
        study_date="2026-01-01",
    )
    assert isinstance(out, list) and len(out) == 1


def test_missing_reference_column_raises_keyerror():
    """If the dataset column name is wrong, fail loudly — not silently."""
    reward = make_reward_fn("bleu", reference_column="refs_col")
    with pytest.raises(KeyError, match="expected reference column 'refs_col'"):
        reward(completions=["hello"], ground_truth=["hello"])


def test_custom_reference_column_works():
    reward = make_reward_fn("bleu", reference_column="my_refs")
    out = reward(completions=["hello world"], my_refs=["hello world"])
    assert len(out) == 1


# ---------- API-based metric warning ----------


def test_api_metric_emits_warning():
    with pytest.warns(UserWarning, match="API calls per sample"):
        make_reward_fn("crimson", provider="openai", model_name="gpt-4o-mini")


# ---------- Empty inputs ----------


def test_empty_completions():
    reward = make_reward_fn("bleu")
    out = reward(completions=[], ground_truth=[])
    assert out == []
