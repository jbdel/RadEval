from RadEval import RadEval
import pytest

def test_do_details():
    """do_details returns flat keys with extra aggregate scores (no per-sample, no annotations)."""
    refs = [
        "Increased mild pulmonary edema and left basal atelectasis.",
    ]

    hyps = [
        "No pleural effusions or pneumothoraces.",
    ]

    evaluator = RadEval(metrics=["radgraph", "bleu", "rouge",
                                 "bertscore", "srrbert",
                                 "f1chexbert", "temporal",
                                 "ratescore", "radcliq",
                                 "radeval_bertscore"],
                      detailed=True)

    results = evaluator(refs=refs, hyps=hyps)

    expected_scalar_keys = {
        "radgraph_simple", "radgraph_partial", "radgraph_complete",
        "bleu", "bleu_1", "bleu_2", "bleu_3",
        "bertscore",
        "rouge1", "rouge2", "rougeL",
        "srrbert_weighted_f1", "srrbert_weighted_precision", "srrbert_weighted_recall",
        "f1chexbert_5_micro_f1", "f1chexbert_all_micro_f1",
        "f1chexbert_5_macro_f1", "f1chexbert_all_macro_f1",
        "f1chexbert_5_weighted_f1", "f1chexbert_all_weighted_f1",
        "ratescore",
        "radcliq_v1",
        "temporal_f1",
        "radeval_bertscore",
    }
    expected_dict_keys = {
        "f1chexbert_label_scores_f1",
        "srrbert_label_scores",
    }

    actual_keys = set(results.keys())
    assert actual_keys == expected_scalar_keys | expected_dict_keys, (
        f"Key mismatch.\n  Missing: {(expected_scalar_keys | expected_dict_keys) - actual_keys}\n"
        f"  Extra: {actual_keys - expected_scalar_keys - expected_dict_keys}")

    for key in expected_scalar_keys:
        assert isinstance(results[key], (int, float)), (
            f"{key} should be a scalar, got {type(results[key])}")

    for key in expected_dict_keys:
        assert isinstance(results[key], dict), (
            f"{key} should be a dict, got {type(results[key])}")

    for value in results.values():
        assert not isinstance(value, list), (
            "do_details should not contain any list values (per-sample data)")

    assert results["bleu_1"] >= results["bleu_2"] >= results["bleu_3"] >= results["bleu"]

    label_f1 = results["f1chexbert_label_scores_f1"]
    assert "f1chexbert_5" in label_f1
    assert "f1chexbert_all" in label_f1
    assert isinstance(label_f1["f1chexbert_5"], dict)
    for label, score in label_f1["f1chexbert_5"].items():
        assert isinstance(score, (int, float))

    srr_labels = results["srrbert_label_scores"]
    assert isinstance(srr_labels, dict)
    for label, info in srr_labels.items():
        for field in ("f1-score", "precision", "recall", "support"):
            assert field in info, f"Missing '{field}' in srrbert label '{label}'"


def test_do_per_sample():
    """Verify do_per_sample returns flat keys with per-sample lists."""
    refs = [
        "Increased mild pulmonary edema and left basal atelectasis.",
        "No acute cardiopulmonary process.",
    ]

    hyps = [
        "No pleural effusions or pneumothoraces.",
        "No acute cardiopulmonary process.",
    ]

    evaluator = RadEval(
        metrics=["radgraph", "bleu", "rouge",
                 "bertscore", "srrbert",
                 "f1chexbert", "temporal",
                 "ratescore", "radcliq",
                 "radeval_bertscore"],
        per_sample=True,
    )

    results = evaluator(refs=refs, hyps=hyps)
    n = len(refs)

    expected_list_keys = {
        "radgraph_simple", "radgraph_partial", "radgraph_complete",
        "bleu",
        "bertscore",
        "rouge1", "rouge2", "rougeL",
        "srrbert_weighted_f1", "srrbert_weighted_precision", "srrbert_weighted_recall",
        "f1chexbert_sample_acc_5", "f1chexbert_sample_acc_all",
        "ratescore",
        "radcliq_v1",
        "temporal_f1",
        "radeval_bertscore",
    }

    for key in expected_list_keys:
        assert key in results, f"Missing key: {key}"
        assert isinstance(results[key], list), f"{key} should be a list, got {type(results[key])}"
        assert len(results[key]) == n, f"{key} length {len(results[key])} != {n}"

    assert set(results.keys()) == expected_list_keys, (
        f"Unexpected keys: {set(results.keys()) - expected_list_keys}")

    for key in ("radgraph_simple", "bertscore", "ratescore", "radeval_bertscore"):
        for val in results[key]:
            assert isinstance(val, (int, float)), f"{key} contains non-numeric: {val}"

    identical_idx = 1
    assert results["bleu"][identical_idx] > results["bleu"][0]


if __name__ == "__main__":
    test_do_details()
    test_do_per_sample()
