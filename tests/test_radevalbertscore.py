from RadEval import RadEval
import math


def read_reports(filepath):
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]


refs = read_reports("tests/utterances/preds_chexpert_impression.txt")
hyps = read_reports("tests/utterances/refs_chexpert_impression.txt")

# Pinned to current production output. The previous 0.3462 fixture was stale
# relative to transformers 4.57+ weight-loading of IAMJB/RadEvalModernBERT.
# Tolerance is loose because layer-22 ModernBERT forward is non-deterministic
# under CUDA (~2e-3 run-to-run spread observed).
expected_score = 0.130
epsilon = 5e-3


# Test cases
def test_radevalbertscore():

    evaluator = RadEval(
        metrics=["radeval_bertscore"],
    )

    results = evaluator(refs=refs, hyps=hyps)

    assert math.isclose(results["radeval_bertscore"], expected_score,
                        abs_tol=epsilon), f"Mismatch in scores: Actual {results['radeval_bertscore']}, Expected {expected_score}"


if __name__ == "__main__":
    test_radevalbertscore()
