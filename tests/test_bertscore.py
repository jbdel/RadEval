from radeval import RadEval
import math


def read_reports(filepath):
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]


refs = read_reports("tests/utterances/preds_chexpert_impression.txt")
hyps = read_reports("tests/utterances/refs_chexpert_impression.txt")

expected_score = 0.5305
epsilon = 1e-5


def test_bertscore():

    evaluator = RadEval(
        metrics=["bertscore"],
    )

    results = evaluator(refs=refs, hyps=hyps)

    assert math.isclose(results["bertscore"], expected_score,
                        rel_tol=epsilon), f"Mismatch in scores: Actual {results['bertscore']}, Expected {expected_score}"


if __name__ == "__main__":
    test_bertscore()
