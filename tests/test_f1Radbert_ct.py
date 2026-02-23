import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def test_f1Radbert_ct_exact_outputs():
    repo_id = "IAMJB/RadBERT-CT"
    tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id)
    model.eval()

    texts = [
        "No acute cardiopulmonary abnormality.",
        "Right lower lobe opacity, suspicious for pneumonia. Pleural effusion present.",
    ]

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits)
        pred_mask = probs > 0.5

    expected_logits = torch.tensor(
        [
            [
                -7.0824,
                -8.3267,
                -8.4071,
                -8.7052,
                -7.7598,
                -7.3826,
                -8.6580,
                -7.1315,
                -8.0535,
                -5.8760,
                -6.7699,
                -6.4723,
                -8.1472,
                -7.7759,
                -8.0363,
                -6.9240,
                -8.0222,
                -7.3431,
            ],
            [
                -3.4248,
                -6.8445,
                -4.1796,
                -4.0292,
                -6.3926,
                -6.6850,
                -5.9953,
                -6.0609,
                -3.6582,
                -4.7553,
                0.2473,
                -6.0450,
                7.5655,
                -4.9244,
                -4.6517,
                -2.8377,
                -6.6161,
                -4.6961,
            ],
        ],
        dtype=logits.dtype,
    )

    assert logits.shape == torch.Size([2, 18])
    assert torch.allclose(logits, expected_logits, atol=1e-4)
    assert pred_mask.tolist() == [
        [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False],
    ]
    assert [[i for i, on in enumerate(row) if on] for row in pred_mask.tolist()] == [[], [10, 12]]

