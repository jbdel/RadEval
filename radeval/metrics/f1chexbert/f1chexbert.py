"""F1CheXbert: CheXbert-based radiology report evaluator (Stanford CheXpert labels)."""
from __future__ import annotations

from typing import Union

import torch
from transformers import AutoConfig, BertModel, BertTokenizer
from huggingface_hub import hf_hub_download
from appdirs import user_cache_dir

from RadEval.metrics._chexbert_base import BaseBertLabeler, BaseCheXbertEvaluator

CACHE_DIR = user_cache_dir("chexbert")


class _CheXbertLabeler(BaseBertLabeler):
    """Standard CheXbert backbone (bert-base-uncased, 13+1 heads)."""

    def __init__(self, *, device):
        config = AutoConfig.from_pretrained("bert-base-uncased")
        backbone = BertModel(config)

        ckpt_path = hf_hub_download(
            repo_id="StanfordAIMI/RRG_scorers",
            filename="chexbert.pth",
            cache_dir=CACHE_DIR,
        )
        raw = torch.load(ckpt_path, map_location="cpu")["model_state_dict"]
        state = {k.replace("module.", ""): v for k, v in raw.items()}

        super().__init__(
            device=device,
            backbone=backbone,
            num_4way_heads=13,
            state_dict=state,
        )


class F1CheXbert(BaseCheXbertEvaluator):
    """CheXbert label extraction + F1 evaluation (14 CheXpert conditions)."""

    CONDITION_NAMES = [
        "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
        "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
        "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
        "Support Devices",
    ]
    NO_FINDING = "No Finding"
    TOP5 = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]

    def __init__(
        self,
        *,
        refs_filename: str | None = None,
        hyps_filename: str | None = None,
        device: Union[str, torch.device] = "cuda",
        batch_size: int = 64,
    ):
        super().__init__()
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = _CheXbertLabeler(device=device).eval()
        self._init_evaluator(
            model=model, tokenizer=tokenizer, device=device,
            batch_size=batch_size, refs_filename=refs_filename,
            hyps_filename=hyps_filename,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["rrg", "classification"], default="rrg")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = F1CheXbert(device=device, batch_size=args.batch_size)

    probes = [
        ("Mild cardiomegaly is present.", "Cardiomegaly"),
        ("Patchy right lower lobe consolidation.", "Consolidation"),
        ("Small left pleural effusion is seen.", "Pleural Effusion"),
    ]
    sentences = [s for s, _ in probes]
    expected = [d for _, d in probes]
    preds = scorer.get_labels(sentences, mode=args.mode)

    for idx, (sentence, exp, row) in enumerate(zip(sentences, expected, preds), 1):
        present = [name for name, value in zip(scorer.TARGET_NAMES, row) if value == 1]
        print(f"[{idx}] {sentence}")
        print(f"  expected: {exp}  received: {present}  match: {exp in present}")
