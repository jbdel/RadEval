"""HopprF1CheXbertCT: per-condition CT report evaluator (ModernBERT).

Classifies each of 16 CT conditions independently per report by formatting
a prompt for each condition and running a 4-way classifier. The 4 classes
(definitely absent / not reported / uncertain / definitely present) are
collapsed to binary (negative = 0,1; positive = 2,3) for F1 evaluation.
"""
from __future__ import annotations

import os
import warnings
from collections import OrderedDict
from typing import List, Sequence, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_DEFAULT_CKPT = (
    "/nfs/cluster/hoppr_vlm_ressources/radeval_checkpoints/"
    "hoppr_f1chexbert_ct_modernbert"
)

CONDITION_NAMES = OrderedDict([
    ("acute_aortic_injury", "Acute aortic injury"),
    ("acute_pulmonary_embolism", "Acute pulmonary embolism"),
    ("acute_rib_fracture", "Acute rib fracture"),
    ("acute_vertebral_fracture", "Acute vertebral fracture"),
    ("aortic_aneurysm", "Aortic aneurysm"),
    ("aortic_atherosclerosis", "Aortic atherosclerosis"),
    ("aortic_valve_calcification", "Aortic valve calcification"),
    ("cardiomegaly", "Cardiomegaly"),
    ("chronic_pulmonary_embolism", "Chronic pulmonary embolism"),
    ("chronic_vertebral_compression_fracture", "Chronic vertebral compression fracture"),
    ("copd_emphysema", "COPD emphysema"),
    ("lung_nodule_or_mass", "Lung nodule or mass"),
    ("pleural_effusion", "Pleural effusion"),
    ("pneumothorax", "Pneumothorax"),
    ("air_space_opacity", "Air space opacity"),
    ("prior_myocardial_infarction", "Prior myocardial infarction"),
])

ID_TO_LABEL = {0: "definitely absent", 1: "not reported",
               2: "uncertain", 3: "definitely present"}


def _format_prompt(condition_pretty: str, findings: str) -> str:
    return (
        f"Condition: {condition_pretty}\n"
        f"Report findings:\n{findings}\n"
        "Classify this condition as one of: "
        "definitely absent, not reported, uncertain, definitely present."
    )


class HopprF1CheXbertCT:
    """Per-condition CT finding classifier for report evaluation.

    For each report, the model is queried once per condition (16 passes).
    4-way predictions are collapsed to binary for computing F1.
    """

    LABELS = list(CONDITION_NAMES.keys())
    LABELS_PRETTY = list(CONDITION_NAMES.values())
    NO_FINDING = "no_finding"

    def __init__(
        self,
        checkpoint_dir: str = _DEFAULT_CKPT,
        device: Union[str, torch.device] = "cuda",
        batch_size: int = 64,
        max_length: int = 1536,
    ):
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(
                f"HopprF1CheXbertCT checkpoint not found: {checkpoint_dir}")

        self.batch_size = batch_size
        self.max_length = max_length
        self.device = torch.device(device) if isinstance(device, str) else device
        if self.device.type == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA requested but unavailable; falling back to CPU.")
            self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_dir,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _predict_label_matrix(
        self, reports: Sequence[str], on_batch_done=None,
    ) -> np.ndarray:
        """Return binary label matrix of shape (N, 16+1).

        For each report, runs 16 condition prompts through the model.
        4-way logits are argmaxed, then collapsed to binary:
          classes 0-1 (absent/not reported) -> 0
          classes 2-3 (uncertain/present)   -> 1
        A 17th "no_finding" column is appended (1 when all others are 0).
        """
        all_prompts = []
        for report in reports:
            for pretty_name in self.LABELS_PRETTY:
                all_prompts.append(_format_prompt(pretty_name, report))

        n_conditions = len(self.LABELS)
        all_preds = []

        for start in range(0, len(all_prompts), self.batch_size):
            batch = all_prompts[start:start + self.batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits
            pred_ids = logits.argmax(dim=-1).cpu()
            binary = (pred_ids >= 2).int()
            all_preds.append(binary)
            if on_batch_done:
                on_batch_done()

        all_preds = torch.cat(all_preds, dim=0)
        matrix = all_preds.view(len(reports), n_conditions)

        no_finding = (~matrix.any(dim=1)).unsqueeze(1).int()
        full = torch.cat([matrix, no_finding], dim=1)
        return full.numpy()

    def __call__(self, hyps: List[str], refs: List[str], on_batch_done=None):
        return self.forward(hyps=hyps, refs=refs, on_batch_done=on_batch_done)

    def forward(
        self, hyps: List[str], refs: List[str], on_batch_done=None,
    ) -> Tuple[float, List[float], dict]:
        if not isinstance(hyps, list) or not isinstance(refs, list):
            raise TypeError("hyps and refs must be of type list")
        if len(hyps) != len(refs):
            raise ValueError("hyps and refs lists don't have the same size")
        if len(hyps) == 0:
            return 0.0, [], {}

        y_pred = self._predict_label_matrix(hyps, on_batch_done=on_batch_done)
        y_true = self._predict_label_matrix(refs, on_batch_done=on_batch_done)

        accuracy = float(accuracy_score(y_true, y_pred))
        per_sample_accuracy = (y_true == y_pred).all(axis=1).astype(float).tolist()
        report = classification_report(
            y_true, y_pred,
            target_names=self.LABELS + [self.NO_FINDING],
            output_dict=True,
            zero_division=0,
        )
        return accuracy, per_sample_accuracy, report
