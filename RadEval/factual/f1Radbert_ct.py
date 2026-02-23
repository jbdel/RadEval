from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import numpy as np
import torch
import warnings
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class F1RadbertCT:
    LABELS = [
        "Medical material",
        "Arterial wall calcification",
        "Cardiomegaly",
        "Pericardial effusion",
        "Coronary artery wall calcification",
        "Hiatal hernia",
        "Lymphadenopathy",
        "Emphysema",
        "Atelectasis",
        "Lung nodule",
        "Lung opacity",
        "Pulmonary fibrotic sequela",
        "Pleural effusion",
        "Mosaic attenuation pattern",
        "Peribronchial thickening",
        "Consolidation",
        "Bronchiectasis",
        "Interlobular septal thickening",
    ]
    NO_FINDING = "No finding"

    def __init__(
        self,
        model_id: str = "IAMJB/RadBERT-CT",
        threshold: float = 0.5,
        device: Union[str, torch.device] = "cuda",
        batch_size: int = 16,
    ):
        self.model_id = model_id
        self.threshold = threshold
        self.batch_size = batch_size
        self.device = torch.device(device) if isinstance(device, str) else device
        if self.device.type == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA requested but unavailable; falling back to CPU.")
            self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id).to(
            self.device
        )
        self.model.eval()

    @torch.no_grad()
    def _predict_label_matrix(self, reports: Sequence[str]) -> np.ndarray:
        chunks = []
        report_list = list(reports)
        for start in range(0, len(report_list), self.batch_size):
            batch_reports = report_list[start : start + self.batch_size]
            enc = self.tokenizer(
                batch_reports,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits
            probs = torch.sigmoid(logits)
            pred = probs > self.threshold

            no_finding = (~pred.any(dim=1)).unsqueeze(1)
            full = torch.cat([pred, no_finding], dim=1)
            chunks.append(full.cpu())

        return torch.cat(chunks, dim=0).numpy().astype(int)

    def __call__(self, hyps: List[str], refs: List[str]):
        return self.forward(hyps=hyps, refs=refs)

    def forward(self, hyps: List[str], refs: List[str]) -> Tuple[float, List[float], dict]:
        if not isinstance(hyps, list) or not isinstance(refs, list):
            raise TypeError("hyps and refs must be of type list")
        if len(hyps) != len(refs):
            raise ValueError("hyps and refs lists don't have the same size")
        if len(hyps) == 0:
            return 0.0, [], {}

        y_pred = self._predict_label_matrix(hyps)
        y_true = self._predict_label_matrix(refs)

        accuracy = float(accuracy_score(y_true, y_pred))
        per_sample_accuracy = (y_true == y_pred).all(axis=1).astype(float).tolist()
        report = classification_report(
            y_true,
            y_pred,
            target_names=self.LABELS + [self.NO_FINDING],
            output_dict=True,
            zero_division=0,
        )
        return accuracy, per_sample_accuracy, report
