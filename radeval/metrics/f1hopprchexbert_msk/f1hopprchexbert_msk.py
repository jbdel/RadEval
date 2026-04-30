"""HopprF1CheXbertMSK: multi-output MSK report evaluator (ModernBERT-large).

A single forward pass classifies 20 musculoskeletal conditions simultaneously.
Each condition head outputs 4-way logits (definitely absent / not reported /
uncertain / definitely present), collapsed to binary for F1 evaluation.
"""
from __future__ import annotations

import os
import warnings
from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass

_DEFAULT_CKPT = (
    "/nfs/cluster/hoppr_vlm_ressources/radeval_checkpoints/f1hopprchexbert_msk"
)

CONDITION_NAMES = OrderedDict([
    ("acute_fracture", "Acute fracture"),
    ("healed_or_chronic_fracture", "Healed or chronic fracture"),
    ("pathologic_fracture", "Pathologic fracture"),
    ("osteoarthritis_or_degenerative_joint_disease", "Osteoarthritis or degenerative joint disease"),
    ("dislocation_or_subluxation", "Dislocation or subluxation"),
    ("joint_effusion", "Joint effusion"),
    ("osteopenia", "Osteopenia"),
    ("soft_tissue_swelling_or_mass", "Soft tissue swelling or mass"),
    ("bone_lesion", "Bone lesion"),
    ("hardware_or_implanted_device", "Hardware or implanted device"),
    ("degenerative_disc_disease", "Degenerative disc disease"),
    ("scoliosis_or_spinal_deformity", "Scoliosis or spinal deformity"),
    ("erosive_or_inflammatory_arthropathy", "Erosive or inflammatory arthropathy"),
    ("osteonecrosis_or_avascular_necrosis", "Osteonecrosis or avascular necrosis"),
    ("periosteal_reaction", "Periosteal reaction"),
    ("chondrocalcinosis", "Chondrocalcinosis"),
    ("soft_tissue_calcification", "Soft tissue calcification"),
    ("osteomyelitis", "Osteomyelitis"),
    ("malalignment_or_deformity", "Malalignment or deformity"),
    ("spondylolisthesis", "Spondylolisthesis"),
])

NUM_CONDITIONS = len(CONDITION_NAMES)
NUM_CLASSES = 4  # definitely absent, not reported, uncertain, definitely present

# ---------------------------------------------------------------------------
# Model definition (must match the training code for from_pretrained to work)
# ---------------------------------------------------------------------------


@dataclass
class MultiOutputClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None


class MSKv2MultiOutputClassifier(PreTrainedModel):
    """20 independent 4-class heads sharing one BERT-style encoder."""

    config_class = AutoConfig
    _keys_to_ignore_on_load_unexpected = [r"cls", r"classifier", r"score"]

    def __init__(self, config):
        super().__init__(config)
        self.encoder = AutoModel.from_config(config)
        hidden = config.hidden_size
        self.heads = nn.ModuleList(
            [nn.Linear(hidden, NUM_CLASSES) for _ in range(NUM_CONDITIONS)]
        )
        self.class_weights = None
        self._class_weights_raw = None
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> MultiOutputClassifierOutput:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0]
        all_logits = torch.stack([head(cls_hidden) for head in self.heads], dim=1)

        loss = None
        if labels is not None:
            total_loss = torch.tensor(0.0, device=all_logits.device,
                                      dtype=all_logits.dtype)
            for i in range(NUM_CONDITIONS):
                total_loss = total_loss + nn.functional.cross_entropy(
                    all_logits[:, i, :], labels[:, i])
            loss = total_loss / NUM_CONDITIONS

        return MultiOutputClassifierOutput(loss=loss, logits=all_logits)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class HopprF1CheXbertMSK:
    """Multi-output MSK finding classifier for report evaluation.

    A single forward pass produces (batch, 20, 4) logits. Predictions are
    collapsed to binary (classes 0-1 = negative, 2-3 = positive) and
    compared via sklearn classification_report.
    """

    LABELS = list(CONDITION_NAMES.keys())
    NO_FINDING = "no_finding"

    def __init__(
        self,
        checkpoint_dir: str = _DEFAULT_CKPT,
        device: Union[str, torch.device] = "cuda",
        batch_size: int = 16,
        max_length: int = 512,
    ):
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(
                f"HopprF1CheXbertMSK checkpoint not found: {checkpoint_dir}")

        self.batch_size = batch_size
        self.max_length = max_length
        self.device = torch.device(device) if isinstance(device, str) else device
        if self.device.type == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA requested but unavailable; falling back to CPU.")
            self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_dir, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token or self.tokenizer.unk_token)

        self.model = MSKv2MultiOutputClassifier.from_pretrained(
            checkpoint_dir, trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _predict_label_matrix(
        self, reports: Sequence[str], on_batch_done=None,
    ) -> np.ndarray:
        """Return binary label matrix of shape (N, 21).

        Logits (N, 20, 4) -> argmax -> binary (classes 2,3 = positive).
        A 21st "no_finding" column is 1 when all 20 conditions are 0.
        """
        all_binary = []
        report_list = list(reports)

        for start in range(0, len(report_list), self.batch_size):
            batch = report_list[start:start + self.batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits  # (B, 20, 4)
            pred_ids = logits.argmax(dim=-1)   # (B, 20)
            binary = (pred_ids >= 2).int().cpu()
            all_binary.append(binary)
            if on_batch_done:
                on_batch_done()

        matrix = torch.cat(all_binary, dim=0)  # (N, 20)
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
