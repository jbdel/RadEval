"""Shared base for CheXbert-family evaluators (F1CheXbert, HopprF1CheXbert).

Subclasses provide:
  - ``CONDITION_NAMES``, ``NO_FINDING``, ``TOP5`` class attributes
  - A constructed ``model`` (BertLabeler variant) and ``tokenizer``
"""
from __future__ import annotations

import json
import os
import warnings
from typing import List, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics._classification import _check_targets
from sklearn.utils.sparsefuncs import count_nonzero


def generate_attention_masks(batch_ids: torch.LongTensor) -> torch.FloatTensor:
    """Create a padding mask: 1 for real tokens, 0 for pads."""
    lengths = (batch_ids != 0).sum(dim=1)
    max_len = batch_ids.size(1)
    idxs = torch.arange(max_len, device=batch_ids.device).unsqueeze(0)
    return (idxs < lengths.unsqueeze(1)).float()


class BaseBertLabeler(nn.Module):
    """Backbone + linear classification heads, shared forward helpers."""

    def __init__(self, *, device, backbone: nn.Module, num_4way_heads: int,
                 state_dict: dict):
        super().__init__()
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.bert = backbone
        hidden = self.bert.config.hidden_size
        self.linear_heads = nn.ModuleList(
            [nn.Linear(hidden, 4) for _ in range(num_4way_heads)])
        self.linear_heads.append(nn.Linear(hidden, 2))
        self.dropout = nn.Dropout(0.1)

        self.load_state_dict(state_dict, strict=True)
        self.to(self.device)
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def cls_logits(self, input_ids, attention_mask=None):
        attn = attention_mask if attention_mask is not None else generate_attention_masks(input_ids)
        outputs = self.bert(input_ids=input_ids, attention_mask=attn)
        cls_repr = self.dropout(outputs.last_hidden_state[:, 0])
        return [head(cls_repr) for head in self.linear_heads]

    @torch.no_grad()
    def cls_embeddings(self, input_ids, attention_mask=None):
        attn = attention_mask if attention_mask is not None else generate_attention_masks(input_ids)
        outputs = self.bert(input_ids=input_ids, attention_mask=attn)
        return outputs.last_hidden_state[:, 0]


class BaseCheXbertEvaluator(nn.Module):
    """Shared evaluation logic for CheXbert-family models."""

    CONDITION_NAMES: List[str] = []
    NO_FINDING: str = ""
    TOP5: List[str] = []

    @property
    def TARGET_NAMES(self):
        return self.CONDITION_NAMES + [self.NO_FINDING]

    def _init_evaluator(self, *, model, tokenizer, device, batch_size,
                        refs_filename=None, hyps_filename=None):
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        if self.device.type == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA requested but unavailable; falling back to CPU.")
            self.device = torch.device("cpu")

        self.refs_filename = refs_filename
        self.hyps_filename = hyps_filename
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.model = model
        self.top5_idx = [self.TARGET_NAMES.index(n) for n in self.TOP5]

    @torch.no_grad()
    def get_embeddings(self, reports: Sequence[str]) -> List[np.ndarray]:
        embeddings: List[np.ndarray] = []
        for i in range(0, len(reports), self.batch_size):
            batch_reports = reports[i:i + self.batch_size]
            encoding = self.tokenizer(
                batch_reports, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            )
            input_ids = encoding.input_ids.to(self.device)
            attention_mask = encoding.attention_mask.to(self.device)
            cls = self.model.cls_embeddings(input_ids, attention_mask=attention_mask)
            embeddings.extend(v.cpu().numpy() for v in cls)
        return embeddings

    @torch.no_grad()
    def get_label(self, report: str, mode: str = "rrg") -> List[int]:
        return self.get_labels([report], mode=mode)[0]

    @torch.no_grad()
    def get_labels(self, reports: Sequence[str], mode: str = "rrg",
                   on_batch_done=None) -> List[List[int]]:
        labels: List[List[int]] = []
        for i in range(0, len(reports), self.batch_size):
            batch_reports = reports[i:i + self.batch_size]
            encoding = self.tokenizer(
                batch_reports, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            )
            input_ids = encoding.input_ids.to(self.device)
            attention_mask = encoding.attention_mask.to(self.device)
            logits_per_head = self.model.cls_logits(input_ids, attention_mask=attention_mask)
            pred_matrix = torch.stack(
                [head.argmax(dim=1) for head in logits_per_head], dim=1)

            for row in pred_matrix.tolist():
                if mode == "rrg":
                    labels.append([1 if c in {1, 3} else 0 for c in row])
                elif mode == "classification":
                    labels.append([1 if c == 1 else -1 if c == 3 else 0 for c in row])
                else:
                    raise ValueError(f"Unknown mode: {mode}")
            if on_batch_done:
                on_batch_done()
        return labels

    def forward(self, hyps: List[str], refs: List[str], on_batch_done=None):
        if self.refs_filename and os.path.exists(self.refs_filename):
            with open(self.refs_filename) as f:
                refs_chexbert = [json.loads(line) for line in f]
        else:
            refs_chexbert = self.get_labels(refs, on_batch_done=on_batch_done)
            if self.refs_filename:
                with open(self.refs_filename, "w") as f:
                    f.write("\n".join(json.dumps(r) for r in refs_chexbert))

        hyps_chexbert = self.get_labels(hyps, on_batch_done=on_batch_done)
        if self.hyps_filename:
            with open(self.hyps_filename, "w") as f:
                f.write("\n".join(json.dumps(h) for h in hyps_chexbert))

        refs5 = [np.array(r)[self.top5_idx] for r in refs_chexbert]
        hyps5 = [np.array(h)[self.top5_idx] for h in hyps_chexbert]

        accuracy = accuracy_score(refs5, hyps5)

        # sklearn >=1.8 returns (y_type, y_true, y_pred, indicator);
        # earlier releases return (y_type, y_true, y_pred). Tolerate both.
        _ct5 = _check_targets(refs5, hyps5)
        y_true5, y_pred5 = _ct5[1], _ct5[2]
        pe_accuracy = (count_nonzero(y_true5 - y_pred5, axis=1) == 0).astype(float)

        sample_label_acc_5 = np.asarray(
            (y_true5 == y_pred5).mean(axis=1)).astype(float).ravel().tolist()

        _ct_full = _check_targets(refs_chexbert, hyps_chexbert)
        y_true_full, y_pred_full = _ct_full[1], _ct_full[2]
        sample_label_acc_full = np.asarray(
            (y_true_full == y_pred_full).mean(axis=1)).astype(float).ravel().tolist()

        cr = classification_report(
            refs_chexbert, hyps_chexbert,
            target_names=self.TARGET_NAMES, output_dict=True,
        )
        cr5 = classification_report(
            refs5, hyps5,
            target_names=self.TOP5, output_dict=True,
        )

        return accuracy, pe_accuracy, cr, cr5, sample_label_acc_full, sample_label_acc_5
