#!/usr/bin/env python
"""CheXbert evaluation utilities – **device‑safe end‑to‑end**

This is a drop‑in replacement for your previous `f1chexbert.py` **and** for the helper
`SemanticEmbeddingScorer`.  All tensors – model weights *and* inputs – are created on
exactly the same device so the             ``Expected all tensors to be on the same device``
run‑time error disappears.  The public API stays identical, so the rest of your
pipeline does not need to change.
"""

from __future__ import annotations

import os
import warnings
import logging
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import numpy as np
from transformers import (
    AutoConfig,
    BertModel,
    BertTokenizer,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.metrics._classification import _check_targets
from sklearn.utils.sparsefuncs import count_nonzero
from huggingface_hub import hf_hub_download
from appdirs import user_cache_dir

# -----------------------------------------------------------------------------
# GLOBALS & UTILITIES
# -----------------------------------------------------------------------------

CACHE_DIR = user_cache_dir("chexbert")
warnings.filterwarnings("ignore")
logging.getLogger("urllib3").setLevel(logging.ERROR)

# Helper ----------------------------------------------------------------------

def _generate_attention_masks(batch_ids: torch.LongTensor) -> torch.FloatTensor:
    """Create a padding mask: 1 for real tokens, 0 for pads."""
    # batch_ids shape: (B, L)
    lengths = (batch_ids != 0).sum(dim=1)  # (B,)
    max_len = batch_ids.size(1)
    idxs = torch.arange(max_len, device=batch_ids.device).unsqueeze(0)  # (1, L)
    return (idxs < lengths.unsqueeze(1)).float()  # (B, L)

# -----------------------------------------------------------------------------
# MODEL COMPONENTS
# -----------------------------------------------------------------------------

class BertLabeler(nn.Module):
    """BERT backbone + 14 small classification heads (CheXbert)."""

    def __init__(self, *, device: Union[str, torch.device]):
        super().__init__()

        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # 1) Backbone on *CPU* first – we'll move to correct device after weights load
        config = AutoConfig.from_pretrained("bert-base-uncased")
        self.bert = BertModel(config)

        hidden = self.bert.config.hidden_size
        # 13 heads with 4‑way logits, + 1 head with 2‑way logits
        self.linear_heads = nn.ModuleList([nn.Linear(hidden, 4) for _ in range(13)])
        self.linear_heads.append(nn.Linear(hidden, 2))

        self.dropout = nn.Dropout(0.1)

        # 2) Load checkpoint weights directly onto CPU first -------------------
        ckpt_path = hf_hub_download(
            repo_id="StanfordAIMI/RRG_scorers",
            filename="chexbert.pth",
            cache_dir=CACHE_DIR,
        )
        state = torch.load(ckpt_path, map_location="cpu")["model_state_dict"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self.load_state_dict(state, strict=True)

        # 3) NOW move the entire module (recursively) to `self.device` ----------
        self.to(self.device)

        # freeze ---------------------------------------------------------------
        for p in self.parameters():
            p.requires_grad = False

    # ---------------------------------------------------------------------
    # forward helpers
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def cls_logits(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | None = None,
    ) -> List[torch.Tensor]:
        """Returns a list of logits for each head (no softmax)."""
        attn = attention_mask if attention_mask is not None else _generate_attention_masks(input_ids)
        outputs = self.bert(input_ids=input_ids, attention_mask=attn)
        cls_repr = self.dropout(outputs.last_hidden_state[:, 0])
        return [head(cls_repr) for head in self.linear_heads]

    @torch.no_grad()
    def cls_embeddings(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | None = None,
    ) -> torch.Tensor:
        """Returns pooled [CLS] representations (B, hidden_size)."""
        attn = attention_mask if attention_mask is not None else _generate_attention_masks(input_ids)
        outputs = self.bert(input_ids=input_ids, attention_mask=attn)
        return outputs.last_hidden_state[:, 0]  # (B, hidden)

# -----------------------------------------------------------------------------
# F1‑CheXbert evaluator
# -----------------------------------------------------------------------------

class F1CheXbert(nn.Module):
    """Generate CheXbert labels + handy evaluation utilities."""

    CONDITION_NAMES = [
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices",
    ]
    NO_FINDING = "No Finding"
    TARGET_NAMES = CONDITION_NAMES + [NO_FINDING]

    TOP5 = [
        "Cardiomegaly",
        "Edema",
        "Consolidation",
        "Atelectasis",
        "Pleural Effusion",
    ]

    def __init__(
        self,
        *,
        refs_filename: str | None = None,
        hyps_filename: str | None = None,
        device: Union[str, torch.device] = "cuda",
        batch_size: int = 64,
    ):
        super().__init__()

        # Resolve device -------------------------------------------------------
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.refs_filename = refs_filename
        self.hyps_filename = hyps_filename
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = batch_size

        # HuggingFace tokenizer (always CPU, we just move tensors later) -------
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # backbone + heads ------------------------------------------------------
        self.model = BertLabeler(device=self.device).eval()

        # indices for the TOP‑5 label subset -----------------------------------
        self.top5_idx = [self.TARGET_NAMES.index(n) for n in self.TOP5]

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def get_embeddings(self, reports: Sequence[str]) -> List[np.ndarray]:
        """Return list[np.ndarray] of pooled [CLS] vectors for each report."""
        embeddings: List[np.ndarray] = []
        for i in range(0, len(reports), self.batch_size):
            batch_reports = reports[i:i + self.batch_size]
            encoding = self.tokenizer(
                batch_reports,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            input_ids = encoding.input_ids.to(self.device)
            attention_mask = encoding.attention_mask.to(self.device)
            cls = self.model.cls_embeddings(input_ids, attention_mask=attention_mask)
            embeddings.extend(v.cpu().numpy() for v in cls)
        return embeddings

    @torch.no_grad()
    def get_label(self, report: str, mode: str = "rrg") -> List[int]:
        """Return 14‑dim binary vector for the given report."""
        return self.get_labels([report], mode=mode)[0]

    @torch.no_grad()
    def get_labels(self, reports: Sequence[str], mode: str = "rrg") -> List[List[int]]:
        """Return 14-dim vectors for each report using batched inference."""
        labels: List[List[int]] = []
        for i in range(0, len(reports), self.batch_size):
            batch_reports = reports[i:i + self.batch_size]
            encoding = self.tokenizer(
                batch_reports,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            input_ids = encoding.input_ids.to(self.device)
            attention_mask = encoding.attention_mask.to(self.device)
            logits_per_head = self.model.cls_logits(input_ids, attention_mask=attention_mask)
            pred_matrix = torch.stack([head.argmax(dim=1) for head in logits_per_head], dim=1)

            for row in pred_matrix.tolist():
                if mode == "rrg":
                    labels.append([1 if c in {1, 3} else 0 for c in row])
                elif mode == "classification":
                    labels.append(
                        [
                            1 if c == 1 else -1 if c == 3 else 0
                            for c in row
                        ]
                    )
                else:
                    raise ValueError(f"Unknown mode: {mode}")
        return labels

    # ---------------------------------------------------------------------
    # Full evaluator – unchanged logic but simplified I/O
    # ---------------------------------------------------------------------

    def forward(self, hyps: List[str], refs: List[str]):

        """
        Return:
        (accuracy, pe_accuracy, cr, cr5, overall_label_acc_full, overall_label_acc_5)

        Notes:
        - accuracy: subset accuracy (exact-match) on TOP-5 labels (scalar in [0,1])
        - pe_accuracy: per-example exact-match on TOP-5 labels (shape (N,), values {0,1})
        - overall_label_acc_full: overall label accuracy across ALL labels (scalar in [0,1])
        - overall_label_acc_5: overall label accuracy across TOP-5 labels (scalar in [0,1])
        """
        # Reference labels -----------------------------------------------------
        if self.refs_filename and os.path.exists(self.refs_filename):
            with open(self.refs_filename) as f:
                refs_chexbert = [eval(line) for line in f]
        else:
            refs_chexbert = self.get_labels(refs)
            if self.refs_filename:
                with open(self.refs_filename, "w") as f:
                    f.write("\n".join(map(str, refs_chexbert)))

        # Hypothesis labels ----------------------------------------------------
        hyps_chexbert = self.get_labels(hyps)
        if self.hyps_filename:
            with open(self.hyps_filename, "w") as f:
                f.write("\n".join(map(str, hyps_chexbert)))

        # TOP‑5 subset arrays --------------------------------------------------
        refs5 = [np.array(r)[self.top5_idx] for r in refs_chexbert]
        hyps5 = [np.array(h)[self.top5_idx] for h in hyps_chexbert]

        # overall accuracy (TOP-5 exact match) --------------------------------
        accuracy = accuracy_score(refs5, hyps5)

        # TOP-5 targets (binarised) -------------------------------------------
        _, y_true5, y_pred5, _ = _check_targets(refs5, hyps5)

        # per-example exact-match accuracy on TOP-5 (0/1 per sample) ----------
        pe_accuracy = (count_nonzero(y_true5 - y_pred5, axis=1) == 0).astype(float)

        # Overall label accuracy on TOP-5 (looser) -----------------------
        sample_label_acc_5 = np.asarray((y_true5 == y_pred5).mean(axis=1)).astype(float).ravel().tolist()

        # Overall label accuracy on FULL labels (looser) -----------------
        _, y_true_full, y_pred_full, _ = _check_targets(refs_chexbert, hyps_chexbert)
        sample_label_acc_full = np.asarray((y_true_full == y_pred_full).mean(axis=1)).astype(float).ravel().tolist()
        
        # full classification reports -----------------------------------------
        cr = classification_report(
            refs_chexbert,
            hyps_chexbert,
            target_names=self.TARGET_NAMES,
            output_dict=True,
        )
        cr5 = classification_report(
            refs5,
            hyps5,
            target_names=self.TOP5,
            output_dict=True,
        )

        return (
            accuracy,
            pe_accuracy,
            cr,
            cr5,
            sample_label_acc_full,
            sample_label_acc_5,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run one-disease probe sentences through F1CheXbert and print expected vs received present labels."
    )
    parser.add_argument(
        "--mode",
        choices=["rrg", "classification"],
        default="rrg",
        help="Output mode to use when calling get_labels().",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for probe inference.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = F1CheXbert(device=device, batch_size=args.batch_size)

    probes: List[Tuple[str, str]] = [
        ("Mild cardiomegaly is present.", "Cardiomegaly"),
        ("Patchy right lower lobe consolidation.", "Consolidation"),
        ("Patchy right air space opacity.", "Lung Opacity"),
        ("Small left pleural effusion is seen.", "Pleural Effusion"),
        ("Left apical pneumothorax is present.", "Pneumothorax"),
        ("Bibasilar subsegmental atelectatic change.", "Atelectasis"),
        ("Pulmonary edema is present.", "Edema"),
        ("A right upper lobe lung lesion is identified.", "Lung Lesion"),
        ("Acute displaced left rib fracture is present.", "Fracture"),
        ("Endotracheal tube and enteric tube are in place.", "Support Devices"),
        ("No focal airspace opacity. Mild right lower lobe pneumonia.", "Pneumonia"),
    ]

    sentences = [s for s, _ in probes]
    expected = [d for _, d in probes]
    preds = scorer.get_labels(sentences, mode=args.mode)

    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print()

    for idx, (sentence, exp, row) in enumerate(zip(sentences, expected, preds), start=1):
        if args.mode == "classification":
            present = [
                name for name, value in zip(scorer.TARGET_NAMES, row) if value in (1, -1)
            ]
        else:
            present = [name for name, value in zip(scorer.TARGET_NAMES, row) if value == 1]

        match = exp in present
        print(f"[{idx}] {sentence}")
        print(f"  expected: {exp}")
        print(f"  received: {present if present else ['<none>']}")
        print(f"  contains_expected: {match}")
        print()
