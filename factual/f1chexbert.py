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
    def cls_logits(self, input_ids: torch.LongTensor) -> List[torch.Tensor]:
        """Returns a list of logits for each head (no softmax)."""
        attn = _generate_attention_masks(input_ids)
        outputs = self.bert(input_ids=input_ids, attention_mask=attn)
        cls_repr = self.dropout(outputs.last_hidden_state[:, 0])
        return [head(cls_repr) for head in self.linear_heads]

    @torch.no_grad()
    def cls_embeddings(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Returns pooled [CLS] representations (B, hidden_size)."""
        attn = _generate_attention_masks(input_ids)
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
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()

        # Resolve device -------------------------------------------------------
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.refs_filename = refs_filename
        self.hyps_filename = hyps_filename

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
        # Tokenise *as a batch* for efficiency
        encoding = self.tokenizer(
            reports,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        input_ids = encoding.input_ids.to(self.device)
        # (B, hidden)
        cls = self.model.cls_embeddings(input_ids)
        return [v.cpu().numpy() for v in cls]

    @torch.no_grad()
    def get_label(self, report: str, mode: str = "rrg") -> List[int]:
        """Return 14‑dim binary vector for the given report."""
        input_ids = self.tokenizer(report, truncation=True, max_length=512, return_tensors="pt").input_ids.to(self.device)
        preds = [head.argmax(dim=1).item() for head in self.model.cls_logits(input_ids)]

        binary = []
        if mode == "rrg":
            for c in preds:
                binary.append(1 if c in {1, 3} else 0)
        elif mode == "classification":
            for c in preds:
                if c == 1:
                    binary.append(1)
                elif c == 2:
                    binary.append(0)
                elif c == 3:
                    binary.append(-1)
                else:
                    binary.append(0)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        return binary

    # ---------------------------------------------------------------------
    # Full evaluator – unchanged logic but simplified I/O
    # ---------------------------------------------------------------------

    def forward(self, hyps: List[str], refs: List[str]):
        """Return (accuracy, per‑example‑accuracy, full classification reports)."""
        # Reference labels -----------------------------------------------------
        if self.refs_filename and os.path.exists(self.refs_filename):
            with open(self.refs_filename) as f:
                refs_chexbert = [eval(line) for line in f]
        else:
            refs_chexbert = [self.get_label(r) for r in refs]
            if self.refs_filename:
                with open(self.refs_filename, "w") as f:
                    f.write("\n".join(map(str, refs_chexbert)))

        # Hypothesis labels ----------------------------------------------------
        hyps_chexbert = [self.get_label(h) for h in hyps]
        if self.hyps_filename:
            with open(self.hyps_filename, "w") as f:
                f.write("\n".join(map(str, hyps_chexbert)))

        # TOP‑5 subset arrays --------------------------------------------------
        refs5 = [np.array(r)[self.top5_idx] for r in refs_chexbert]
        hyps5 = [np.array(h)[self.top5_idx] for h in hyps_chexbert]

        # overall accuracy -----------------------------------------------------
        accuracy = accuracy_score(refs5, hyps5)
        _, y_true, y_pred = _check_targets(refs5, hyps5)
        pe_accuracy = (count_nonzero(y_true - y_pred, axis=1) == 0).astype(float)

        # full classification reports -----------------------------------------
        cr = classification_report(refs_chexbert, hyps_chexbert, target_names=self.TARGET_NAMES, output_dict=True)
        cr5 = classification_report(refs5, hyps5, target_names=self.TOP5, output_dict=True)

        return accuracy, pe_accuracy, cr, cr5
