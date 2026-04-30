"""BERTScorer — trimmed from bert_score.scorer (MIT License)."""
from __future__ import annotations

import os
import warnings
from collections import defaultdict
from typing import Optional

import pandas as pd
import torch

from .utils import (
    VERSION,
    bert_cos_score_idf,
    get_idf_dict,
    get_model,
    get_tokenizer,
)


class BERTScorer:
    """Minimal BERTScorer preserving the public API RadEval uses."""

    def __init__(
        self,
        model_type: Optional[str] = None,
        num_layers: Optional[int] = None,
        batch_size: int = 64,
        nthreads: int = 4,
        all_layers: bool = False,
        idf: bool = False,
        idf_sents: Optional[list] = None,
        device: Optional[str] = None,
        lang: Optional[str] = None,
        rescale_with_baseline: bool = False,
        baseline_path: Optional[str] = None,
        use_fast_tokenizer: bool = True,
    ):
        assert model_type is not None, (
            "model_type must be specified (lang-only default lookup was removed "
            "from the vendored bert_score)."
        )
        assert num_layers is not None, (
            "num_layers must be specified (model2layers default table was "
            "removed from the vendored bert_score)."
        )
        if rescale_with_baseline:
            assert lang is not None, (
                "Need to specify lang when rescaling with baseline."
            )

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._lang = lang
        self._rescale_with_baseline = rescale_with_baseline
        self._idf = idf
        self.batch_size = batch_size
        self.nthreads = nthreads
        self.all_layers = all_layers
        self._model_type = model_type
        self._num_layers = num_layers
        self._use_fast_tokenizer = use_fast_tokenizer

        self._tokenizer = get_tokenizer(model_type, use_fast=use_fast_tokenizer)
        self._model = get_model(model_type, num_layers, all_layers=all_layers)
        self._model.to(self.device)

        self._idf_dict = None
        if idf_sents is not None:
            self.compute_idf(idf_sents)

        self._baseline_vals = None
        self.baseline_path = baseline_path
        self.use_custom_baseline = self.baseline_path is not None
        if self.baseline_path is None and lang is not None:
            # Look up vendored baseline TSVs first.
            vendor_path = os.path.join(
                os.path.dirname(__file__),
                "rescale_baseline",
                str(lang),
                f"{model_type}.tsv",
            )
            if os.path.isfile(vendor_path):
                self.baseline_path = vendor_path

    # -- properties (preserved for API compatibility) -----------------

    @property
    def lang(self):
        return self._lang

    @property
    def idf(self):
        return self._idf

    @property
    def model_type(self):
        return self._model_type

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def rescale_with_baseline(self):
        return self._rescale_with_baseline

    @property
    def use_fast_tokenizer(self):
        return self._use_fast_tokenizer

    @property
    def baseline_vals(self):
        if self._baseline_vals is None:
            if self.baseline_path and os.path.isfile(self.baseline_path):
                if not self.all_layers:
                    self._baseline_vals = torch.from_numpy(
                        pd.read_csv(self.baseline_path)
                        .iloc[self.num_layers]
                        .to_numpy()
                    )[1:].float()
                else:
                    self._baseline_vals = (
                        torch.from_numpy(
                            pd.read_csv(self.baseline_path).to_numpy()
                        )[:, 1:]
                        .unsqueeze(1)
                        .float()
                    )
            else:
                raise ValueError(
                    f"Baseline not found for {self.model_type} "
                    f"at {self.baseline_path}"
                )
        return self._baseline_vals

    @property
    def hash(self):
        parts = [
            self._model_type,
            f"L{self._num_layers}",
            "idf" if self._idf else "no-idf",
            f"version={VERSION}",
        ]
        if self._rescale_with_baseline:
            parts.append("custom-rescaled" if self.use_custom_baseline else "rescaled")
        if self._use_fast_tokenizer:
            parts.append("fast-tokenizer")
        return "_".join(parts)

    # -- computation --------------------------------------------------

    def compute_idf(self, sents):
        if self._idf_dict is not None:
            warnings.warn("Overwriting previous importance weights.")
        self._idf_dict = get_idf_dict(sents, self._tokenizer, nthreads=self.nthreads)

    def score(self, cands, refs, verbose: bool = False,
              batch_size: int = 64, return_hash: bool = False):
        """Return (P, R, F1), each a 1-D tensor of length N.

        We do NOT support nested reference lists in this vendored copy —
        RadEval always passes `list[str]`.
        """
        assert isinstance(refs, list) and isinstance(cands, list)
        assert len(refs) == len(cands)
        assert not refs or isinstance(refs[0], str), (
            "Vendored bert_score does not support nested reference lists."
        )

        if self.idf:
            assert self._idf_dict, "IDF weights are not computed"
            idf_dict = self._idf_dict
        else:
            idf_dict = defaultdict(lambda: 1.0)
            idf_dict[self._tokenizer.sep_token_id] = 0
            idf_dict[self._tokenizer.cls_token_id] = 0

        all_preds = bert_cos_score_idf(
            self._model,
            refs,
            cands,
            self._tokenizer,
            idf_dict,
            verbose=verbose,
            device=self.device,
            batch_size=batch_size,
            all_layers=self.all_layers,
        ).cpu()

        if self.rescale_with_baseline:
            all_preds = (all_preds - self.baseline_vals) / (1 - self.baseline_vals)

        out = all_preds[..., 0], all_preds[..., 1], all_preds[..., 2]  # P, R, F
        if return_hash:
            return out, self.hash
        return out

    def __repr__(self):
        return (
            f"BERTScorer(hash={self.hash}, batch_size={self.batch_size}, "
            f"nthreads={self.nthreads})"
        )
