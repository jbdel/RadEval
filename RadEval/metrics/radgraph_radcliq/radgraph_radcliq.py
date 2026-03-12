"""RadGraph-RadCliQ: entity + relation F1 as used in the RadCliQ-v1 composite.

This metric isolates the RadGraph sub-score from the RadCliQ-v1 pipeline
(rajpurkarlab/CXR-Report-Metric). It differs from the official ``F1RadGraph``
metric (from the ``radgraph`` package) in several ways:

**Official F1RadGraph** (``do_radgraph``):
  - Uses ``radgraph-xl`` (larger, more accurate model)
  - Computes entity overlap at three reward levels: simple, partial, complete
  - Returns three aggregate F1 scores across all pairs

**RadGraph-RadCliQ** (``do_radgraph_radcliq``):
  - Uses the original ``radgraph`` model (the one RadCliQ-v1 was trained with)
  - Extracts entity *and* relation sets from the RadGraph output
  - Computes per-pair ``(entity_f1 + relation_f1) / 2``
  - Returns per-sample scores (useful for correlation analysis and details)

Use ``do_radgraph`` for the standard metric and ``do_radgraph_radcliq``
when you need per-pair scores or exact alignment with the RadCliQ-v1 composite.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from radgraph import RadGraph


def _compute_f1(test_set: set, retrieved_set: set) -> float:
    tp = len(test_set & retrieved_set)
    fp = len(retrieved_set) - tp
    fn = len(test_set) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def _extract_entities(output: dict) -> set:
    return {
        (tuple(ent["tokens"]), ent["label"])
        for ent in output.get("entities", {}).values()
    }


def _extract_relations(output: dict) -> set:
    rels = set()
    entities = output.get("entities", {})
    for ent in entities.values():
        src = (tuple(ent["tokens"]), ent["label"])
        for rel_type, tgt_idx in ent.get("relations", []):
            tgt_ent = entities.get(tgt_idx)
            if tgt_ent:
                tgt = (tuple(tgt_ent["tokens"]), tgt_ent["label"])
                rels.add((src, tgt, rel_type))
    return rels


class RadGraphRadCliQ:
    """Per-pair (entity_f1 + relation_f1) / 2 using the RadGraph model."""

    def __init__(self, model_type: str = "radgraph"):
        self._radgraph = RadGraph(model_type=model_type)

    def __call__(
        self, hyps: List[str], refs: List[str], on_sample_done=None,
    ) -> Tuple[float, List[float]]:
        return self.forward(hyps=hyps, refs=refs, on_sample_done=on_sample_done)

    def forward(
        self, hyps: List[str], refs: List[str], on_sample_done=None,
    ) -> Tuple[float, List[float]]:
        if not isinstance(hyps, list) or not isinstance(refs, list):
            raise TypeError("hyps and refs must be of type list")
        if len(hyps) != len(refs):
            raise ValueError("hyps and refs lists don't have the same size")
        if len(hyps) == 0:
            return 0.0, []

        gt_outputs = self._radgraph(refs)
        pred_outputs = self._radgraph(hyps)

        scores = []
        for i in range(len(refs)):
            gt_out = gt_outputs.get(str(i), {})
            pred_out = pred_outputs.get(str(i), {})
            ent_f1 = _compute_f1(_extract_entities(gt_out), _extract_entities(pred_out))
            rel_f1 = _compute_f1(_extract_relations(gt_out), _extract_relations(pred_out))
            scores.append((ent_f1 + rel_f1) / 2)
            if on_sample_done:
                on_sample_done()

        mean_score = sum(scores) / len(scores)
        return mean_score, scores
