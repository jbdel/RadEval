"""RadGraph entity+relation F1 (RadCliQ variant).

Computes per-pair (entity_F1 + relation_F1) / 2 using the RadGraph model,
matching the RadGraph sub-metric used inside RadCliQ-v1 from:
  https://github.com/rajpurkarlab/CXR-Report-Metric

This differs from the official F1RadGraph metric (``do_radgraph``) which
uses radgraph-xl and returns simple/partial/complete reward-level scores.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple, Union

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
    """Per-pair RadGraph entity+relation F1 (RadCliQ variant).

    For each (ref, hyp) pair:
      entity_f1  = F1 over the set of (tokens, label) tuples
      relation_f1 = F1 over the set of (src, tgt, relation) tuples
      score = (entity_f1 + relation_f1) / 2
    """

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
            ent_f1 = _compute_f1(
                _extract_entities(gt_out), _extract_entities(pred_out))
            rel_f1 = _compute_f1(
                _extract_relations(gt_out), _extract_relations(pred_out))
            scores.append((ent_f1 + rel_f1) / 2)
            if on_sample_done:
                on_sample_done()

        mean_score = sum(scores) / len(scores)
        return mean_score, scores
