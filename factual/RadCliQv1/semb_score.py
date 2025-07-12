from typing import Sequence, Union

import numpy as np
import torch

from RadEval.factual.f1chexbert import F1CheXbert


def semantic_embedding_scores(
    refs: Sequence[str],
    hyps: Sequence[str],
    *,
    device: Union[str, torch.device] = "cpu",
) -> np.ndarray:
    """Return per‑pair cosine similarities between `refs` and `hyps`.

    All heavy math is vectorised; no Python loops.

    Args:
        refs: Iterable of ground‑truth report strings.
        hyps: Iterable of predicted report strings (must match `refs` length).
        device: Computation device (e.g. "cpu", "cuda", "cuda:0").

    Returns
    -------
    np.ndarray
        Shape ``(N,)`` – cosine similarity for each pair, where
        ``N == len(refs) == len(hyps)``.

    Raises
    ------
    ValueError
        If `refs` and `hyps` are of different lengths.
    """

    if len(refs) != len(hyps):
        raise ValueError(f"refs ({len(refs)}) and hyps ({len(hyps)}) differ in length")

    labeler = F1CheXbert(device=device)

    # Stack embeddings into (N, dim) matrices
    gt_embeds = np.vstack(labeler.get_embeddings(refs))   # (N, dim)
    pred_embeds = np.vstack(labeler.get_embeddings(hyps))  # (N, dim)

    # Cosine similarity – fully vectorised
    dot = np.einsum("nd,nd->n", gt_embeds, pred_embeds)
    norms = np.linalg.norm(gt_embeds, axis=1) * np.linalg.norm(pred_embeds, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sims = np.where(norms > 0, dot / norms, 0.0)

    return sims


def mean_semantic_score(scores: np.ndarray) -> float:
    """Convenience helper: mean of an array of scores."""
    return float(scores.mean())


if __name__ == "__main__":
    _refs = [
        "No evidence of pneumothorax following chest tube removal.",
        "There is a left pleural effusion.",
        "No evidence of pneumothorax following chest tube removal.",

    ]
    _hyps = [
        "No pneumothorax detected.",
        "Left pleural effusion is present.",
        "Left pleural effusion is present.",
    ]

    _scores = semantic_embedding_scores(_refs, _hyps, device="cpu")
    print("Per‑pair cosine:", _scores)
    print("Mean:", mean_semantic_score(_scores))
