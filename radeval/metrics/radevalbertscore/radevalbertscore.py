import torch
from RadEval.metrics.bertscore.bertscore import BertScoreBase


def _get_default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RadEvalBERTScorer(BertScoreBase):
    """BERTScore wrapper using a custom radiology BERT model."""

    def __init__(self,
                 model_type: str = "IAMJB/RadEvalModernBERT",
                 num_layers: int = 22,
                 use_fast_tokenizer: bool = True,
                 rescale_with_baseline: bool = False,
                 device: torch.device = None,
                 batch_size: int = 64):
        self.model_type = model_type
        self.num_layers = num_layers
        device = device or _get_default_device()
        super().__init__(
            model_type=model_type, num_layers=num_layers,
            batch_size=batch_size, rescale_with_baseline=rescale_with_baseline,
            use_fast_tokenizer=use_fast_tokenizer, device=device,
        )

    def score(self, refs: list[str], hyps: list[str], on_batch_done=None):
        F1 = self._score_batched(refs, hyps, on_batch_done)
        return F1.mean().item(), F1


if __name__ == "__main__":
    refs = ["Chronic mild to moderate cardiomegaly and pulmonary venous hypertension."]
    hyps = ["Mild left basal atelectasis; no pneumonia."]
    scorer = RadEvalBERTScorer(num_layers=23)
    f1_score = scorer.score(refs, hyps)
    print(f"Mean F1 score: {f1_score[0]:.4f}")
