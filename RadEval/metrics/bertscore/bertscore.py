import torch
import torch.nn as nn
from bert_score import BERTScorer


class BertScoreBase(nn.Module):
    """Shared BERTScorer wrapper with batch-level progress callbacks."""

    def __init__(self, *, model_type, num_layers, batch_size=64,
                 rescale_with_baseline=True, idf=False,
                 use_fast_tokenizer=False, device=None):
        super().__init__()
        self.batch_size = batch_size
        with torch.no_grad():
            self.bert_scorer = BERTScorer(
                model_type=model_type,
                num_layers=num_layers,
                batch_size=batch_size,
                nthreads=4,
                all_layers=False,
                idf=idf,
                device=device,
                lang='en',
                rescale_with_baseline=rescale_with_baseline,
                baseline_path=None,
                use_fast_tokenizer=use_fast_tokenizer,
            )

    def _score_batched(self, refs, hyps, on_batch_done=None):
        all_f = []
        for i in range(0, len(refs), self.batch_size):
            p, r, f = self.bert_scorer.score(
                cands=hyps[i:i + self.batch_size],
                refs=refs[i:i + self.batch_size],
                verbose=False,
                batch_size=self.batch_size,
            )
            all_f.append(f)
            if on_batch_done:
                on_batch_done()
        return torch.cat(all_f)


class BertScore(BertScoreBase):
    def __init__(self, model_type='distilbert-base-uncased', num_layers=5,
                 rescale_with_baseline=True, idf=False, batch_size=64):
        super().__init__(
            model_type=model_type, num_layers=num_layers,
            batch_size=batch_size, rescale_with_baseline=rescale_with_baseline,
            idf=idf,
        )

    def forward(self, refs, hyps, on_batch_done=None):
        f = self._score_batched(refs, hyps, on_batch_done)
        return torch.mean(f).item(), f.tolist()


if __name__ == '__main__':
    x, y = BertScore()(
        hyps=["nothing to do lol", "nothing to do x"],
        refs=["heart size is moderately enlarged.", "heart size is mildly enlarged."],
    )
    print(x)
    print(y)
