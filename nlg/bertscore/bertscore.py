import torch
import torch.nn as nn
from bert_score import BERTScorer


class BertScore(nn.Module):
    def __init__(self,
                 model_type='distilbert-base-uncased',
                 num_layers=5,
                 rescale_with_baseline=True,
                 idf=False,
                 ):
        super(BertScore, self).__init__()
        with torch.no_grad():
            self.bert_scorer = BERTScorer(model_type=model_type,
                                          num_layers=num_layers,
                                          batch_size=64,
                                          nthreads=4,
                                          all_layers=False,
                                          idf=idf,
                                          device=None,
                                          lang='en',
                                          rescale_with_baseline=rescale_with_baseline,
                                          baseline_path=None)

    def forward(self, refs, hyps):
        p, r, f = self.bert_scorer.score(
            cands=hyps,
            refs=refs,
            verbose=False,
            batch_size=64,
        )
        return torch.mean(f).item(), f.tolist()


if __name__ == '__main__':
    x, y = (BertScore()(
        hyps=[
            "nothing to do lol",
            "nothing to do x",
            'there are moderate bilateral pleural effusions with overlying atelectasis,  underlying consolidation not excluded. mild prominence of the interstitial  markings suggests mild pulmonary edema. the cardiac silhouette is mildly  enlarged. the mediastinal contours are unremarkable. there is no evidence of  pneumothorax.'
        ],
        refs=[
            'heart size is moderately enlarged. the mediastinal and hilar contours are unchanged. there is no pulmonary edema. small left pleural effusion is present. patchy opacities in the lung bases likely reflect atelectasis. no pneumothorax is seen. there are no acute osseous abnormalities.',
            'heart size is mildly enlarged. the mediastinal and hilar contours are normal. there is mild pulmonary edema. moderate bilateral pleural effusions are present, left greater than right. bibasilar airspace opacities likely reflect atelectasis. no pneumothorax is seen. there are no acute osseous abnormalities.',
            'heart size is mildly enlarged. the mediastinal and hilar contours are normal. there is mild pulmonary edema. moderate bilateral pleural effusions are present, left greater than right. bibasilar airspace opacities likely reflect atelectasis. no pneumothorax is seen. there are no acute osseous abnormalities.'
        ])
    )
    print(x)
    print(y)
