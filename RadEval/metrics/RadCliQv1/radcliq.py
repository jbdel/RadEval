import numpy as np
import torch
from radgraph import RadGraph
from RadEval.metrics.f1chexbert import F1CheXbert
from sklearn.preprocessing import StandardScaler
from RadEval.metrics.bleu.bleu import Bleu


def compute_f1(test_set, retrieved_set):
    """Helper to compute F1 between two sets of items."""
    tp = len(test_set & retrieved_set)
    fp = len(retrieved_set) - tp
    fn = len(test_set) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def extract_entities(output):
    """Extracts set of (tokens, label) tuples from RadGraph output."""
    return {(tuple(ent["tokens"]), ent["label"]) for ent in output.get("entities", {}).values()}


def extract_relations(output):
    """Extracts set of (src, tgt, relation) tuples from RadGraph output."""
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


class CompositeMetric:
    def __init__(self):
        scaler = StandardScaler(with_mean=True, with_std=True)
        # learnt parameters, infered from 
        # https://github.com/rajpurkarlab/CXR-Report-Metric/blob/main/CXRMetric/run_eval.py#L219
        scaler.mean_            = np.array([0.53792312, 0.61757256, 0.76479421, 0.44738335])
        scaler.scale_           = np.array([0.30282584, 0.22430938, 0.25394391, 0.29892717])
        scaler.var_             = np.array([0.09170349, 0.05031470, 0.06448751, 0.08935745])
        scaler.n_samples_seen_  = 160
        scaler.n_features_in_   = 4

        self.scaler = scaler
        self.coefs  = np.array([
                        -3.77083683e-01,   # radgraph weight
                        -3.70300100e-01,   # bertscore weight
                        -2.52616218e-01,   # s-emb weight
                        4.31504841e-12,   # bleu weight
                        2.46655256e-10    # intercept / bias
                    ])
        self.cols = ["radgraph", "bertscore", "semb_score", "bleu_score"]

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._radgraph = RadGraph(model_type='radgraph')
        self._chexbert = F1CheXbert(device=device)
        self._bleu = Bleu(n=2)
        self._bertscore_cache_key = None
        self._bert_scorer = None

    def _get_bert_scorer(self, refs):
        """Return a BERTScorer with IDF computed from *refs*."""
        cache_key = id(refs)
        if self._bert_scorer is None or self._bertscore_cache_key != cache_key:
            from bert_score import BERTScorer
            self._bert_scorer = BERTScorer(
                model_type='distilroberta-base',
                rescale_with_baseline=True,
                idf=True,
                idf_sents=refs,
                batch_size=64,
                nthreads=4,
                all_layers=False,
                lang='en',
                device=None,
            )
            self._bertscore_cache_key = cache_key
        return self._bert_scorer

    def _compute_sub_metrics(self, refs, hyps):
        scorer = self._get_bert_scorer(refs)
        _, _, f1 = scorer.score(cands=hyps, refs=refs, verbose=False, batch_size=64)
        bert_scores = f1.numpy()

        gt_outputs = self._radgraph(refs)
        pred_outputs = self._radgraph(hyps)
        rad_scores = []
        for i in range(len(refs)):
            gt_out = gt_outputs.get(str(i), {})
            pred_out = pred_outputs.get(str(i), {})
            ent_f1 = compute_f1(extract_entities(gt_out), extract_entities(pred_out))
            rel_f1 = compute_f1(extract_relations(gt_out), extract_relations(pred_out))
            rad_scores.append((ent_f1 + rel_f1) / 2)
        rad_scores = np.array(rad_scores)

        gt_embs = np.vstack(self._chexbert.get_embeddings(refs))
        pred_embs = np.vstack(self._chexbert.get_embeddings(hyps))
        dot = np.einsum("nd,nd->n", gt_embs, pred_embs)
        norms = np.linalg.norm(gt_embs, axis=1) * np.linalg.norm(pred_embs, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            sem_scores = np.where(norms > 0, dot / norms, 0.0)

        bleu_scores = self._bleu(refs, hyps)[1]

        return {
            'bertscore': bert_scores,
            'radgraph': rad_scores,
            'semb_score': sem_scores,
            'bleu_score': bleu_scores,
        }

    def _build_matrix(self, metrics: dict[str, np.ndarray]) -> np.ndarray:
        """Stack features in the canonical column order."""
        return np.column_stack([metrics[c] for c in self.cols])

    def predict(self, refs, hyps) -> np.ndarray:
        metrics = self._compute_sub_metrics(refs, hyps)
        X = self._build_matrix(metrics)
        Xn = self.scaler.transform(X)
        Xn = np.hstack([Xn, np.ones((Xn.shape[0], 1))])
        scores = Xn @ self.coefs
        return 1/scores.mean(), scores


if __name__ == "__main__":
    refs = [
        "No evidence of pneumothorax following chest tube removal.",
        "There is a left pleural effusion.",
        "There is a left pleural effusion."
    ]
    hyps = [
        "No pneumothorax detected.",
        "Left pleural effusion is present.",
        "No pneumothorax detected.",
    ]

    radcliq = CompositeMetric()
    mean_scores, detail_scores = radcliq.predict(refs, hyps)
    for i, s in enumerate(detail_scores, 1):
        print(f"Pair {i}: RadCliQ-v1 = {s:.4f}")
    
    print(f"RadCliQ-v1 score: {mean_scores:.4f}")
