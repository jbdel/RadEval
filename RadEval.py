from collections import defaultdict
import warnings
import re
from nlg.rouge.rouge import Rouge
from nlg.bleu.bleu import Bleu
from nlg.bertscore.bertscore import BertScore
from radgraph import F1RadGraph
from factual.StruxtBert import StruxtBert
from factual.constants import leaves_mapping
from torch import nn
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.exceptions import UndefinedMetricWarning
import json
from f1chexbert import F1CheXbert

# Suppress UndefinedMetricWarning
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)


class RadEval():
    def __init__(self,
                 do_radgraph=True,
                 do_green=True,
                 do_bleu=True,
                 do_rouge=True,
                 do_bertscore=True,
                 do_diseases=True,
                 do_chexbert=True
                 ):
        super(RadEval, self).__init__()

        self.do_radgraph = do_radgraph
        self.do_green = do_green
        self.do_bleu = do_bleu
        self.do_rouge = do_rouge
        self.do_bertscore = do_bertscore
        self.do_diseases = do_diseases
        self.do_chexbert = do_chexbert

        # Initialize scorers only once
        if self.do_radgraph:
            self.radgraph_scorer = F1RadGraph(reward_level="all", model_type="radgraph-xl")
        if self.do_bleu:
            self.bleu_scorer = Bleu()
        if self.do_bertscore:
            self.bertscore_scorer = BertScore(model_type='distilbert-base-uncased',
                                              num_layers=5)
        if self.do_green:
            # Initialize green scorer here if needed
            pass
        if self.do_rouge:
            self.rouge_scorers = {
                "rouge1": Rouge(rouges=["rouge1"]),
                "rouge2": Rouge(rouges=["rouge2"]),
                "rougeL": Rouge(rouges=["rougeL"])
            }
        if self.do_diseases:
            model = "StanfordAIMI/CXR-BERT-Leaves-Diseases-Only"
            self.diseases_model = StruxtBert(model_id_or_path=model, mapping=leaves_mapping)

        if self.do_chexbert:
            self.chexbert_scorer = F1CheXbert()

        # Store the metric keys
        self.metric_keys = []
        if self.do_radgraph:
            self.metric_keys.extend(["radgraph_simple", "radgraph_partial", "radgraph_complete"])
        if self.do_bleu:
            self.metric_keys.append("bleu")
        if self.do_green:
            self.metric_keys.append("green")
        if self.do_bertscore:
            self.metric_keys.append("bertscore")
        if self.do_rouge:
            self.metric_keys.extend(self.rouge_scorers.keys())
        if self.do_diseases:
            self.metric_keys.extend(["samples_avg_precision", "samples_avg_recall", "samples_avg_f1-score"])

        if self.do_chexbert:
            self.metric_keys.extend([
                "chexbert-5_micro avg_f1-score",
                "chexbert-all_micro avg_f1-score",
                "chexbert-5_macro avg_f1-score",
                "chexbert-all_macro avg_f1-score"
            ])

    def __call__(self, refs, hyps):
        if not (isinstance(hyps, list) and isinstance(refs, list)):
            raise TypeError("hyps and refs must be of type list")
        if len(hyps) != len(refs):
            raise ValueError("hyps and refs lists don't have the same size")

        scores = self.compute_scores(refs=refs, hyps=hyps)
        return scores

    def compute_scores(self, refs, hyps):
        if not (isinstance(hyps, list) and isinstance(refs, list)):
            raise TypeError("hyps and refs must be of type list")
        if len(hyps) != len(refs):
            raise ValueError("hyps and refs lists don't have the same size")

        scores = {}
        if self.do_radgraph:
            radgraph_scores = self.radgraph_scorer(refs=refs, hyps=hyps)
            radgraph_scores = radgraph_scores[0]
            scores["radgraph_simple"] = radgraph_scores[0]
            scores["radgraph_partial"] = radgraph_scores[1]
            scores["radgraph_complete"] = radgraph_scores[2]

        if self.do_bleu:
            scores["bleu"] = self.bleu_scorer(refs, hyps)[0]

        if self.do_bertscore:
            scores["bertscore"] = self.bertscore_scorer(refs, hyps)[0]

        if self.do_green:
            # Compute green score here if needed
            pass

        if self.do_rouge:
            for key, scorer in self.rouge_scorers.items():
                scores[key] = scorer(refs, hyps)[0]

        if self.do_diseases:
            outputs, _ = self.diseases_model(sentences=refs + hyps)

            refs_preds = outputs[:len(refs)]
            hyps_preds = outputs[len(refs):]

            classification_dict = classification_report(refs_preds, hyps_preds, output_dict=True)
            scores["samples_avg_precision"] = classification_dict["samples avg"]["precision"]
            scores["samples_avg_recall"] = classification_dict["samples avg"]["recall"]
            scores["samples_avg_f1-score"] = classification_dict["samples avg"]["f1-score"]

        if self.do_chexbert:
            accuracy, accuracy_per_sample, chexbert_all, chexbert_5 = self.chexbert_scorer(hyps, refs)
            scores["chexbert-5_micro avg_f1-score"] = chexbert_5["micro avg"]["f1-score"]
            scores["chexbert-all_micro avg_f1-score"] = chexbert_all["micro avg"]["f1-score"]
            scores["chexbert-5_macro avg_f1-score"] = chexbert_5["macro avg"]["f1-score"]
            scores["chexbert-all_macro avg_f1-score"] = chexbert_all["macro avg"]["f1-score"]

        return scores


def main():
    refs = [
        "No acute cardiopulmonary process.",
        "No radiographic findings to suggest pneumonia.",
        "1.Status post median sternotomy for CABG with stable cardiac enlargement and calcification of the aorta consistent with atherosclerosis.Relatively lower lung volumes with no focal airspace consolidation appreciated.Crowding of the pulmonary vasculature with possible minimal perihilar edema, but no overt pulmonary edema.No pleural effusions or pneumothoraces.",
        "1. Left PICC tip appears to terminate in the distal left brachiocephalic vein.2. Mild pulmonary vascular congestion.3. Interval improvement in aeration of the lung bases with residual streaky opacity likely reflective of atelectasis.Interval resolution of the left pleural effusion.",
        "No definite acute cardiopulmonary process.Enlarged cardiac silhouette could be accentuated by patient's positioning.",
        "Increased mild pulmonary edema and left basal atelectasis.",
    ]

    hyps = [
        "No acute cardiopulmonary process.",
        "No radiographic findings to suggest pneumonia.",
        "Status post median sternotomy for CABG with stable cardiac enlargement and calcification of the aorta consistent with atherosclerosis.",
        "Relatively lower lung volumes with no focal airspace consolidation appreciated.",
        "Crowding of the pulmonary vasculature with possible minimal perihilar edema, but no overt pulmonary edema.",
        "No pleural effusions or pneumothoraces.",
    ]

    evaluator = RadEval(do_radgraph=True,
                        do_green=False,
                        do_bleu=True,
                        do_rouge=True,
                        do_bertscore=True,
                        do_diseases=False,
                        do_chexbert=True)

    results = evaluator(refs=refs, hyps=hyps)
    print(json.dumps(results, indent=4))


if __name__ == '__main__':
    main()
