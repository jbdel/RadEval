from collections import defaultdict
import warnings
import re
from nlg.rouge.rouge import Rouge
from nlg.bleu.bleu import Bleu
from nlg.bertscore.bertscore import BertScore
from radgraph import F1RadGraph
# from factual.StructBert import StructBert
# from factual.constants import leaves_mapping
from factual.RaTEScore import RaTEScore
from torch import nn
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.exceptions import UndefinedMetricWarning
import json
from factual.f1chexbert import F1CheXbert
import nltk
from utils import clean_numbered_list
from factual.RadCliQv1.radcliq import CompositeMetric
# Suppress UndefinedMetricWarning
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)




class RadEval():
    def __init__(self,
                 do_radgraph=False,
                 do_green=False,
                 do_bleu=False,
                 do_rouge=False,
                 do_bertscore=False,
                 do_diseases=False,
                 do_chexbert=False,
                 do_ratescore=False,
                 do_radcliq=False,
                 ):
        super(RadEval, self).__init__()

        self.do_radgraph = do_radgraph
        self.do_green = do_green
        self.do_bleu = do_bleu
        self.do_rouge = do_rouge
        self.do_bertscore = do_bertscore
        self.do_diseases = do_diseases
        self.do_chexbert = do_chexbert
        self.do_ratescore = do_ratescore
        self.do_radcliq = do_radcliq

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
            # nltk.download('punkt_tab')
            # model = "StanfordAIMI/CXR-BERT-Leaves-Diseases-Only"
            # self.diseases_model = StructBert(model_id_or_path=model, mapping=leaves_mapping)
            pass

        if self.do_chexbert:
            self.chexbert_scorer = F1CheXbert()

        if self.do_ratescore:
            self.ratescore_scorer = RaTEScore()

        if self.do_radcliq:
            self.radcliq_scorer = CompositeMetric()

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

        if self.do_ratescore:
            self.metric_keys.append("ratescore")
        if self.do_radcliq:
            self.metric_keys.append("radcliqv1")

    def __call__(self, refs, hyps):
        if not (isinstance(hyps, list) and isinstance(refs, list)):
            raise TypeError("hyps and refs must be of type list")
        if len(hyps) != len(refs):
            raise ValueError("hyps and refs lists don't have the same size")
        if len(refs) == 0:
            return {}
        
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
            # Clean reports before tokenization
            parsed_refs = [clean_numbered_list(ref) for ref in refs]
            parsed_hyps = [clean_numbered_list(hyp) for hyp in hyps]
      
       
            section_level_hyps_pred = []
            section_level_refs_pred = []
            for parsed_hyp, parsed_ref in zip(parsed_hyps, parsed_refs):
                outputs, _ = self.diseases_model(sentences=parsed_ref + parsed_hyp)

                refs_preds = outputs[:len(parsed_ref)]
                hyps_preds = outputs[len(parsed_ref):]

                merged_refs_preds = np.any(refs_preds, axis=0).astype(int)
                merged_hyps_preds = np.any(hyps_preds, axis=0).astype(int)

                section_level_hyps_pred.append(merged_hyps_preds)
                section_level_refs_pred.append(merged_refs_preds)

            classification_dict = classification_report(section_level_refs_pred,
                                                        section_level_hyps_pred,
                                                        output_dict=True,
                                                        zero_division=0)
            scores["samples_avg_precision"] = classification_dict["samples avg"]["precision"]
            scores["samples_avg_recall"] = classification_dict["samples avg"]["recall"]
            scores["samples_avg_f1-score"] = classification_dict["samples avg"]["f1-score"]
        
       

        if self.do_chexbert:
            accuracy, accuracy_per_sample, chexbert_all, chexbert_5 = self.chexbert_scorer(hyps, refs)
            scores["chexbert-5_micro avg_f1-score"] = chexbert_5["micro avg"]["f1-score"]
            scores["chexbert-all_micro avg_f1-score"] = chexbert_all["micro avg"]["f1-score"]
            scores["chexbert-5_macro avg_f1-score"] = chexbert_5["macro avg"]["f1-score"]
            scores["chexbert-all_macro avg_f1-score"] = chexbert_all["macro avg"]["f1-score"]

        if self.do_ratescore:
            scores["ratescore"] = sum(self.ratescore_scorer.compute_score(refs, hyps)) / len(refs)

        if self.do_radcliq:
            scores["radcliqv1"] = self.radcliq_scorer.predict(refs, hyps)[0]

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
