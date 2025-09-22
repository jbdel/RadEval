from collections import defaultdict
import stanza
import warnings
import re
from nlg.rouge.rouge import Rouge
from nlg.bleu.bleu import Bleu
from nlg.bertscore.bertscore import BertScore
from radgraph import F1RadGraph
from factual.green import GREEN
from factual.RaTEScore import RaTEScore
from factual.f1temporal import F1Temporal
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
from factual.SRRBert.srr_bert import SRRBert, srr_bert_parse_sentences
from nlg.radevalbertscore import RadEvalBERTScorer
# Suppress UndefinedMetricWarning
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)




class RadEval():
    def __init__(self,
                 do_radgraph=False,
                 do_green=False,
                 do_bleu=False,
                 do_rouge=False,
                 do_bertscore=False,
                 do_srr_bert=False,
                 do_chexbert=False,
                 do_ratescore=False,
                 do_radcliq=False,
                 do_radeval_bertsore=False,
                 do_temporal=False,
                 ):
        super(RadEval, self).__init__()

        self.do_radgraph = do_radgraph
        self.do_green = do_green
        self.do_bleu = do_bleu
        self.do_rouge = do_rouge
        self.do_bertscore = do_bertscore
        self.do_srr_bert = do_srr_bert
        self.do_chexbert = do_chexbert
        self.do_ratescore = do_ratescore
        self.do_radcliq = do_radcliq
        self.do_temporal = do_temporal
        self.do_radeval_bertsore = do_radeval_bertsore

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
            self.green_scorer = GREEN("StanfordAIMI/GREEN-radllama2-7b", 
                                      output_dir=".")

        if self.do_rouge:
            self.rouge_scorers = {
                "rouge1": Rouge(rouges=["rouge1"]),
                "rouge2": Rouge(rouges=["rouge2"]),
                "rougeL": Rouge(rouges=["rougeL"])
            }
        if self.do_srr_bert:
            nltk.download('punkt_tab')
            self.srr_bert_scorer = SRRBert(model_type="leaves_with_statuses")
            

        if self.do_chexbert:
            self.chexbert_scorer = F1CheXbert()

        if self.do_ratescore:
            self.ratescore_scorer = RaTEScore()

        if self.do_radcliq:
            self.radcliq_scorer = CompositeMetric()

        if self.do_temporal:
            stanza.download('en', package='radiology', processors={'ner': 'radiology'})
            self.F1Temporal = F1Temporal

        if self.do_radeval_bertsore:
            self.radeval_bertsore = RadEvalBERTScorer(
                model_type="IAMJB/RadEvalModernBERT", 
                num_layers=22,
                use_fast_tokenizer=True,
                rescale_with_baseline=False)
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
        if self.do_srr_bert:
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
        if self.do_temporal:
            self.metric_keys.append("temporal_f1")
        if self.do_radeval_bertsore:
            self.metric_keys.append("radeval_bertsore")

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
            print("Computing radgraph scores")
            radgraph_scores = self.radgraph_scorer(refs=refs, hyps=hyps)
            radgraph_scores = radgraph_scores[0]
            scores["radgraph_simple"] = radgraph_scores[0]
            scores["radgraph_partial"] = radgraph_scores[1]
            scores["radgraph_complete"] = radgraph_scores[2]

        if self.do_bleu:
            print("Computing bleu scores")
            scores["bleu"] = self.bleu_scorer(refs, hyps)[0]

        if self.do_bertscore:
            print("Computing bertscore scores")
            scores["bertscore"] = self.bertscore_scorer(refs, hyps)[0]

        if self.do_green:
            print("Computing green scores")
            # Use the initialized green scorer
            mean, std, green_scores, summary, results_df = self.green_scorer(refs, hyps)
            scores["green"] = mean

        if self.do_rouge:
            print("Computing rouge scores")
            for key, scorer in self.rouge_scorers.items():
                scores[key] = scorer(refs, hyps)[0]

        if self.do_srr_bert:            
            print("Computing srr_bert scores")
            # Clean reports before tokenization
            parsed_refs = [srr_bert_parse_sentences(ref) for ref in refs]
            parsed_hyps = [srr_bert_parse_sentences(hyp) for hyp in hyps]

       
            section_level_hyps_pred = []
            section_level_refs_pred = []
            for parsed_hyp, parsed_ref in zip(parsed_hyps, parsed_refs):
                outputs, _ = self.srr_bert_scorer(sentences=parsed_ref + parsed_hyp)

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
            scores["srr_bert_weighted_f1"] = classification_dict["weighted avg"]["f1-score"]
            scores["srr_bert_weighted_precision"] = classification_dict["weighted avg"]["precision"]
            scores["srr_bert_weighted_recall"] = classification_dict["weighted avg"]["recall"]

       

        if self.do_chexbert:
            print("Computing chexbert scores")
            accuracy, accuracy_per_sample, chexbert_all, chexbert_5 = self.chexbert_scorer(hyps, refs)
            scores["chexbert-5_micro avg_f1-score"] = chexbert_5["micro avg"]["f1-score"]
            scores["chexbert-all_micro avg_f1-score"] = chexbert_all["micro avg"]["f1-score"]
            scores["chexbert-5_macro avg_f1-score"] = chexbert_5["macro avg"]["f1-score"]
            scores["chexbert-all_macro avg_f1-score"] = chexbert_all["macro avg"]["f1-score"]
            scores["chexbert-5_weighted_f1"] = chexbert_5["weighted avg"]["f1-score"]
            scores["chexbert-all_weighted_f1"] = chexbert_all["weighted avg"]["f1-score"]

        if self.do_ratescore:
            print("Computing ratescore scores")
            scores["ratescore"] = sum(self.ratescore_scorer.compute_score(refs, hyps)) / len(refs)

        if self.do_radcliq:
            print("Computing radcliq scores")
            scores["radcliq-v1"] = self.radcliq_scorer.predict(refs, hyps)[0]

        if self.do_temporal:
            print("Computing temporal scores")
            scores["temporal_f1"] = self.F1Temporal(predictions=hyps, references=refs)["f1"]

        if self.do_radeval_bertsore:
            print("Computing radeval_bertsore scores")
            scores["radeval_bertsore"] = self.radeval_bertsore.score(refs=refs, hyps=hyps)
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
                        do_srr_bert=True,
                        do_chexbert=True,
                        do_temporal=True,
                        do_ratescore=True,
                        do_radcliq=True,
                        do_radeval_bertsore=True)

    results = evaluator(refs=refs, hyps=hyps)
    print(json.dumps(results, indent=4))


if __name__ == '__main__':
    main()
