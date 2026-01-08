from collections import defaultdict
import stanza
import warnings
import logging
import os
import re
from .nlg.rouge.rouge import Rouge
from .nlg.bleu.bleu import Bleu
from .nlg.bertscore.bertscore import BertScore
from radgraph import F1RadGraph
from .factual.green_score import GREEN
from .factual.RaTEScore import RaTEScore
from .factual.f1temporal import F1Temporal
from torch import nn
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.exceptions import UndefinedMetricWarning
import json
from .factual.f1chexbert import F1CheXbert
import nltk
from .utils import clean_numbered_list
from .utils import multilabel_prf_per_sample
from .factual.RadCliQv1.radcliq import CompositeMetric
from .factual.SRRBert.srr_bert import SRRBert, srr_bert_parse_sentences
from .nlg.radevalbertscore import RadEvalBERTScorer
# Suppress Warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.ERROR)




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
                 do_radeval_bertscore=False,
                 do_temporal=False,
                 do_details=False,
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
        self.do_radeval_bertscore = do_radeval_bertscore
        self.do_details = do_details

        # Initialize scorers only once
        if self.do_radgraph:
            self.radgraph_scorer = F1RadGraph(reward_level="all", model_type="radgraph-xl")
        if self.do_bleu:
            self.bleu_scorer = Bleu()
            self.bleu_scorer_1 = Bleu(n=1)
            self.bleu_scorer_2 = Bleu(n=2)
            self.bleu_scorer_3 = Bleu(n=3)
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
            nltk.download('punkt_tab', quiet=True)
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

        if self.do_radeval_bertscore:
            self.radeval_bertscore = RadEvalBERTScorer(
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
        if self.do_radeval_bertscore:
            self.metric_keys.append("radeval_bertscore")

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

            if self.do_details:
                f1_scores = radgraph_scores[0]
                individual_scores = radgraph_scores[1]
                hyps_entities = radgraph_scores[2]
                refs_entities = radgraph_scores[3]

                scores["radgraph"] = {
                    "radgraph_simple": f1_scores[0],
                    "radgraph_partial": f1_scores[1], 
                    "radgraph_complete": f1_scores[2],
                    "sample_scores": individual_scores,
                    "hypothesis_annotation_lists": hyps_entities,
                    "reference_annotation_lists": refs_entities
                }

            else:
                radgraph_scores = radgraph_scores[0]
                scores["radgraph_simple"] = round(radgraph_scores[0],2)
                scores["radgraph_partial"] = round(radgraph_scores[1],2)
                scores["radgraph_complete"] = round(radgraph_scores[2],2)

        if self.do_bleu:
            if self.do_details:
                bleu_1_score, bleu_1_samples = self.bleu_scorer_1(refs, hyps)
                bleu_2_score, bleu_2_samples = self.bleu_scorer_2(refs, hyps)
                bleu_3_score, bleu_3_samples = self.bleu_scorer_3(refs, hyps)
                bleu_4_score, bleu_4_samples = self.bleu_scorer(refs, hyps)
                
                scores["bleu"] = {
                    "bleu_1": {"mean_score": bleu_1_score, "sample_scores": bleu_1_samples},
                    "bleu_2": {"mean_score": bleu_2_score, "sample_scores": bleu_2_samples},
                    "bleu_3": {"mean_score": bleu_3_score, "sample_scores": bleu_3_samples},
                    "bleu_4": {"mean_score": bleu_4_score, "sample_scores": bleu_4_samples}
                }
            else:
                scores["bleu"] = round(self.bleu_scorer(refs, hyps)[0], 2)

        if self.do_bertscore:
            if self.do_details:
                bertscore_scores, sample_scores = self.bertscore_scorer(refs, hyps)
                scores["bertscore"] = {
                    "mean_score": bertscore_scores,
                    "sample_scores": sample_scores
                }
            else:
                scores["bertscore"] = round(self.bertscore_scorer(refs, hyps)[0], 2)

        if self.do_green:
            # Use the initialized green scorer
            mean, std, sample_scores, summary, _ = self.green_scorer(refs, hyps)
            if self.do_details:
                scores["green"] = {
                    "mean": mean,
                    "std": std,
                    "sample_scores": sample_scores,
                    "summary": summary
                }
            else:
                scores["green"] = round(mean, 2)

        if self.do_rouge:
            if self.do_details:
                rouge_scores = {}
                for key, scorer in self.rouge_scorers.items():
                    mean, sample_scores  = scorer(refs, hyps)
                    rouge_scores[key] = {
                        "mean_score": mean,
                        "sample_scores": sample_scores
                    }

                scores["rouge"] = rouge_scores
            else:
                for key, scorer in self.rouge_scorers.items():
                    scores[key] = round(scorer(refs, hyps)[0], 2)

        if self.do_srr_bert:            
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

            label_names = [label for label, idx in sorted(self.srr_bert_scorer.mapping.items(), key=lambda x: x[1])]
            classification_dict = classification_report(section_level_refs_pred,
                                                        section_level_hyps_pred,
                                                        target_names=label_names,
                                                        output_dict=True,
                                                        zero_division=0)
            
            sample_precision, sample_recall, sample_f1 = multilabel_prf_per_sample(
                section_level_refs_pred,
                section_level_hyps_pred
            )
            
            if self.do_details:
                label_scores = {}
                for label in label_names:
                    if label in classification_dict:
                        f1 = classification_dict[label]["f1-score"]
                        support = classification_dict[label]["support"]
                        if f1 > 0 or support > 0:
                            label_scores[label] = {
                                "f1-score": f1,
                                "precision": classification_dict[label]["precision"],
                                "recall": classification_dict[label]["recall"],
                                "support": support
                            }

                scores["srr_bert"] = {
                    "srr_bert_weighted_f1": {
                        "weighted_mean_score": classification_dict["weighted avg"]["f1-score"],
                        "sample_scores": sample_f1.tolist(),
                    },
                    "srr_bert_weighted_precision": {
                        "weighted_mean_score": classification_dict["weighted avg"]["precision"],
                        "sample_scores": sample_precision.tolist(),
                    },
                    "srr_bert_weighted_recall": {
                        "weighted_mean_score": classification_dict["weighted avg"]["recall"],
                        "sample_scores": sample_recall.tolist(),
                    },
                    "label_scores": label_scores
                }
            else:
                scores["srr_bert_weighted_f1"] = round(classification_dict["weighted avg"]["f1-score"], 2)
                scores["srr_bert_weighted_precision"] = round(classification_dict["weighted avg"]["precision"], 2)
                scores["srr_bert_weighted_recall"] = round(classification_dict["weighted avg"]["recall"], 2)

       

        if self.do_chexbert:
            accuracy, accuracy_per_sample, chexbert_all, chexbert_5 = self.chexbert_scorer(hyps, refs)
            if self.do_details:
                chexbert_5_labels = {
                    k: v["f1-score"]
                    for k, v in list(chexbert_5.items())[:-4]
                }

                chexbert_all_labels = {
                    k: v["f1-score"]
                    for k, v in list(chexbert_all.items())[:-4]
                }

                scores["chexbert"] = {
                    "chexbert-5_micro avg_f1-score": chexbert_5["micro avg"]["f1-score"],
                    "chexbert-all_micro avg_f1-score": chexbert_all["micro avg"]["f1-score"],
                    "chexbert-5_macro avg_f1-score": chexbert_5["macro avg"]["f1-score"],
                    "chexbert-all_macro avg_f1-score": chexbert_all["macro avg"]["f1-score"],
                    "chexbert-5_weighted_f1": chexbert_5["weighted avg"]["f1-score"],
                    "chexbert-all_weighted_f1": chexbert_all["weighted avg"]["f1-score"],
                    "chexbert_accuracy": {"mean_score": accuracy, "sample_scores": accuracy_per_sample},
                    "label_scores_f1-score": {
                        "chexbert-5": chexbert_5_labels,
                        "chexbert_all": chexbert_all_labels
                    }
                }
            else:
                scores["chexbert-5_micro avg_f1-score"] = round(chexbert_5["micro avg"]["f1-score"], 2)
                scores["chexbert-all_micro avg_f1-score"] = round(chexbert_all["micro avg"]["f1-score"], 2)
                scores["chexbert-5_macro avg_f1-score"] = round(chexbert_5["macro avg"]["f1-score"], 2)
                scores["chexbert-all_macro avg_f1-score"] = round(chexbert_all["macro avg"]["f1-score"], 2)
                scores["chexbert-5_weighted_f1"] = round(chexbert_5["weighted avg"]["f1-score"], 2)
                scores["chexbert-all_weighted_f1"] = round(chexbert_all["weighted avg"]["f1-score"], 2)

        if self.do_ratescore:
            rate_score, pred_pairs_raw ,gt_pairs_raw = self.ratescore_scorer.compute_score(candidate_list=hyps, reference_list=refs)
            f1_ratescore = float(np.mean(rate_score))
            if self.do_details:
                pred_pairs = [
                    {ent: label for ent, label in sample}
                    for sample in pred_pairs_raw
                ]
                gt_pairs = [
                    {ent: label for ent, label in sample}
                    for sample in gt_pairs_raw
                ]
                scores["ratescore"] = {
                    "f1-score": f1_ratescore,
                    "sample_scores": rate_score,
                    "hyps_pairs": pred_pairs,
                    "refs_pairs": gt_pairs
                }
            else:
                scores["ratescore"] = round(f1_ratescore, 2)

        if self.do_radcliq:
            mean_scores, detail_scores = self.radcliq_scorer.predict(refs, hyps)
            if self.do_details:
                scores["radcliq-v1"] = {
                    "mean_score": mean_scores,
                    "sample_scores": detail_scores.tolist()
                }
            else:
                scores["radcliq-v1"] = round(mean_scores, 2)

        if self.do_temporal:
            temporal_scores = self.F1Temporal(predictions=hyps, references=refs)
            if self.do_details:
                hyp_entities = [
                    sorted(list(group)) if group else []
                    for group in temporal_scores.get("prediction_entities", [])
                ]
                ref_entities = [
                    sorted(list(group)) if group else []
                    for group in temporal_scores.get("reference_entities", [])
                ]
                scores["temporal_f1"] = {
                    "f1-score": temporal_scores["f1"],
                    "sample_scores": temporal_scores["sample_scores"],
                    "hyps_entities": hyp_entities,
                    "refs_entities": ref_entities
                }
            else:
                scores["temporal_f1"] = round(temporal_scores["f1"], 2)

        if self.do_radeval_bertscore:
            radeval_bertscores = self.radeval_bertscore.score(refs=refs, hyps=hyps)
            if self.do_details:
                scores["radeval_bertscore"] = {
                    "f1-score": radeval_bertscores[0],
                    "sample_scores": radeval_bertscores[1].tolist()
                }
            else:
                scores["radeval_bertscore"] = round(radeval_bertscores[0], 2)

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
                        do_radeval_bertscore=True)

    results = evaluator(refs=refs, hyps=hyps)
    print(json.dumps(results, indent=4))


if __name__ == '__main__':
    main()
