import math
import warnings
import logging
import os
import json
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    MofNCompleteColumn, TimeElapsedColumn,
)


class RadEval():
    def __init__(self,
                 do_radgraph=False,
                 do_green=False,
                 do_mammo_green=False,
                 mammo_green_model="gpt-4o-mini",
                 mammo_green_api_key=None,
                 do_bleu=False,
                 do_rouge=False,
                 do_bertscore=False,
                 do_srr_bert=False,
                 do_chexbert=False,
                 do_f1radbert_ct=False,
                 do_ratescore=False,
                 do_radcliq=False,
                 do_radeval_bertscore=False,
                 do_temporal=False,
                 do_details=False,
                 show_progress=True,
                 do_crimson=False,
                 crimson_api="hf",
                 crimson_api_key=None,
                 crimson_batch_size=1,
                 ):
        super(RadEval, self).__init__()

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        warnings.filterwarnings('ignore')
        logging.getLogger("RadEval").setLevel(logging.ERROR)

        self.do_radgraph = do_radgraph
        self.do_green = do_green
        self.do_crimson = do_crimson
        self.crimson_api = crimson_api
        self.crimson_api_key = crimson_api_key
        self.crimson_batch_size = crimson_batch_size
        self.do_mammo_green = do_mammo_green
        self.mammo_green_model = mammo_green_model
        self.mammo_green_api_key = mammo_green_api_key
        self.do_bleu = do_bleu
        self.do_rouge = do_rouge
        self.do_bertscore = do_bertscore
        self.do_srr_bert = do_srr_bert
        self.do_chexbert = do_chexbert
        self.do_f1radbert_ct = do_f1radbert_ct
        self.do_ratescore = do_ratescore
        self.do_radcliq = do_radcliq
        self.do_temporal = do_temporal
        self.do_radeval_bertscore = do_radeval_bertscore
        self.do_details = do_details
        self.show_progress = show_progress

        if self.do_radgraph:
            from radgraph import F1RadGraph
            self.radgraph_scorer = F1RadGraph(
                reward_level="all", model_type="radgraph-xl")
        if self.do_bleu:
            from .metrics.bleu.bleu import Bleu
            self.bleu_scorer = Bleu()
            self.bleu_scorer_1 = Bleu(n=1)
            self.bleu_scorer_2 = Bleu(n=2)
            self.bleu_scorer_3 = Bleu(n=3)
        if self.do_bertscore:
            from .metrics.bertscore.bertscore import BertScore
            self.bertscore_scorer = BertScore(model_type='distilbert-base-uncased',
                                              num_layers=5)
        if self.do_green:
            from .metrics.green_score import GREEN
            self.green_scorer = GREEN("StanfordAIMI/GREEN-radllama2-7b",
                                      output_dir=".")

        if self.do_crimson:
            from .metrics.crimson import CRIMSON
            self.crimson_scorer = CRIMSON(
                api=self.crimson_api,
                api_key=self.crimson_api_key,
                batch_size=self.crimson_batch_size,
            )

        if self.do_mammo_green:
            from .metrics.green_score import MammoGREEN
            self.mammo_green_scorer = MammoGREEN(
                model_name=self.mammo_green_model,
                api_key=self.mammo_green_api_key,
                output_dir="."
            )

        if self.do_rouge:
            from rouge_score import rouge_scorer
            self._rouge_types = ["rouge1", "rouge2", "rougeL"]
            self._rouge_scorer = rouge_scorer.RougeScorer(
                self._rouge_types, use_stemmer=True)

        if self.do_srr_bert:
            import nltk
            from .metrics.SRRBert.srr_bert import SRRBert
            nltk.download('punkt_tab', quiet=True)
            self.srr_bert_scorer = SRRBert(model_type="leaves_with_statuses")

        if self.do_chexbert:
            from .metrics.f1chexbert import F1CheXbert
            self.chexbert_scorer = F1CheXbert()

        if self.do_f1radbert_ct:
            from .metrics.f1Radbert_ct import F1RadbertCT
            self.f1radbert_ct_scorer = F1RadbertCT(
                model_id="IAMJB/RadBERT-CT",
                threshold=0.5,
                batch_size=16,
            )

        if self.do_ratescore:
            from .metrics.RaTEScore import RaTEScore
            self.ratescore_scorer = RaTEScore()

        if self.do_radcliq:
            from .metrics.RadCliQv1.radcliq import CompositeMetric
            self.radcliq_scorer = CompositeMetric()

        if self.do_temporal:
            import stanza
            from .metrics.f1temporal import F1Temporal
            stanza.download('en', package='radiology',
                            processors={'ner': 'radiology'})
            self.F1Temporal = F1Temporal

        if self.do_radeval_bertscore:
            from .metrics.radevalbertscore import RadEvalBERTScorer
            self.radeval_bertscore = RadEvalBERTScorer(
                model_type="IAMJB/RadEvalModernBERT",
                num_layers=22,
                use_fast_tokenizer=True,
                rescale_with_baseline=False)

        self.metric_keys = []
        if self.do_radgraph:
            self.metric_keys.extend(
                ["radgraph_simple", "radgraph_partial", "radgraph_complete"])
        if self.do_bleu:
            self.metric_keys.append("bleu")
        if self.do_green:
            self.metric_keys.append("green")
        if self.do_crimson:
            self.metric_keys.append("crimson")
        if self.do_mammo_green:
            self.metric_keys.append("mammo_green")
        if self.do_bertscore:
            self.metric_keys.append("bertscore")
        if self.do_rouge:
            self.metric_keys.extend(["rouge1", "rouge2", "rougeL"])
        if self.do_srr_bert:
            self.metric_keys.extend(
                ["samples_avg_precision", "samples_avg_recall", "samples_avg_f1-score"])
        if self.do_chexbert:
            self.metric_keys.extend([
                "chexbert-5_micro avg_f1-score",
                "chexbert-all_micro avg_f1-score",
                "chexbert-5_macro avg_f1-score",
                "chexbert-all_macro avg_f1-score"
            ])
        if self.do_f1radbert_ct:
            self.metric_keys.extend([
                "f1radbert_ct_accuracy",
                "f1radbert_ct_micro avg_f1-score",
                "f1radbert_ct_macro avg_f1-score",
                "f1radbert_ct_weighted_f1",
            ])

        if self.do_ratescore:
            self.metric_keys.append("ratescore")
        if self.do_radcliq:
            self.metric_keys.append("radcliqv1")
        if self.do_temporal:
            self.metric_keys.append("temporal_f1")
        if self.do_radeval_bertscore:
            self.metric_keys.append("radeval_bertscore")

    def __call__(self, refs, hyps, crimson_patient_contexts=None):
        if not (isinstance(hyps, list) and isinstance(refs, list)):
            raise TypeError("hyps and refs must be of type list")
        if len(hyps) != len(refs):
            raise ValueError("hyps and refs lists don't have the same size")
        if len(refs) == 0:
            return {}

        scores = self.compute_scores(refs=refs, hyps=hyps, crimson_patient_contexts=crimson_patient_contexts)
        return scores

    def _make_progress(self):
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            disable=not self.show_progress,
        )

    def _score_chexbert(self, scores, prefix, scorer, hyps, refs,
                        n_samples, progress, metric_task):
        display = prefix[0].upper() + prefix[1:]
        progress.update(metric_task, description=f"Computing {display}")
        n_batches = math.ceil(n_samples / scorer.batch_size) * 2
        batch_task = progress.add_task("  [dim]Batches", total=n_batches)
        _, _, cr_all, cr_5, sample_acc_full, sample_acc_5 = scorer.forward(
            hyps, refs, on_batch_done=lambda: progress.advance(batch_task))
        progress.remove_task(batch_task)
        if self.do_details:
            labels_5 = {k: v["f1-score"] for k, v in list(cr_5.items())[:-4]}
            labels_all = {k: v["f1-score"] for k, v in list(cr_all.items())[:-4]}
            scores[prefix] = {
                "sample_scores": {
                    "all_labels": sample_acc_full,
                    "5_labels": sample_acc_5,
                },
                f"{prefix}-5_micro avg_f1-score": cr_5["micro avg"]["f1-score"],
                f"{prefix}-all_micro avg_f1-score": cr_all["micro avg"]["f1-score"],
                f"{prefix}-5_macro avg_f1-score": cr_5["macro avg"]["f1-score"],
                f"{prefix}-all_macro avg_f1-score": cr_all["macro avg"]["f1-score"],
                f"{prefix}-5_weighted_f1": cr_5["weighted avg"]["f1-score"],
                f"{prefix}-all_weighted_f1": cr_all["weighted avg"]["f1-score"],
                "label_scores_f1-score": {
                    f"{prefix}-5": labels_5,
                    f"{prefix}_all": labels_all,
                },
            }
        else:
            scores[f"{prefix}-5_micro avg_f1-score"] = round(cr_5["micro avg"]["f1-score"], 4)
            scores[f"{prefix}-all_micro avg_f1-score"] = round(cr_all["micro avg"]["f1-score"], 4)
            scores[f"{prefix}-5_macro avg_f1-score"] = round(cr_5["macro avg"]["f1-score"], 4)
            scores[f"{prefix}-all_macro avg_f1-score"] = round(cr_all["macro avg"]["f1-score"], 4)
            scores[f"{prefix}-5_weighted_f1"] = round(cr_5["weighted avg"]["f1-score"], 4)
            scores[f"{prefix}-all_weighted_f1"] = round(cr_all["weighted avg"]["f1-score"], 4)
        progress.advance(metric_task)

    def compute_scores(self, refs, hyps, crimson_patient_contexts=None):
        scores = {}
        n_samples = len(refs)

        enabled = []
        if self.do_radgraph:        enabled.append("RadGraph")
        if self.do_bleu:            enabled.append("BLEU")
        if self.do_bertscore:       enabled.append("BERTScore")
        if self.do_green:           enabled.append("GREEN")
        if self.do_crimson:         enabled.append("CRIMSON")
        if self.do_mammo_green:     enabled.append("MammoGREEN")
        if self.do_rouge:           enabled.append("ROUGE")
        if self.do_srr_bert:        enabled.append("SRR-BERT")
        if self.do_chexbert:        enabled.append("CheXbert")
        if self.do_f1radbert_ct:    enabled.append("F1RadBERT-CT")

        if self.do_ratescore:       enabled.append("RaTEScore")
        if self.do_radcliq:         enabled.append("RadCliQ-v1")
        if self.do_temporal:        enabled.append("Temporal F1")
        if self.do_radeval_bertscore: enabled.append("RadEval-BERTScore")

        with self._make_progress() as progress:
            metric_task = progress.add_task(
                "Starting...", total=len(enabled))

            # ----------------------------------------------------------
            if self.do_radgraph:
                progress.update(metric_task, description="Computing RadGraph")
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
                    scores["radgraph_simple"] = round(radgraph_scores[0], 4)
                    scores["radgraph_partial"] = round(radgraph_scores[1], 4)
                    scores["radgraph_complete"] = round(radgraph_scores[2], 4)
                progress.advance(metric_task)

            # ----------------------------------------------------------
            if self.do_bleu:
                progress.update(metric_task, description="Computing BLEU")
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
                    scores["bleu"] = round(self.bleu_scorer(refs, hyps)[0], 4)
                progress.advance(metric_task)

            # ----------------------------------------------------------
            if self.do_bertscore:
                progress.update(metric_task, description="Computing BERTScore")
                n_batches = math.ceil(n_samples / self.bertscore_scorer.batch_size)
                batch_task = progress.add_task("  [dim]Batches", total=n_batches)
                cb = lambda: progress.advance(batch_task)
                if self.do_details:
                    bertscore_scores, sample_scores = self.bertscore_scorer(
                        refs, hyps, on_batch_done=cb)
                    scores["bertscore"] = {
                        "mean_score": bertscore_scores,
                        "sample_scores": sample_scores
                    }
                else:
                    scores["bertscore"] = round(
                        self.bertscore_scorer(refs, hyps, on_batch_done=cb)[0], 4)
                progress.remove_task(batch_task)
                progress.advance(metric_task)

            # ----------------------------------------------------------
            if self.do_green:
                progress.update(metric_task, description="Computing GREEN")
                mean, std, sample_scores, _ = self.green_scorer(refs, hyps)
                if self.do_details:
                    scores["green"] = {
                        "mean": mean,
                        "std": std,
                        "sample_scores": sample_scores,
                    }
                else:
                    scores["green"] = round(mean, 4)
                progress.advance(metric_task)

            # ----------------------------------------------------------
            if self.do_crimson:
                progress.update(metric_task, description="Computing CRIMSON")
                mean, std, sample_scores, results_df = self.crimson_scorer(refs, hyps, patient_contexts=crimson_patient_contexts)
                if self.do_details:
                    error_columns = [
                        "false_findings",
                        "missing_findings",
                        "attribute_errors",
                        "location_errors",
                        "severity_errors",
                        "descriptor_errors",
                        "measurement_errors",
                        "certainty_errors",
                        "unspecific_errors",
                        "overinterpretation_errors",
                        "temporal_errors",
                    ]
                    if len(results_df):
                        error_counts = results_df[error_columns].to_dict(orient="records")
                    else:
                        error_counts = []
                    scores["crimson"] = {
                        "mean": mean,
                        "std": std,
                        "sample_scores": sample_scores,
                        "error_counts": error_counts,
                    }
                else:
                    scores["crimson"] = round(mean, 4)
                progress.advance(metric_task)

            # ----------------------------------------------------------
            if self.do_mammo_green:
                progress.update(metric_task, description="Computing MammoGREEN")
                mean, std, sample_scores, results_df = self.mammo_green_scorer(
                    refs, hyps)
                if self.do_details:
                    scores["mammo_green"] = {
                        "mean": mean,
                        "std": std,
                        "sample_scores": sample_scores,
                        "error_counts": results_df[[
                            "matched_findings", "false_finding", "missing_finding",
                            "mischaracterization", "wrong_location_laterality",
                            "incorrect_birads", "insignificant_errors"
                        ]].to_dict(orient="records"),
                    }
                else:
                    scores["mammo_green"] = round(mean, 4)
                progress.advance(metric_task)

            # ----------------------------------------------------------
            if self.do_rouge:
                progress.update(metric_task, description="Computing ROUGE")
                raw_scores = [
                    self._rouge_scorer.score(ref, hyp)
                    for ref, hyp in zip(refs, hyps)
                ]
                if self.do_details:
                    rouge_results = {}
                    for rt in self._rouge_types:
                        f1s = [s[rt].fmeasure for s in raw_scores]
                        rouge_results[rt] = {
                            "mean_score": sum(f1s) / len(f1s),
                            "sample_scores": f1s,
                        }
                    scores["rouge"] = rouge_results
                else:
                    for rt in self._rouge_types:
                        f1s = [s[rt].fmeasure for s in raw_scores]
                        scores[rt] = round(sum(f1s) / len(f1s), 4)
                progress.advance(metric_task)

            # ----------------------------------------------------------
            if self.do_srr_bert:
                progress.update(metric_task, description="Computing SRR-BERT")
                sample_task = progress.add_task(
                    "  [dim]Samples", total=n_samples)

                classification_dict, sample_precision, sample_recall, sample_f1 = \
                    self.srr_bert_scorer.evaluate(
                        refs, hyps,
                        on_sample_done=lambda: progress.advance(sample_task),
                    )
                progress.remove_task(sample_task)

                if self.do_details:
                    label_names = [
                        label for label, idx in sorted(
                            self.srr_bert_scorer.mapping.items(),
                            key=lambda x: x[1])
                    ]
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
                    scores["srr_bert_weighted_f1"] = round(
                        classification_dict["weighted avg"]["f1-score"], 4)
                    scores["srr_bert_weighted_precision"] = round(
                        classification_dict["weighted avg"]["precision"], 4)
                    scores["srr_bert_weighted_recall"] = round(
                        classification_dict["weighted avg"]["recall"], 4)
                progress.advance(metric_task)

            # ----------------------------------------------------------
            if self.do_chexbert:
                self._score_chexbert(
                    scores, "chexbert", self.chexbert_scorer,
                    hyps, refs, n_samples, progress, metric_task)

            # ----------------------------------------------------------
            if self.do_f1radbert_ct:
                progress.update(metric_task, description="Computing F1RadBERT-CT")
                n_batches = math.ceil(n_samples / self.f1radbert_ct_scorer.batch_size) * 2
                batch_task = progress.add_task("  [dim]Batches", total=n_batches)
                f1radbert_ct_accuracy, f1radbert_ct_sample_acc, f1radbert_ct_report = self.f1radbert_ct_scorer(
                    hyps, refs, on_batch_done=lambda: progress.advance(batch_task))
                progress.remove_task(batch_task)
                if self.do_details:
                    f1radbert_ct_labels = {
                        k: v["f1-score"]
                        for k, v in list(f1radbert_ct_report.items())[:-4]
                    }
                    scores["f1radbert_ct"] = {
                        "f1radbert_ct_accuracy": f1radbert_ct_accuracy,
                        "f1radbert_ct_micro avg_f1-score": f1radbert_ct_report["micro avg"]["f1-score"],
                        "f1radbert_ct_macro avg_f1-score": f1radbert_ct_report["macro avg"]["f1-score"],
                        "f1radbert_ct_weighted_f1": f1radbert_ct_report["weighted avg"]["f1-score"],
                        "sample_scores": {
                            "all_labels": f1radbert_ct_sample_acc,
                        },
                        "label_scores_f1-score": f1radbert_ct_labels,
                    }
                else:
                    scores["f1radbert_ct_accuracy"] = round(
                        f1radbert_ct_accuracy, 4)
                    scores["f1radbert_ct_micro avg_f1-score"] = round(
                        f1radbert_ct_report["micro avg"]["f1-score"], 4)
                    scores["f1radbert_ct_macro avg_f1-score"] = round(
                        f1radbert_ct_report["macro avg"]["f1-score"], 4)
                    scores["f1radbert_ct_weighted_f1"] = round(
                        f1radbert_ct_report["weighted avg"]["f1-score"], 4)
                progress.advance(metric_task)

            # ----------------------------------------------------------
            if self.do_ratescore:
                progress.update(metric_task, description="Computing RaTEScore")
                sample_task = progress.add_task(
                    "  [dim]Samples", total=n_samples)

                rate_score, pred_pairs_raw, gt_pairs_raw = self.ratescore_scorer.compute_score(
                    candidate_list=hyps, reference_list=refs,
                    on_sample_done=lambda: progress.advance(sample_task),
                )
                progress.remove_task(sample_task)

                f1_ratescore = sum(rate_score) / len(rate_score)
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
                    scores["ratescore"] = round(f1_ratescore, 4)
                progress.advance(metric_task)

            # ----------------------------------------------------------
            if self.do_radcliq:
                progress.update(metric_task, description="Computing RadCliQ-v1")
                mean_scores, detail_scores = self.radcliq_scorer.predict(
                    refs, hyps)
                if self.do_details:
                    scores["radcliq-v1"] = {
                        "mean_score": mean_scores,
                        "sample_scores": detail_scores.tolist()
                    }
                else:
                    scores["radcliq-v1"] = round(mean_scores, 4)
                progress.advance(metric_task)

            # ----------------------------------------------------------
            if self.do_temporal:
                progress.update(metric_task, description="Computing Temporal F1")
                sample_task = progress.add_task(
                    "  [dim]Samples", total=n_samples)

                temporal_scores = self.F1Temporal(
                    predictions=hyps, references=refs,
                    on_sample_done=lambda: progress.advance(sample_task),
                )
                progress.remove_task(sample_task)

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
                    scores["temporal_f1"] = round(temporal_scores["f1"], 4)
                progress.advance(metric_task)

            # ----------------------------------------------------------
            if self.do_radeval_bertscore:
                progress.update(metric_task, description="Computing RadEval-BERTScore")
                n_batches = math.ceil(n_samples / self.radeval_bertscore.batch_size)
                batch_task = progress.add_task("  [dim]Batches", total=n_batches)
                radeval_bertscores = self.radeval_bertscore.score(
                    refs=refs, hyps=hyps,
                    on_batch_done=lambda: progress.advance(batch_task))
                progress.remove_task(batch_task)
                if self.do_details:
                    scores["radeval_bertscore"] = {
                        "mean_score": radeval_bertscores[0],
                        "sample_scores": radeval_bertscores[1].tolist()
                    }
                else:
                    scores["radeval_bertscore"] = round(radeval_bertscores[0], 4)
                progress.advance(metric_task)

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
