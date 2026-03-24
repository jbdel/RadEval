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
                 openai_api_key=None,
                 gemini_api_key=None,
                 do_radgraph=False,
                 do_green=False,
                 do_mammo_green=False,
                 mammo_green_model="gpt-4o-mini",
                 mammo_green_max_concurrent=50,
                 do_bleu=False,
                 do_rouge=False,
                 do_bertscore=False,
                 do_srrbert=False,
                 do_f1chexbert=False,
                 do_f1radbert_ct=False,
                 do_ratescore=False,
                 do_radgraph_radcliq=False,
                 do_radcliq=False,
                 do_radeval_bertscore=False,
                 do_temporal=False,
                 do_radfact_ct=False,
                 radfact_ct_model="gpt-4o-mini",
                 radfact_ct_filter_negatives=False,
                 radfact_ct_max_concurrent=50,
                 do_crimson=False,
                 crimson_api="hf",
                 crimson_model=None,
                 crimson_batch_size=1,
                 crimson_max_concurrent=50,
                 hoppr_crimson_ct_api="openai",
                 hoppr_crimson_ct_model=None,
                 do_per_sample=False,
                 do_details=False,
                 show_progress=True,
                 ):
        super(RadEval, self).__init__()

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        warnings.filterwarnings('ignore')
        logging.getLogger("RadEval").setLevel(logging.ERROR)

        self.do_radgraph = do_radgraph
        self.do_green = do_green
        self.do_mammo_green = do_mammo_green
        self.mammo_green_model = mammo_green_model
        self.mammo_green_max_concurrent = mammo_green_max_concurrent
        self.openai_api_key = openai_api_key
        self.gemini_api_key = gemini_api_key
        self.do_bleu = do_bleu
        self.do_rouge = do_rouge
        self.do_bertscore = do_bertscore
        self.do_srrbert = do_srrbert
        self.do_f1chexbert = do_f1chexbert
        self.do_f1radbert_ct = do_f1radbert_ct
        self.do_ratescore = do_ratescore
        self.do_radgraph_radcliq = do_radgraph_radcliq
        self.do_radcliq = do_radcliq
        self.do_temporal = do_temporal
        self.do_radfact_ct = do_radfact_ct
        self.radfact_ct_model = radfact_ct_model
        self.radfact_ct_filter_negatives = radfact_ct_filter_negatives
        self.radfact_ct_max_concurrent = radfact_ct_max_concurrent
        self.do_crimson = do_crimson
        self.crimson_api = crimson_api
        self.crimson_model = crimson_model
        self.crimson_batch_size = crimson_batch_size
        self.crimson_max_concurrent = crimson_max_concurrent
        self.hoppr_crimson_ct_api = hoppr_crimson_ct_api
        self.hoppr_crimson_ct_model = hoppr_crimson_ct_model
        self.do_radeval_bertscore = do_radeval_bertscore
        self.do_per_sample = do_per_sample
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

        if self.do_mammo_green:
            from .metrics.green_score import MammoGREEN
            self.mammo_green_scorer = MammoGREEN(
                model_name=self.mammo_green_model,
                openai_api_key=self.openai_api_key,
                gemini_api_key=self.gemini_api_key,
                max_concurrent=self.mammo_green_max_concurrent,
                output_dir="."
            )

        if self.do_rouge:
            from rouge_score import rouge_scorer
            self._rouge_types = ["rouge1", "rouge2", "rougeL"]
            self._rouge_scorer = rouge_scorer.RougeScorer(
                self._rouge_types, use_stemmer=True)

        if self.do_srrbert:
            import nltk
            from .metrics.SRRBert.srr_bert import SRRBert
            nltk.download('punkt_tab', quiet=True)
            self.srrbert_scorer = SRRBert(model_type="leaves_with_statuses")

        if self.do_f1chexbert:
            from .metrics.f1chexbert import F1CheXbert
            self.f1chexbert_scorer = F1CheXbert()

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

        if self.do_radgraph_radcliq:
            from .metrics.radgraph_radcliq import RadGraphRadCliQ
            self.radgraph_radcliq_scorer = RadGraphRadCliQ()

        if self.do_radcliq:
            from .metrics.RadCliQv1.radcliq import CompositeMetric
            self.radcliq_scorer = CompositeMetric()

        if self.do_temporal:
            import stanza
            from .metrics.f1temporal import F1Temporal
            stanza.download('en', package='radiology',
                            processors={'ner': 'radiology'})
            self.F1Temporal = F1Temporal

        if self.do_radfact_ct:
            try:
                from .metrics.radfact_ct import RadFactCT
                if RadFactCT is None:
                    raise ImportError("RadFactCT is not available")
                self.radfact_ct_scorer = RadFactCT(
                    model_name=self.radfact_ct_model,
                    openai_api_key=self.openai_api_key,
                    filter_negatives=self.radfact_ct_filter_negatives,
                    max_concurrent=self.radfact_ct_max_concurrent,
                )
            except (ImportError, EnvironmentError) as e:
                warnings.warn(
                    f"RadFactCT unavailable ({e}); disabling do_radfact_ct.")
                self.do_radfact_ct = False

        if self.do_crimson:
            try:
                from .metrics.crimson import CRIMSON
                if CRIMSON is None:
                    raise ImportError("CRIMSON is not available")
                self.crimson_scorer = CRIMSON(
                    provider=self.crimson_api,
                    model_name=self.crimson_model,
                    openai_api_key=self.openai_api_key,
                    gemini_api_key=self.gemini_api_key,
                    batch_size=self.crimson_batch_size,
                    max_concurrent=self.crimson_max_concurrent,
                )
            except (ImportError, EnvironmentError, OSError) as e:
                warnings.warn(
                    f"CRIMSON unavailable ({e}); disabling do_crimson.")
                self.do_crimson = False


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
        if self.do_mammo_green:
            self.metric_keys.append("mammo_green")
        if self.do_bertscore:
            self.metric_keys.append("bertscore")
        if self.do_rouge:
            self.metric_keys.extend(["rouge1", "rouge2", "rougeL"])
        if self.do_srrbert:
            self.metric_keys.extend(
                ["samples_avg_precision", "samples_avg_recall", "samples_avg_f1-score"])
        if self.do_f1chexbert:
            self.metric_keys.extend([
                "f1chexbert_5_micro_f1",
                "f1chexbert_all_micro_f1",
                "f1chexbert_5_macro_f1",
                "f1chexbert_all_macro_f1"
            ])
        if self.do_f1radbert_ct:
            self.metric_keys.extend([
                "f1radbert_ct_accuracy",
                "f1radbert_ct_micro_f1",
                "f1radbert_ct_macro_f1",
                "f1radbert_ct_weighted_f1",
            ])


        if self.do_ratescore:
            self.metric_keys.append("ratescore")
        if self.do_radgraph_radcliq:
            self.metric_keys.append("radgraph_radcliq")
        if self.do_radcliq:
            self.metric_keys.append("radcliqv1")
        if self.do_temporal:
            self.metric_keys.append("temporal_f1")
        if self.do_radfact_ct:
            self.metric_keys.extend([
                "radfact_ct_precision", "radfact_ct_recall", "radfact_ct_f1"])
        if self.do_crimson:
            self.metric_keys.append("crimson")

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
            scores[f"{prefix}_5_micro_f1"] = cr_5["micro avg"]["f1-score"]
            scores[f"{prefix}_all_micro_f1"] = cr_all["micro avg"]["f1-score"]
            scores[f"{prefix}_5_macro_f1"] = cr_5["macro avg"]["f1-score"]
            scores[f"{prefix}_all_macro_f1"] = cr_all["macro avg"]["f1-score"]
            scores[f"{prefix}_5_weighted_f1"] = cr_5["weighted avg"]["f1-score"]
            scores[f"{prefix}_all_weighted_f1"] = cr_all["weighted avg"]["f1-score"]
            scores[f"{prefix}_label_scores_f1"] = {
                f"{prefix}_5": labels_5,
                f"{prefix}_all": labels_all,
            }
        elif self.do_per_sample:
            scores[f"{prefix}_sample_acc_5"] = (
                sample_acc_5.tolist() if hasattr(sample_acc_5, 'tolist')
                else list(sample_acc_5))
            scores[f"{prefix}_sample_acc_all"] = (
                sample_acc_full.tolist() if hasattr(sample_acc_full, 'tolist')
                else list(sample_acc_full))
        else:
            scores[f"{prefix}_5_micro_f1"] = round(cr_5["micro avg"]["f1-score"], 4)
            scores[f"{prefix}_all_micro_f1"] = round(cr_all["micro avg"]["f1-score"], 4)
            scores[f"{prefix}_5_macro_f1"] = round(cr_5["macro avg"]["f1-score"], 4)
            scores[f"{prefix}_all_macro_f1"] = round(cr_all["macro avg"]["f1-score"], 4)
            scores[f"{prefix}_5_weighted_f1"] = round(cr_5["weighted avg"]["f1-score"], 4)
            scores[f"{prefix}_all_weighted_f1"] = round(cr_all["weighted avg"]["f1-score"], 4)
        progress.advance(metric_task)

    def compute_scores(self, refs, hyps):
        scores = {}
        n_samples = len(refs)

        enabled = []
        if self.do_radgraph:        enabled.append("RadGraph")
        if self.do_bleu:            enabled.append("BLEU")
        if self.do_bertscore:       enabled.append("BERTScore")
        if self.do_green:           enabled.append("GREEN")
        if self.do_mammo_green:     enabled.append("MammoGREEN")
        if self.do_rouge:           enabled.append("ROUGE")
        if self.do_srrbert:        enabled.append("SRRBert")
        if self.do_f1chexbert:        enabled.append("F1CheXbert")
        if self.do_f1radbert_ct:    enabled.append("F1RadBERT-CT")


        if self.do_ratescore:       enabled.append("RaTEScore")
        if self.do_radgraph_radcliq: enabled.append("RadGraph-RadCliQ")
        if self.do_radcliq:         enabled.append("RadCliQ-v1")
        if self.do_temporal:        enabled.append("Temporal F1")
        if self.do_radfact_ct:      enabled.append("RadFact-CT")
        if self.do_crimson:         enabled.append("CRIMSON")

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
                    scores["radgraph_simple"] = f1_scores[0]
                    scores["radgraph_partial"] = f1_scores[1]
                    scores["radgraph_complete"] = f1_scores[2]
                elif self.do_per_sample:
                    per_level = radgraph_scores[1]
                    scores["radgraph_simple"] = list(per_level[0])
                    scores["radgraph_partial"] = list(per_level[1])
                    scores["radgraph_complete"] = list(per_level[2])
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
                    scores["bleu"] = round(self.bleu_scorer(refs, hyps)[0], 4)
                    scores["bleu_1"] = round(self.bleu_scorer_1(refs, hyps)[0], 4)
                    scores["bleu_2"] = round(self.bleu_scorer_2(refs, hyps)[0], 4)
                    scores["bleu_3"] = round(self.bleu_scorer_3(refs, hyps)[0], 4)
                elif self.do_per_sample:
                    _, bleu_samples = self.bleu_scorer(refs, hyps)
                    scores["bleu"] = bleu_samples
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
                    scores["bertscore"] = round(
                        self.bertscore_scorer(refs, hyps, on_batch_done=cb)[0], 4)
                elif self.do_per_sample:
                    _, sample_scores = self.bertscore_scorer(
                        refs, hyps, on_batch_done=cb)
                    scores["bertscore"] = sample_scores
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
                    scores["green"] = round(mean, 4)
                    scores["green_std"] = round(std, 4)
                elif self.do_per_sample:
                    scores["green"] = sample_scores
                else:
                    scores["green"] = round(mean, 4)
                progress.advance(metric_task)

            # ----------------------------------------------------------
            if self.do_mammo_green:
                progress.update(metric_task, description="Computing MammoGREEN")
                sample_task = progress.add_task(
                    "  [dim]Samples", total=n_samples)
                mg_cost = getattr(self.mammo_green_scorer, 'cost_tracker', None)

                def _mg_sample_done():
                    progress.advance(sample_task)
                    if mg_cost:
                        progress.update(sample_task,
                                        description=f"  [dim]Samples (${mg_cost.cost:.2f})")

                mean, std, sample_scores, results_df = self.mammo_green_scorer(
                    refs, hyps, on_sample_done=_mg_sample_done)
                if mg_cost:
                    progress.update(sample_task,
                                    description=f"  [dim]Samples (${mg_cost.cost:.2f})")
                progress.remove_task(sample_task)
                if self.do_details:
                    scores["mammo_green"] = round(mean, 4)
                    scores["mammo_green_std"] = round(std, 4)
                elif self.do_per_sample:
                    scores["mammo_green"] = sample_scores
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
                    for rt in self._rouge_types:
                        f1s = [s[rt].fmeasure for s in raw_scores]
                        scores[rt] = round(sum(f1s) / len(f1s), 4)
                elif self.do_per_sample:
                    for rt in self._rouge_types:
                        scores[rt] = [s[rt].fmeasure for s in raw_scores]
                else:
                    for rt in self._rouge_types:
                        f1s = [s[rt].fmeasure for s in raw_scores]
                        scores[rt] = round(sum(f1s) / len(f1s), 4)
                progress.advance(metric_task)

            # ----------------------------------------------------------
            if self.do_srrbert:
                progress.update(metric_task, description="Computing SRR-BERT")
                sample_task = progress.add_task(
                    "  [dim]Samples", total=n_samples)

                classification_dict, sample_precision, sample_recall, sample_f1 = \
                    self.srrbert_scorer.evaluate(
                        refs, hyps,
                        on_sample_done=lambda: progress.advance(sample_task),
                    )
                progress.remove_task(sample_task)

                if self.do_details:
                    label_names = [
                        label for label, idx in sorted(
                            self.srrbert_scorer.mapping.items(),
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

                    scores["srrbert_weighted_f1"] = round(
                        classification_dict["weighted avg"]["f1-score"], 4)
                    scores["srrbert_weighted_precision"] = round(
                        classification_dict["weighted avg"]["precision"], 4)
                    scores["srrbert_weighted_recall"] = round(
                        classification_dict["weighted avg"]["recall"], 4)
                    scores["srrbert_label_scores"] = label_scores
                elif self.do_per_sample:
                    scores["srrbert_weighted_f1"] = sample_f1.tolist()
                    scores["srrbert_weighted_precision"] = sample_precision.tolist()
                    scores["srrbert_weighted_recall"] = sample_recall.tolist()
                else:
                    scores["srrbert_weighted_f1"] = round(
                        classification_dict["weighted avg"]["f1-score"], 4)
                    scores["srrbert_weighted_precision"] = round(
                        classification_dict["weighted avg"]["precision"], 4)
                    scores["srrbert_weighted_recall"] = round(
                        classification_dict["weighted avg"]["recall"], 4)
                progress.advance(metric_task)

            # ----------------------------------------------------------
            if self.do_f1chexbert:
                self._score_chexbert(
                    scores, "f1chexbert", self.f1chexbert_scorer,
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
                    scores["f1radbert_ct_accuracy"] = round(f1radbert_ct_accuracy, 4)
                    scores["f1radbert_ct_micro_f1"] = round(
                        f1radbert_ct_report["micro avg"]["f1-score"], 4)
                    scores["f1radbert_ct_macro_f1"] = round(
                        f1radbert_ct_report["macro avg"]["f1-score"], 4)
                    scores["f1radbert_ct_weighted_f1"] = round(
                        f1radbert_ct_report["weighted avg"]["f1-score"], 4)
                    scores["f1radbert_ct_label_scores_f1"] = f1radbert_ct_labels
                elif self.do_per_sample:
                    scores["f1radbert_ct_sample_acc"] = (
                        f1radbert_ct_sample_acc.tolist()
                        if hasattr(f1radbert_ct_sample_acc, 'tolist')
                        else list(f1radbert_ct_sample_acc))
                else:
                    scores["f1radbert_ct_accuracy"] = round(
                        f1radbert_ct_accuracy, 4)
                    scores["f1radbert_ct_micro_f1"] = round(
                        f1radbert_ct_report["micro avg"]["f1-score"], 4)
                    scores["f1radbert_ct_macro_f1"] = round(
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
                    scores["ratescore"] = round(f1_ratescore, 4)
                elif self.do_per_sample:
                    scores["ratescore"] = rate_score
                else:
                    scores["ratescore"] = round(f1_ratescore, 4)
                progress.advance(metric_task)

            # ----------------------------------------------------------
            if self.do_radgraph_radcliq:
                progress.update(metric_task, description="Computing RadGraph-RadCliQ")
                sample_task = progress.add_task(
                    "  [dim]Samples", total=n_samples)
                mean_rg, sample_rg = self.radgraph_radcliq_scorer(
                    hyps, refs,
                    on_sample_done=lambda: progress.advance(sample_task))
                progress.remove_task(sample_task)
                if self.do_details:
                    scores["radgraph_radcliq"] = round(mean_rg, 4)
                elif self.do_per_sample:
                    scores["radgraph_radcliq"] = sample_rg
                else:
                    scores["radgraph_radcliq"] = round(mean_rg, 4)
                progress.advance(metric_task)

            # ----------------------------------------------------------
            if self.do_radcliq:
                progress.update(metric_task, description="Computing RadCliQ-v1")
                mean_scores, detail_scores = self.radcliq_scorer.predict(
                    refs, hyps)
                if self.do_details:
                    scores["radcliq_v1"] = round(mean_scores, 4)
                elif self.do_per_sample:
                    scores["radcliq_v1"] = detail_scores.tolist()
                else:
                    scores["radcliq_v1"] = round(mean_scores, 4)
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
                    scores["temporal_f1"] = round(temporal_scores["f1"], 4)
                elif self.do_per_sample:
                    scores["temporal_f1"] = temporal_scores["sample_scores"]
                else:
                    scores["temporal_f1"] = round(temporal_scores["f1"], 4)
                progress.advance(metric_task)

            # ----------------------------------------------------------
            if self.do_radfact_ct:
                progress.update(metric_task, description="Computing RadFact-CT")
                sample_task = progress.add_task(
                    "  [dim]Samples", total=n_samples)
                cost = self.radfact_ct_scorer.cost_tracker

                def _radfact_sample_done():
                    progress.advance(sample_task)
                    progress.update(sample_task,
                                    description=f"  [dim]Samples (${cost.cost:.2f})")

                radfact_agg, radfact_per_sample = self.radfact_ct_scorer(
                    hyps, refs, on_sample_done=_radfact_sample_done)
                progress.update(sample_task,
                                description=f"  [dim]Samples (${cost.cost:.2f})")
                progress.remove_task(sample_task)
                if self.do_details:
                    scores["radfact_ct_precision"] = radfact_agg["logical_precision"]
                    scores["radfact_ct_recall"] = radfact_agg["logical_recall"]
                    scores["radfact_ct_f1"] = radfact_agg["logical_f1"]
                elif self.do_per_sample:
                    scores["radfact_ct_precision"] = [
                        s["logical_precision"] for s in radfact_per_sample]
                    scores["radfact_ct_recall"] = [
                        s["logical_recall"] for s in radfact_per_sample]
                    scores["radfact_ct_f1"] = [
                        s["logical_f1"] for s in radfact_per_sample]
                else:
                    scores["radfact_ct_precision"] = radfact_agg["logical_precision"]
                    scores["radfact_ct_recall"] = radfact_agg["logical_recall"]
                    scores["radfact_ct_f1"] = radfact_agg["logical_f1"]
                progress.advance(metric_task)

            # ----------------------------------------------------------
            if self.do_crimson:
                progress.update(metric_task, description="Computing CRIMSON")
                sample_task = progress.add_task(
                    "  [dim]Samples", total=n_samples)
                crimson_cost = getattr(self.crimson_scorer, 'cost_tracker', None)

                def _crimson_sample_done():
                    progress.advance(sample_task)
                    if crimson_cost:
                        progress.update(sample_task,
                                        description=f"  [dim]Samples (${crimson_cost.cost:.2f})")

                mean, std, sample_scores, results_df = self.crimson_scorer(
                    refs, hyps,
                    on_sample_done=_crimson_sample_done)
                if crimson_cost:
                    progress.update(sample_task,
                                    description=f"  [dim]Samples (${crimson_cost.cost:.2f})")
                progress.remove_task(sample_task)
                if self.do_details:
                    scores["crimson"] = round(mean, 4)
                    scores["crimson_std"] = round(std, 4)
                elif self.do_per_sample:
                    scores["crimson"] = sample_scores
                else:
                    scores["crimson"] = round(mean, 4)
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
                    scores["radeval_bertscore"] = round(radeval_bertscores[0], 4)
                elif self.do_per_sample:
                    scores["radeval_bertscore"] = radeval_bertscores[1].tolist()
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
                        do_srrbert=True,
                        do_f1chexbert=True,
                        do_temporal=True,
                        do_ratescore=True,
                        do_radcliq=True,
                        do_radeval_bertscore=True)

    results = evaluator(refs=refs, hyps=hyps)
    print(json.dumps(results, indent=4))


if __name__ == '__main__':
    main()
