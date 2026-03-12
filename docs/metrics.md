[Back to README](../README.md)

# Metrics Reference

All metrics are enabled via `do_<name>=True` in the `RadEval` constructor. Each metric returns a scalar score by default; pass `do_details=True` to get per-sample breakdowns.

---

## Lexical Metrics

### BLEU (`do_bleu`)

N-gram overlap between hypothesis and reference. Returns BLEU-4 by default.

| Mode | Output key | Value |
|------|-----------|-------|
| Default | `bleu` | float |
| Details | `bleu.bleu_1` ... `bleu.bleu_4` | `{mean_score, sample_scores}` |

### ROUGE (`do_rouge`)

Recall-oriented n-gram evaluation. Computes ROUGE-1, ROUGE-2, and ROUGE-L in a single pass.

| Mode | Output keys | Value |
|------|------------|-------|
| Default | `rouge1`, `rouge2`, `rougeL` | float each |
| Details | `rouge.rouge1`, `rouge.rouge2`, `rouge.rougeL` | `{mean_score, sample_scores}` |

---

## Semantic Metrics

### BERTScore (`do_bertscore`)

Contextual embedding similarity using `distilbert-base-uncased` (layer 5).

| Mode | Output key | Value |
|------|-----------|-------|
| Default | `bertscore` | float (mean F1) |
| Details | `bertscore` | `{mean_score, sample_scores}` |

### RadEval BERTScore (`do_radeval_bertscore`)

Same architecture as BERTScore but using [IAMJB/RadEvalModernBERT](https://huggingface.co/IAMJB/RadEvalModernBERT), a domain-adapted radiology encoder with strong zero-shot retrieval performance.

| Mode | Output key | Value |
|------|-----------|-------|
| Default | `radeval_bertscore` | float (mean F1) |
| Details | `radeval_bertscore` | `{mean_score, sample_scores}` |

---

## Clinical Metrics

### F1CheXbert (`do_chexbert`)

Classifies reports into 14 CheXpert conditions using a BERT-based labeler, then computes multi-label F1 (micro/macro) between reference and hypothesis labels.

| Mode | Output keys | Value |
|------|------------|-------|
| Default | `chexbert-5_micro avg_f1-score`, `chexbert-all_micro avg_f1-score`, etc. | float |
| Details | `chexbert` | `{sample_scores, label_scores_f1-score, ...}` |

### HopprCheXbert (`do_hopprchexbert`)

Same architecture as F1CheXbert but with a ModernBERT backbone and 27 Hoppr-specific conditions. Output structure mirrors F1CheXbert with `hopprchexbert-` prefix.

### F1RadBERT-CT (`do_f1radbert_ct`)

Multi-label classification of 18 CT-specific findings using [IAMJB/RadBERT-CT](https://huggingface.co/IAMJB/RadBERT-CT).

| Mode | Output keys | Value |
|------|------------|-------|
| Default | `f1radbert_ct_accuracy`, `f1radbert_ct_micro avg_f1-score`, etc. | float |
| Details | `f1radbert_ct` | `{sample_scores, label_scores_f1-score, ...}` |

### F1RadGraph (`do_radgraph`)

Extracts clinical entities and relations as a knowledge graph using [RadGraph-XL](https://physionet.org/content/radgraph/1.0.0/), then computes entity/relation F1 at three levels: simple, partial, complete.

| Mode | Output keys | Value |
|------|------------|-------|
| Default | `radgraph_simple`, `radgraph_partial`, `radgraph_complete` | float |
| Details | `radgraph` | `{sample_scores, hypothesis_annotation_lists, reference_annotation_lists}` |

### RaTEScore (`do_ratescore`)

Entity-aware metric that extracts medical entities via NER, computes synonym-aware embeddings, and scores precision/recall using a learned affinity matrix.

| Mode | Output key | Value |
|------|-----------|-------|
| Default | `ratescore` | float (mean F1) |
| Details | `ratescore` | `{f1-score, sample_scores, hyps_pairs, refs_pairs}` |

---

## Specialized Metrics

### RadGraph-RadCliQ (`do_radgraph_radcliq`)

The RadGraph sub-score as computed in the RadCliQ-v1 pipeline. Unlike the official `F1RadGraph` metric (`do_radgraph`), which uses `radgraph-xl` and entity-matching at three reward levels, this metric:

- Uses the original `radgraph` model
- Extracts entity and relation sets from each report
- Computes per-pair `(entity_f1 + relation_f1) / 2`
- Returns per-sample scores

Use `do_radgraph` for the standard metric. Use `do_radgraph_radcliq` when you need per-pair entity+relation F1 scores or exact alignment with the RadCliQ-v1 composite.

| Mode | Output key | Value |
|------|-----------|-------|
| Default | `radgraph_radcliq` | float (mean score) |
| Details | `radgraph_radcliq` | `{mean_score, sample_scores}` |

### RadCliQ-v1 (`do_radcliq`)

Composite metric that combines BERTScore, RadGraph, semantic embeddings, and BLEU via a learned linear model. Lower values indicate better reports.

| Mode | Output key | Value |
|------|-----------|-------|
| Default | `radcliq-v1` | float |
| Details | `radcliq-v1` | `{mean_score, sample_scores}` |

### SRR-BERT (`do_srr_bert`)

Structured Radiology Report evaluation. Parses each report into sentences, classifies each sentence into 163 finding/status labels, merges at report level, and computes weighted precision/recall/F1.

| Mode | Output keys | Value |
|------|------------|-------|
| Default | `srr_bert_weighted_f1`, `srr_bert_weighted_precision`, `srr_bert_weighted_recall` | float |
| Details | `srr_bert` | `{srr_bert_weighted_f1: {weighted_mean_score, sample_scores}, ..., label_scores}` |

### Temporal F1 (`do_temporal`)

Extracts temporal entities (e.g. "stable", "worsening", "new") via Stanza NER and keyword matching, then computes F1 on the entity sets.

| Mode | Output key | Value |
|------|-----------|-------|
| Default | `temporal_f1` | float |
| Details | `temporal_f1` | `{f1-score, sample_scores, hyps_entities, refs_entities}` |

### GREEN (`do_green`)

LLM-based evaluation using [GREEN-radllama2-7B](https://huggingface.co/StanfordAIMI/GREEN-radllama2-7b). The model generates a natural language comparison, from which a score is extracted. Supports multi-GPU inference.

| Mode | Output key | Value |
|------|-----------|-------|
| Default | `green` | float (mean score) |
| Details | `green` | `{mean, std, sample_scores}` |

### MammoGREEN (`do_mammo_green`)

Mammography-specific LLM-as-judge metric. Calls an OpenAI or Gemini model to count clinically significant errors (false findings, missing findings, mischaracterization, wrong location, incorrect BI-RADS, incorrect breast density).

Requires `pip install RadEval[api]` and an API key (`mammo_green_api_key` or `OPENAI_API_KEY` / `GOOGLE_API_KEY` env var).

| Mode | Output key | Value |
|------|-----------|-------|
| Default | `mammo_green` | float (mean score) |
| Details | `mammo_green` | `{mean, std, sample_scores, error_counts}` |

### CRIMSON (`do_crimson`)

LLM-based clinical radiology report scoring from the [Rajpurkar Lab](https://github.com/rajpurkarlab/CRIMSON). Evaluates report quality by comparing predicted findings against reference findings, identifying errors (false findings, missing findings, attribute errors) and weighting them by clinical significance.

Supports two backends:
- **OpenAI** (`crimson_api="openai"`): uses `gpt-5.2` by default
- **HuggingFace** (`crimson_api="hf"`): uses [MedGemma-CRIMSON](https://huggingface.co/CRIMSONScore/medgemma-4b-it-crimson) locally

Requires `pip install RadEval[api]` for OpenAI, or `torch` + `transformers` for HuggingFace.

| Mode | Output key | Value |
|------|-----------|-------|
| Default | `crimson` | float (mean CRIMSON score, range [-1, 1]) |
| Details | `crimson` | `{mean, std, sample_scores, error_counts}` |

### RadFact-CT (`do_radfact_ct`)

LLM-based factual evaluation for CT reports, ported from [microsoft/RadFact](https://github.com/microsoft/RadFact). Splits reports into atomic phrases, then runs bidirectional entailment verification (is each candidate phrase supported by the reference, and vice versa).

Two modes:
- **RadFact +/-** (default): evaluates all phrases including negatives ("no pneumothorax")
- **RadFact +** (`radfact_ct_filter_negatives=True`): filters out negative/normal findings first

Requires `pip install RadEval[api]` and `OPENAI_API_KEY`. Uses `gpt-4o-mini` by default.

| Mode | Output keys | Value |
|------|------------|-------|
| Default | `radfact_ct_precision`, `radfact_ct_recall`, `radfact_ct_f1` | float (percentages) |
| Details | `radfact_ct` | `{logical_precision, logical_recall, logical_f1, per_sample}` |
