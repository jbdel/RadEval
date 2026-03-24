[Back to README](../README.md)

# Metrics Reference

All metrics are enabled via `do_<name>=True` in the `RadEval` constructor. Three output modes:

| Mode | Flag | Output |
|------|------|--------|
| Default | -- | Flat dict of scalar scores |
| Per-Sample | `do_per_sample=True` | Same flat keys, values are `list[float]` (one per report) |
| Details | `do_details=True` | Default keys + extra aggregate scores (label breakdowns, std, bleu_1/2/3) |

---

## Lexical Metrics

### BLEU (`do_bleu`)

N-gram overlap between hypothesis and reference. Returns BLEU-4 (4-gram) by default.

| Mode | Output keys | Value |
|------|------------|-------|
| Default | `bleu` | float (BLEU-4) |
| Per-Sample | `bleu` | list[float] (BLEU-4 per sample) |
| Details | `bleu`, `bleu_1`, `bleu_2`, `bleu_3` | float each (adds BLEU-1/2/3) |

```python
from RadEval import RadEval

evaluator = RadEval(do_bleu=True)
results = evaluator(refs=refs, hyps=hyps)
print(results["bleu"])  # 0.3605
```

### ROUGE (`do_rouge`)

Recall-oriented n-gram evaluation. Computes ROUGE-1, ROUGE-2, and ROUGE-L in a single pass.

| Mode | Output keys | Value |
|------|------------|-------|
| Default | `rouge1`, `rouge2`, `rougeL` | float each |
| Per-Sample | `rouge1`, `rouge2`, `rougeL` | list[float] each |
| Details | `rouge1`, `rouge2`, `rougeL` | same as default |

```python
evaluator = RadEval(do_rouge=True)
results = evaluator(refs=refs, hyps=hyps)
print(results["rouge1"], results["rouge2"], results["rougeL"])
```

---

## Semantic Metrics

### BERTScore (`do_bertscore`)

Contextual embedding similarity using `distilbert-base-uncased` (layer 5).

| Mode | Output key | Value |
|------|-----------|-------|
| Default | `bertscore` | float (mean F1) |
| Per-Sample | `bertscore` | list[float] |
| Details | `bertscore` | same as default |

```python
evaluator = RadEval(do_bertscore=True)
results = evaluator(refs=refs, hyps=hyps)
print(results["bertscore"])  # 0.6327
```

### RadEval BERTScore (`do_radeval_bertscore`)

Same architecture as BERTScore but using [IAMJB/RadEvalModernBERT](https://huggingface.co/IAMJB/RadEvalModernBERT), a domain-adapted radiology encoder with strong zero-shot retrieval performance.

| Mode | Output key | Value |
|------|-----------|-------|
| Default | `radeval_bertscore` | float (mean F1) |
| Per-Sample | `radeval_bertscore` | list[float] |
| Details | `radeval_bertscore` | same as default |

```python
evaluator = RadEval(do_radeval_bertscore=True)
results = evaluator(refs=refs, hyps=hyps)
print(results["radeval_bertscore"])  # 0.3462
```

---

## Clinical Metrics

### F1CheXbert (`do_f1chexbert`)

Classifies reports into 14 CheXpert conditions using a BERT-based labeler, then computes multi-label F1 (micro/macro) between reference and hypothesis labels.

| Mode | Output keys | Value |
|------|------------|-------|
| Default | `f1chexbert_5_micro_f1`, `f1chexbert_all_micro_f1`, etc. (6 keys) | float |
| Per-Sample | `f1chexbert_sample_acc_5`, `f1chexbert_sample_acc_all` | list[float] |
| Details | default keys + `f1chexbert_label_scores_f1` | adds per-label F1 dict |

```python
evaluator = RadEval(do_f1chexbert=True)
results = evaluator(refs=refs, hyps=hyps)
print(results["f1chexbert_5_micro_f1"])
print(results["f1chexbert_all_micro_f1"])
```

### F1RadBERT-CT (`do_f1radbert_ct`)

Multi-label classification of 18 CT-specific findings using [IAMJB/RadBERT-CT](https://huggingface.co/IAMJB/RadBERT-CT).

| Mode | Output keys | Value |
|------|------------|-------|
| Default | `f1radbert_ct_accuracy`, `f1radbert_ct_micro_f1`, etc. (4 keys) | float |
| Per-Sample | `f1radbert_ct_sample_acc` | list[float] |
| Details | default keys + `f1radbert_ct_label_scores_f1` | adds per-label F1 dict |

```python
evaluator = RadEval(do_f1radbert_ct=True)
results = evaluator(refs=refs, hyps=hyps)
print(results["f1radbert_ct_accuracy"])
print(results["f1radbert_ct_micro_f1"])
```

### F1RadGraph (`do_radgraph`)

Extracts clinical entities and relations as a knowledge graph using [RadGraph-XL](https://physionet.org/content/radgraph/1.0.0/), then computes entity/relation F1 at three levels: simple, partial, complete.

| Mode | Output keys | Value |
|------|------------|-------|
| Default | `radgraph_simple`, `radgraph_partial`, `radgraph_complete` | float |
| Per-Sample | `radgraph_simple`, `radgraph_partial`, `radgraph_complete` | list[float] each |
| Details | `radgraph_simple`, `radgraph_partial`, `radgraph_complete` | same as default |

```python
evaluator = RadEval(do_radgraph=True)
results = evaluator(refs=refs, hyps=hyps)
print(results["radgraph_simple"])    # 0.7222
print(results["radgraph_partial"])   # 0.6111
print(results["radgraph_complete"])  # 0.6111
```

### RaTEScore (`do_ratescore`)

Entity-aware metric that extracts medical entities via NER, computes synonym-aware embeddings, and scores precision/recall using a learned affinity matrix.

| Mode | Output key | Value |
|------|-----------|-------|
| Default | `ratescore` | float (mean F1) |
| Per-Sample | `ratescore` | list[float] |
| Details | `ratescore` | same as default |

```python
evaluator = RadEval(do_ratescore=True)
results = evaluator(refs=refs, hyps=hyps)
print(results["ratescore"])  # 0.5878
```

---

## Specialized Metrics

### RadGraph-RadCliQ (`do_radgraph_radcliq`)

The RadGraph sub-score as computed in the RadCliQ-v1 pipeline. Unlike the official `F1RadGraph` metric (`do_radgraph`), which uses `radgraph-xl` and entity-matching at three reward levels, this metric:

- Uses the original `radgraph` model (the one RadCliQ-v1 was trained with)
- Extracts entity **and** relation sets from each report
- Computes per-pair `(entity_f1 + relation_f1) / 2`
- Returns per-sample scores

Use `do_radgraph` for the standard metric. Use `do_radgraph_radcliq` when you need per-pair entity+relation F1 scores or exact alignment with the RadCliQ-v1 composite.

| Mode | Output key | Value |
|------|-----------|-------|
| Default | `radgraph_radcliq` | float (mean score) |
| Per-Sample | `radgraph_radcliq` | list[float] |
| Details | `radgraph_radcliq` | same as default |

```python
evaluator = RadEval(do_radgraph_radcliq=True)
results = evaluator(refs=refs, hyps=hyps)
print(results["radgraph_radcliq"])  # 0.2576
```

### RadCliQ-v1 (`do_radcliq`)

Composite metric that combines BERTScore (with IDF), RadGraph, semantic embeddings, and BLEU-2 via a learned linear model. Returns `1/mean(raw_scores)` so that higher values indicate better reports. Validated against the [reference implementation](https://github.com/rajpurkarlab/CXR-Report-Metric).

| Mode | Output key | Value |
|------|-----------|-------|
| Default | `radcliq_v1` | float |
| Per-Sample | `radcliq_v1` | list[float] |
| Details | `radcliq_v1` | same as default |

```python
evaluator = RadEval(do_radcliq=True)
results = evaluator(refs=refs, hyps=hyps)
print(results["radcliq_v1"])  # higher is better (returns 1/mean of raw scores)
```

### SRRBert (`do_srrbert`)

Structured Radiology Report evaluation. Parses each report into sentences, classifies each sentence into 163 finding/status labels, merges at report level, and computes weighted precision/recall/F1.

| Mode | Output keys | Value |
|------|------------|-------|
| Default | `srrbert_weighted_f1`, `srrbert_weighted_precision`, `srrbert_weighted_recall` | float |
| Per-Sample | `srrbert_weighted_f1`, `srrbert_weighted_precision`, `srrbert_weighted_recall` | list[float] each |
| Details | default keys + `srrbert_label_scores` | adds per-label P/R/F1 dict |

```python
evaluator = RadEval(do_srrbert=True)
results = evaluator(refs=refs, hyps=hyps)
print(results["srrbert_weighted_f1"])
print(results["srrbert_weighted_precision"])
print(results["srrbert_weighted_recall"])
```

### Temporal F1 (`do_temporal`)

Extracts temporal entities (e.g. "stable", "worsening", "new") via Stanza NER and keyword matching, then computes F1 on the entity sets.

| Mode | Output key | Value |
|------|-----------|-------|
| Default | `temporal_f1` | float |
| Per-Sample | `temporal_f1` | list[float] |
| Details | `temporal_f1` | same as default |

```python
evaluator = RadEval(do_temporal=True)
results = evaluator(refs=refs, hyps=hyps)
print(results["temporal_f1"])  # 0.5
```

### GREEN (`do_green`)

LLM-based evaluation using [GREEN-radllama2-7B](https://huggingface.co/StanfordAIMI/GREEN-radllama2-7b). The model generates a natural language comparison, from which a score is extracted. Supports multi-GPU inference.

| Mode | Output keys | Value |
|------|------------|-------|
| Default | `green` | float (mean score) |
| Per-Sample | `green` | list[float] |
| Details | `green`, `green_std` | adds standard deviation |

```python
# Set CUDA_VISIBLE_DEVICES for multi-GPU inference
evaluator = RadEval(do_green=True)
results = evaluator(refs=refs, hyps=hyps)
print(results["green"])  # 0.875
```

### MammoGREEN (`do_mammo_green`)

Mammography-specific LLM-as-judge metric. Calls an OpenAI or Gemini model to count clinically significant errors (false findings, missing findings, mischaracterization, wrong location, incorrect BI-RADS, incorrect breast density).

Requires `pip install RadEval[api]` and an API key (`openai_api_key` or `OPENAI_API_KEY` / `GOOGLE_API_KEY` env var).

| Mode | Output keys | Value |
|------|------------|-------|
| Default | `mammo_green` | float (mean score) |
| Per-Sample | `mammo_green` | list[float] |
| Details | `mammo_green`, `mammo_green_std` | adds standard deviation |

```python
# OpenAI (default: gpt-4o-mini)
evaluator = RadEval(do_mammo_green=True, openai_api_key="sk-...")

# Gemini
evaluator = RadEval(
    do_mammo_green=True,
    mammo_green_model="gemini-2.5-flash",
    gemini_api_key="AIza...",
)

results = evaluator(refs=refs, hyps=hyps)
print(results["mammo_green"])
```

### CRIMSON (`do_crimson`)

LLM-based clinical radiology report scoring from the [Rajpurkar Lab](https://github.com/rajpurkarlab/CRIMSON). Evaluates report quality by comparing predicted findings against reference findings, identifying errors (false findings, missing findings, attribute errors) and weighting them by clinical significance.

Supports two backends:
- **HuggingFace** (default, `crimson_api="hf"`): uses [MedGemma-CRIMSON](https://huggingface.co/CRIMSONScore/medgemma-4b-it-crimson) locally
- **OpenAI** (`crimson_api="openai"`): uses `gpt-5.2` by default

Requires `torch` + `transformers` for HuggingFace, or `pip install RadEval[api]` for OpenAI.

| Mode | Output keys | Value |
|------|------------|-------|
| Default | `crimson` | float (mean CRIMSON score, range [-1, 1]) |
| Per-Sample | `crimson` | list[float] |
| Details | `crimson`, `crimson_std` | adds standard deviation |

```python
# HuggingFace MedGemma (default, as in the original paper, runs locally on GPU)
evaluator = RadEval(do_crimson=True)

# OpenAI API
evaluator = RadEval(
    do_crimson=True,
    crimson_api="openai",
    openai_api_key="sk-...",
)

results = evaluator(refs=refs, hyps=hyps)
print(results["crimson"])  # range [-1, 1]
```

### RadFact-CT (`do_radfact_ct`)

LLM-based factual evaluation for CT reports, ported from [microsoft/RadFact](https://github.com/microsoft/RadFact). Splits reports into atomic phrases, then runs bidirectional entailment verification (is each candidate phrase supported by the reference, and vice versa).

Two modes:
- **RadFact +/-** (default): evaluates all phrases including negatives ("no pneumothorax")
- **RadFact +** (`radfact_ct_filter_negatives=True`): filters out negative/normal findings first

Requires `pip install RadEval[api]` and `openai_api_key` or `OPENAI_API_KEY`. Uses `gpt-4o-mini` by default.

Evaluates samples concurrently (default 10) with live cost tracking in the progress bar. Control via `radfact_ct_max_concurrent`.

| Mode | Output keys | Value |
|------|------------|-------|
| Default | `radfact_ct_precision`, `radfact_ct_recall`, `radfact_ct_f1` | float (percentages) |
| Per-Sample | `radfact_ct_precision`, `radfact_ct_recall`, `radfact_ct_f1` | list[float] each |
| Details | `radfact_ct_precision`, `radfact_ct_recall`, `radfact_ct_f1` | same as default |

```python
# RadFact +/- (default, 10 concurrent samples)
evaluator = RadEval(do_radfact_ct=True, openai_api_key="sk-...")

# RadFact + (filter negatives)
evaluator = RadEval(do_radfact_ct=True, radfact_ct_filter_negatives=True)

# Custom concurrency (higher for Tier 2+ accounts)
evaluator = RadEval(do_radfact_ct=True, radfact_ct_max_concurrent=20)

results = evaluator(refs=refs, hyps=hyps)
print(results["radfact_ct_precision"])  # 66.67
print(results["radfact_ct_recall"])     # 83.33
print(results["radfact_ct_f1"])         # 74.07
```

---

## Provider Support Matrix

| Metric | OpenAI | Gemini | Local HF |
|--------|--------|--------|----------|
| CRIMSON | yes | -- | yes |
| MammoGREEN | yes | yes | -- |
| RadFact-CT | yes | -- | -- |
| GREEN | -- | -- | yes |
