[Back to README](../README.md)

# Configuration Reference

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `do_radgraph` | bool | `False` | F1RadGraph -- knowledge-graph clinical evaluation |
| `do_green` | bool | `False` | GREEN -- LLM-based quality scorer (loads a 7B model) |
| `do_mammo_green` | bool | `False` | MammoGREEN -- mammography-specific LLM scorer |
| `mammo_green_model` | str | `"gpt-4o-mini"` | Model name for MammoGREEN (OpenAI or Gemini) |
| `mammo_green_api_key` | str / None | `None` | API key for MammoGREEN (falls back to env vars) |
| `do_bleu` | bool | `False` | BLEU-4 n-gram overlap |
| `do_rouge` | bool | `False` | ROUGE-1, ROUGE-2, ROUGE-L |
| `do_bertscore` | bool | `False` | BERTScore with distilbert-base-uncased |
| `do_srr_bert` | bool | `False` | SRR-BERT structured report evaluation |
| `do_chexbert` | bool | `False` | F1CheXbert label classification (14 CheXpert labels) |
| `do_f1radbert_ct` | bool | `False` | F1RadBERT-CT multi-label CT classification |
| `do_hopprchexbert` | bool | `False` | HopprCheXbert (ModernBERT, 27 Hoppr labels) |
| `do_ratescore` | bool | `False` | RaTEScore entity-aware scoring |
| `do_radcliq` | bool | `False` | RadCliQ-v1 composite metric |
| `do_radeval_bertscore` | bool | `False` | RadEval BERTScore (domain-adapted ModernBERT) |
| `do_temporal` | bool | `False` | Temporal entity F1 |
| `do_radfact_ct` | bool | `False` | RadFact-CT LLM-based factual evaluation (requires API key) |
| `do_crimson` | bool | `False` | CRIMSON clinical significance scoring |
| `crimson_api` | str | `"openai"` | Backend: `"openai"` or `"hf"` (HuggingFace MedGemma) |
| `crimson_api_key` | str / None | `None` | API key for OpenAI backend (falls back to `OPENAI_API_KEY`) |
| `crimson_model` | str / None | `None` | Override model name (defaults per backend) |
| `crimson_batch_size` | int | `1` | Batch size for HuggingFace inference |
| `radfact_ct_model` | str | `"gpt-4o-mini"` | LLM model for RadFact-CT |
| `radfact_ct_api_key` | str / None | `None` | API key for RadFact-CT (falls back to `OPENAI_API_KEY`) |
| `radfact_ct_filter_negatives` | bool | `False` | RadFact+ mode: filter negative findings before scoring |
| `do_details` | bool | `False` | Return per-sample scores, label breakdowns, entity annotations |
| `show_progress` | bool | `True` | Display rich progress bars during scoring |

## Example Presets

### Lightweight (fast, no GPU models)

```python
evaluator = RadEval(do_bleu=True, do_rouge=True)
```

### Clinical accuracy

```python
evaluator = RadEval(
    do_radgraph=True,
    do_chexbert=True,
    do_f1radbert_ct=True,
)
```

### Full evaluation

```python
evaluator = RadEval(
    do_radgraph=True,
    do_green=True,
    do_bleu=True,
    do_rouge=True,
    do_bertscore=True,
    do_srr_bert=True,
    do_chexbert=True,
    do_f1radbert_ct=True,
    do_temporal=True,
    do_ratescore=True,
    do_radcliq=True,
    do_radeval_bertscore=True,
    do_details=True,
)
```

### GPU tips

- **GREEN** loads a 7B-parameter LLM. Set `CUDA_VISIBLE_DEVICES` to control which GPUs it uses for multi-GPU inference.
- **MammoGREEN** calls an external API and does not require a GPU.
- All other metrics load smaller models (< 1GB) and run on a single GPU or CPU.
- Metrics are loaded lazily -- only the ones you enable consume memory.
