"""Vendored, trimmed BERTScore implementation.

This is a small fork of https://github.com/Tiiiger/bert_score (MIT License),
trimmed to the subset of functionality RadEval actually uses and patched to be
compatible with transformers v5+, which removed:
  - `tokenizer.batch_encode_plus` and `encode_plus`
  - legacy slow tokenizer classes (`GPT2Tokenizer`, `RobertaTokenizer`, ...)
  - `tokenizer.build_inputs_with_special_tokens` on the fast/TokenizersBackend

We only keep the call path RadEval exercises:
  - `BERTScorer(model_type, num_layers, batch_size, rescale_with_baseline,
               use_fast_tokenizer, device, lang, idf, idf_sents, baseline_path)`
  - `scorer.score(cands, refs, verbose=False, batch_size=N)` -> (P, R, F)

What was removed vs upstream 0.3.13:
  - multi-reference handling via nested ref lists (we always pass list[str])
  - `plot_example` (matplotlib dependency)
  - SciBERT auto-download
  - Language defaults table (caller always passes `model_type`)
  - `model2layers` table (caller always passes `num_layers`)
  - `GPT2Tokenizer`/`RobertaTokenizer` instance checks — replaced with a
    duck-typed lookup on `tokenizer.__class__.__name__` that preserves the
    `add_prefix_space=True` code path for RoBERTa/GPT-2 family tokenizers.

Numerics are bit-identical to upstream bert-score 0.3.13 for BERT/RoBERTa
family models (verified against RadEval's regression test fixtures).
"""
from .scorer import BERTScorer

__all__ = ["BERTScorer"]
