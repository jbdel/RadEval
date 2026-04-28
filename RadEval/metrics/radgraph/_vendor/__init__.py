"""Vendored, trimmed Stanford-AIMI/radgraph package (originally BSD-3-clause).

Source: https://github.com/Stanford-AIMI/radgraph (PyPI: radgraph==0.1.18)

Why this is vendored:
  The upstream `radgraph` package ships an archived-AllenNLP fork whose
  tokenizer layer calls two transformers APIs that were removed in v5:
    - `tokenizer.build_inputs_with_special_tokens(segment)` (in
      allennlp/data/token_indexers/pretrained_transformer_indexer.py)
    - `tokenizer.encode_plus(a, b, ...)` (in
      allennlp/data/tokenizers/pretrained_transformer_tokenizer.py)
  and in transformers >=4.44 it also collides with an `add_special_tokens`
  kwarg name. This vendor contains the three fixes and is the only way
  RadEval 2.1+ can run RadGraph-family metrics against transformers 5.x.

Patches vs upstream 0.1.18 (see PATCHES.md in this folder for details):
  1. PretrainedTransformerTokenizer uses tokenizer(...) instead of
     encode_plus(...), with a ModernBERT-safe fallback for token_type_ids.
  2. PretrainedTransformerIndexer precomputes single/pair special-token
     prefix/suffix at init time from tokenizer.encode("") (fast-tokenizer
     compatible) rather than calling build_inputs_with_special_tokens.
  3. add_special_tokens is stripped from tokenizer_kwargs before
     AutoTokenizer.from_pretrained (the 4.44+ issue #5 fix).
"""
from .core import RadGraph, F1RadGraph
from .radgpt import get_radgraph_processed_annotations

__all__ = ["RadGraph", "F1RadGraph", "get_radgraph_processed_annotations"]
