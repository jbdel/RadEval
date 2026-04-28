"""Vendored RadGraph inference — exposes RadGraph and F1RadGraph.

RadEval 2.1+ vendors a patched copy of Stanford-AIMI/radgraph
(/_vendor/) so the library can run on transformers 5.x without relying on
the upstream `radgraph` PyPI package (which pins transformers<5 via an
archived AllenNLP fork). See _vendor/__init__.py for patch details.
"""
from ._vendor import RadGraph, F1RadGraph

__all__ = ["RadGraph", "F1RadGraph"]
