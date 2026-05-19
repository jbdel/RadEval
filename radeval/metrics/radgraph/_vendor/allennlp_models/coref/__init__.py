"""
Coreference resolution is defined as follows: given a document, find and cluster entity mentions.
"""

from radeval.metrics.radgraph._vendor.allennlp_models.coref.dataset_readers.conll import ConllCorefReader
from radeval.metrics.radgraph._vendor.allennlp_models.coref.dataset_readers.preco import PrecoReader
from radeval.metrics.radgraph._vendor.allennlp_models.coref.dataset_readers.winobias import WinobiasReader
from radeval.metrics.radgraph._vendor.allennlp_models.coref.models.coref import CoreferenceResolver
