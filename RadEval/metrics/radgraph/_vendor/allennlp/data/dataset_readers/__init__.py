"""
A :class:`~allennlp.data.dataset_readers.dataset_reader.DatasetReader`
reads a file and converts it to a collection of
:class:`~allennlp.data.instance.Instance` s.
The various subclasses know how to read specific filetypes
and produce datasets in the formats required by specific models.
"""


from RadEval.metrics.radgraph._vendor.allennlp.data.dataset_readers.conll2003 import Conll2003DatasetReader
from RadEval.metrics.radgraph._vendor.allennlp.data.dataset_readers.dataset_reader import (
    DatasetReader,
    AllennlpDataset,
    AllennlpLazyDataset,
)
from RadEval.metrics.radgraph._vendor.allennlp.data.dataset_readers.interleaving_dataset_reader import InterleavingDatasetReader
from RadEval.metrics.radgraph._vendor.allennlp.data.dataset_readers.sequence_tagging import SequenceTaggingDatasetReader
from RadEval.metrics.radgraph._vendor.allennlp.data.dataset_readers.sharded_dataset_reader import ShardedDatasetReader
from RadEval.metrics.radgraph._vendor.allennlp.data.dataset_readers.babi import BabiReader
from RadEval.metrics.radgraph._vendor.allennlp.data.dataset_readers.text_classification_json import TextClassificationJsonReader
