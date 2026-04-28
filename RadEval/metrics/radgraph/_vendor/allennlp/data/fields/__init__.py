"""
A :class:`~allennlp.data.fields.field.Field` is some piece of data instance
that ends up as an array in a model.
"""

from RadEval.metrics.radgraph._vendor.allennlp.data.fields.field import Field
from RadEval.metrics.radgraph._vendor.allennlp.data.fields.adjacency_field import AdjacencyField
from RadEval.metrics.radgraph._vendor.allennlp.data.fields.array_field import ArrayField
from RadEval.metrics.radgraph._vendor.allennlp.data.fields.flag_field import FlagField
from RadEval.metrics.radgraph._vendor.allennlp.data.fields.index_field import IndexField
from RadEval.metrics.radgraph._vendor.allennlp.data.fields.label_field import LabelField
from RadEval.metrics.radgraph._vendor.allennlp.data.fields.list_field import ListField
from RadEval.metrics.radgraph._vendor.allennlp.data.fields.metadata_field import MetadataField
from RadEval.metrics.radgraph._vendor.allennlp.data.fields.multilabel_field import MultiLabelField
from RadEval.metrics.radgraph._vendor.allennlp.data.fields.namespace_swapping_field import NamespaceSwappingField
from RadEval.metrics.radgraph._vendor.allennlp.data.fields.sequence_field import SequenceField
from RadEval.metrics.radgraph._vendor.allennlp.data.fields.sequence_label_field import SequenceLabelField
from RadEval.metrics.radgraph._vendor.allennlp.data.fields.span_field import SpanField
from RadEval.metrics.radgraph._vendor.allennlp.data.fields.text_field import TextField
