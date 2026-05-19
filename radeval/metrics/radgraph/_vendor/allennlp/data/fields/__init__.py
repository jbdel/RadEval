"""
A :class:`~allennlp.data.fields.field.Field` is some piece of data instance
that ends up as an array in a model.
"""

from radeval.metrics.radgraph._vendor.allennlp.data.fields.field import Field
from radeval.metrics.radgraph._vendor.allennlp.data.fields.adjacency_field import AdjacencyField
from radeval.metrics.radgraph._vendor.allennlp.data.fields.array_field import ArrayField
from radeval.metrics.radgraph._vendor.allennlp.data.fields.flag_field import FlagField
from radeval.metrics.radgraph._vendor.allennlp.data.fields.index_field import IndexField
from radeval.metrics.radgraph._vendor.allennlp.data.fields.label_field import LabelField
from radeval.metrics.radgraph._vendor.allennlp.data.fields.list_field import ListField
from radeval.metrics.radgraph._vendor.allennlp.data.fields.metadata_field import MetadataField
from radeval.metrics.radgraph._vendor.allennlp.data.fields.multilabel_field import MultiLabelField
from radeval.metrics.radgraph._vendor.allennlp.data.fields.namespace_swapping_field import NamespaceSwappingField
from radeval.metrics.radgraph._vendor.allennlp.data.fields.sequence_field import SequenceField
from radeval.metrics.radgraph._vendor.allennlp.data.fields.sequence_label_field import SequenceLabelField
from radeval.metrics.radgraph._vendor.allennlp.data.fields.span_field import SpanField
from radeval.metrics.radgraph._vendor.allennlp.data.fields.text_field import TextField
