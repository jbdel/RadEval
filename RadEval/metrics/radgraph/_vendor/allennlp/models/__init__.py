"""
These submodules contain the classes for AllenNLP models,
all of which are subclasses of `Model`.
"""

from RadEval.metrics.radgraph._vendor.allennlp.models.model import Model
from RadEval.metrics.radgraph._vendor.allennlp.models.archival import archive_model, load_archive, Archive
from RadEval.metrics.radgraph._vendor.allennlp.models.simple_tagger import SimpleTagger
from RadEval.metrics.radgraph._vendor.allennlp.models.basic_classifier import BasicClassifier
