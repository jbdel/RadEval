"""
This module contains classes representing regularization schemes
as well as a class for applying regularization to parameters.
"""

from radeval.metrics.radgraph._vendor.allennlp.nn.regularizers.regularizer import Regularizer
from radeval.metrics.radgraph._vendor.allennlp.nn.regularizers.regularizers import L1Regularizer
from radeval.metrics.radgraph._vendor.allennlp.nn.regularizers.regularizers import L2Regularizer
from radeval.metrics.radgraph._vendor.allennlp.nn.regularizers.regularizer_applicator import RegularizerApplicator
