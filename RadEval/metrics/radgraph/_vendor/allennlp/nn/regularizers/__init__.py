"""
This module contains classes representing regularization schemes
as well as a class for applying regularization to parameters.
"""

from RadEval.metrics.radgraph._vendor.allennlp.nn.regularizers.regularizer import Regularizer
from RadEval.metrics.radgraph._vendor.allennlp.nn.regularizers.regularizers import L1Regularizer
from RadEval.metrics.radgraph._vendor.allennlp.nn.regularizers.regularizers import L2Regularizer
from RadEval.metrics.radgraph._vendor.allennlp.nn.regularizers.regularizer_applicator import RegularizerApplicator
