"""
Custom PyTorch
`Module <https://pytorch.org/docs/master/nn.html#torch.nn.Module>`_ s
that are used as components in AllenNLP `Model` s.
"""

from radeval.metrics.radgraph._vendor.allennlp.modules.attention import Attention
from radeval.metrics.radgraph._vendor.allennlp.modules.bimpm_matching import BiMpmMatching
from radeval.metrics.radgraph._vendor.allennlp.modules.conditional_random_field import ConditionalRandomField
from radeval.metrics.radgraph._vendor.allennlp.modules.elmo import Elmo
from radeval.metrics.radgraph._vendor.allennlp.modules.feedforward import FeedForward
from radeval.metrics.radgraph._vendor.allennlp.modules.gated_sum import GatedSum
from radeval.metrics.radgraph._vendor.allennlp.modules.highway import Highway
from radeval.metrics.radgraph._vendor.allennlp.modules.input_variational_dropout import InputVariationalDropout
from radeval.metrics.radgraph._vendor.allennlp.modules.layer_norm import LayerNorm
from radeval.metrics.radgraph._vendor.allennlp.modules.matrix_attention import MatrixAttention
from radeval.metrics.radgraph._vendor.allennlp.modules.maxout import Maxout
from radeval.metrics.radgraph._vendor.allennlp.modules.residual_with_layer_dropout import ResidualWithLayerDropout
from radeval.metrics.radgraph._vendor.allennlp.modules.scalar_mix import ScalarMix
from radeval.metrics.radgraph._vendor.allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from radeval.metrics.radgraph._vendor.allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from radeval.metrics.radgraph._vendor.allennlp.modules.text_field_embedders import TextFieldEmbedder
from radeval.metrics.radgraph._vendor.allennlp.modules.time_distributed import TimeDistributed
from radeval.metrics.radgraph._vendor.allennlp.modules.token_embedders import TokenEmbedder, Embedding
from radeval.metrics.radgraph._vendor.allennlp.modules.softmax_loss import SoftmaxLoss
