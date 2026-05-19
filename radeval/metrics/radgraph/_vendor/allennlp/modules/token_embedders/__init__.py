"""
A `TokenEmbedder` is a `Module` that
embeds one-hot-encoded tokens as vectors.
"""

from radeval.metrics.radgraph._vendor.allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from radeval.metrics.radgraph._vendor.allennlp.modules.token_embedders.embedding import Embedding
from radeval.metrics.radgraph._vendor.allennlp.modules.token_embedders.token_characters_encoder import TokenCharactersEncoder
from radeval.metrics.radgraph._vendor.allennlp.modules.token_embedders.elmo_token_embedder import ElmoTokenEmbedder
from radeval.metrics.radgraph._vendor.allennlp.modules.token_embedders.empty_embedder import EmptyEmbedder
from radeval.metrics.radgraph._vendor.allennlp.modules.token_embedders.bag_of_word_counts_token_embedder import (
    BagOfWordCountsTokenEmbedder,
)
from radeval.metrics.radgraph._vendor.allennlp.modules.token_embedders.pass_through_token_embedder import PassThroughTokenEmbedder
from radeval.metrics.radgraph._vendor.allennlp.modules.token_embedders.pretrained_transformer_embedder import (
    PretrainedTransformerEmbedder,
)
from radeval.metrics.radgraph._vendor.allennlp.modules.token_embedders.pretrained_transformer_mismatched_embedder import (
    PretrainedTransformerMismatchedEmbedder,
)
