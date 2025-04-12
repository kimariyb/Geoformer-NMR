from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

""" Geoformer model configuration"""

logger = logging.get_logger(__name__)


class GeoformerConfig(PretrainedConfig):
    model_type = "geoformer"

    def __init__(
        self,
        max_z: int = 100,
        embedding_dim: int = 512,
        ffn_embedding_dim: int = 2048,
        pred_hidden_dim: int = 256,
        num_layers: int = 9,
        num_heads: int = 8,
        cutoff: int = 5.0,
        num_rbf: int = 64,
        rbf_trainable: bool = True,
        norm_type: str = "max_min",
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        dataset_root=None,
        mean=None,
        std=None,
        **kwargs
    ):
        self.max_z = max_z
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.pred_hidden_dim = pred_hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.rbf_trainable = rbf_trainable
        self.norm_type = norm_type
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.dataset_root = dataset_root
        self.mean = mean
        self.std = std

        super(GeoformerConfig, self).__init__(**kwargs)
