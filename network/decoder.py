import torch
import torch.nn as nn

from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

from network.encoder import GeoformerEncoder

    
class GeoformerDecoder(nn.Module):
    r"""
    Geoformer decoder module.
    """
    def __init__(self, embedding_dim=128) -> None:
        super(GeoformerDecoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.act = nn.LeakyReLU(negative_slope=0.1)
        
        self.readout = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2), self.act, nn.Dropout(0.1),
            nn.Linear(self.embedding_dim // 2, self.embedding_dim // 4), self.act, nn.Dropout(0.1),
            nn.Linear(self.embedding_dim // 4, 1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.readout[0].weight, a=1)
        nn.init.kaiming_uniform_(self.readout[3].weight, a=1)
        nn.init.kaiming_uniform_(self.readout[6].weight, a=1)
        self.readout[0].bias.data.fill_(0)
        self.readout[3].bias.data.fill_(0)
        self.readout[6].bias.data.fill_(0)

    def forward(
        self,
        x: torch.Tensor, 
        edge_attr: torch.Tensor,  
        **kwargs
    ):
        edge_embedding = torch.mean(edge_attr, dim=2)
        
        # join node and edge embedding
        augmented_node_embedding = x + edge_embedding
        
        logits = self.readout(augmented_node_embedding)

        return logits


class GeoformerPretrainedModel(PreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super(GeoformerPretrainedModel, self).__init__(config)

        self.geo_encoder = GeoformerEncoder(
            pad_token_id=config.pad_token_id,
            max_z=config.max_z,
            embedding_dim=config.embedding_dim,
            ffn_embedding_dim=config.ffn_embedding_dim,
            num_layers=config.num_layers,
            num_rbf=config.num_rbf,
            rbf_trainable=config.rbf_trainable,
            cutoff=config.cutoff,
            num_attention_heads=config.num_attention_heads,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation_dropout=config.activation_dropout,
            norm_type=config.norm_type,
        )
        
        self.geo_decoder = GeoformerDecoder()

        self.post_init()
        
    def init_weights(self):
        self.geo_encoder.reset_parameters()
        self.geo_decoder.reset_parameters()

    def forward(
        self,
        z: torch.Tensor,  
        pos: torch.Tensor, 
        mask: torch.Tensor, 
    ):
        x, edge_attr = self.geo_encoder(z=z, pos=pos)
        logits = self.geo_decoder(x=x, edge_attr=edge_attr) 
        
        # (B, N, 1) -> (B * N, 1)
        logits = logits.view(-1, 1)[:, 0]
            
        return logits[mask]
    
    
class GeoformerConfig(PretrainedConfig):
    model_type = "geoformer"
    
    def __init__(
        self,
        max_z: int = 100,
        embedding_dim: int = 512,
        ffn_embedding_dim: int = 2048,
        num_layers: int = 9,
        num_attention_heads: int = 8,
        cutoff: int = 5.0,
        num_rbf: int = 64,
        rbf_trainable: bool = True,
        norm_type: str = "max_min",
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        dataset_root=None,
        pad_token_id: int = 0,
        **kwargs
    ):
        self.max_z = max_z
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.rbf_trainable = rbf_trainable
        self.norm_type = norm_type
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.dataset_root = dataset_root

        super(GeoformerConfig, self).__init__(
            pad_token_id=pad_token_id, **kwargs
        )


def create_model(config) -> GeoformerPretrainedModel:
    model_config = GeoformerConfig(
        max_z=config.max_z,
        embedding_dim=config.embedding_dim,
        ffn_embedding_dim=config.ffn_embedding_dim,
        num_layers=config.num_layers,
        num_attention_heads=config.num_heads,
        cutoff=config.cutoff,
        num_rbf=config.num_rbf,
        rbf_trainable=config.trainable_rbf,
        norm_type=config.norm_type,
        dropout=config.dropout,
        attention_dropout=config.attention_dropout,
        activation_dropout=config.activation_dropout,
        dataset_root=config.dataset_root,
        pad_token_id=config.pad_token_id,
    )

    return GeoformerPretrainedModel(model_config)