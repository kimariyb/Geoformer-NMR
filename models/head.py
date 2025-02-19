import torch
import torch.nn as nn

from einops import rearrange

from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

from models.encoder import GeoformerEncoder
from models.config import GeoformerConfig


class GeoformerPredictorHead(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super(GeoformerPredictorHead, self).__init__(*args, **kwargs)
        
        self.embedding_dim = config.embedding_dim
        self.dropout = nn.Dropout(config.dropout)
        self.act = nn.LeakyReLU(negative_slope=0.1)
        
        self.readout = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2), self.act, self.dropout,
            nn.Linear(self.embedding_dim // 2, self.embedding_dim // 4), self.act, self.dropout,
            nn.Linear(self.embedding_dim // 4, self.embedding_dim // 8), self.act, self.dropout,
            nn.Linear(self.embedding_dim // 8, 1),
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.readout[0].weight)
        nn.init.xavier_uniform_(self.readout[3].weight)
        nn.init.xavier_uniform_(self.readout[6].weight)
        nn.init.xavier_uniform_(self.readout[9].weight)
        self.readout[0].bias.data.fill_(0.0)
        self.readout[3].bias.data.fill_(0.0)
        self.readout[6].bias.data.fill_(0.0)
        self.readout[9].bias.data.fill_(0.0)

    def forward(
        self,
        x: torch.Tensor,  # (B, N, F)
        edge_attr: torch.Tensor,  # (B, N, N, F)
        **kwargs,
    ):
        x = self.readout(x) + edge_attr.sum() * 0  # (B, N, 1)
        
        return x  # (B, N, 1)
    

class GeoformerModel(PreTrainedModel):
    def __init__(self, config: PretrainedConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.encoder = GeoformerEncoder(config)
        self.head = GeoformerPredictorHead(config)
        
        self.post_init()
    
    def init_weights(self):
        self.encoder.reset_parameters()
        self.head.reset_parameters()
        
        
class GeoformerForNmrPrediction(GeoformerModel):
    def __init__(self, config, *args, **kwargs) -> None:
        super(GeoformerForNmrPrediction, self).__init__(config, *args, **kwargs)
        self.config = config
        self.pad_token_id = config.pad_token_id
    
    def forward(self, batch):
        z, pos, mask = batch["z"], batch["pos"], batch["mask"]
        x, edge_attr = self.encoder(z=z, pos=pos)

        padding_mask = z == self.pad_token_id  # (B, N)

        # (B, N, 1) 
        x = self.head(x=x, edge_attr=edge_attr)

        logits = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)  # (B, N, 1)
        
        # (B, N) -> (B*N)
        logits = rearrange(logits, "b n 1 -> (b n)")
    

        return logits[mask]

    
def create_model(config) -> GeoformerForNmrPrediction:
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
        dataset=config.dataset,
        pad_token_id=config.pad_token_id,
    )

    return GeoformerForNmrPrediction(config=model_config)