
import torch
import torch.nn as nn

from einops import rearrange, repeat
from torch_geometric.nn import Set2Set

from transformers import PreTrainedModel
from typing import Optional

from geoformer.config import GeoformerConfig
from geoformer.layers import CosineCutoff, ExpNormalSmearing, VecLayerNorm


class GeoformerMultiHeadAttention(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super(GeoformerMultiHeadAttention, self).__init__(*args, **kwargs)

        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads

        if not (
            self.head_dim * config.num_heads == self.embedding_dim
        ):
            raise AssertionError(
                "The embedding_dim must be divisible by num_heads."
            )

        self.act = nn.SiLU()
        self.cutoff = CosineCutoff(config.cutoff)

        self.dropout_module = nn.Dropout(
            p=config.attention_dropout, inplace=False
        )

        self.k_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.q_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.v_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.dk_proj = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.du_update_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.du_norm = VecLayerNorm(
            self.embedding_dim, trainable=False, norm_type=config.norm_type
        )
        self.dihedral_proj = nn.Linear(
            self.embedding_dim, 2 * self.embedding_dim, bias=False
        )
        self.edge_attr_update = nn.Linear(
            self.embedding_dim, self.embedding_dim
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.dk_proj.weight)
        self.dk_proj.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.du_update_proj.weight)
        self.du_update_proj.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.dihedral_proj.weight)
        nn.init.xavier_uniform_(self.edge_attr_update.weight)
        self.edge_attr_update.bias.data.fill_(0.0)

    def forward(
        self,
        x: torch.Tensor,  # (N, F)
        vec: Optional[torch.Tensor],  # (N, N, 3)
        dist: Optional[torch.Tensor],  # (N, N)
        edge_attr: Optional[torch.Tensor],  # (N, N, F)
        **kwargs,
    ):
        q = rearrange(
            self.q_proj(x), "n (h d) -> h n d", h=self.num_heads
        )  # (H, N, D)
        k = rearrange(
            self.k_proj(x), "n (h d) -> h n d", h=self.num_heads
        )  # (H, N, D)
        v = rearrange(
            self.v_proj(x), "n (h d) -> h n d", h=self.num_heads
        )  # (H, N, D)
        dk = rearrange(
            self.act(self.dk_proj(edge_attr)),
            "n m (h d) -> h n m d",
            h=self.num_heads,
        )  # (H, N, N, D)

        attn_weights = torch.einsum(
            "h n d, h n e, h n m d -> h n m", q, k, dk
        )  # (H, N, N)

        attn_scale = repeat(
            self.cutoff(dist), "n m -> h n m", h=self.num_heads
        )  # (H, N, N)
        attn_scale = rearrange(
            attn_scale, "h n m -> (h) n m", h=self.num_heads
        )  # H, N, N)
        attn_probs = self.act(attn_weights) * attn_scale  # (H, N, N)

        attn_per_nodes = attn_probs.unsqueeze(-1) * v.unsqueeze(
            -3
        )  # (H, N, N, D)
        attn_per_nodes = rearrange(
            attn_per_nodes, "(h) n m d -> n m (h d)", h=self.num_heads
        )  # (N, N, F)
        attn = attn_per_nodes.sum(dim=1)  # (N, F)

        du = (
            self.du_update_proj(attn_per_nodes)
            .unsqueeze(-2)
            * vec.unsqueeze(-1)
        ).sum(
            dim=-3
        )  # (N, 3, F)
        du = self.du_norm(du)  # (N, 3, F)
        ws, wt = torch.split(
            self.dihedral_proj(du), self.embedding_dim, dim=-1
        )  # (N, 3, F)
        ipe = (wt.unsqueeze(0) * ws.unsqueeze(1)).sum(dim=-2)  # (N, N, F)
        ipe = self.act(self.edge_attr_update(edge_attr)) * ipe  # (N, N, F)

        return attn, ipe


class GeoformerAttnBlock(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super(GeoformerAttnBlock, self).__init__(*args, **kwargs)

        self.embedding_dim = config.embedding_dim
        self.dropout_module = nn.Dropout(p=config.dropout, inplace=False)

        self.act = nn.SiLU()

        self.self_attn = GeoformerMultiHeadAttention(config)

        self.ffn = nn.Sequential(
            nn.Linear(self.embedding_dim, config.ffn_embedding_dim),
            self.act,
            nn.Dropout(p=config.activation_dropout, inplace=False),
            nn.Linear(config.ffn_embedding_dim, self.embedding_dim),
        )

        self.attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.self_attn.reset_parameters()
        nn.init.xavier_uniform_(self.ffn[0].weight)
        self.ffn[0].bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.ffn[3].weight)
        self.ffn[3].bias.data.fill_(0.0)
        self.attn_layer_norm.reset_parameters()
        self.final_layer_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,  # (N, F)
        vec: torch.Tensor,  # (N, N, 3)
        dist: torch.Tensor,  # (N, N)
        edge_attr: torch.Tensor,  # (N, N, F)
        **kwargs,
    ):
        # attention
        dx, dedge_attr = x, edge_attr
        x, edge_attr = self.self_attn(
            x=x,
            vec=vec,
            dist=dist,
            edge_attr=edge_attr,
        )

        x = self.dropout_module(x)
        x = x + dx
        x = self.attn_layer_norm(x)

        # ipe update
        edge_attr = edge_attr + dedge_attr

        # ffn
        dx = x
        x = self.ffn(x)
        x = self.dropout_module(x)
        x = x + dx

        x = self.final_layer_norm(x)

        return x, edge_attr


class GeoformerEncoder(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super(GeoformerEncoder, self).__init__(*args, **kwargs)
        self.embedding_dim = config.embedding_dim
        self.cutoff = config.cutoff

        self.embedding = nn.Embedding(config.max_z, self.embedding_dim)
        
        self.distance_expansion = ExpNormalSmearing(
            cutoff=config.cutoff,
            num_rbf=config.num_rbf,
            trainable=config.rbf_trainable,
        )
        self.dist_proj = nn.Linear(config.num_rbf, self.embedding_dim)
        self.act = nn.SiLU()

        self.layers = nn.ModuleList(
            [GeoformerAttnBlock(config) for _ in range(config.num_layers)]
        )

        self.x_in_layernorm = nn.LayerNorm(self.embedding_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        nn.init.xavier_uniform_(self.dist_proj.weight)
        self.dist_proj.bias.data.fill_(0.0)
        for layer in self.layers:
            layer.reset_parameters()
        self.x_in_layernorm.reset_parameters()

    def forward(
        self,
        z: torch.Tensor,  # (N)
        pos: torch.Tensor,  # (N, 3)
        **kwargs,
    ):      
        dist = torch.norm(
            pos.unsqueeze(1) - pos.unsqueeze(0),
            dim=-1,
        ) # (N, N)
    
        vec = (pos.unsqueeze(1) - pos.unsqueeze(0)) / (
            dist.unsqueeze(-1) + 1e-8
        )  # (N, N, 3)

        x = self.embedding(z)  # (N, F)
        x = self.x_in_layernorm(x) # (N, F)
        edge_attr = self.distance_expansion(dist)  # (N, N, num_rbf)
        edge_attr = self.act(self.dist_proj(edge_attr))  # (N, N, F)
        
        for layer in self.layers:
            x, edge_attr = layer(
                x=x,
                vec=vec,
                dist=dist,
                edge_attr=edge_attr,
            )

        return x, edge_attr


class GeoformerSpectraRegression(PreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super(GeoformerSpectraRegression, self).__init__(
            config, *inputs, **kwargs
        )
        self.config = config
        self.geo_encoder = GeoformerEncoder(config)
        
        self.pred_hidden_feats = config.pred_hid_feats

        self.readout_n = nn.Sequential(
            nn.Linear(self.embedding_dim, self.pred_hidden_dim), nn.PReLU(), nn.Dropout(0.2),
            nn.Linear(self.pred_hidden_dim, self.pred_hidden_dim), nn.PReLU(), nn.Dropout(0.2),
            nn.Linear(self.pred_hidden_dim, self.pred_hidden_dim), nn.PReLU(), nn.Dropout(0.2),
            nn.Linear(self.pred_hidden_dim, 1)
        )
        
        self.readout_g = Set2Set(
            in_channels=self.embedding_dim, processing_steps=3, num_layers=1
        )

        mean = torch.scalar_tensor(0) if config.mean is None else config.mean 
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean).float()
        self.register_buffer("mean", mean)

        std = torch.scalar_tensor(1) if config.std is None else config.std
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std).float()
        self.register_buffer("std", std)
        
        self.init_weights() 
        
    def init_weights(self):
        self.geo_encoder.reset_parameters()

        nn.init.xavier_uniform_(self.readout_n[0].weight)
        self.readout_n[0].bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.readout_n[3].weight)
        self.readout_n[3].bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.readout_n[6].weight)
        self.readout_n[6].bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.readout_n[9].weight)
        self.readout_n[9].bias.data.fill_(0.0)

    def forward(self, data, **kwargs):
        z, pos, mask, batch = data.z, data.pos, data.mask, data.batch
        
        node_embed_feats, _ = self.geo_encoder(z=z, pos=pos) # (N, F)

        graph_embed_feats = self.readout_g(node_embed_feats, batch) # (B, N, F)
        graph_embed_feats = torch.repeat_interleave(
            graph_embed_feats, repeats=node_embed_feats.shape[0] // graph_embed_feats.shape[0], dim=0
        ) # (N, F)
        
        pred = self.readout_n(
            torch.hstack([node_embed_feats, graph_embed_feats])[mask]
        )
       
        if self.std is not None:
            pred = pred * self.std

        if self.mean is not None:
            pred = pred + self.mean

        return pred[:, 0]


def create_model(config) -> GeoformerSpectraRegression:
    model_config = GeoformerConfig(
        max_z=config.max_z,
        embedding_dim=config.embedding_dim,
        ffn_embedding_dim=config.ffn_embedding_dim,
        pred_hidden_dim=config.pred_hidden_dim,
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
        mean=config.mean,
        std=config.std,
    )

    return GeoformerSpectraRegression(config=model_config)
