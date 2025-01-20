import torch
import torch.nn as nn

from typing import Optional
from einops import rearrange, repeat

from network.layers import CosineCutoff, ExpNormalSmearing, VecLayerNorm
                                                    


class GeoformerMultiHeadAttention(nn.Module):
    r"""
    Multi-head attention layer for Geoformer.

    Parameters
    ----------
    embedding_dim : int
        Size of the embedding dimension.
    num_attention_heads : int
        Number of attention heads.
    attention_dropout : float
        Dropout rate for attention weights.
    cutoff : float
        Cutoff distance for attention weights.
    norm_type : str, optional
        Type of normalization for the output of the attention layer.
        Default: "max_min"
    """
    def __init__(
        self, 
        embedding_dim: int,
        num_attention_heads: int,
        attention_dropout: float,
        cutoff: float,
        norm_type: str = "max_min",
    ) -> None:
        super(GeoformerMultiHeadAttention, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_attention_heads
        self.head_dim = embedding_dim // num_attention_heads


        self.act = nn.GELU()
        self.cutoff = CosineCutoff(cutoff)

        self.dropout_module = nn.Dropout(
            p=attention_dropout, inplace=False
        )

        self.k_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.q_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.v_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.dk_proj = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.du_update_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.du_norm = VecLayerNorm(
            self.embedding_dim, trainable=False, norm_type=norm_type
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
        x: torch.Tensor,  # (B, N, F)
        vec: Optional[torch.Tensor],  # (B, N, N, 3)
        dist: Optional[torch.Tensor],  # (B, N, N)
        edge_attr: Optional[torch.Tensor],  # (B, N, N, F)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, N)
        **kwargs,
    ):
        q = rearrange(
            self.q_proj(x), "b n (h d) -> (b h) n d", h=self.num_heads
        )  # (BH, N, D)
        k = rearrange(
            self.k_proj(x), "b n (h d) -> (b h) n d", h=self.num_heads
        )  # (BH, N, D)
        v = rearrange(
            self.v_proj(x), "b n (h d) -> (b h) n d", h=self.num_heads
        )  # (BH, N, D)
        dk = rearrange(
            self.act(self.dk_proj(edge_attr)),
            "b n m (h d) -> (b h) n m d",
            h=self.num_heads,
        )  # (BH, N, N, D)

        attn_weights = ((q.unsqueeze(-2) * k.unsqueeze(-3)) * dk).sum(
            dim=-1
        )  # (BH, N, N)

        if key_padding_mask is not None:
            attn_weights = rearrange(
                attn_weights, "(b h) n m -> b h n m", h=self.num_heads
            )
            attn_weights = attn_weights.masked_fill(
                rearrange(key_padding_mask, "b n m -> b () n m"),
                0.0,
            )
            attn_weights = rearrange(attn_weights, "b h n m -> (b h) n m")

        attn_scale = repeat(
            self.cutoff(dist), "b n m -> b h n m", h=self.num_heads
        )  # (BH, N, N)
        attn_scale = rearrange(
            attn_scale, "b h n m -> (b h) n m", h=self.num_heads
        )  # (BH, N, N)
        attn_probs = self.act(attn_weights) * attn_scale  # (BH, N, N)

        attn_per_nodes = attn_probs.unsqueeze(-1) * v.unsqueeze(
            -3
        )  # (BH, N, N, D)
        attn_per_nodes = rearrange(
            attn_per_nodes, "(b h) n m d -> b n m (h d)", h=self.num_heads
        )  # (B, N, N, F)
        attn = attn_per_nodes.sum(dim=2)  # (B, N, F)

        du = (
            self.du_update_proj(attn_per_nodes)
            .masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
            .unsqueeze(-2)
            * vec.unsqueeze(-1)
        ).sum(
            dim=-3
        )  # (B, N, 3, F)
        du = self.du_norm(du)  # (B, N, 3, F)
        ws, wt = torch.split(
            self.dihedral_proj(du), self.embedding_dim, dim=-1
        )  # (B, N, 3, F)
        ipe = (wt.unsqueeze(1) * ws.unsqueeze(2)).sum(dim=-2)  # (B, N, N, F)
        ipe = self.act(self.edge_attr_update(edge_attr)) * ipe  # (B, N, N, F)

        return attn, ipe


class GeoformerAttnBlock(nn.Module):
    def __init__(
        self, 
        embedding_dim: int,
        ffn_embedding_dim: int,
        num_attention_heads: int,
        dropout: float,
        attention_dropout: float,
        activation_dropout: float,
        cutoff: float,
        norm_type: str = "max_min",
    ) -> None:
        super(GeoformerAttnBlock, self).__init__()

        self.embedding_dim = embedding_dim
        self.dropout_module = nn.Dropout(p=dropout, inplace=False)

        self.act = nn.GELU()

        self.self_attn = GeoformerMultiHeadAttention(
            embedding_dim=embedding_dim,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            cutoff=cutoff,
            norm_type=norm_type,
        )

        self.ffn = nn.Sequential(
            nn.Linear(self.embedding_dim, ffn_embedding_dim),
            self.act,
            nn.Dropout(p=activation_dropout, inplace=False),
            nn.Linear(ffn_embedding_dim, self.embedding_dim),
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
        x: torch.Tensor,  # (B, N, F)
        vec: torch.Tensor,  # (B, N, N, 3)
        dist: torch.Tensor,  # (B, N, N)
        edge_attr: torch.Tensor,  # (B, N, N, ?)
        key_padding_mask: Optional[
            torch.Tensor
        ],  # [padding, cutoff] (B, N, N)
        **kwargs,
    ):
        # attention
        dx, dedge_attr = x, edge_attr
        x, edge_attr = self.self_attn(
            x=x,
            vec=vec,
            dist=dist,
            edge_attr=edge_attr,
            key_padding_mask=key_padding_mask,
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
    r"""
    Geoformer encoder.

    Parameters
    ----------
    pad_token_id : int
        Padding token id.
    max_z : int
        Maximum number of atoms in the dataset.
    embedding_dim : int
        Size of the embedding dimension.
    ffn_embedding_dim : int
        Size of the feedforward network embedding dimension.
    num_layers : int
        Number of layers.
    num_rbf : int
        Number of radial basis functions for distance expansion.
    rbf_trainable : bool
        Whether the radial basis functions are trainable.
    cutoff : float
        Cutoff distance for attention weights.
    num_attention_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate for the output of each layer.
    attention_dropout : float
        Dropout rate for attention weights.
    activation_dropout : float
        Dropout rate for activation weights.
    norm_type : str, optional
        Type of normalization for the output of the attention layer.
        Default: "max_min"
    """
    def __init__(
        self,
        pad_token_id: int,
        max_z: int,
        embedding_dim: int,
        ffn_embedding_dim: int,
        num_layers: int,
        num_rbf: int,
        rbf_trainable: bool,
        cutoff: float,
        num_attention_heads: int,
        dropout: float,
        attention_dropout: float,
        activation_dropout: float,
        norm_type: str = "max_min",
    ) -> None:
        super(GeoformerEncoder, self).__init__()

        self.pad_token_id = pad_token_id
        self.embedding_dim = embedding_dim
        self.cutoff = cutoff

        self.embedding = nn.Embedding(
            max_z, self.embedding_dim, padding_idx=self.pad_token_id
        )
        self.distance_expansion = ExpNormalSmearing(
            cutoff=cutoff,
            num_rbf=num_rbf,
            trainable=rbf_trainable,
        )
        self.dist_proj = nn.Linear(num_rbf, self.embedding_dim)
        self.act = nn.GELU()

        self.layers = nn.ModuleList([
            GeoformerAttnBlock(
                embedding_dim=embedding_dim,
                ffn_embedding_dim=ffn_embedding_dim,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                cutoff=cutoff,
                norm_type=norm_type,        
            ) for _ in range(num_layers)
        ])

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
        z: torch.Tensor,  # (B, N)
        pos: torch.Tensor,  # (B, N, 3)
        **kwargs,
    ):
        B, N, *_ = z.shape
        # generate mask
        padding_mask = z == self.pad_token_id  # (B, N)
        pos_mask = ~(
            padding_mask.unsqueeze(1) | padding_mask.unsqueeze(2)
        )  # (B, N, N)
        dist = torch.norm(
            pos.unsqueeze(1) - pos.unsqueeze(2), dim=-1
        )  # (B, N, N)
        loop_mask = torch.eye(N, dtype=torch.bool, device=dist.device)
        loop_mask = repeat(loop_mask, "n m -> b n m", b=B)  # (B, N, N)
        dist = dist.masked_fill(loop_mask, 0.0)  # (B, N, N)
        adj_mask = (dist < self.cutoff) & pos_mask  # (B, N, N)
        loop_adj_mask = ~loop_mask & adj_mask  # (B, N, N)

        vec = (pos.unsqueeze(1) - pos.unsqueeze(2)) / (
            dist.unsqueeze(-1) + 1e-8
        )  # (B, N, N, 3)
        vec = vec.masked_fill(
            ~loop_adj_mask.unsqueeze(-1), 0.0
        )  # (B, N, N, 3)

        key_padding_mask = (
            (~adj_mask)
            .masked_fill(padding_mask.unsqueeze(-1), False)
            .masked_fill(padding_mask.unsqueeze(-2), True)
        )

        x = self.embedding(z)  # (B, N, F)
        x = self.x_in_layernorm(x)
        edge_attr = self.distance_expansion(dist)  # (B, N, N, num_rbf)
        edge_attr = self.act(self.dist_proj(edge_attr))  # (B, N, N, F)
        edge_attr = edge_attr.masked_fill(
            ~adj_mask.unsqueeze(-1), 0.0
        )  # (B, N, N, F)

        for layer in self.layers:
            x, edge_attr = layer(
                x=x,
                vec=vec,
                dist=dist,
                edge_attr=edge_attr,
                key_padding_mask=key_padding_mask,
            )

        return x, edge_attr


