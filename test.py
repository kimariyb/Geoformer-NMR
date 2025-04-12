from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from data.carbon import CarbonDataset
from geoformer.layers import CosineCutoff, ExpNormalSmearing, VecLayerNorm


if __name__ == "__main__":
    dataset = CarbonDataset(root="./data/dataset/carbon")
    
    data = dataset[0]
    z = data["z"] # (N)
    pos = data["pos"] # (N, 3)
    
    embedding = nn.Embedding(55, 128)
    x_in_layernorm = nn.LayerNorm(128)
    x = embedding(z) # (N, 128)
    x = x_in_layernorm(x) # (N, 128)
    print(x.shape)
    
    dist = torch.norm(
        pos.unsqueeze(1) - pos.unsqueeze(0),
        dim=-1,
    ) # (N, N)
    print(dist.shape)
    
    distance_expansion = ExpNormalSmearing(
        cutoff=5.0,
        num_rbf=32,
        trainable=True,
    )
    
    edge_attr = distance_expansion(dist) # (N, N, 32)
    dist_proj = nn.Linear(32, 128)
    edge_attr = dist_proj(edge_attr) # (N, N, 128)
    print(edge_attr.shape)
    
       
    vec = (pos.unsqueeze(1) - pos.unsqueeze(0)) / (
        dist.unsqueeze(-1) + 1e-8
    )  # (N, N, 3)
    print(vec.shape)
    
    
    k_proj = nn.Linear(128, 128)
    q_proj = nn.Linear(128, 128)
    v_proj = nn.Linear(128, 128)
    dk_proj = nn.Linear(128, 128)
    
    q = rearrange(
        q_proj(x), "n (h d) -> h n d", h=8
    )  # (H, N, D)
    k = rearrange(
        k_proj(x), "n (h d) -> h n d", h=8
    )  # (H, N, D)
    v = rearrange(
        v_proj(x), "n (h d) -> h n d", h=8
    )  # (H, N, D)
    dk = rearrange(
        dk_proj(edge_attr),
        "n m (h d) -> h n m d",
        h=8,
    )  # (H, N, N, D)
    print(q.shape, k.shape, v.shape, dk.shape)
    
    attn_weights = torch.einsum(
        "h n d, h n e, h n m d -> h n m", q, k, dk
    )  # (H, N, N)
    print(attn_weights.shape)
    
    cutoff = CosineCutoff(5.0)
    attn_scale = repeat(
        cutoff(dist), "n m -> h n m", h=8
    )
    attn_scale = rearrange(
        attn_scale, "h n m -> (h) n m", h=8
    )  # (H, N, N)
    print(attn_scale.shape)
    
    attn_probs = attn_weights * attn_scale  # (H, N, N)
    attn_per_nodes = attn_probs.unsqueeze(-1) * v.unsqueeze(-3)  # (H, N, N, D)
    attn_per_nodes = rearrange(
        attn_per_nodes, 
        "(h) n m d -> n m (h d)", h=8
    )  # (N, N, F)
    print(attn_per_nodes.shape)
    attn = attn_per_nodes.sum(dim=1)  # (N, F)
    print(attn.shape)

    du_update_proj =  nn.Linear(128, 128)
    du = (
        du_update_proj(attn_per_nodes) # (N, N, F)
        .unsqueeze(-2) # (N, N, 1, F)
        * vec.unsqueeze(-1) # (N, N, 3, F)
    ).sum(
        dim=-3
    )  # (N, 3, F)
    
    du_norm = VecLayerNorm(
        128, trainable=False, norm_type="max_min"
    )
    du = du_norm(du)  # (N, 3, F)
    print(du.shape)
    
    dihedral_proj = nn.Linear(128, 2 * 128, bias=False)
    ws, wt = torch.split(
        dihedral_proj(du), 128, dim=-1
    ) 
    print(ws.shape, wt.shape)
    
    ipe = (wt.unsqueeze(0) * ws.unsqueeze(1)).sum(dim=-2)  # (N, N, F)
    print(ipe.shape)