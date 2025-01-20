import torch
import torch.nn as nn
import torch.nn.functional as F
import ase

from transformers import PreTrainedModel

from network.encoder import GeoformerEncoder


class GeoformerBaseDecoder(nn.Module):
    def __init__(self, embedding_dim=128, num_classes=1) -> None:
        super(GeoformerBaseDecoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.act = nn.LeakyReLU(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            self.act,
            nn.Linear(self.embedding_dim // 2, self.num_classes),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.classifier[0].weight)
        self.classifier[0].bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.classifier[2].weight)
        self.classifier[2].bias.data.fill_(0.0)

    def forward(
        self,
        x: torch.Tensor,  # (B, N, F)
        edge_attr: torch.Tensor,  # (B, N, N, F)
        **kwargs,
    ):
        return self.classifier(x) + edge_attr.sum() * 0


class GeoformerDecoder(GeoformerBaseDecoder):
    def __init__(self):
        super(GeoformerDecoder, self).__init__()
        atomic_mass = torch.from_numpy(ase.data.atomic_masses).float()
        self.register_buffer("atomic_mass", atomic_mass)

    def forward(
        self,
        x: torch.Tensor,  # (B, N, F)
        edge_attr: torch.Tensor,  # (B, N, N, F)
        **kwargs,
    ):
        x = self.classifier(x) + edge_attr.sum() * 0  # (B, N, 1)

        # Get center of mass.
        z = kwargs["z"]  # (B, N)
        pos = kwargs["pos"]  # (B, N, 3)
        padding_mask = kwargs["padding_mask"]  # (B, N)
        mass = (
            self.atomic_mass[z].masked_fill(padding_mask, 0.0).unsqueeze(-1)
        )  # (B, N, 1)
        
        c = torch.sum(mass * pos, dim=-2) / torch.sum(mass, dim=-2)
        x = torch.norm(pos - c.unsqueeze(-2), dim=-1, keepdim=True) ** 2 * x
        
        return x  # (B, N, 1)


class GeoformerModel(PreTrainedModel):
    def __init__(self):
        super(GeoformerModel, self).__init__()

        self.geo_encoder = GeoformerEncoder()
        self.geo_decoder = GeoformerDecoder()

        self.post_init()

    def init_weights(self):
        self.geo_encoder.reset_parameters()
        self.geo_decoder.reset_parameters()


class ChemicalShiftRegression(GeoformerModel):
    def __init__(self, config, *inputs, **kwargs):
        super(ChemicalShiftRegression, self).__init__(
            config, *inputs, **kwargs
        )

        self.config = config
        self.aggr = config.aggr
        self.pad_token_id = config.pad_token_id

    def forward(
        self,
        z: torch.Tensor,  # (B, N)
        pos: torch.Tensor,  # (B, N, 3)
        **kwargs,
    ):
        x, edge_attr = self.geo_encoder(z=z, pos=pos)

        padding_mask = z == self.pad_token_id  # (B, N)

        # (B, N, 1) or (B, N, 3)
        x = self.geo_decoder(
            x=x, edge_attr=edge_attr, z=z, pos=pos, padding_mask=padding_mask
        )

        logits = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)  # (B, N, 1)

        if self.std is not None:
            logits = logits * self.std

        logits = (
            self.prior_model(logits, z)
            if self.prior_model is not None
            else logits
        )

        if self.aggr == "sum":
            logits = logits.sum(dim=1)  # (B, 1)
        elif self.aggr == "mean":
            logits = logits.sum(dim=1) / (~padding_mask).sum(dim=-1).unsqueeze(
                -1
            )  # (B, 1)
        else:
            NotImplementedError(f"Unknown aggregation method: {self.aggr}")

        return logits



