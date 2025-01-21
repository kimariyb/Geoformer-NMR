import torch
import torch.nn as nn

from transformers import PreTrainedModel

from network.encoder import GeoformerEncoder

    
class GeoformerDecoder(nn.Module):
    r"""
    Geoformer decoder module.
    """
    def __init__(self, embedding_dim=128) -> None:
        super(GeoformerDecoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.act = nn.LeakyReLU(negative_slope=0.1)
        self.node = nn.Embedding(embedding_dim=self.embedding_dim)
        self.edge = nn.Embedding(embedding_dim=self.embedding_dim)
        self.readout = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim), self.act, nn.Dropout(0.1),
            nn.Linear(self.embedding_dim, self.embedding_dim), self.act, nn.Dropout(0.1),
            nn.Linear(self.embedding_dim, self.embedding_dim), self.act, nn.Dropout(0.1),
            nn.Linear(self.embedding_dim, 1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.node.weight, a=1)
        nn.init.kaiming_uniform_(self.edge.weight, a=1)
        nn.init.kaiming_uniform_(self.readout[0].weight, a=1)
        nn.init.kaiming_uniform_(self.readout[3].weight, a=1)
        nn.init.kaiming_uniform_(self.readout[6].weight, a=1)
        nn.init.kaiming_uniform_(self.readout[9].weight, a=1)
        

    def forward(
        self,
        x: torch.Tensor,  # (B, N, F)
        edge_attr: torch.Tensor,  # (B, N, N, F)
        **kwargs
    ):
        # (B, N, F)
        node_embedding = self.node(x)
        # (B, N, F)
        edge_embedding = self.edge(edge_attr)
        
        # Augmented node embedding
        # (B, N, F)
        augmented_node_embedding = node_embedding + edge_embedding

        # (B, N, 1)
        logits = self.readout(augmented_node_embedding)

        return logits


class GeoformerModel(PreTrainedModel):
    def __init__(self):
        super(GeoformerModel, self).__init__()

        self.geo_encoder = GeoformerEncoder()
        self.geo_decoder = GeoformerDecoder()

        self.post_init()
        
    def save_pretrained(self, save_directory, is_main_process = True, state_dict = None, save_function = torch.save, push_to_hub = False, max_shard_size = "5GB", safe_serialization = True, variant = None, token = None, save_peft_format = True, **kwargs):
        return self.save_pretrained(save_directory, is_main_process, state_dict, save_function, push_to_hub, max_shard_size, safe_serialization, variant, token, save_peft_format, **kwargs)

    def load_pretrained(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        return self.load_pretrained(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def init_weights(self):
        self.geo_encoder.reset_parameters()
        self.geo_decoder.reset_parameters()


    def forward(
        self,
        z: torch.Tensor,  # (B, N, F)
        pos: torch.Tensor,  # (B, N, 3)
        mask: torch.Tensor,  # (B, N)
    ):
        x, edge_attr = self.geo_encoder(z=z, pos=pos)
        logits = self.geo_decoder(x=x, edge_attr=edge_attr) # (B, N, 1)
        
        return logits[mask]
