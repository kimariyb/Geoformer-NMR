from typing import Any, Tuple, List

import torch
import numpy as np


class GeoformerDataCollator:
    def __init__(self, max_nodes=None) -> None:
        self.max_nodes = max_nodes

    @staticmethod
    def _pad_feats(feats: torch.Tensor, max_node: int) -> torch.Tensor:
        N, *_ = feats.shape
        if N <= max_node:
            feats_padded = torch.zeros([max_node, *_], dtype=feats.dtype)
            feats_padded[:N] = feats
        else:
            print(
                f"Warning: max_node {max_node} is too small to hold all nodes {N} in a batch"
            )
            print("Play truncation...")

        return feats_padded

    def __call__(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        graph_list, shift_list, masks_list = map(list, zip(*batch))

        max_node = (
            max(len(graph.z) for graph in graph_list)
            if self.max_nodes is None
            else self.max_nodes
        )

        
        batch_z = torch.stack(
            [self._pad_feats(torch.LongTensor(graph.z), max_node) for graph in graph_list]
        )

        batch_pos = torch.stack(
            [self._pad_feats(torch.FloatTensor(graph.pos), max_node) for graph in graph_list]
        )

        batch_label = torch.stack(
            [self._pad_feats(torch.FloatTensor(shift), max_node) for shift in shift_list]
        )

        batch_mask = torch.stack(
            [self._pad_feats(torch.BoolTensor(mask), max_node) for mask in masks_list]
        )

        return batch_z, batch_pos, batch_label, batch_mask
