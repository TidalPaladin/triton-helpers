import torch
import torch.nn as nn
from torch import Tensor

from .kernel import PI_1, PI_2, PI_3, compute_embedding_counts, hash_encoding


class HashEncoding(nn.Module):

    def __init__(
        self,
        max_entries_per_level: int = 2**14,
        num_levels: int = 16,
        d_in: int = 3,
        d_embed: int = 2,
        min_res: int = 16,
        max_res: int = 512,
    ):
        super().__init__()
        t = compute_embedding_counts(num_levels, max_entries_per_level, d_in, min_res, max_res)
        self.embeddings = nn.Parameter(torch.cat([torch.randn(int(t[i]), d_embed) for i in range(num_levels)], dim=0))
        self.register_buffer("pi", torch.tensor([PI_1, PI_2, PI_3], dtype=torch.int64))
        self.max_entries_per_level = max_entries_per_level
        self.num_levels = num_levels
        self.min_res = min_res
        self.max_res = max_res

    def forward(self, coords: Tensor, features: Tensor | None = None) -> Tensor:
        return hash_encoding(
            coords,
            self.embeddings,
            features,
            self.pi,
            self.max_entries_per_level,
            self.min_res,
            self.max_res,
            self.num_levels,
        )
