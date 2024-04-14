from dataclasses import dataclass
from typing import Dict, Any

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch import Tensor

from ...benchmark import CLI, KernelExecutor
from .kernel import hash_encoding
from .module import HashEncoding

try:
    from tinycudann.modules import Encoding
except ImportError:
    Encoding = None


@dataclass
class TinyCudaNN(KernelExecutor):

    def prepare_inputs(self, L: int, D: int, F: int, T: int, N: int, X: int, Y: int, **kwargs) -> Dict[str, Any]:
        config = {
            "otype": "Grid",
            "type": "Hash",
            "n_levels": N,
            "n_features_per_level": F,
            "log2_hashmap_size": int(math.log2(T)),
            "base_resolution": X,
            "per_level_scale": math.log2(Y // X),
            "interpolation": "Linear"
        }
        device = kwargs.get("device", "cuda")
        module = Encoding(D, config, dtype=kwargs.get("dtype", torch.float32)).to(device)
        x = self.rand((32, L, D), **kwargs)
        return {
            "x": x,
            "module": module,
        }

    def forward(
        self,
        x: Tensor,
        module: nn.Module,
    ) -> Tensor:
        return module(x.view(-1, x.size(-1)))



@dataclass
class Triton(KernelExecutor):

    def prepare_inputs(self, L: int, D: int, F: int, T: int, N: int, X: int, Y: int, **kwargs) -> Dict[str, Tensor | None]:
        x = self.rand((32, L, D), **kwargs)
        layer = HashEncoding(T, N, F, X, Y).to(x.device)
        dtype = kwargs.get("dtype", torch.float32)
        return {
            "x": x,
            "e": layer.embeddings.to(dtype),
            "pi": layer.pi,
        }

    def forward(
        self,
        x: Tensor,
        e: Tensor,
        pi: Tensor,
    ) -> Tensor:
        return hash_encoding(x, e, None, pi)


if __name__ == "__main__":
    CLI.entrypoint(
        "InstantNGP Hash Encoding",
        [TinyCudaNN("tiny-cuda-nn"), Triton("triton-hash")],
        dims={
            "L": ((512, 2048, 8192, 16384, 32768, 65536), "values"),
            "D": (3, "values"),
            "F": (2, "values"),
            "T": (2**14, "values"),
            "N": (16, "values"),
            "X": (16, "values"),
            "Y": (32, "values"),
        },
    )