import math
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
from torch import Tensor

from ...benchmark import CLI, KernelExecutor
from .kernel import hash_encoding
from .module import HashEncoding


try:
    from tinycudann.modules import Encoding  # type: ignore
except ImportError:
    Encoding = None


@dataclass
class TinyCudaNN(KernelExecutor):

    def prepare_inputs(self, L: int, D: int, F: int, T: int, N: int, X: int, Y: int, **kwargs) -> Dict[str, Any]:
        assert Encoding is not None, f"{self.__class__.__name__} requires TinyCudaNN to be installed."
        b = math.exp((math.log(Y) - math.log(X)) / (N - 1))
        config = {
            "otype": "HashGrid",
            "n_levels": N,
            "n_features_per_level": F,
            "log2_hashmap_size": int(math.log2(T) + 1e-5),
            "base_resolution": X,
            "per_level_scale": b + 1e-5,
            "interpolation": "Linear",
        }
        device = kwargs.get("device", "cuda")
        module = Encoding(D, config, dtype=kwargs.get("dtype", torch.float32)).to(device)
        torch.random.manual_seed(0)
        x = self.rand((32, L, D), **kwargs)
        return {
            "x": x.detach(),
            "module": module,
        }

    def forward(self, x: Tensor, module: nn.Module) -> Tensor:
        return module(x.view(-1, x.size(-1)))


@dataclass
class Triton(KernelExecutor):

    @torch.no_grad()
    def prepare_inputs(
        self, L: int, D: int, F: int, T: int, N: int, X: int, Y: int, **kwargs
    ) -> Dict[str, Tensor | None]:
        torch.random.manual_seed(0)
        x = self.rand((32, L, D), **kwargs)
        layer = HashEncoding(T, N, D, F, X, Y).to(kwargs.get("device", "cuda"))
        dtype = kwargs.get("dtype", torch.float32)
        e = layer.embeddings.to(dtype)
        e.requires_grad = kwargs.get("requires_grad", False)
        assert layer.pi is not None
        return {
            "x": x.detach(),
            "e": e,
            "pi": layer.pi.detach(),
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
            "L": ((512, 2048, 8192, 16384, 32768, 65536, 65536 * 2), "values"),
            "D": (3, "values"),
            "F": (2, "values"),
            "T": (2**14, "values"),
            "N": (16, "values"),
            "X": (16, "values"),
            "Y": (512, "values"),
        },
    )
