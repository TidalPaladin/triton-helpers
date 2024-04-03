import pytest
import torch
import torch.nn as nn
from torch import Tensor
from torch.testing import assert_close

from triton_helpers.layers.reversible_fused_mlp import reversible_fused_mlp


class ReversibleMLP(nn.Module):

    def __init__(self, D: int, depth: int = 1, activation: nn.Module = nn.ReLU()):
        super().__init__()
        self.ff = nn.ModuleList([nn.Linear(D, D) for _ in range(depth)])
        self.activation = activation
        self.depth = depth

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.uniform_(layer.bias, -0.1, 0.1)

    def forward(self, x: Tensor) -> Tensor:
        lagging = x
        for i, layer in enumerate(self.ff):
            act = self.activation if i < self.depth - 1 else nn.Identity()
            tmp = x
            x = lagging + act(layer(x))
            lagging = tmp
        return x


@pytest.mark.cuda
@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize(
    "dtype,fp16_acc,tol",
    [
        (torch.float16, False, 1e-2),
        (torch.bfloat16, False, 2e-2),
        (torch.float16, True, 1e-2),
        (torch.bfloat16, True, 2e-2),
    ],
)
def test_forward(dtype, tol, fp16_acc, depth):
    torch.random.manual_seed(0)
    D = 32
    B, L = 1, 32

    x = torch.randn((B, L, D), device="cuda")
    layer = ReversibleMLP(D, depth=depth).to("cuda")

    w = torch.cat([l.weight for l in layer.ff], dim=0)
    b = torch.cat([l.bias.view(1, -1) for l in layer.ff], dim=0)

    baseline = layer(x)
    actual = reversible_fused_mlp(
        x.to(dtype),
        w.to(dtype),
        b.to(dtype),
        fp16_acc=fp16_acc,
    )
    assert_close(baseline, actual, rtol=tol, atol=tol, check_dtype=False)
