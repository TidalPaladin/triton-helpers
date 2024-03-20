import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.testing import assert_close

from triton_helpers.layers.fully_fused_mlp import fully_fused_mlp


class MLP(nn.Module):

    def __init__(self, D_in: int, D_hidden: int, D_out: int, depth: int = 1):
        super(MLP, self).__init__()
        self.layer_in = nn.Linear(D_in, D_hidden)
        self.layer_hidden = nn.ModuleList([nn.Linear(D_hidden, D_hidden) for _ in range(depth - 1)])
        self.layer_out = nn.Linear(D_hidden, D_out)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.layer_in(x))
        for layer in self.layer_hidden:
            x = F.relu(layer(x))
        x = self.layer_out(x)
        return x


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype,fp16_acc,tol",
    [
        (torch.float16, False, 1e-2),
        (torch.bfloat16, False, 1e-2),
        (torch.float16, True, 1e-2),
        (torch.bfloat16, True, 1e-2),
    ],
)
def test_forward_shallow(dtype, tol, fp16_acc):
    torch.random.manual_seed(0)
    D_in = 1
    D_hidden = 32
    D_out = 5
    B, L = 3, 40

    x = torch.randn((B, L, D_in), device="cuda")
    layer = MLP(D_in, D_hidden, D_out).to("cuda")

    baseline = layer(x)
    actual = fully_fused_mlp(
        x.to(dtype),
        layer.layer_in.weight.to(dtype),
        layer.layer_in.bias.to(dtype),
        layer.layer_out.weight.to(dtype),
        layer.layer_out.bias.to(dtype),
        fp16_acc=fp16_acc,
    )
    assert_close(baseline, actual, rtol=tol, atol=tol, check_dtype=False)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype,fp16_acc,tol",
    [
        (torch.float16, False, 1e-2),
        (torch.bfloat16, False, 1e-2),
        (torch.float16, True, 1e-2),
        (torch.bfloat16, True, 1e-2),
    ],
)
def test_forward_deep(dtype, tol, fp16_acc):
    torch.random.manual_seed(0)
    D_in = 1
    D_hidden = 32
    D_out = 5
    B, L = 3, 40

    x = torch.randn((B, L, D_in), device="cuda")
    layer = MLP(D_in, D_hidden, D_out, depth=3).to("cuda")

    w_hid = torch.cat([l.weight for l in layer.layer_hidden], dim=0)
    b_hid = torch.cat([l.bias.view(1, -1) for l in layer.layer_hidden], dim=0)

    baseline = layer(x)
    actual = fully_fused_mlp(
        x.to(dtype),
        layer.layer_in.weight.to(dtype),
        layer.layer_in.bias.to(dtype),
        layer.layer_out.weight.to(dtype),
        layer.layer_out.bias.to(dtype),
        w_hid.to(dtype),
        b_hid.to(dtype),
        fp16_acc=fp16_acc,
    )
    assert_close(baseline, actual, rtol=tol, atol=tol, check_dtype=False)
