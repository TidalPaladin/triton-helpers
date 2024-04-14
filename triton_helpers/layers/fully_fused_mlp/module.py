import torch
import torch.nn as nn
import triton
from torch import Tensor

from .kernel import fully_fused_mlp


class FullyFusedMLP(nn.Module):

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        depth: int,
        activation: str | triton.JITFunction = "relu",
        fp16_acc: bool = False,
    ):
        super().__init__()
        # Input
        layer = nn.Linear(d_in, d_hidden)
        self.w_in = layer.weight
        self.b_in = layer.bias

        # Hidden
        if depth > 1:
            layer = nn.ModuleList([nn.Linear(d_hidden, d_hidden) for _ in range(depth - 1)])
            self.w_hid = nn.Parameter(torch.cat([layer.weight for layer in layer], 0))
            self.b_hid = nn.Parameter(torch.stack([layer.bias for layer in layer], 0))
        else:
            self.w_hid = self.b_hid = None

        # Output
        layer = nn.Linear(d_hidden, d_out)
        self.w_out = layer.weight
        self.b_out = layer.bias

        self.activation = activation
        self.fp16_acc = fp16_acc

    def forward(self, x: Tensor) -> Tensor:
        return fully_fused_mlp(
            x,
            self.w_in,
            self.b_in,
            self.w_out,
            self.b_out,
            self.w_hid,
            self.b_hid,
            self.activation,
            fp16_acc=self.fp16_acc,
        )
