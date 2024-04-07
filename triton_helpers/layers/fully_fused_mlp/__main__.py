from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from ...benchmark import CLI, KernelExecutor
from .kernel import fully_fused_mlp


@dataclass
class Baseline(KernelExecutor):

    def prepare_inputs(self, L: int, D: int, H: int, **kwargs) -> Dict[str, Tensor | None]:
        x = self.randn((32, L, D), **kwargs)
        w_in = self.randn((D, D), **kwargs)
        b_in = self.randn((D,), **kwargs)
        w_out = self.randn((D, D), **kwargs)
        b_out = self.randn((D,), **kwargs)

        if H > 1:
            w_hid = self.randn((D * H, D), **kwargs)
            b_hid = self.randn((H, D), **kwargs)
        else:
            w_hid = None
            b_hid = None

        return {
            "x": x,
            "w_in": w_in,
            "b_in": b_in,
            "w_out": w_out,
            "b_out": b_out,
            "w_hid": w_hid,
            "b_hid": b_hid,
            "H": torch.tensor(H),
        }

    def forward(
        self,
        x: Tensor,
        w_in: Tensor,
        b_in: Tensor,
        w_out: Tensor,
        b_out: Tensor,
        w_hid: Tensor | None,
        b_hid: Tensor | None,
        H: Tensor,
    ) -> Tensor:
        x = F.linear(x, w_in, b_in)
        x = F.relu(x)

        D_h = w_in.shape[1]
        for i in range(H - 1):
            w = w_hid[D_h * i : D_h * (i + 1), :]  # type: ignore
            b = b_hid[i : i + 1, :]  # type: ignore
            x = F.linear(x, w, b)
            x = F.relu(x)

        x = F.linear(x, w_out, b_out)
        return x


@dataclass
class Triton(Baseline):
    fp16_acc: bool = False

    def forward(
        self,
        x: Tensor,
        w_in: Tensor,
        b_in: Tensor,
        w_out: Tensor,
        b_out: Tensor,
        w_hid: Tensor | None,
        b_hid: Tensor | None,
        H: Tensor,
    ) -> Tensor:
        return fully_fused_mlp(x, w_in, b_in, w_out, b_out, w_hid, b_hid, fp16_acc=self.fp16_acc)


if __name__ == "__main__":
    CLI.entrypoint(
        "MLP",
        [Baseline("baseline"), Triton("fully-fused"), Triton("fully-fused-acc16", fp16_acc=True)],
        dims={
            "H": ((1, 10, 1), "linspace"),
            "D": (64, "values"),
            "L": (1024, "values"),
        },
    )
