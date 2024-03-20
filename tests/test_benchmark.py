from dataclasses import dataclass
from typing import Dict

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from triton_helpers.benchmark import CLI, KernelExecutor


@dataclass
class Baseline(KernelExecutor):

    def prepare_inputs(self, L: int, D: int, **kwargs) -> Dict[str, Tensor]:
        x = self.randn((4, L, D), **kwargs)
        w = self.randn((D,), **kwargs)
        b = self.randn((D,), **kwargs)
        return {"x": x, "w": w, "b": b}

    def forward(self, x: Tensor, w: Tensor, b: Tensor) -> Tensor:
        return F.layer_norm(x, (x.shape[-1],), w, b)


@pytest.fixture
def entrypoint():
    def fn(argv):
        CLI.entrypoint(
            "LayerNorm",
            [Baseline("torch")],
            dims={
                "D": ((6, 15, 10), "logspace"),
                "L": (256, "values"),
            },
            argv=argv,
        )

    return fn


@pytest.mark.parametrize("line_arg", ["kernel", "dtype", "mode", "L"])
@pytest.mark.parametrize("mode", ["fwd", "bwd", "fwd-bwd"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.cuda),
    ],
)
def test_benchmark(mocker, entrypoint, tmp_path, device, mode, line_arg):
    spy = mocker.spy(Baseline, "forward")
    if not torch.cuda.is_available():
        mocker.patch("torch.cuda.synchronize")
    argv = [
        "benchmark",
        "-o",
        str(tmp_path),
        "-w",
        str(1),
        "-r",
        str(5),
        "--device",
        str(device),
        "--dtype",
        "fp32" if device == "cpu" else "fp16",
        "-m",
        mode,
        "-l",
        line_arg,
    ]
    entrypoint(argv)
    spy.assert_called()
    files = list(tmp_path.glob("*"))
    assert len(files)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.cuda),
    ],
)
def test_profile(mocker, entrypoint, device):
    if torch.cuda.is_available():
        start = mocker.spy(torch.cuda.cudart(), "cudaProfilerStart")
        stop = mocker.spy(torch.cuda.cudart(), "cudaProfilerStop")
    argv = [
        "profile",
        "--device",
        str(device),
        "--dtype",
        "fp32" if device == "cpu" else "fp16",
    ]
    entrypoint(argv)
    if torch.cuda.is_available():
        start.assert_called()  # type: ignore
        stop.assert_called()  # type: ignore
