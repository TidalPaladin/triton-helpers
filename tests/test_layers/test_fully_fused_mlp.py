from typing import cast

import pytest
import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch import Tensor
from torch.testing import assert_close

from triton_helpers.layers.fully_fused_mlp import (
    FullyFusedMLP,
    feedforward_bwd_dw,
    feedforward_bwd_dx,
    feedforward_bwd_dz,
    fully_fused_mlp,
)
from triton_helpers.ops import offset_grid, relu, silu


class MLP(nn.Module):

    def __init__(self, D_in: int, D_hidden: int, D_out: int, depth: int = 1, activation: nn.Module = nn.ReLU()):
        super(MLP, self).__init__()
        self.layer_in = nn.Linear(D_in, D_hidden)
        self.layer_hidden = nn.ModuleList([nn.Linear(D_hidden, D_hidden) for _ in range(depth - 1)])
        self.layer_out = nn.Linear(D_hidden, D_out)
        self.activation = activation

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.uniform_(layer.bias, -0.1, 0.1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.layer_in(x))
        for layer in self.layer_hidden:
            x = self.activation(layer(x))
        x = self.layer_out(x)
        return x


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype,fp16_acc,tol",
    [
        (torch.float32, False, 1e-3),
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
@pytest.mark.parametrize("depth", [8, 32])
@pytest.mark.parametrize(
    "dtype,fp16_acc,tol",
    [
        (torch.float32, False, 1e-3),
        (torch.float16, False, 1e-2),
        (torch.bfloat16, False, 1e-2),
        (torch.float16, True, 1e-2),
        (torch.bfloat16, True, 1e-2),
    ],
)
def test_forward_deep(dtype, tol, fp16_acc, depth):
    torch.random.manual_seed(0)
    D_in = 1
    D_hidden = 32
    D_out = 5
    B, L = 3, 40

    x = torch.randn((B, L, D_in), device="cuda")
    layer = MLP(D_in, D_hidden, D_out, depth=depth).to("cuda")

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


@pytest.mark.cuda
@pytest.mark.parametrize("torch_act,act", [(nn.ReLU(), relu), (nn.SiLU(), silu)])
def test_forward_custom_act(torch_act, act):
    torch.random.manual_seed(0)
    D_in = 1
    D_hidden = 32
    D_out = 5
    B, L = 3, 40

    dtype = torch.float16
    x = torch.randn((B, L, D_in), device="cuda")
    layer = MLP(D_in, D_hidden, D_out, activation=torch_act).to("cuda")

    baseline = layer(x)
    actual = fully_fused_mlp(
        x.to(dtype),
        layer.layer_in.weight.to(dtype),
        layer.layer_in.bias.to(dtype),
        layer.layer_out.weight.to(dtype),
        layer.layer_out.bias.to(dtype),
        activation=act,
    )
    assert_close(baseline, actual, rtol=0, atol=0.01, check_dtype=False)


@pytest.mark.cuda
@pytest.mark.parametrize("torch_act,act", [(nn.Identity(), "none"), (nn.ReLU(), "relu"), (nn.SiLU(), "silu")])
@pytest.mark.parametrize("dtype,tol", [(torch.float16, 1e-2), (torch.bfloat16, 1e-2)])
def test_backward_dz(torch_act, act, dtype, tol):
    @triton.jit
    def kernel(x_p, do_p, dz_p, BLOCK: tl.constexpr, ACTIVATION: tl.constexpr):
        offsets = offset_grid(BLOCK, BLOCK)
        x = tl.load(x_p + offsets)
        do = tl.load(do_p + offsets)
        dz = feedforward_bwd_dz(x, do, ACTIVATION=ACTIVATION)
        tl.store(dz_p + offsets, dz)

    # Setup x and baseline layer
    torch.random.manual_seed(0)
    D = 16
    L = 16
    x = torch.randn((L, D), dtype=dtype, device="cuda", requires_grad=True)
    layer = torch_act.to("cuda")

    # Baseline grads
    y = layer(x)
    do = torch.randn_like(y)
    y.backward(do)
    baseline_dz = x.grad

    # Kernel grads
    dz = torch.zeros_like(cast(Tensor, baseline_dz), dtype=dtype)
    kernel[(1,)](x, do, dz, L, act)  # type: ignore

    assert_close(dz, baseline_dz, atol=tol, rtol=0, check_dtype=False)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-2), (torch.float16, 1e-2), (torch.bfloat16, 3e-2)])
def test_backward_dw(dtype, tol):
    @triton.jit
    def kernel(x_p, dw_p, dz_p, BLOCK: tl.constexpr):
        offsets = offset_grid(BLOCK, BLOCK)
        x = tl.load(x_p + offsets)
        dz = tl.load(dz_p + offsets)
        dw = feedforward_bwd_dw(x, dz)
        tl.store(dw_p + offsets, dw)

    # Setup x and baseline layer
    torch.random.manual_seed(0)
    D_1, D_2 = 16, 16
    L = 16
    x = torch.randn((L, D_1), dtype=dtype, device="cuda", requires_grad=True)
    linear = nn.Linear(D_1, D_2).to("cuda")

    # Baseline grads
    y = linear(x.float())
    dz = torch.randn_like(y)
    y.backward(dz)
    baseline_dw = linear.weight.grad

    # Kernel grads
    dw = torch.empty_like(linear.weight, dtype=dtype)
    kernel[(1,)](x, dw, dz.to(dtype), L)  # type: ignore

    assert_close(dw, baseline_dw, atol=tol, rtol=0, check_dtype=False)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype,tol", [(torch.float16, 1e-2), (torch.bfloat16, 1e-2)])
def test_backward_dx(dtype, tol):
    @triton.jit
    def kernel(w_p, dx_p, dz_p, BLOCK: tl.constexpr):
        offsets = offset_grid(BLOCK, BLOCK)
        w = tl.load(w_p + offsets)
        dz = tl.load(dz_p + offsets)
        dx = feedforward_bwd_dx(w, dz)
        tl.store(dx_p + offsets, dx)

    # Setup x and baseline layer
    torch.random.manual_seed(0)
    D_1, D_2 = 16, 16
    L = 16
    x = torch.randn((L, D_1), dtype=dtype, device="cuda", requires_grad=True)
    linear = nn.Linear(D_1, D_2).to("cuda")

    # Baseline grads
    y = linear(x.float())
    y.retain_grad()
    dz = torch.randn_like(y)
    y.backward(dz)
    baseline_dx = x.grad

    # Kernel grads
    dx = torch.empty_like(x, dtype=dtype)
    kernel[(1,)](linear.weight.to(dtype), dx, dz.to(dtype), L)  # type: ignore

    assert_close(dx, baseline_dx, atol=tol, rtol=0, check_dtype=False)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype,fp16_acc,tol",
    [
        (torch.float32, False, 1e-2),
        (torch.float16, False, 1e-2),
        (torch.bfloat16, False, 1e-2),
        (torch.float16, True, 1e-2),
        (torch.bfloat16, True, 1e-2),
    ],
)
def test_backward_shallow(dtype, tol, fp16_acc):
    torch.random.manual_seed(0)
    D_in = 8
    D_hidden = 16
    D_out = 4
    B, L = 2, 2

    x = torch.randn((B, L, D_in), device="cuda", requires_grad=True)
    layer = MLP(D_in, D_hidden, D_out).to("cuda")

    # Baseline grads
    y = layer(x.float())
    do = torch.randn_like(y)
    y.backward(do)
    baseline_dx = x.grad
    baseline_dw_in = layer.layer_in.weight.grad
    baseline_db_in = layer.layer_in.bias.grad
    baseline_dw_out = layer.layer_out.weight.grad
    baseline_db_out = layer.layer_out.bias.grad

    x.grad = None
    layer.layer_in.weight.grad = None
    layer.layer_in.bias.grad = None
    layer.layer_out.weight.grad = None
    layer.layer_out.bias.grad = None

    y = fully_fused_mlp(
        x.to(dtype),
        layer.layer_in.weight.to(dtype),
        layer.layer_in.bias.to(dtype),
        layer.layer_out.weight.to(dtype),
        layer.layer_out.bias.to(dtype),
        fp16_acc=fp16_acc,
    )
    y.backward(do)
    dx = x.grad
    dw_in = layer.layer_in.weight.grad
    db_in = layer.layer_in.bias.grad
    dw_out = layer.layer_out.weight.grad
    db_out = layer.layer_out.bias.grad

    assert_close(baseline_db_out, db_out, rtol=tol, atol=tol, check_dtype=False)
    assert_close(baseline_dw_out, dw_out, rtol=tol, atol=tol, check_dtype=False)

    assert_close(baseline_db_in, db_in, rtol=tol, atol=tol, check_dtype=False)
    assert_close(baseline_dw_in, dw_in, rtol=tol, atol=tol, check_dtype=False)

    assert_close(baseline_dx, dx, rtol=tol, atol=tol, check_dtype=False)


@pytest.mark.cuda
@pytest.mark.parametrize("depth", [3, 8, 16, 32])
@pytest.mark.parametrize(
    "dtype,fp16_acc,tol",
    [
        (torch.float16, False, 1e-2),
        (torch.bfloat16, False, 1e-2),
        (torch.float16, True, 1e-2),
        (torch.bfloat16, True, 1e-2),
    ],
)
def test_backward_deep(dtype, tol, fp16_acc, depth):
    torch.random.manual_seed(0)
    D_in = 8
    D_hidden = 16
    D_out = 4
    B, L = 2, 2

    x = torch.randn((B, L, D_in), device="cuda", requires_grad=True)
    layer = MLP(D_in, D_hidden, D_out, depth=depth).to("cuda")

    # Baseline grads
    y = layer(x.float())
    do = torch.randn_like(y)
    y.backward(do)
    baseline_dx = x.grad
    baseline_dw_in = layer.layer_in.weight.grad
    baseline_db_in = layer.layer_in.bias.grad
    baseline_dw_hid = torch.cat([l.weight.grad for l in layer.layer_hidden], dim=0)
    baseline_db_hid = torch.cat([l.bias.grad for l in layer.layer_hidden], dim=0)
    baseline_dw_out = layer.layer_out.weight.grad
    baseline_db_out = layer.layer_out.bias.grad

    x.grad = None
    layer.layer_in.weight.grad = None
    layer.layer_in.bias.grad = None
    for l in layer.layer_hidden:
        l.weight.grad = None
        l.bias.grad = None
    layer.layer_out.weight.grad = None
    layer.layer_out.bias.grad = None

    w_hid = torch.cat([l.weight for l in layer.layer_hidden], dim=0)
    b_hid = torch.cat([l.bias.view(1, -1) for l in layer.layer_hidden], dim=0)

    y = fully_fused_mlp(
        x.to(dtype),
        layer.layer_in.weight.to(dtype),
        layer.layer_in.bias.to(dtype),
        layer.layer_out.weight.to(dtype),
        layer.layer_out.bias.to(dtype),
        w_hid.to(dtype),
        b_hid.to(dtype),
        fp16_acc=fp16_acc,
    )
    y.backward(do)
    dx = x.grad
    dw_in = layer.layer_in.weight.grad
    db_in = layer.layer_in.bias.grad
    dw_hid = torch.cat([l.weight.grad for l in layer.layer_hidden], dim=0)
    db_hid = torch.cat([l.bias.grad for l in layer.layer_hidden], dim=0)
    dw_out = layer.layer_out.weight.grad
    db_out = layer.layer_out.bias.grad

    assert_close(baseline_db_out, db_out, rtol=tol, atol=tol, check_dtype=False)
    assert_close(baseline_dw_out, dw_out, rtol=tol, atol=tol, check_dtype=False)

    assert_close(baseline_db_hid, db_hid, rtol=tol, atol=tol, check_dtype=False)
    assert_close(baseline_dw_hid, dw_hid, rtol=tol, atol=tol, check_dtype=False)

    assert_close(baseline_db_in, db_in, rtol=tol, atol=tol, check_dtype=False)
    assert_close(baseline_dw_in, dw_in, rtol=tol, atol=tol, check_dtype=False)

    assert_close(baseline_dx, dx, rtol=tol, atol=tol, check_dtype=False)


@pytest.mark.cuda
@pytest.mark.parametrize("depth", [1, 4, 8])
def test_module(depth):
    torch.random.manual_seed(0)
    D_in = 8
    D_hidden = 16
    D_out = 4
    B, L = 2, 2

    tol = 1e-2
    x = torch.randn((B, L, D_in), device="cuda", requires_grad=True)
    torch.random.manual_seed(0)
    layer = MLP(D_in, D_hidden, D_out, depth=depth).to("cuda")
    torch.random.manual_seed(0)
    layer2 = FullyFusedMLP(D_in, D_hidden, D_out, depth=depth).to("cuda")

    # Baseline output
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        baseline_y = layer(x)
    baseline_y.sum().backward()

    # Triton output
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        y = layer2(x)
    y.sum().backward()

    assert_close(baseline_y, y, rtol=tol, atol=tol, check_dtype=False)
