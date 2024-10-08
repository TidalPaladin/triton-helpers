import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.testing import assert_close

from triton_helpers.ops import (
    diag,
    multiply_mod,
    norm_coeff,
    offset_grid,
    relu,
    relu2,
    relu2_bwd,
    relu_bwd,
    silu,
    silu_bwd,
    to_tensor,
)


@pytest.mark.cuda
def test_to_tensor():
    @triton.jit
    def kernel(o_p, X: tl.constexpr = 1.0):  # type: ignore
        x = tl.load(o_p + tl.arange(0, 1))
        y = x + to_tensor(X, dtype=tl.float16)
        tl.store(o_p + tl.arange(0, 1), y)

    o = torch.zeros(1, dtype=torch.float16, device="cuda")
    kernel[(1,)](o)  # type: ignore
    assert o.item() == 1


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [tl.int32, tl.uint32])
def test_offset_grid(dtype):
    @triton.jit
    def kernel(o_p, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, DTYPE: tl.constexpr):
        grid = offset_grid(BLOCK_M, BLOCK_K, DTYPE)
        tl.device_assert(tl.constexpr(grid.dtype) == DTYPE)
        tl.store(o_p + grid, grid.to(tl.float32))

    M = 16
    K = 4
    o = torch.zeros(M, K, dtype=torch.float16, device="cuda")
    kernel[(1,)](o, M, K, dtype)  # type: ignore
    assert_close(o.flatten(), torch.arange(0, M * K), check_device=False, check_dtype=False)


@pytest.mark.cuda
def test_norm_coeff():
    @triton.jit
    def kernel(i_p, o_p, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
        grid = offset_grid(BLOCK_M, BLOCK_K)
        x = tl.load(i_p + grid)
        c = norm_coeff(x)
        y = x * c[:, None]
        tl.store(o_p + grid, y.to(o_p.dtype.element_ty))

    M = 16
    K = 4
    i = torch.randn(M, K, device="cuda")
    o = torch.empty_like(i)
    kernel[(1,)](i, o, M, K)  # type: ignore
    assert o.norm(dim=1).allclose(o.new_ones(M), atol=1e-3)


@pytest.mark.cuda
def test_diag():
    @triton.jit
    def kernel(i_p, o_p, BLOCK_M: tl.constexpr):
        grid = offset_grid(BLOCK_M, BLOCK_M)
        x = tl.load(i_p + grid)
        x = diag(x, BLOCK_M)
        tl.store(o_p + tl.arange(0, BLOCK_M), x)

    M = 16
    i = torch.randn(M, M, device="cuda")
    o = i.new_empty(M)
    kernel[(1,)](i, o, M)  # type: ignore
    assert_close(o, i.diag())


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype, tol",
    [
        (torch.float32, 1e-4),
        (torch.float16, 1e-2),
        (torch.bfloat16, 1e-2),
    ],
)
def test_relu(dtype, tol):
    @triton.jit
    def kernel(i_p, o_p, BLOCK: tl.constexpr):
        x = tl.load(i_p + tl.arange(0, BLOCK))
        x = relu(x)
        tl.store(o_p + tl.arange(0, BLOCK), x)

    M = 64
    torch.random.manual_seed(0)
    i = torch.randn(M, device="cuda", dtype=dtype)
    o = i.new_empty(M)
    kernel[(1,)](i, o, M)  # type: ignore
    assert_close(o, F.relu(i), atol=tol, rtol=0)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype, tol",
    [
        (torch.float32, 1e-4),
        (torch.float16, 1e-2),
        (torch.bfloat16, 1e-2),
    ],
)
def test_relu_bwd(dtype, tol):
    @triton.jit
    def kernel(i_p, grad_p, o_p, BLOCK: tl.constexpr):
        x = tl.load(i_p + tl.arange(0, BLOCK))
        grad = tl.load(grad_p + tl.arange(0, BLOCK))
        dx = relu_bwd(x, grad)
        tl.store(o_p + tl.arange(0, BLOCK), dx)

    M = 64
    torch.random.manual_seed(0)
    i = torch.randn(M, device="cuda", dtype=dtype, requires_grad=True)
    o = i.new_empty(M)
    y = F.relu(i)
    do = torch.randn_like(y)
    y.backward(do)
    baseline = i.grad
    kernel[(1,)](i, do, o, M)  # type: ignore
    assert_close(o, baseline, atol=tol, rtol=0)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype, tol",
    [
        (torch.float32, 1e-4),
        (torch.float16, 1e-2),
        (torch.bfloat16, 1e-2),
    ],
)
def test_silu(dtype, tol):
    @triton.jit
    def kernel(i_p, o_p, BLOCK: tl.constexpr):
        x = tl.load(i_p + tl.arange(0, BLOCK))
        x = silu(x)
        tl.store(o_p + tl.arange(0, BLOCK), x)

    M = 64
    torch.random.manual_seed(0)
    i = torch.randn(M, device="cuda", dtype=dtype)
    o = i.new_empty(M)
    kernel[(1,)](i, o, M)  # type: ignore
    assert_close(o, F.silu(i), atol=tol, rtol=0)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype, tol",
    [
        (torch.float32, 1e-4),
        (torch.float16, 1e-2),
        (torch.bfloat16, 1e-2),
    ],
)
def test_silu_bwd(dtype, tol):
    @triton.jit
    def kernel(i_p, grad_p, o_p, BLOCK: tl.constexpr):
        x = tl.load(i_p + tl.arange(0, BLOCK))
        grad = tl.load(grad_p + tl.arange(0, BLOCK))
        dx = silu_bwd(x, grad)
        tl.store(o_p + tl.arange(0, BLOCK), dx)

    M = 64
    torch.random.manual_seed(0)
    i = torch.randn(M, device="cuda", dtype=dtype, requires_grad=True)
    o = i.new_empty(M)
    y = F.silu(i)
    do = torch.randn_like(y)
    y.backward(do)
    baseline = i.grad
    kernel[(1,)](i, do, o, M)  # type: ignore
    assert_close(o, baseline, atol=tol, rtol=0)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype, tol",
    [
        (torch.float32, 1e-4),
        (torch.float16, 1e-2),
        (torch.bfloat16, 1e-2),
    ],
)
def test_relu2(dtype, tol):
    @triton.jit
    def kernel(i_p, o_p, BLOCK: tl.constexpr):
        x = tl.load(i_p + tl.arange(0, BLOCK))
        x = relu2(x)
        tl.store(o_p + tl.arange(0, BLOCK), x)

    M = 64
    torch.random.manual_seed(0)
    i = torch.randn(M, device="cuda", dtype=dtype)
    o = i.new_empty(M)
    kernel[(1,)](i, o, M)  # type: ignore
    assert_close(o, F.relu(i) * F.relu(i), atol=tol, rtol=0)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype, tol",
    [
        (torch.float32, 1e-4),
        (torch.float16, 1e-2),
        (torch.bfloat16, 1e-2),
    ],
)
def test_relu2_bwd(dtype, tol):
    @triton.jit
    def kernel(i_p, grad_p, o_p, BLOCK: tl.constexpr):
        x = tl.load(i_p + tl.arange(0, BLOCK))
        grad = tl.load(grad_p + tl.arange(0, BLOCK))
        dx = relu2_bwd(x, grad)
        tl.store(o_p + tl.arange(0, BLOCK), dx)

    M = 64
    torch.random.manual_seed(0)
    i = torch.randn(M, device="cuda", dtype=dtype, requires_grad=True)
    o = i.new_empty(M)
    y = F.relu(i) * F.relu(i)
    do = torch.randn_like(y)
    y.backward(do)
    baseline = i.grad
    kernel[(1,)](i, do, o, M)  # type: ignore
    assert_close(o, baseline, atol=tol, rtol=0)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype, triton_dtype, x, y, m, exp",
    [
        (torch.int32, tl.int32, 1, 10, 7, (1 * 10) % 7),
        (torch.int32, tl.int32, 2**10, 2**10, 2**5, (2**10 * 2**10) % 2**5),
        (torch.int32, tl.uint32, 1, 10, 7, (1 * 10) % 7),
        (torch.int32, tl.uint32, 2**10, 2**10, 2**5, (2**10 * 2**10) % 2**5),
        (torch.int64, tl.uint32, 2_654_435_761, 8, 700, (2_654_435_761 * 8) % 700),
    ],
)
def test_multiply_mod(dtype, triton_dtype, x, y, m, exp):
    @triton.jit
    def kernel(x_p, y_p, m_p, o_p, DTYPE: tl.constexpr):  # type: ignore
        x = tl.load(x_p + tl.arange(0, 1)).to(DTYPE)
        y = tl.load(y_p + tl.arange(0, 1)).to(DTYPE)
        m = tl.load(m_p + tl.arange(0, 1)).to(DTYPE)
        z = multiply_mod(x, y, m)
        tl.store(o_p + tl.arange(0, 1), z.to(o_p.dtype.element_ty))

    x = torch.tensor(x, dtype=dtype, device="cuda")
    y = x.new_tensor(y)
    m = x.new_tensor(m)
    o = x.new_empty(1)
    kernel[(1,)](x, y, m, o, triton_dtype)  # type: ignore

    assert o.item() == exp
