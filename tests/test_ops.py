import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.testing import assert_close

from triton_helpers.ops import diag, norm_coeff, offset_grid, relu, relu_bwd, silu, silu_bwd, to_tensor


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
def test_offset_grid():
    @triton.jit
    def kernel(o_p, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
        grid = offset_grid(BLOCK_M, BLOCK_K)
        tl.store(o_p + grid, grid.to(tl.float32))

    M = 16
    K = 4
    o = torch.zeros(M, K, dtype=torch.float16, device="cuda")
    kernel[(1,)](o, M, K)  # type: ignore
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
    y.retain_grad()
    y.sin().sum().backward()

    baseline = i.grad
    kernel[(1,)](i, y.grad, o, M)  # type: ignore
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
    y.retain_grad()
    y.sin().sum().backward()
    baseline = i.grad
    kernel[(1,)](i, y.grad, o, M)  # type: ignore
    assert_close(o, baseline, atol=tol, rtol=0)
