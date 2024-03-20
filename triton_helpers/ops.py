import triton
import triton.language as tl


@triton.jit
def to_tensor(val, dtype: tl.constexpr) -> tl.tensor:
    r"""Promote a scalar to a tensor with a given dtype."""
    return tl.full((1,), val, dtype=dtype)


@triton.jit
def offset_grid(BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr) -> tl.tensor:
    r"""Create a 2D offset grid of shape :math:`(BLOCK_M, BLOCK_K)` given block sizes."""
    return (tl.arange(0, BLOCK_M) * BLOCK_K)[:, None] + tl.arange(0, BLOCK_K)[None, :]


@triton.jit
def norm_coeff(t: tl.tensor) -> tl.tensor:
    r"""Compute the L2 normalization coefficient for a tensor."""
    sos = tl.sum((t * t), 1)
    return tl.math.rsqrt(sos.to(tl.float32)).to(t.dtype)


@triton.jit
def diag(t: tl.tensor, SIZE: tl.constexpr) -> tl.tensor:
    r"""Extract the diagonal of a square matrix."""
    block_idx = tl.arange(0, SIZE)
    output = tl.zeros((SIZE, SIZE), dtype=t.dtype)
    output = tl.where(block_idx[:, None] == block_idx, t, output)
    return tl.sum(output, 1)


@triton.jit
def relu(x: tl.tensor) -> tl.tensor:
    return tl.where(x < 0, to_tensor(0, x.dtype), x)


@triton.jit
def silu(x: tl.tensor) -> tl.tensor:
    return x * tl.sigmoid(x.to(tl.float32)).to(x.dtype)
