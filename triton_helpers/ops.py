from importlib import import_module
from typing import Any, Iterable

import triton
import triton.language as tl


@triton.jit
def to_tensor(val, dtype: tl.constexpr) -> tl.tensor:
    r"""Promote a scalar to a tensor with a given dtype."""
    return tl.full((1,), val, dtype=dtype)


@triton.jit
def offset_grid(BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, DTYPE: tl.constexpr = tl.int32) -> tl.tensor:
    r"""Create a 2D offset grid of shape :math:`(BLOCK_M, BLOCK_K)` given block sizes."""
    return (tl.arange(0, BLOCK_M).to(DTYPE) * BLOCK_K)[:, None] + tl.arange(0, BLOCK_K).to(DTYPE)[None, :]


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
def relu_bwd(x: tl.tensor, grad: tl.tensor) -> tl.tensor:
    return tl.where(x > 0, grad, to_tensor(0, grad.dtype))


@triton.jit
def silu(x: tl.tensor) -> tl.tensor:
    return x * tl.sigmoid(x.to(tl.float32)).to(x.dtype)


@triton.jit
def silu_bwd(x: tl.tensor, grad: tl.tensor) -> tl.tensor:
    s = tl.sigmoid(x.to(tl.float32)).to(x.dtype)
    # Try to set up a FMA
    return grad * s * (x * (1 - s) + 1)


@triton.jit
def high_low_mod(high: tl.tensor, low: tl.tensor, m: tl.tensor) -> tl.tensor:
    r"""Computes a modulo from the high and low bits of an integer and m."""
    # H = (high * B + low) % m
    #   = ((high * B) % m + low % m) % m
    #   = ([(high % m) * (B % m)] % m) % m + low % m) % m
    # NOTE: Offset of 1 seems necessary to prevent overflow in temporary base
    SHIFT: tl.constexpr = (1 << tl.constexpr(high.dtype.int_bitwidth)) - 1
    base = (to_tensor(SHIFT, m.dtype) % m + to_tensor(1, m.dtype)) % m
    high_mod = (high % m) * (base % m)
    low_mod = low % m
    return (high_mod + low_mod) % m


@triton.jit
def multiply_mod(x: tl.tensor, y: tl.tensor, m: tl.tensor) -> tl.tensor:
    r"""Computes (x * y) % m for integer tensors x, y, and m."""
    low = x * y
    high = tl.math.mulhi(x, y)
    return high_low_mod(high, low, m)


def ensure_str(fn: str | triton.JITFunction, choices: Iterable[str] | None = None) -> str:
    r"""Ensure a function is a string, optionally checking it against a set of choices.

    Args:
        fn: Function to check. Can be a string or JIT function. If a JIT function, its module and name will be used.
        choices: Optional set of choices to check against.

    Returns:
        Function as a string.
    """
    if isinstance(fn, str):
        if (choices is not None) and fn not in (choices := set(choices)):
            raise ValueError(f"Function {fn} not in {choices}")
        return fn
    else:
        fn = f"{fn.__module__}.{fn.__name__}"
    return fn


def import_path(path: str | tl.constexpr) -> Any:
    r"""Import a function from a module given its path.

    This will generally be used to parameterize a function within a kernel, e.g. for a custom activation function.

    Args:
        path: The path to the function to import.

    Returns:
        Imported function.

    Example:
        >>> FN: tl.constexpr = tl.constexpr(import_path("triton_helpers.ops.relu"))
    """
    if isinstance(path, tl.constexpr):
        path = path.value  # type: ignore
    assert isinstance(path, str)
    func_name = path.split(".")[-1]
    module_name = ".".join(path.split(".")[:-1])
    return getattr(import_module(module_name), func_name)
