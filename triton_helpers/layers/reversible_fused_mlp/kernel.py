from typing import Any, cast

import triton
import triton.language as tl
from torch import Tensor
from torch.autograd import Function

from ...heuristics import PowerOfTwoHeuristic
from ...ops import ensure_str
from ..fully_fused_mlp.kernel import FULLY_FUSED_MAX_DIM, feedforward


@triton.heuristics(
    dict(
        BLOCK_D=PowerOfTwoHeuristic("D", 16),
        BLOCK_L=PowerOfTwoHeuristic("L", min_val=16, max_val=64),
        num_warps=lambda _: 4,
    )
)
@triton.jit
def _fwd_kernel(
    # fmt: off
    # Inputs
    x_p, w_p, b_p, o1_p, o2_p,
    # Sizes
    L: int, D: int,
    # Strides
    stride_x_l: int, stride_o_l: int,
    # Params
    ACTIVATION: tl.constexpr, DTYPE: tl.constexpr, DEPTH: tl.constexpr,
    # Blocks
    BLOCK_L: tl.constexpr, BLOCK_D: tl.constexpr,
    # fmt: on
):
    # Select offsets for this block
    start = tl.program_id(0) * BLOCK_L

    # Update input and output pointer
    x_p += start * stride_x_l
    o1_p += start * stride_o_l
    o2_p += start * stride_o_l

    # Load input (x2 is initially a copy of x1)
    X_block_ptr = tl.make_block_ptr(
        x_p,
        (L, D),
        (stride_x_l, 1),
        (0, 0),
        (BLOCK_L, BLOCK_D),
        (1, 0),
    )
    x = tl.load(X_block_ptr, boundary_check=(0, 1))
    lagging = x

    # Set up weight and bias pointers
    W_block_ptr = tl.make_block_ptr(
        w_p,
        (DEPTH * BLOCK_D, BLOCK_D),
        (BLOCK_D, 1),
        (0, 0),
        (BLOCK_D, BLOCK_D),
        (1, 0),
    )
    B_block_ptr = tl.make_block_ptr(
        b_p,
        (DEPTH, BLOCK_D),
        (BLOCK_D, 1),
        (0, 0),
        (1, BLOCK_D),
        (1, 0),
    )

    for d in range(DEPTH):
        # Load weight / bias
        w = tl.load(W_block_ptr)
        b = tl.view(tl.load(B_block_ptr), (BLOCK_D,))

        # Do residual feedforward
        tmp = x
        if d < DEPTH - 1:
            x = lagging + feedforward(x, w, b, ACTIVATION=ACTIVATION, DTYPE=DTYPE)
        else:
            x = lagging + feedforward(x, w, b, ACTIVATION="none", DTYPE=DTYPE)
        lagging = tmp

        # Advance pointers
        W_block_ptr = tl.advance(W_block_ptr, (BLOCK_D, 0))
        B_block_ptr = tl.advance(B_block_ptr, (1, 0))

    # Store outputs
    O1_block_ptr = tl.make_block_ptr(
        o1_p,
        (L, D),
        (stride_o_l, 1),
        (0, 0),
        (BLOCK_L, BLOCK_D),
        (1, 0),
    )
    tl.store(O1_block_ptr, x, boundary_check=(0, 1), eviction_policy="evict_first")
    O2_block_ptr = tl.make_block_ptr(
        o2_p,
        (L, D),
        (stride_o_l, 1),
        (0, 0),
        (BLOCK_L, BLOCK_D),
        (1, 0),
    )
    tl.store(O2_block_ptr, lagging, boundary_check=(0, 1), eviction_policy="evict_first")


class _reversible_mlp_forward(Function):

    @staticmethod
    def forward(
        # fmt: off
        ctx, 
        x: Tensor, 
        w: Tensor, b: Tensor, 
        activation: str | triton.JITFunction = "relu",  
        fp16_acc: bool = False,
    ) -> Tensor:
        B, L, D = x.shape

        # Check dimension size
        assert D <= FULLY_FUSED_MAX_DIM, f"Input dimension {D} too large for full fusion"
        assert D & (D - 1) == 0, "Dimension must be a power of 2"
        assert D >= 16, "Dimension must be at least 16"

        # Check weight and bias
        depth = w.shape[0] // D
        assert w.shape == (depth * D, D), "Hidden weight shape mismatch"
        assert b.shape == (depth, D), "Hidden bias shape mismatch"

        # Handle non-str activation
        activation = ensure_str(activation, choices=["relu", "silu", "none"])

        # Init output
        o1 = x.new_zeros((B, L, D))
        o2 = x.new_zeros((B, L, D))

        def grid(META):
            return (triton.cdiv(B * L, META["BLOCK_L"]),)

        cast(Any, _fwd_kernel)[grid](
            x,
            w,
            b,
            o1,
            o2,
            B * L,
            D,
            x.stride(1),
            o1.stride(1),
            activation,
            DTYPE=tl.float16 if fp16_acc else tl.float32,
            DEPTH=depth,
        )

        ctx.save_for_backward(x, w, b, o1, o2)
        ctx.activation = activation
        ctx.fp16_acc = fp16_acc
        ctx.depth = depth

        return o1

    @staticmethod
    def backward(ctx, do: Tensor):
        raise NotImplementedError("Backward pass not implemented for fully fused mlp")  # pragma: no cover


def reversible_fused_mlp(
    # fmt: off
    x: Tensor, 
    w: Tensor, b: Tensor, 
    activation: str | triton.JITFunction = "relu", 
    fp16_acc: bool = False,
    # fmt: on
) -> Tensor:
    r"""Perform a reversible fully fused forward pass of a multi-layer perceptron (MLP).

    This implementation of a fully fused MLP is inspired by the approach described in [1]. Scaling is particularly
    effective with multiple layers.

    .. [1] Müller, T., Evans, B., Schied, C., & Keller, A. (2021). Real-time Neural Radiance Caching for Path Tracing. arXiv preprint arXiv:2106.12372.
        Available at https://arxiv.org/abs/2106.12372

    .. note::
        Backward pass is not yet implemented for this operation.

    Args:
        x: Input tensor
        w: Weight tensor
        b: Bias tensor
        activation: Activation function to use. Can be "relu", "silu", "none", or a custom Triton function.
        fp16_acc: Use FP16 for accumulation. This will reduce precision and may lead to numerical instability, but
            is faster.

    Shapes:
        - x: :math:`(B, L, D_in)`
        - w: :math:`(depth * D_hidden, D_in)`
        - b: :math:`(depth,)`
        - Output: :math:`(B, L, D_out)`

    Returns:
        Output of MLP forward pass.
    """
    return _reversible_mlp_forward.apply(x, w, b, activation, fp16_acc)  # type: ignore
