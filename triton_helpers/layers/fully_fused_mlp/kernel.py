from typing import Any, Final, cast

import triton
import triton.language as tl
from torch import Tensor
from torch.autograd import Function

from ...heuristics import PowerOfTwoHeuristic
from ...ops import relu, silu


# Above this size we can't keep weights in SRAM
FULLY_FUSED_MAX_DIM: Final = 64


@triton.jit
def feedforward(
    # fmt: off
    # Inputs
    x: tl.tensor, w: tl.tensor, b: tl.tensor,
    # Params
    HAS_BIAS: tl.constexpr = tl.constexpr(True),
    ACTIVATION: tl.constexpr = tl.constexpr("relu"), 
    DTYPE: tl.constexpr = tl.constexpr(tl.float32),
    # fmt: on
) -> tl.tensor:
    # x @ w.T + b
    x = tl.dot(x, tl.trans(w), out_dtype=DTYPE).to(x.dtype)
    if HAS_BIAS:
        x += b

    # nonlinearity
    if ACTIVATION == "relu":
        x = relu(x)
    elif ACTIVATION == "silu":
        x = silu(x)
    elif ACTIVATION == "none":
        pass
    else:
        tl.static_assert(False, "Unknown activation")

    return x


@triton.heuristics(
    dict(
        BLOCK_D_IN=PowerOfTwoHeuristic("D_in", 16),
        BLOCK_D_HIDDEN=PowerOfTwoHeuristic("D_hidden", 16),
        BLOCK_D_OUT=PowerOfTwoHeuristic("D_out", 16),
        BLOCK_L=PowerOfTwoHeuristic("L", min_val=16, max_val=128),
        num_warps=lambda _: 4,
    )
)
@triton.jit
def _fwd_kernel(
    # fmt: off
    # Inputs
    x_p, w_in_p, b_in_p, w_hid_p, b_hid_p, w_out_p, b_out_p, o_p,
    # Sizes
    L: int, D_in: int, D_hidden: int, D_out: int,
    # Strides
    stride_x_l: int, stride_o_l: int,
    # Params
    ACTIVATION: tl.constexpr, DTYPE: tl.constexpr, DEPTH: tl.constexpr,
    # Blocks
    BLOCK_L: tl.constexpr, BLOCK_D_IN: tl.constexpr, BLOCK_D_HIDDEN: tl.constexpr, BLOCK_D_OUT: tl.constexpr,
    # fmt: on
):
    # Select offsets for this block
    start = tl.program_id(0) * BLOCK_L

    # Update input and output pointer
    x_p += start * stride_x_l
    o_p += start * stride_o_l

    # Load input layer weights / bias
    ptr = tl.make_block_ptr(
        w_in_p,
        (D_hidden, D_in),
        (D_in, 1),
        (0, 0),
        (BLOCK_D_HIDDEN, BLOCK_D_IN),
        (1, 0),
    )
    w_in = tl.load(ptr, boundary_check=(0, 1))
    ptr = tl.make_block_ptr(b_in_p, (D_hidden,), (1,), (0,), (BLOCK_D_HIDDEN,), (0,))
    b_in = tl.load(ptr, boundary_check=(0,))

    # Load input
    X_block_ptr = tl.make_block_ptr(
        x_p,
        (L, D_in),
        (stride_x_l, 1),
        (0, 0),
        (BLOCK_L, BLOCK_D_IN),
        (1, 0),
    )
    x = tl.load(X_block_ptr, boundary_check=(0, 1))

    # Do input feedforward layer
    x = feedforward(x, w_in, b_in, ACTIVATION=ACTIVATION, DTYPE=DTYPE)

    # Handle multiple hidden layers
    if DEPTH > 1:
        H: tl.constexpr = DEPTH - 1
        W_block_ptr = tl.make_block_ptr(
            w_hid_p,
            (H * BLOCK_D_HIDDEN, BLOCK_D_HIDDEN),
            (BLOCK_D_HIDDEN, 1),
            (0, 0),
            (BLOCK_D_HIDDEN, BLOCK_D_HIDDEN),
            (1, 0),
        )
        B_block_ptr = tl.make_block_ptr(
            b_hid_p,
            (H, BLOCK_D_HIDDEN),
            (BLOCK_D_HIDDEN, 1),
            (0, 0),
            (1, BLOCK_D_HIDDEN),
            (1, 0),
        )
        for _ in range(DEPTH - 1):
            # No boundary checks, we know D_hidden = BLOCK_D_HIDDEN
            w = tl.load(W_block_ptr)
            b = tl.view(tl.load(B_block_ptr), (BLOCK_D_HIDDEN,))
            x = feedforward(x, w, b, ACTIVATION=ACTIVATION, DTYPE=DTYPE)

            # Advance pointers
            W_block_ptr = tl.advance(W_block_ptr, (BLOCK_D_HIDDEN, 0))
            B_block_ptr = tl.advance(B_block_ptr, (1, 0))

    # Load output layer weights / bias
    ptr = tl.make_block_ptr(
        w_out_p,
        (D_out, D_hidden),
        (D_hidden, 1),
        (0, 0),
        (BLOCK_D_OUT, BLOCK_D_HIDDEN),
        (1, 0),
    )
    w_out = tl.load(ptr, boundary_check=(0, 1))
    ptr = tl.make_block_ptr(b_out_p, (D_out,), (1,), (0,), (BLOCK_D_OUT,), (0,))
    b_out = tl.load(ptr, boundary_check=(0,))

    # Do output feedforward layer
    x = feedforward(x, w_out, b_out, ACTIVATION="none", DTYPE=DTYPE)

    # Store output
    O_block_ptr = tl.make_block_ptr(
        o_p,
        (L, D_out),
        (stride_o_l, 1),
        (0, 0),
        (BLOCK_L, BLOCK_D_OUT),
        (1, 0),
    )
    tl.store(O_block_ptr, x, boundary_check=(0, 1), eviction_policy="evict_first")


class _mlp_forward(Function):

    @staticmethod
    def forward(
        # fmt: off
        ctx, 
        x: Tensor, 
        w_in: Tensor, b_in: Tensor, 
        w_out: Tensor, b_out: Tensor, 
        w_hid: Tensor | None, b_hid: Tensor | None,
        activation: str = "relu",  
        fp16_acc: bool = False,
    ) -> Tensor:
        B, L, D = x.shape
        D_hidden, D_in = w_in.shape
        D_out, _ = w_out.shape
        assert D_in == D, "Input dimension mismatch"
        assert D_hidden == w_out.shape[-1], "Hidden dimension mismatch"

        # Check size for full fusion
        assert D_in <= FULLY_FUSED_MAX_DIM, f"Input dimension {D_in} too large for full fusion"
        assert D_hidden <= FULLY_FUSED_MAX_DIM, f"Hidden dimension {D_hidden} too large for full fusion"
        assert D_out <= FULLY_FUSED_MAX_DIM, f"Output dimension {D_out} too large for full fusion"

        # Check hidden dim is power of 2 and >= 16
        assert D_hidden & (D_hidden - 1) == 0, "Hidden dimension must be a power of 2"
        assert D_hidden >= 16, "Hidden dimension must be at least 16"

        if w_hid is not None and b_hid is not None:
            depth = w_hid.shape[0] // D_hidden
            assert w_hid.shape == (depth * D_hidden, D_hidden), "Hidden weight shape mismatch"
            assert b_hid.shape == (depth, D_hidden), "Hidden bias shape mismatch"
            depth += 1
        else:
            depth = 1
            w_hid = x.new_empty(
                0,
            )
            b_hid = x.new_empty(
                0,
            )

        o = x.new_empty((B, L, D_out))

        def grid(META):
            return (triton.cdiv(B * L, META["BLOCK_L"]),)

        cast(Any, _fwd_kernel)[grid](
            x,
            w_in,
            b_in,
            w_hid,
            b_hid,
            w_out,
            b_out,
            o,
            B * L,
            D_in,
            D_hidden,
            D_out,
            x.stride(1),
            o.stride(1),
            activation,
            DTYPE=tl.float16 if fp16_acc else tl.float32,
            DEPTH=depth,
        )

        return o

    @staticmethod
    def backward(ctx, *args, **kwargs):
        raise NotImplementedError("Backward pass not implemented for fully fused mlp")  # pragma: no cover


def fully_fused_mlp(
    # fmt: off
    x: Tensor, 
    w_in: Tensor, b_in: Tensor, 
    w_out: Tensor, b_out: Tensor, 
    w_hid: Tensor | None = None, b_hid: Tensor | None = None, 
    activation: str = "relu", 
    fp16_acc: bool = False,
    # fmt: on
) -> Tensor:
    r"""Perform a fully fused forward pass of a multi-layer perceptron (MLP).

    This implementation of a fully fused MLP is inspired by the approach described in [1]. Scaling is particularly
    effective with multiple layers.

    .. [1] Müller, T., Evans, B., Schied, C., & Keller, A. (2021). Real-time Neural Radiance Caching for Path Tracing. arXiv preprint arXiv:2106.12372.
        Available at https://arxiv.org/abs/2106.12372

    .. note::
        Backward pass is not yet implemented for this operation.

    Args:
        x: Input tensor
        w_in: Weight tensor for the input layer
        b_in: Bias tensor for the input layer
        w_out: Weight tensor for the output layer
        b_out: Bias tensor for the output layer, or None if there are no additional layers
        w_hid: Weight tensor for the hidden layers, or None if there are no additional layers
        activation: Activation function to use. Can be "relu" or "silu".
        fp16_acc: Use FP16 for accumulation. This will reduce precision and may lead to numerical instability, but
            is faster.

    Shapes:
        - x: :math:`(B, L, D_in)`
        - w_in: :math:`(D_hidden, D_in)`
        - b_in: :math:`(D_hidden,)`
        - w_out: :math:`(D_out, D_hidden)`
        - b_out: :math:`(D_out,)`
        - w_hid: :math:`(depth * D_hidden, D_hidden)`
        - b_hid: :math:`(depth, D_hidden)`
        - Output: :math:`(B, L, D_out)`

    Returns:
        Output of MLP forward pass.
    """
    return _mlp_forward.apply(x, w_in, b_in, w_out, b_out, w_hid, b_hid, activation, fp16_acc)  # type: ignore
