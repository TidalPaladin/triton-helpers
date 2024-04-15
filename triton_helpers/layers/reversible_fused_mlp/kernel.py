from typing import Any, cast

import triton
import triton.language as tl
from torch import Tensor
from torch.autograd import Function

from ...heuristics import PowerOfTwoHeuristic
from ...ops import ensure_str
from ..fully_fused_mlp.kernel import FULLY_FUSED_MAX_DIM, feedforward, feedforward_bwd_dw, feedforward_bwd_dx, feedforward_bwd_dz


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

    for d in tl.static_range(DEPTH):
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
        if d < DEPTH - 1:
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


@triton.jit
def _bwd_inner(
    # fmt: off
    # Inputs
    x, w, b,
    # Derivatives
    do, dw_p, db_p,
    # Lock
    lock_p,
    # Params
    ACTIVATION: tl.constexpr, DTYPE: tl.constexpr, USE_LOCK: tl.constexpr = tl.constexpr(True),
    # fmt: on
) -> tl.tensor:
    # Compute z from x
    z = feedforward(x, w, b, ACTIVATION="none", DTYPE=DTYPE)

    # Compute dz
    dz = feedforward_bwd_dz(z, do, ACTIVATION=ACTIVATION).to(do.dtype)

    # Compute dw and db
    db = tl.sum(dz, axis=0).to(db_p.dtype.element_ty)
    db = tl.expand_dims(db, axis=0)
    dw = feedforward_bwd_dw(x, dz, DTYPE=DTYPE).to(dw_p.dtype.element_ty)

    # Atomic update of dw and db
    if USE_LOCK:
        # TODO: Maybe try atomic_add, it's generally slower though
        while tl.atomic_cas(lock_p, 0, 1) == 1:
            pass
        dw += tl.load(dw_p, boundary_check=(0, 1))
        tl.store(dw_p, dw, boundary_check=(0, 1))
        db += tl.load(db_p, boundary_check=(1,))
        tl.store(db_p, db, boundary_check=(1,))
        tl.atomic_xchg(lock_p, 0)
    else:
        tl.store(dw_p, dw, boundary_check=(0, 1), eviction_policy="evict_first")
        tl.store(db_p, db, boundary_check=(1,), eviction_policy="evict_first")

    # Compute and return dx
    dx = feedforward_bwd_dx(w, dz, DTYPE=DTYPE)
    return dx


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_L": 16}),
        triton.Config({"BLOCK_L": 32}),
        triton.Config({"BLOCK_L": 64}),
    ],
    key=["L", "D_in", "D_hidden", "D_out", "DEPTH"],
    reset_to_zero=["dw_in_p", "db_in_p", "dw_hid_p", "db_hid_p", "dw_out_p", "db_out_p"],
)
@triton.heuristics(
    dict(
        BLOCK_D_IN=PowerOfTwoHeuristic("D_in", 16),
        BLOCK_D_HIDDEN=PowerOfTwoHeuristic("D_hidden", 16),
        BLOCK_D_OUT=PowerOfTwoHeuristic("D_out", 16),
        BLOCK_L=PowerOfTwoHeuristic("L", min_val=16, max_val=64),
        num_warps=lambda _: 4,
        num_stages=lambda _: 2,
    )
)
@triton.jit
def _bwd_kernel_rfs(
    # fmt: off
    # Inputs
    x_p, w_in_p, b_in_p, w_hid_p, b_hid_p, w_out_p, b_out_p,
    # Derivatives
    dx_p, do_p, dw_in_p, db_in_p, dw_hid_p, db_hid_p, dw_out_p, db_out_p,
    # Lock
    lock_p,
    # Sizes
    L: int, D_in: int, D_hidden: int, D_out: int,
    # Strides
    stride_x_l: int, stride_do_l: int,
    # Params
    ACTIVATION: tl.constexpr, DTYPE: tl.constexpr, DEPTH: tl.constexpr, USE_LOCK: tl.constexpr,
    # Blocks
    BLOCK_L: tl.constexpr, BLOCK_D_IN: tl.constexpr, BLOCK_D_HIDDEN: tl.constexpr, BLOCK_D_OUT: tl.constexpr,
    # fmt: on
):
    # This kernel implements the backward pass by recomputing the forward pass in reverse DEPTH-1 times.

    # Select offsets for this block
    start = tl.program_id(0) * BLOCK_L

    # Update input and output pointer
    x_p += start * stride_x_l
    dx_p += start * stride_x_l
    do_p += start * stride_do_l
    if not USE_LOCK:
        pid = tl.program_id(0)
        dw_in_p += pid * D_in * D_hidden
        db_in_p += pid * D_hidden
        dw_hid_p += pid * D_hidden * D_hidden
        db_hid_p += pid * D_hidden
        dw_out_p += pid * D_hidden * D_out
        db_out_p += pid * D_out

    # Load x
    X_block_ptr = tl.make_block_ptr(
        x_p,
        (L, D_in),
        (stride_x_l, 1),
        (0, 0),
        (BLOCK_L, BLOCK_D_IN),
        (1, 0),
    )
    x = tl.load(X_block_ptr, boundary_check=(0, 1))

    # Load do
    DO_block_ptr = tl.make_block_ptr(
        do_p,
        (L, D_out),
        (stride_do_l, 1),
        (0, 0),
        (BLOCK_L, BLOCK_D_OUT),
        (1, 0),
    )
    do = tl.load(DO_block_ptr, boundary_check=(0, 1))

    # Backward through output layer
    x_prev = _fwd_inner(
        # fmt: off
        x, w_in_p, b_in_p, w_hid_p, b_hid_p, 
        D_in, D_hidden,
        ACTIVATION, DTYPE, DEPTH,
        BLOCK_D_IN, BLOCK_D_HIDDEN,
        # fmt: on
    )
    w = tl.load(
        _make_w_block_ptr(w_out_p, D_hidden, D_out, BLOCK_D_HIDDEN, BLOCK_D_OUT),
        boundary_check=(0, 1),
    )
    b = tl.load(
        _make_b_block_ptr(b_out_p, D_out, BLOCK_D_OUT),
        boundary_check=(1,),
    )
    DW_block_ptr = _make_w_block_ptr(dw_out_p, D_hidden, D_out, BLOCK_D_HIDDEN, BLOCK_D_OUT)
    DB_block_ptr = _make_b_block_ptr(db_out_p, D_out, BLOCK_D_OUT)
    do = _bwd_inner(
        # fmt: off
        x_prev, w, b,
        do, DW_block_ptr, DB_block_ptr,
        lock_p,
        "none", DTYPE, USE_LOCK,
        # fmt: on
    )
    lock_p += 1

    # Init hidden dw / db block pointers at the end of the block.
    # We will work backwards through the hidden layers.
    #
    # NOTE: The inner loop duplicates parts of _fwd_inner. Calling the function would be cleaner,
    # but results in `DEPTH-1` recompilations of _fwd_inner, which is very slow.
    # It's not clear why - efforts to avoid DEPTH being a tl.constexpr in _fwd_inner did not help.
    H: tl.constexpr = DEPTH - 1
    DW_block_ptr = _make_w_block_ptr(dw_hid_p, D_hidden, D_hidden, BLOCK_D_HIDDEN, BLOCK_D_HIDDEN, H, True)
    DB_block_ptr = _make_b_block_ptr(db_hid_p, D_hidden, BLOCK_D_HIDDEN, H, True)
    for i in tl.static_range(H):
        # Input layer
        w_in = tl.load(
            _make_w_block_ptr(w_in_p, D_in, D_hidden, BLOCK_D_IN, BLOCK_D_HIDDEN),
            boundary_check=(0, 1),
        )
        b_in = tl.load(
            _make_b_block_ptr(b_in_p, D_hidden, BLOCK_D_HIDDEN),
            boundary_check=(1,),
        )
        x_prev = feedforward(x, w_in, b_in, ACTIVATION=ACTIVATION, DTYPE=DTYPE)

        # Possible hidden layers
        W_block_ptr = _make_w_block_ptr(w_hid_p, D_hidden, D_hidden, BLOCK_D_HIDDEN, BLOCK_D_HIDDEN, H)
        B_block_ptr = _make_b_block_ptr(b_hid_p, D_hidden, BLOCK_D_HIDDEN, H)
        for _ in range(H - (i + 1)):
            # Hidden forward
            w_hid = tl.load(W_block_ptr, boundary_check=(0, 1))
            b_hid = tl.load(B_block_ptr, boundary_check=(1,))
            x_prev = feedforward(x_prev, w_hid, b_hid, ACTIVATION=ACTIVATION, DTYPE=DTYPE)

            # Advance pointers
            W_block_ptr = tl.advance(W_block_ptr, (BLOCK_D_HIDDEN, 0))
            B_block_ptr = tl.advance(B_block_ptr, (1, 0))

        w_h = tl.load(W_block_ptr, boundary_check=(0, 1), eviction_policy="evict_first")
        b_h = tl.load(B_block_ptr, boundary_check=(1,), eviction_policy="evict_first")
        do = _bwd_inner(
            # fmt: off
            x_prev, w_h, b_h,
            do, DW_block_ptr, DB_block_ptr,
            lock_p,
            ACTIVATION, DTYPE, USE_LOCK,
            # fmt: on
        )

        # Update pointers
        # NOTE: Conditional is needed to silence warnings about tl.advance
        if i < H - 1:
            DW_block_ptr = tl.advance(DW_block_ptr, (-BLOCK_D_HIDDEN, 0))
            DB_block_ptr = tl.advance(DB_block_ptr, (-1, 0))
        lock_p += 1

    # Backward through input layer
    w = tl.load(
        _make_w_block_ptr(w_in_p, D_in, D_hidden, BLOCK_D_IN, BLOCK_D_HIDDEN),
        boundary_check=(0, 1),
    )
    b = tl.load(
        _make_b_block_ptr(b_in_p, D_hidden, BLOCK_D_HIDDEN),
        boundary_check=(1,),
    )
    DW_block_ptr = _make_w_block_ptr(dw_in_p, D_in, D_hidden, BLOCK_D_IN, BLOCK_D_HIDDEN)
    DB_block_ptr = _make_b_block_ptr(db_in_p, D_hidden, BLOCK_D_HIDDEN)
    do = _bwd_inner(
        # fmt: off
        x, w, b,
        do, DW_block_ptr, DB_block_ptr,
        lock_p,
        ACTIVATION, DTYPE, USE_LOCK,
        # fmt: on
    )

    # Final state of do is now dx, store it
    DX_block_ptr = tl.make_block_ptr(
        dx_p,
        (L, D_in),
        (stride_x_l, 1),
        (0, 0),
        (BLOCK_L, BLOCK_D_IN),
        (1, 0),
    )
    tl.store(
        DX_block_ptr,
        do.to(dx_p.dtype.element_ty),
        boundary_check=(0, 1),
        eviction_policy="evict_first",
    )


class ReversibleFusedMLP(Function):

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
    return ReversibleFusedMLP.apply(x, w, b, activation, fp16_acc)  # type: ignore

