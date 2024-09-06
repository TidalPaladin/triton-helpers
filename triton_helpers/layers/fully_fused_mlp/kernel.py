from typing import Any, Final, Sequence, cast

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd import Function

from ...heuristics import PowerOfTwoHeuristic
from ...ops import relu, relu2, relu2_bwd, relu_bwd, silu, silu_bwd


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
    # z = x @ w.T + b
    dtype = x.dtype
    z = tl.dot(x, tl.trans(w), out_dtype=DTYPE)
    if HAS_BIAS:
        z += b

    if ACTIVATION == "relu":
        y = relu(z)
    elif ACTIVATION == "silu":
        y = silu(z)
    elif ACTIVATION == "none":
        y = z
    elif ACTIVATION == "relu2":
        y = relu2(z)
    else:
        tl.static_assert(False, f"Invalid activation function: {ACTIVATION}")
        # For type checker
        y = z

    return y.to(dtype)


@triton.jit
def feedforward_bwd_dz(z: tl.tensor, do: tl.tensor, ACTIVATION: tl.constexpr) -> tl.tensor:
    if ACTIVATION == "relu":
        dz = relu_bwd(z, do)
    elif ACTIVATION == "silu":
        dz = silu_bwd(z, do)
    elif ACTIVATION == "none":
        dz = do
    elif ACTIVATION == "relu2":
        dz = relu2_bwd(z, do)
    else:
        tl.static_assert(False, f"Invalid activation function: {ACTIVATION}")
        # For type checker
        dz = do
    return dz.to(do.dtype)


@triton.jit
def feedforward_bwd_dw(x: tl.tensor, dz: tl.tensor, DTYPE: tl.constexpr = tl.constexpr(tl.float32)) -> tl.tensor:
    return tl.dot(tl.trans(dz), x, out_dtype=DTYPE).to(dz.dtype)


@triton.jit
def feedforward_bwd_dx(w: tl.tensor, dz: tl.tensor, DTYPE: tl.constexpr = tl.constexpr(tl.float32)) -> tl.tensor:
    return tl.dot(dz, w, out_dtype=DTYPE).to(dz.dtype)


@triton.jit
def _make_w_block_ptr(
    # fmt: off
    p: tl.pointer_type,
    D_IN: tl.constexpr, D_OUT: tl.constexpr, 
    BLOCK_D_IN: tl.constexpr, BLOCK_D_OUT: tl.constexpr,
    STACK_SIZE: tl.constexpr = tl.constexpr(1),
    REVERSE: tl.constexpr = tl.constexpr(False),
    # fmt: on
):
    if REVERSE:
        return tl.make_block_ptr(
            p,
            (STACK_SIZE * D_OUT, D_IN),
            (D_IN, 1),
            (D_OUT * (STACK_SIZE - 1), 0),
            (BLOCK_D_OUT, BLOCK_D_IN),
            (1, 0),
        )

    else:
        return tl.make_block_ptr(
            p,
            (STACK_SIZE * D_OUT, D_IN),
            (D_IN, 1),
            (0, 0),
            (BLOCK_D_OUT, BLOCK_D_IN),
            (1, 0),
        )


@triton.jit
def _make_b_block_ptr(
    # fmt: off
    p: tl.pointer_type,
    D_OUT: tl.constexpr, 
    BLOCK_D_OUT: tl.constexpr,
    STACK_SIZE: tl.constexpr = tl.constexpr(1),
    REVERSE: tl.constexpr = tl.constexpr(False),
    # fmt: on
):
    if REVERSE:
        return tl.make_block_ptr(
            p,
            (STACK_SIZE, D_OUT),
            (D_OUT, 1),
            (STACK_SIZE - 1, 0),
            (1, BLOCK_D_OUT),
            (1, 0),
        )

    else:
        return tl.make_block_ptr(
            p,
            (STACK_SIZE, D_OUT),
            (D_OUT, 1),
            (0, 0),
            (1, BLOCK_D_OUT),
            (1, 0),
        )


@triton.jit
def _fwd_inner(
    # fmt: off
    # Inputs
    x, w_in_p, b_in_p, w_hid_p, b_hid_p,
    # Sizes
    D_in: int, D_hidden: int,
    # Params
    ACTIVATION: tl.constexpr, DTYPE: tl.constexpr, DEPTH: tl.constexpr,
    # Blocks
    BLOCK_D_IN: tl.constexpr, BLOCK_D_HIDDEN: tl.constexpr,
    # fmt: on
) -> tl.tensor:
    # Input layer
    w_in = tl.load(
        _make_w_block_ptr(w_in_p, D_in, D_hidden, BLOCK_D_IN, BLOCK_D_HIDDEN),
        boundary_check=(0, 1),
    )
    b_in = tl.load(
        _make_b_block_ptr(b_in_p, D_hidden, BLOCK_D_HIDDEN),
        boundary_check=(1,),
    )
    x = feedforward(x, w_in, b_in, ACTIVATION=ACTIVATION, DTYPE=DTYPE)

    # Possible hidden layers
    H: tl.constexpr = DEPTH - 1
    W_block_ptr = _make_w_block_ptr(w_hid_p, D_hidden, D_hidden, BLOCK_D_HIDDEN, BLOCK_D_HIDDEN, H)
    B_block_ptr = _make_b_block_ptr(b_hid_p, D_hidden, BLOCK_D_HIDDEN, H)
    for i in tl.static_range(H):
        # Hidden forward
        w_hid = tl.load(W_block_ptr, boundary_check=(0, 1))
        b_hid = tl.load(B_block_ptr, boundary_check=(1,))
        x = feedforward(x, w_hid, b_hid, ACTIVATION=ACTIVATION, DTYPE=DTYPE)

        # Advance pointers
        if i < H - 1:
            W_block_ptr = tl.advance(W_block_ptr, (BLOCK_D_HIDDEN, 0))
            B_block_ptr = tl.advance(B_block_ptr, (1, 0))

    return x


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_L": 16}),
        triton.Config({"BLOCK_L": 32}),
        triton.Config({"BLOCK_L": 64}),
        triton.Config({"BLOCK_L": 128}),
    ],
    key=["L", "D_in", "D_hidden", "D_out", "DEPTH"],
)
@triton.heuristics(
    dict(
        BLOCK_D_IN=PowerOfTwoHeuristic("D_in", 16),
        BLOCK_D_HIDDEN=PowerOfTwoHeuristic("D_hidden", 16),
        BLOCK_D_OUT=PowerOfTwoHeuristic("D_out", 16),
        BLOCK_L=PowerOfTwoHeuristic("L", min_val=16, max_val=128),
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
    stride_x_l: int, stride_x_d: int, stride_o_l: int, stride_o_d: int,
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

    # Load input
    X_block_ptr = tl.make_block_ptr(
        x_p,
        (L, D_in),
        (stride_x_l, stride_x_d),
        (0, 0),
        (BLOCK_L, BLOCK_D_IN),
        (1, 0),
    )
    x = tl.load(X_block_ptr, boundary_check=(0, 1))

    # Forward through input and hidden layers
    x = _fwd_inner(
        # fmt: off
        x, w_in_p, b_in_p, w_hid_p, b_hid_p, 
        D_in, D_hidden,
        ACTIVATION, DTYPE, DEPTH,
        BLOCK_D_IN, BLOCK_D_HIDDEN,
        # fmt: on
    )

    # Output layer
    w_out = tl.load(
        _make_w_block_ptr(w_out_p, D_hidden, D_out, BLOCK_D_HIDDEN, BLOCK_D_OUT),
        boundary_check=(0, 1),
    )
    b_out = tl.load(
        _make_b_block_ptr(b_out_p, D_out, BLOCK_D_OUT),
        boundary_check=(1,),
    )
    x = feedforward(x, w_out, b_out, ACTIVATION="none", DTYPE=DTYPE)

    # Store output
    O_block_ptr = tl.make_block_ptr(
        o_p,
        (L, D_out),
        (stride_o_l, stride_o_d),
        (0, 0),
        (BLOCK_L, BLOCK_D_OUT),
        (1, 0),
    )
    tl.store(O_block_ptr, x, boundary_check=(0, 1), eviction_policy="evict_first")


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
    stride_x_l: int, stride_x_d: int, stride_do_l: int, stride_do_d: int,
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
        (stride_x_l, stride_x_d),
        (0, 0),
        (BLOCK_L, BLOCK_D_IN),
        (1, 0),
    )
    x = tl.load(X_block_ptr, boundary_check=(0, 1))

    # Load do
    DO_block_ptr = tl.make_block_ptr(
        do_p,
        (L, D_out),
        (stride_do_l, stride_do_d),
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
        (stride_x_l, stride_x_d),
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


class _fully_fused_mlp(Function):

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
    @torch.no_grad()
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

        # Init hidden layer weights
        if w_hid is not None and b_hid is not None:
            depth = w_hid.shape[0] // D_hidden
            assert w_hid.shape == (depth * D_hidden, D_hidden), "Hidden weight shape mismatch"
            assert b_hid.shape == (depth, D_hidden), "Hidden bias shape mismatch"
            depth += 1
        else:
            depth = 1
            w_hid = x.new_empty(0)
            b_hid = x.new_empty(0)

        # Handle non-str activation
        assert activation in ["relu", "relu2", "silu", "none"], f"Invalid activation function: {activation}"

        # Init output
        o = x.new_empty((B, L, D_out))

        def grid(META):
            return (triton.cdiv(B * L, META["BLOCK_L"]),)

        cast(Any, _fwd_kernel)[grid](
            # fmt: off
            x, w_in, b_in, w_hid, b_hid, w_out, b_out, o,
            B * L, D_in, D_hidden, D_out,
            x.stride(1), x.stride(2), o.stride(1), o.stride(2),
            activation, 
            DTYPE=tl.float16 if fp16_acc else tl.float32,
            DEPTH=depth,
            # fmt: on
        )

        ctx.save_for_backward(x, w_in, b_in, w_out, b_out, w_hid, b_hid)
        ctx.activation = activation
        ctx.fp16_acc = fp16_acc
        ctx.depth = depth

        return o

    @torch.cuda.amp.custom_bwd
    @torch.no_grad()
    @staticmethod
    def backward(ctx, do: Tensor):
        x, w_in, b_in, w_out, b_out, w_hid, b_hid = cast(Sequence[Tensor], ctx.saved_tensors)
        assert x.dtype != torch.float32 or ctx.depth < 4, "FP32 backward not supported for depth >= 4"

        B, L, D_in = x.shape
        D_hidden, _ = w_in.shape
        D_out, _ = w_out.shape

        dx = x.new_empty(x.shape)

        # Init of these tensors depends on choice of locking, which depends on depth
        # Deadlocks seem to become an issue at small depths, so we only use locking for depth > 5.
        # NOTE: There are still deadlock issues so this is currently disabled
        # if USE_LOCK := ctx.depth > 5:
        if USE_LOCK := False:
            dw_in = torch.zeros_like(w_in)
            db_in = torch.zeros_like(b_in)
            dw_hid = torch.zeros_like(w_hid)
            db_hid = torch.zeros_like(b_hid)
            dw_out = torch.zeros_like(w_out)
            db_out = torch.zeros_like(b_out)
            locks = torch.zeros(ctx.depth + 1, dtype=torch.int32, device=x.device)
            # For some reason FP16 acc is much slower when locking is used at small L
            fp16_acc = True
        else:
            G_max = triton.cdiv(B * L, 16)
            dw_in = w_in.new_empty(G_max, *w_in.shape)
            db_in = b_in.new_empty(G_max, *b_in.shape)
            dw_hid = w_hid.new_empty(G_max, *w_hid.shape)
            db_hid = b_hid.new_empty(G_max, *b_hid.shape)
            dw_out = w_out.new_empty(G_max, *w_out.shape)
            db_out = b_out.new_empty(G_max, *b_out.shape)
            locks = torch.empty(0, dtype=torch.int32, device=x.device)
            fp16_acc = ctx.fp16_acc

        def grid(META):
            return (triton.cdiv(B * L, META["BLOCK_L"]),)

        cast(Any, _bwd_kernel_rfs)[grid](
            # fmt: off
            x, w_in, b_in, w_hid, b_hid, w_out, b_out,
            dx, do, dw_in, db_in, dw_hid, db_hid, dw_out, db_out,
            locks,
            B * L, D_in, D_hidden, D_out,
            x.stride(1), x.stride(2), do.stride(1), do.stride(2),
            ctx.activation,
            DTYPE=tl.float16 if fp16_acc else tl.float32,
            DEPTH=ctx.depth,
            USE_LOCK=USE_LOCK,
            # fmt: on
        )

        # If not locking, perform reduction
        if not USE_LOCK:
            dw_in = dw_in.sum(dim=0)
            db_in = db_in.sum(dim=0)
            dw_hid = dw_hid.sum(dim=0)
            db_hid = db_hid.sum(dim=0)
            dw_out = dw_out.sum(dim=0)
            db_out = db_out.sum(dim=0)

        dw_hid = dw_hid if ctx.depth > 1 else None
        db_hid = db_hid if ctx.depth > 1 else None
        return dx, dw_in, db_in, dw_out, db_out, dw_hid, db_hid, None, None


def fully_fused_mlp(
    # fmt: off
    x: Tensor, 
    w_in: Tensor, b_in: Tensor, 
    w_out: Tensor, b_out: Tensor, 
    w_hid: Tensor | None = None, b_hid: Tensor | None = None, 
    activation: str | triton.JITFunction = "relu", 
    fp16_acc: bool = False,
    # fmt: on
) -> Tensor:
    r"""Perform a fully fused forward pass of a multi-layer perceptron (MLP).

    This implementation of a fully fused MLP is inspired by the approach described in [1]. Scaling is particularly
    effective with multiple layers.

    .. note::
        The forward-backward latency of this kernel is only significantly superior to the non-fused version for small inputs
        or for large depth. Forward latency of htis kernel is superior for all inputs.

    .. [1] Mller, T., Evans, B., Schied, C., & Keller, A. (2021). Real-time Neural Radiance Caching for Path Tracing. arXiv preprint arXiv:2106.12372.
        Available at https://arxiv.org/abs/2106.12372

    Args:
        x: Input tensor
        w_in: Weight tensor for the input layer
        b_in: Bias tensor for the input layer
        w_out: Weight tensor for the output layer
        b_out: Bias tensor for the output layer, or None if there are no additional layers
        w_hid: Weight tensor for the hidden layers, or None if there are no additional layers
        activation: Activation function to use. Can be "relu", "relu2", "silu", or "none".
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
    return _fully_fused_mlp.apply(x, w_in, b_in, w_out, b_out, w_hid, b_hid, activation, fp16_acc)  # type: ignore
