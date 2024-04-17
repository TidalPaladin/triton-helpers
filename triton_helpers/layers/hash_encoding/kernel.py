import math
from typing import Any, Dict, Final

import torch
import triton
import triton.language as tl
from torch import Tensor

from triton_helpers.heuristics import PowerOfTwoHeuristic
from triton_helpers.ops import high_low_mod

from .helpers import (
    compute_b,
    compute_embedding_counts,
    compute_resolutions,
    get_first_hash_level,
    seek_to_level_embeddings,
)


# Three primes chosen as in the paper, one prime per spatial dimension
PI_1: Final = 1
PI_2: Final = 2_654_435_761
PI_3: Final = 805_459_861


@triton.jit
def prod(x: tl.tensor, y: tl.tensor) -> tl.tensor:
    return x * y


@triton.jit
def create_corner_offsets(BLOCK: tl.constexpr) -> tl.tensor:
    r"""Creates binary offsets to the 2**BLOCK corners of a hypercube.

    Shape:
        (2**BLOCK, BLOCK)
    """
    return (tl.arange(0, 2**BLOCK)[:, None] >> tl.arange(0, BLOCK)[None, :]) & 1


@triton.jit
def get_interpolation_weights(x: tl.tensor, D: tl.constexpr, BLOCK: tl.constexpr) -> tl.tensor:
    r"""Gets interpolation weights for a point in a hypercube.

    Shape:
        (N, 2**BLOCK)
    """
    # Get corner of x hypercube
    x_rd = tl.math.float2int_rd(x).to(x.dtype)

    # Get corner offsets (1, 2**D, D)
    corner_offsets = create_corner_offsets(BLOCK)[None, :, :]

    # Compute interpolation weights (N, 2**D, 1)
    w = (x - x_rd)[:, None, :]
    w = tl.where(corner_offsets == 0, 1 - w, w)

    # Set out of bounds weights along last dim to 1 and product reduce along last dim
    w = tl.where(tl.arange(0, BLOCK)[None, None, :] < D, w, 1)
    w = tl.reduce(w, 2, prod)

    # Set out of bounds weights along new last dim to 0
    w = tl.where(tl.arange(0, 2**BLOCK)[None, :] < 2**D, w, 0)

    return w


@triton.jit
def interpolate(x: tl.tensor, e: tl.tensor, D: tl.constexpr, BLOCK_D: tl.constexpr) -> tl.tensor:
    r"""Interpolates feature vectors for a point in a hypercube.

    Shape:
        (N, F)
    """
    w = get_interpolation_weights(x, D, BLOCK_D)[:, :, None]
    return tl.sum(w * e, axis=1).to(e.dtype)


@triton.jit
def interpolate_bwd(
    x: tl.tensor, do: tl.tensor, D: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_F: tl.constexpr
) -> tl.tensor:
    # Recompute the interpolation weights (N, 2**D, F)
    w = get_interpolation_weights(x, D, BLOCK_D)[:, :, None] + tl.arange(0, BLOCK_F)[None, None, :]

    # Compute de (N, 2**D, F)
    de = do[:, None, :] * w


@triton.jit
def embedding_lookup(
    # fmt: off
    x: tl.tensor, pi: tl.tensor,
    D: int,
    # Table params
    T_l, N_l,
    # Block sizes
    BLOCK_D: tl.constexpr,
    # Hash needed
    NEEDS_HASH: tl.constexpr,
    T_POW_2: tl.constexpr = tl.constexpr(False),
    # Working dtype
    DTYPE: tl.constexpr = tl.constexpr(tl.uint32),
    # fmt: on
) -> tl.tensor:
    r"""Looks up embedding indices for the hybercube corners about a spatial coordinate.

    Shape:
        (N, 2**D)
    """
    # Debugging this function is a nightmare, so many assertions are provided that will trigger
    # only when TRITON_DEBUG=1. Otherwise these add no runtime latency. Use assertions liberally, otherwise
    # outputs will be silently wrong with little indication of why.
    # NOTE: Be very careful with truncation and casting. Everything needs to use uint32, but it's
    # easy to overflow in intermediate products.
    tl.device_assert((x >= 0), "x must be non-negative")
    tl.device_assert((x < N_l), "x must be less than N_l")
    tl.device_assert((T_l < 2**32), "T_l must be less than 2**32")
    tl.device_assert((N_l < 2**32), "N_l must be less than 2**32")
    tl.device_assert(
        (tl.math.pow((N_l + 1.0).to(tl.float64), D) > T_l) == NEEDS_HASH, "Hashing condition set incorrectly"
    )

    # Scale x by this level's resolution and round down to get the lower corner
    # TODO: Should probably change this to support only uint32
    if DTYPE is tl.uint64:
        x = tl.math.float2ull_rd(x)
    elif DTYPE is tl.uint32:
        x = tl.math.float2uint_rd(x)
    else:
        tl.static_assert(False, "Hash must be either uint or int, 32 or 64")

    # Map x to 2**D vertices for each corner
    corners = x[:, None, :] + create_corner_offsets(BLOCK_D).to(DTYPE)

    # At coarse resolution hashing isn't needed, mapping is 1:1
    if not NEEDS_HASH:  # type: ignore
        # Scale dimension D_i by (N_l + 1) ** i (this is basically a stride)
        scale = tl.math.pow(N_l, tl.arange(0, BLOCK_D).to(tl.float32))
        scale = tl.where(tl.arange(0, BLOCK_D) < D, scale, 0)
        h = tl.sum(corners * scale.to(DTYPE)[None, None, :], axis=2)

    # Otherwise compute hash function
    else:
        # If T is a power of two we can do the modulo very fast with a bitwise and
        T_l = T_l.to(DTYPE)
        if T_POW_2:
            tl.device_assert(tl.math.popc(T_l.to(tl.int32)) == 1, "T must be a power of 2")
            h = tl.xor_sum(corners * pi[None, None, :], 2) & (T_l - 1)
        else:
            low = tl.xor_sum(corners * pi[None, None, :], 2)
            high = tl.xor_sum(tl.math.mulhi(corners, pi[None, None, :]), 2)
            h = high_low_mod(high, low, T_l)

    tl.device_assert((h < T_l) & (h >= 0), f"Embedding index out of bounds")
    return h


def _compute_b(args: Dict[str, Any]) -> int:
    return compute_b(args["MIN_RES"], args["MAX_RES"], args["L"])


def _get_first_hash_level(args: Dict[str, Any]) -> int:
    result = get_first_hash_level(args["MIN_RES"], args["MAX_RES"], args["L"], args["T"], args["D"])
    return result


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_N": 64}, num_warps=8),
        triton.Config({"BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_N": 128}, num_warps=8),
        triton.Config({"BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_N": 256}, num_warps=8),
        triton.Config({"BLOCK_N": 256}, num_warps=16),
    ],
    key=["D", "F", "L", "T", "MIN_RES", "MAX_RES"],
)
@triton.heuristics(
    values={
        "BLOCK_D": PowerOfTwoHeuristic("D"),
        "BLOCK_F": PowerOfTwoHeuristic("F"),
        # "BLOCK_N": PowerOfTwoHeuristic("N", min_val=16, max_val=16),
        "B": _compute_b,
        "FIRST_HASH_LEVEL": _get_first_hash_level,
    }
)
@triton.jit
def _fwd_kernel(
    # fmt: off
    # Input
    x_p, pi_p, e_p, o_p,
    # Strides
    stride_x_n: int, stride_x_d: int,
    stride_e_t: int,
    stride_o_n: int, stride_o_f: int,
    # Sizes
    N: int, D: tl.constexpr, F: tl.constexpr,
    # Hash parameters
    T: tl.constexpr, L: tl.constexpr, MIN_RES: tl.constexpr, MAX_RES: tl.constexpr,
    # Block sizes
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_F: tl.constexpr,
    # Derived parameters
    B: tl.constexpr, FIRST_HASH_LEVEL: tl.constexpr,
    # Dtypes
    INT_DTYPE: tl.constexpr = tl.uint32,
    # fmt: on
):
    # Input validation and constant setup
    tl.static_assert(
        FIRST_HASH_LEVEL > 0, "Hashing requested at first level, this indicates a bug or poor table parameters"
    )
    T_POW_2: tl.constexpr = T & (T - 1) == 0

    # Set pointers to this program's start
    start = tl.program_id(0) * BLOCK_N

    # Load input
    X_block_ptr = tl.make_block_ptr(
        x_p,
        (N, D),
        (stride_x_n, stride_x_d),
        (start, 0),
        (BLOCK_N, BLOCK_D),
        (1, 0),
    )
    x = tl.load(X_block_ptr, boundary_check=(0, 1))

    # Load pi
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D
    pi = tl.load(pi_p + offsets, mask=mask).to(INT_DTYPE)

    # Iterate over hash table levels
    # NOTE: It is empirically faster to run this as a static range, accumulate into a single buffer,
    # and then do one write at the end. The expectation is that this table will be used with a fully
    # fused MLP, in which case L * F should be small (typically 32 for L=16, F=2) and fit fully in SRAM.
    N_l_prev = tl.constexpr(1)
    o = tl.zeros((BLOCK_N, L, BLOCK_F), dtype=o_p.dtype.element_ty)
    for l in tl.static_range(L):
        # Resolution and number of array entries for this level
        N_l = tl.constexpr(MIN_RES * B**l)
        T_l = tl.constexpr(min((MIN_RES * B**l + 1) ** D, T))

        # Apply scaling to x
        X_SCALE = tl.constexpr(N_l // N_l_prev)
        x = x * X_SCALE
        N_l_prev = tl.constexpr(N_l)

        # Look up embeddings
        embedding_idx = embedding_lookup(x, pi, D, T_l, N_l, BLOCK_D, l >= FIRST_HASH_LEVEL, T_POW_2, INT_DTYPE)
        embedding_idx = (embedding_idx * F)[:, :, None] + tl.arange(0, BLOCK_F).to(INT_DTYPE)[None, None, :]
        emb_mask = (
            (tl.arange(0, BLOCK_N) < N)[:, None, None]
            & (tl.arange(0, 2**BLOCK_D) < 2**D)[None, :, None]
            & (tl.arange(0, BLOCK_F) < F)[None, None, :]
        )
        embedding_idx = tl.where(emb_mask, embedding_idx, 0)
        e = tl.load(e_p + embedding_idx, mask=emb_mask)

        # Interpolate embeddings
        e = interpolate(x, e, D, BLOCK_D)[:, None, :]

        # Update output
        mask = tl.arange(0, L) == l
        o = tl.where(mask[None, :, None], e, o)

        # Advance pointers
        e_p += T_l * stride_e_t

        # Synchronize workers at this level to ensure a high cache hit rate.
        # This yields a very tangible speedup.
        tl.debug_barrier()

    o = tl.reshape(o, (BLOCK_N, L * BLOCK_F))
    start = tl.program_id(0) * BLOCK_N
    O_block_ptr = tl.make_block_ptr(
        o_p,
        (N, L * F),
        (stride_o_n, stride_o_f),
        (start, 0),
        (BLOCK_N, L * BLOCK_F),
        (1, 0),
    )
    tl.store(O_block_ptr, o, boundary_check=(0, 1), eviction_policy="evict_first")


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_N": 64}, num_warps=8),
        triton.Config({"BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_N": 128}, num_warps=8),
        triton.Config({"BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_N": 256}, num_warps=8),
        triton.Config({"BLOCK_N": 256}, num_warps=16),
    ],
    key=["D", "F", "L", "T", "MIN_RES", "MAX_RES"],
)
@triton.heuristics(
    values={
        # "BLOCK_N": PowerOfTwoHeuristic("N", max_val=64),
        "BLOCK_D": PowerOfTwoHeuristic("D"),
        "BLOCK_F": PowerOfTwoHeuristic("F"),
        "B": _compute_b,
        "FIRST_HASH_LEVEL": _get_first_hash_level,
    }
)
@triton.jit
def _bwd_kernel(
    # fmt: off
    # Input
    x_p, pi_p, do_p, de_p,
    # Strides
    stride_x_n: int, stride_x_d: int,
    stride_do_n: int, stride_do_f: int,
    stride_de_t: int,
    # Sizes
    N: int, D: tl.constexpr, F: tl.constexpr,
    # Hash parameters
    T: tl.constexpr, L: tl.constexpr, MIN_RES: tl.constexpr, MAX_RES: tl.constexpr,
    # Block sizes
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_F: tl.constexpr,
    # Derived parameters
    B: tl.constexpr, FIRST_HASH_LEVEL: tl.constexpr,
    # Dtypes
    INT_DTYPE: tl.constexpr = tl.uint32,
    # fmt: on
):
    # Input validation and constant setup
    tl.static_assert(
        FIRST_HASH_LEVEL > 0, "Hashing requested at first level, this indicates a bug or poor table parameters"
    )
    T_POW_2: tl.constexpr = T & (T - 1) == 0

    # Set pointers to this program's start
    start = tl.program_id(0) * BLOCK_N

    # Load input
    X_block_ptr = tl.make_block_ptr(
        x_p,
        (N, D),
        (stride_x_n, stride_x_d),
        (start, 0),
        (BLOCK_N, BLOCK_D),
        (1, 0),
    )
    x = tl.load(X_block_ptr, boundary_check=(0, 1))

    # Load pi
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D
    pi = tl.load(pi_p + offsets, mask=mask).to(INT_DTYPE)

    # Initialize DO pointer
    DO_block_ptr = tl.make_block_ptr(
        do_p,
        (N, F * L),
        (stride_do_n, stride_do_f),
        (start, 0),
        (BLOCK_N, BLOCK_F),
        (1, 0),
    )

    # Iterate over hash table levels
    N_l_prev = tl.constexpr(1)
    for l in tl.static_range(L):
        # Resolution and number of array entries for this level
        N_l = tl.constexpr(MIN_RES * B**l)
        T_l = tl.constexpr(min((MIN_RES * B**l + 1) ** D, T))

        # Apply scaling to x
        X_SCALE = tl.constexpr(N_l // N_l_prev)
        x = x * X_SCALE
        N_l_prev = tl.constexpr(N_l)

        # Look up embeddings
        embedding_idx = embedding_lookup(x, pi, D, T_l, N_l, BLOCK_D, l >= FIRST_HASH_LEVEL, T_POW_2, INT_DTYPE)
        embedding_idx = (embedding_idx * F)[:, :, None] + tl.arange(0, BLOCK_F).to(INT_DTYPE)[None, None, :]
        mask = (
            (tl.arange(0, BLOCK_N) < N)[:, None, None]
            & (tl.arange(0, 2**BLOCK_D) < 2**D)[None, :, None]
            & (tl.arange(0, BLOCK_F) < F)[None, None, :]
        )
        embedding_idx = tl.where(mask, embedding_idx, 0)

        # Get interpolation weights (N, 2**D)
        w = get_interpolation_weights(x, D, BLOCK_D)[:, :, None]

        # Load do for this level (N, F)
        do = tl.load(DO_block_ptr, boundary_check=(0, 1))

        # Compute de (N, 2**D, F)
        de = do[:, None, :] * w

        # Store
        tl.atomic_add(de_p + embedding_idx, de, mask=mask)

        # Advance pointers
        if l < L - 1:
            DO_block_ptr = tl.advance(DO_block_ptr, (0, BLOCK_F))
            de_p += T_l * stride_de_t


def _cpu_create_corner_offsets(d: int, **kwargs) -> Tensor:
    result = (torch.arange(2**d, dtype=torch.int32).view(-1, 1) >> torch.arange(d, dtype=torch.int32).view(1, -1)) & 1
    return result.to(**kwargs)


@torch.no_grad()
def _cpu_embedding_lookup(x: Tensor, pi: Tensor, D: int, T_l: int, N_l: int) -> Tensor:
    x_rd = torch.floor(x * N_l).to(torch.int64)
    offsets = _cpu_create_corner_offsets(D, device=x_rd.device, dtype=torch.int64)
    corners = x_rd[:, None, :] + offsets[None, :, :]
    if (N_l + 1) ** D <= T_l:
        scale = N_l ** torch.arange(D, device=pi.device)
        h = corners * scale
        return h.sum(-1)
    else:
        t = corners * pi
        h = torch.bitwise_xor(t[..., 0], t[..., 1])
        for i in range(2, D):
            h = torch.bitwise_xor(h, t[..., i])
        return h % T_l


def _cpu_interpolate(x: Tensor, e: Tensor, D: int, N_l: int) -> Tensor:
    x = x * N_l
    x_rd = torch.floor(x).to(torch.int64)

    # Get corner offsets (1, 2**D, D)
    corner_offsets = _cpu_create_corner_offsets(D, device=x_rd.device, dtype=torch.int64)

    # Compute interpolation weights (N, 2**D, 1)
    w = (x - x_rd)[:, None, :]
    w = torch.where(corner_offsets == 0, 1 - w, w)

    # Set out of bounds weights along last dim to 1 and product reduce along last dim
    w = w.prod(-1)

    return (w[..., None] * e).sum(1)


def _cpu_hash_encoding(
    # fmt: off
    x: Tensor, e: Tensor, pi: Tensor, o: Tensor,
    D: int, F: int,
    T: int, L: int, N_min: int, N_max: int,
    # fmt: on
) -> Tensor:
    assert 1 <= D <= 3, "D must be 1, 2, or 3"
    resolutions = compute_resolutions(L, N_min, N_max)
    t_vals = compute_embedding_counts(L, T, D, N_min, N_max)
    x.shape[0]
    for l_i in range(L):
        n_i = resolutions[l_i]
        t_i = t_vals[l_i]
        emb_idx = _cpu_embedding_lookup(x, pi, D, t_i, n_i)
        # assert (emb_idx >= 0).all()
        # assert (emb_idx < t_i).all()
        # foo = emb_idx.flatten().view(-1, 1) * F + torch.arange(F, device=emb_idx.device).view(1, -1)
        # print(foo.flatten())
        e_i = seek_to_level_embeddings(e, l_i, L, T, D, N_min, N_max)
        e_i = e_i[emb_idx]
        # print(e_i.flatten())
        start = l_i * F
        end = start + F
        e_i = _cpu_interpolate(x, e_i, D, n_i)
        # print(e_i.flatten())
        o[..., start:end] = e_i
    return o


class HashEncoding(torch.autograd.Function):

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
    @torch.no_grad()
    @staticmethod
    def forward(
        ctx,
        coords: Tensor,
        embeddings: Tensor,
        pi: Tensor | None,
        features: Tensor | None,
        max_entries_per_level: int,
        min_res: int,
        max_res: int,
        levels: int,
    ) -> Tensor:
        # Establish dimensions
        D_in = coords.shape[-1]
        D_feature = features.shape[-1] if features is not None else 0
        D_embed = embeddings.shape[-1]
        L = math.prod(coords.shape[:-1])

        # Validate coords and ensure float32
        assert coords.is_floating_point(), "Coords must be float32"
        coords = coords.float()

        # Create pi if not provided
        if pi is None:
            if D_in > 3:
                raise ValueError("Pi must be provided for more than 3 dimensions")
            pi = coords.new_tensor([PI_1, PI_2, PI_3], dtype=torch.int64)

        # Validate pi
        assert isinstance(pi, Tensor)
        assert pi.ndim == 1, "Pi must be a 1-dimensional tensor"
        assert pi.numel() >= D_in, "Pi must have at least D_in elements"

        # Validate features if provided
        if features is not None:
            assert isinstance(features, Tensor)
            assert (
                features.shape[:-1] == coords.shape[:-1]
            ), "Features must have the same shape as coords, except for the last dimension"

        # Validate gradient state (only embeddings receive grads)
        assert not coords.requires_grad, "Gradients not supported for coords"
        assert features is None or not features.requires_grad, "Gradients not supported for features"
        assert not pi.requires_grad, "Gradients not supported for pi"

        # Validate device
        devices = {coords.device, embeddings.device, pi.device, features.device if features is not None else None}
        devices = {d for d in devices if d is not None}
        assert len(devices) == 1 and all(
            d.type == "cuda" for d in devices
        ), f"All tensors must be on the same device: {devices}"

        # Initialize output buffer
        D_out = D_embed * levels + D_feature
        out_shape = coords.shape[:-1] + (D_out,)
        o = embeddings.new_empty(*out_shape)

        def grid(META):
            return (triton.cdiv(L, META["BLOCK_N"]),)

        _fwd_kernel[grid](  # type: ignore
            # fmt: off
            coords, pi, embeddings, o,
            coords.stride(-2), coords.stride(-1),
            embeddings.stride(-2),
            o.stride(-2), o.stride(-1),
            L, D_in, D_embed,
            max_entries_per_level, levels, min_res, max_res,
            # fmt: on
        )

        # Copy features to output if provided
        if features is not None:
            o[..., D_embed * levels :] = features

        ctx.save_for_backward(coords, embeddings, pi)
        ctx.max_entries_per_level = max_entries_per_level
        ctx.min_res = min_res
        ctx.max_res = max_res
        ctx.levels = levels

        return o

    @torch.cuda.amp.custom_bwd
    @torch.no_grad()
    @staticmethod
    def backward(ctx, do: Tensor):
        coords, embeddings, pi = ctx.saved_tensors

        D_in = coords.shape[-1]
        D_embed = embeddings.shape[-1]
        L = math.prod(coords.shape[:-1])

        de = torch.zeros_like(embeddings)

        def grid(META):
            return (triton.cdiv(L, META["BLOCK_N"]),)

        _bwd_kernel[grid](  # type: ignore
            # fmt: off
            coords, pi, do, de,
            coords.stride(-2), coords.stride(-1),
            do.stride(-2), do.stride(-1),
            de.stride(-2),
            L, D_in, D_embed,
            ctx.max_entries_per_level, ctx.levels, ctx.min_res, ctx.max_res,
            # fmt: on
        )

        return None, de, None, None, None, None, None, None


def hash_encoding(
    coord: Tensor,
    embeddings: Tensor,
    features: Tensor | None = None,
    pi: Tensor | None = None,
    max_entries_per_level: int = 2**14,
    min_res: int = 16,
    max_res: int = 512,
    levels: int = 16,
) -> Tensor:
    return HashEncoding.apply(
        coord,
        embeddings,
        pi,
        features,
        max_entries_per_level,
        min_res,
        max_res,
        levels,
    )
