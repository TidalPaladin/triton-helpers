import math
from functools import partial
from typing import Any, Dict, Final, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor

from triton_helpers.heuristics import PowerOfTwoHeuristic, SelectHeuristic, SMHeuristic
from triton_helpers.ops import high_low_mod, to_tensor

from .helpers import (
    compute_b,
    compute_embedding_counts,
    compute_level_embedding_offset,
    compute_resolutions,
    get_first_hash_level,
    seek_to_level_embeddings,
)


# Three primes chosen as in the paper, one prime per spatial dimension
PI_1: Final = 1
PI_2: Final = 2_654_435_761
PI_3: Final = 805_459_861

RTX_3090_CACHE_SIZE_MB: Final = 6.144_000


@triton.jit
def prod(x: tl.tensor, y: tl.tensor) -> tl.tensor:
    return x * y


@triton.jit
def create_corner_offsets(BLOCK: tl.constexpr) -> tl.tensor:
    r"""Creates binary offsets to the 2**BLOCK corners of a hypercube.

    Shape:
        (2**BLOCK, BLOCK)
    """
    tl.static_assert(BLOCK <= 32, "BLOCK must be <= 32")
    result = (tl.arange(0, 2**BLOCK)[:, None] >> tl.arange(0, BLOCK)[None, :]) & 1
    if BLOCK <= 8:
        result = result.to(tl.uint8)
    elif BLOCK <= 16:
        result = result.to(tl.uint16)
    return result


@triton.jit
def get_interpolation_weights(x: tl.tensor, D: tl.constexpr, BLOCK: tl.constexpr) -> tl.tensor:
    r"""Gets interpolation weights for a point in a hypercube.

    Shape:
        (N, 2**BLOCK)
    """
    # Compute interpolation weights (N, 2**D, 1)
    w = tl.math.fmod(x, 1.0)[:, None, :]
    corner_offsets = create_corner_offsets(BLOCK)[None, :, :]
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
    # This is marginally faster to cast w to e.dtype before multiply/sum, but
    # it's not worth the loss of precision.
    return tl.sum(w * e, axis=1).to(e.dtype)


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
    T_POW_2: tl.constexpr = False,
    # Working dtype
    DTYPE: tl.constexpr = tl.uint32,
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
    tl.static_assert(DTYPE is tl.uint32, f"Hash dtype must be uint32, got {DTYPE}")
    tl.device_assert((x >= 0), "x must be non-negative")
    tl.device_assert((x <= N_l), "x must be less than N_l")
    tl.device_assert((T_l < 2**32), "T_l must be less than 2**32")
    tl.device_assert((N_l < 2**32), "N_l must be less than 2**32")
    tl.device_assert(
        (tl.math.pow((N_l + 1.0).to(tl.float64), D) > T_l) == NEEDS_HASH,
        f"Hashing condition set incorrectly, hash={NEEDS_HASH}",
    )

    # Round x down by casting to uint type
    x = x.to(DTYPE)

    # Map x to 2**D vertices for each corner
    corners = x[:, None, :] + (create_corner_offsets(BLOCK_D) & (2**D - 1)).to(DTYPE)

    # At coarse resolution hashing isn't needed, mapping is 1:1
    if not NEEDS_HASH:  # type: ignore
        # Scale dimension D_i by (N_l + 1) ** i (this is basically a stride)
        scale = tl.math.pow(N_l, tl.arange(0, BLOCK_D).to(tl.float32))
        scale = tl.where(tl.arange(0, BLOCK_D) < D, scale, 0)
        # Float precision can be an issue - round instead of truncating
        scale = tl.math.float2uint_rn(scale)
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


@triton.heuristics(
    values={
        "BLOCK_D": PowerOfTwoHeuristic("D"),
        "BLOCK_F": PowerOfTwoHeuristic("F"),
        "BLOCK_N": SelectHeuristic(
            lambda args: (args["END_L"] - args["START_L"]) < 16,
            SMHeuristic("x_p", "N", max_size=512),
            SMHeuristic("x_p", "N", max_size=256),
        ),
        "num_warps": lambda args: 16 if args["BLOCK_N"] == 512 else 8 if args["BLOCK_N"] >= 128 else 4,
        "HASH_LEVEL": lambda args: get_first_hash_level(
            args["MIN_RES"], args["MAX_RES"], args["L"], args["T"], args["D"]
        ),
        "B": lambda args: compute_b(args["MIN_RES"], args["MAX_RES"], args["L"]),
        "E_START": lambda args: compute_level_embedding_offset(
            args.get("START_L", 0), args["L"], args["T"], args["D"], args["MIN_RES"], args["MAX_RES"]
        ),
        "SYNCHRONIZE": lambda _: True,
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
    # Operating range and optimization
    START_L: tl.constexpr, END_L: tl.constexpr, SYNCHRONIZE: tl.constexpr,
    # Block sizes
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_F: tl.constexpr,
    # Derived
    HASH_LEVEL: tl.constexpr, B: tl.constexpr, E_START: tl.constexpr,
    SCALE: tl.constexpr = 1.0,
    # fmt: on
):
    # Input validation and constant setup
    T_POW_2: tl.constexpr = T & (T - 1) == 0
    INT_DTYPE: tl.constexpr = tl.uint32
    tl.static_assert(B > 1, f"B must be greater than 1, got B={B}")
    tl.static_assert(START_L < L, f"START_L={START_L} must be less than L={L}")
    tl.static_assert(START_L < END_L, f"START_L={START_L} must be less than END_L={END_L}")
    LOOP_SIZE: tl.constexpr = END_L - START_L if END_L < L else L - START_L
    ANY_HASH: tl.constexpr = HASH_LEVEL < START_L + LOOP_SIZE
    tl.static_assert(START_L + LOOP_SIZE <= L, "Loop size exceeds L")
    tl.static_assert(0 <= E_START, "E_START must be non-negative")
    tl.static_assert(LOOP_SIZE & (LOOP_SIZE - 1) == 0, "Loop size must be a power of 2")

    # Set pointers to this program's start
    start = tl.program_id(0) * BLOCK_N
    # Seek embedding pointer to match the starting level
    e_p += E_START * stride_e_t

    # Load input and apply initial scale
    X_block_ptr = tl.make_block_ptr(
        x_p,
        (N, D),
        (stride_x_n, stride_x_d),
        (start, 0),
        (BLOCK_N, BLOCK_D),
        (1, 0),
    )
    # NOTE: Representation quality is higher with FP32 coordinates, expecially at high
    # resolution. Computation is faster with FP32, probably due to casting required for
    # various constituent operations.
    x = tl.load(X_block_ptr, boundary_check=(0, 1)).to(tl.float32)
    tl.device_assert(x >= 0, "x must be non-negative")
    tl.device_assert(x <= SCALE, f"x must be less than SCALE={SCALE}")
    INITIAL_SCALE: tl.constexpr = tl.constexpr(int(MIN_RES * B**START_L)) / SCALE
    x *= INITIAL_SCALE

    # Load pi if it will be needed
    if ANY_HASH:
        offsets = tl.arange(0, BLOCK_D)
        mask = offsets < D
        pi = tl.load(pi_p + offsets, mask=mask, eviction_policy="evict_last").to(INT_DTYPE)
    else:
        pi = tl.arange(0, BLOCK_D).to(INT_DTYPE)

    # Iterate over hash table levels
    # NOTE: It is empirically faster to run this as a static range, accumulate into a single buffer,
    # and then do one write at the end. The expectation is that this table will be used with a fully
    # fused MLP, in which case L * F should be small (typically 32 for L=16, F=2) and fit fully in SRAM.
    o = tl.zeros((BLOCK_N, LOOP_SIZE, BLOCK_F), dtype=o_p.dtype.element_ty)
    for l in tl.static_range(START_L, START_L + LOOP_SIZE):
        # Resolution and number of array entries for this level
        N_l = tl.constexpr(int(MIN_RES * B**l))
        T_l = tl.constexpr(min((tl.constexpr(int(MIN_RES * B**l)) + 1) ** D, T))
        N_l = N_l.to(tl.uint32)
        T_l = T_l.to(tl.uint32)

        # Scale x for this level
        if l > START_L:
            N_l_prev = tl.constexpr(int(MIN_RES * B ** (l - 1)))
            scale = N_l.to(x.dtype) / N_l_prev.to(x.dtype)
            x *= scale.to(x.dtype)

        # Look up embeddings
        if l >= HASH_LEVEL:
            tl.device_assert(T_l >= T, "T_l should be >= T when hashing")
            embedding_idx = embedding_lookup(x, pi, D, T_l, N_l, BLOCK_D, True, T_POW_2, INT_DTYPE)
            embedding_idx = (embedding_idx * F)[:, :, None] + tl.arange(0, BLOCK_F).to(tl.uint8)[None, None, :]
        else:
            tl.device_assert(T_l < T, "T_l should be < T when not hashing")
            embedding_idx = embedding_lookup(x, pi, D, T_l, N_l, BLOCK_D, False, T_POW_2, INT_DTYPE)
            embedding_idx = (embedding_idx * F)[:, :, None] + tl.arange(0, BLOCK_F).to(tl.uint8)[None, None, :]
        tl.static_assert(tl.constexpr(embedding_idx.dtype) == INT_DTYPE, f"Embedding index must be {INT_DTYPE}")

        # Load embeddings
        # NOTE: Since this is an uncoalesced load it helps to apply masking to the pointers as well as the load result.
        # This keeps us from polluting the cache with values that will be discarded.
        emb_mask = (
            (tl.arange(0, BLOCK_N) < N)[:, None, None]
            & (tl.arange(0, 2**BLOCK_D) < 2**D)[None, :, None]
            & (tl.arange(0, BLOCK_F) < F)[None, None, :]
        )
        e = tl.load(e_p + embedding_idx, mask=emb_mask, eviction_policy="evict_last")

        # Interpolate embeddings
        e = interpolate(x, e, D, BLOCK_D)[:, None, :]
        tl.static_assert(
            tl.constexpr(e.dtype) == tl.constexpr(e_p.dtype.element_ty), "Embedding dtype should match output"
        )

        # Update output
        mask = tl.arange(0, LOOP_SIZE) == l - START_L
        o = tl.where(mask[None, :, None], e, o)

        # Advance pointers
        e_p += T_l * stride_e_t

        # Synchronize workers at this level to ensure a high cache hit rate.
        if SYNCHRONIZE:
            tl.debug_barrier()

    start = tl.program_id(0) * BLOCK_N
    o = tl.reshape(o, (BLOCK_N, LOOP_SIZE * BLOCK_F))
    O_block_ptr = tl.make_block_ptr(
        o_p,
        (N, L * F),
        (stride_o_n, stride_o_f),
        (start, F * START_L),
        (BLOCK_N, LOOP_SIZE * BLOCK_F),
        (1, 0),
    )
    tl.store(O_block_ptr, o, boundary_check=(0, 1), eviction_policy="evict_first")


@triton.heuristics(
    values={
        "BLOCK_D": PowerOfTwoHeuristic("D"),
        "BLOCK_F": PowerOfTwoHeuristic("F"),
        "BLOCK_N": SMHeuristic("x_p", "N", max_size=512),
        "num_warps": lambda args: 16 if args["BLOCK_N"] == 512 else 8 if args["BLOCK_N"] >= 128 else 4,
        "HASH_LEVEL": lambda args: get_first_hash_level(
            args["MIN_RES"], args["MAX_RES"], args["L"], args["T"], args["D"]
        ),
        "B": lambda args: compute_b(args["MIN_RES"], args["MAX_RES"], args["L"]),
        "SYNCHRONIZE": lambda _: False,
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
    # Operating range and optimization
    START_L: tl.constexpr, END_L: tl.constexpr,
    # Block sizes
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_F: tl.constexpr,
    # Derived
    HASH_LEVEL: tl.constexpr, B: tl.constexpr,
    SYNCHRONIZE: tl.constexpr,
    SCALE: tl.constexpr = 1.0,
    # fmt: on
):
    # Input validation and constant setup
    T_POW_2: tl.constexpr = T & (T - 1) == 0
    INT_DTYPE: tl.constexpr = tl.uint32
    tl.static_assert(B > 1, f"B must be greater than 1, got B={B}")
    tl.static_assert(START_L < L, f"START_L={START_L} must be less than L={L}")
    tl.static_assert(START_L < END_L, f"START_L={START_L} must be less than END_L={END_L}")
    LOOP_SIZE: tl.constexpr = END_L - START_L if END_L < L else L - START_L
    ANY_HASH: tl.constexpr = HASH_LEVEL < START_L + LOOP_SIZE
    tl.static_assert(START_L + LOOP_SIZE <= L, "Loop size exceeds L")
    tl.static_assert(LOOP_SIZE & (LOOP_SIZE - 1) == 0, "Loop size must be a power of 2")

    # Set pointers to this program's start
    start = tl.program_id(0) * BLOCK_N

    # Load input and apply initial scale
    X_block_ptr = tl.make_block_ptr(
        x_p,
        (N, D),
        (stride_x_n, stride_x_d),
        (start, 0),
        (BLOCK_N, BLOCK_D),
        (1, 0),
    )
    x = tl.load(X_block_ptr, boundary_check=(0, 1)).to(tl.float32)
    tl.device_assert(x >= 0, "x must be non-negative")
    tl.device_assert(x <= SCALE, f"x must be less than SCALE={SCALE}")
    INITIAL_SCALE: tl.constexpr = tl.constexpr(int(MIN_RES * B**START_L)) / SCALE
    x *= to_tensor(INITIAL_SCALE, x.dtype)

    # Load pi if it will be needed
    if ANY_HASH:
        offsets = tl.arange(0, BLOCK_D)
        mask = offsets < D
        pi = tl.load(pi_p + offsets, mask=mask, eviction_policy="evict_last").to(INT_DTYPE)
    else:
        pi = tl.arange(0, BLOCK_D).to(INT_DTYPE)

    # Initialize DO pointer
    DO_block_ptr = tl.make_block_ptr(
        do_p,
        (N, F * L),
        (stride_do_n, stride_do_f),
        (start, F * START_L),
        (BLOCK_N, BLOCK_F),
        (1, 0),
    )

    # Iterate over hash table levels
    for l in tl.static_range(START_L, START_L + LOOP_SIZE):
        # Resolution and number of array entries for this level
        N_l = tl.constexpr(int(MIN_RES * B**l))
        T_l = tl.constexpr(min((tl.constexpr(int(MIN_RES * B**l)) + 1) ** D, T))
        N_l = N_l.to(tl.uint32)
        T_l = T_l.to(tl.uint32)

        # Scale x for this level
        if l > START_L:
            N_l_prev = tl.constexpr(int(MIN_RES * B ** (l - 1)))
            scale = N_l.to(x.dtype) / N_l_prev.to(x.dtype)
            x *= scale.to(x.dtype)

        # Look up embeddings
        if l >= HASH_LEVEL:
            tl.device_assert(T_l >= T, "T_l should be >= T when hashing")
            embedding_idx = embedding_lookup(x, pi, D, T_l, N_l, BLOCK_D, True, T_POW_2, INT_DTYPE)
            embedding_idx = (embedding_idx * F)[:, :, None] + tl.arange(0, BLOCK_F).to(tl.uint8)[None, None, :]
        else:
            tl.device_assert(T_l < T, "T_l should be < T when not hashing")
            embedding_idx = embedding_lookup(x, pi, D, T_l, N_l, BLOCK_D, False, T_POW_2, INT_DTYPE)
            embedding_idx = (embedding_idx * F)[:, :, None] + tl.arange(0, BLOCK_F).to(tl.uint8)[None, None, :]
        tl.static_assert(tl.constexpr(embedding_idx.dtype) == INT_DTYPE, f"Embedding index must be {INT_DTYPE}")

        # Get interpolation weights (N, 2**D)
        w = get_interpolation_weights(x, D, BLOCK_D)[:, :, None]

        # Load do for this level (N, F)
        do = tl.load(DO_block_ptr, boundary_check=(0, 1))

        # Compute de (N, 2**D, F)
        de = do[:, None, :] * w.to(do.dtype)

        # Store
        mask = (
            (tl.arange(0, BLOCK_N) < N)[:, None, None]
            & (tl.arange(0, 2**BLOCK_D) < 2**D)[None, :, None]
            & (tl.arange(0, BLOCK_F) < F)[None, None, :]
        )
        tl.atomic_add(de_p + embedding_idx, de, mask=mask)

        # Advance pointers
        if l < START_L + LOOP_SIZE - 1:
            DO_block_ptr = tl.advance(DO_block_ptr, (0, BLOCK_F))
            de_p += T_l * stride_de_t

        if SYNCHRONIZE:
            tl.debug_barrier()


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
    for l_i in range(L):
        n_i = resolutions[l_i]
        t_i = t_vals[l_i]
        emb_idx = _cpu_embedding_lookup(x, pi, D, t_i, n_i)
        e_i = seek_to_level_embeddings(e, l_i, L, T, D, N_min, N_max)
        e_i = e_i[emb_idx]
        start = l_i * F
        end = start + F
        e_i = _cpu_interpolate(x, e_i, D, n_i)
        o[..., start:end] = e_i
    return o


DIVIDER_CONFIGS: Dict[Any, int] = {}
BWD_DIVIDER_CONFIGS: Dict[Any, int] = {}


class HashEncoding(torch.autograd.Function):

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
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
        divider: int | None = None,
        bwd_divider: int | None = None,
        scale: float = 1.0,
    ) -> Tensor:
        # Establish dimensions
        D_in = coords.shape[-1]
        D_feature = features.shape[-1] if features is not None else 0
        D_embed = embeddings.shape[-1]
        L = math.prod(coords.shape[:-1])

        # Validate coords and ensure float32
        assert coords.is_floating_point(), "Coords must be float"

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
        assert levels & (levels - 1) == 0, "Levels must be a power of 2"
        D_out = D_embed * levels + D_feature
        out_shape = coords.shape[:-1] + (D_out,)
        o = embeddings.new_empty(*out_shape, requires_grad=embeddings.requires_grad)

        def grid(META):
            return (triton.cdiv(L, META["BLOCK_N"]),)

        def _launch(divider: int):
            assert 1 <= divider <= levels, f"Divider must be between 1 and {levels}"
            assert divider & (divider - 1) == 0, "Divider must be a power of 2"
            for i in range(levels // divider):
                _fwd_kernel[grid](  # type: ignore
                    # fmt: off
                    coords, pi, embeddings, o,
                    coords.stride(-2), coords.stride(-1),
                    embeddings.stride(-2),
                    o.stride(-2), o.stride(-1),
                    L, D_in, D_embed,
                    max_entries_per_level, levels, min_res, max_res,
                    i*divider, (i+1)*divider,
                    SCALE=scale,
                    # fmt: on
                )

        # Determine how many chunks to process the various levels in.
        # This is mostly a function of the hardware's L2 cache and memory usage.
        # If we can maintain a high L2 hit rate for uncoalesced embedding reads,
        # we can run the kernel in one pass. Otherwise, it is optimal to break it up into multiple passes.
        # The number of levels in a pass must be a power of 2.
        #
        # It seems like the only way to reliably choose a good divider is with autotuning.
        # Here we choose a divider if one is not already provided
        if divider is None:
            key = (L, D_in, D_embed, max_entries_per_level, levels, min_res, coords.dtype, embeddings.dtype)
            if key in DIVIDER_CONFIGS:
                divider = DIVIDER_CONFIGS[key]
            else:
                dividers = HashEncoding._possible_dividers(levels)
                runtimes = [triton.testing.do_bench(partial(_launch, divider)) for divider in dividers]
                divider = DIVIDER_CONFIGS[key] = dividers[runtimes.index(min(runtimes))]
        ctx.bwd_divider = bwd_divider

        # Launch kernel and work through the levels
        _launch(divider)

        # Copy features to output if provided
        if features is not None:
            o[..., D_embed * levels :] = features

        ctx.save_for_backward(coords, embeddings, pi)
        ctx.max_entries_per_level = max_entries_per_level
        ctx.min_res = min_res
        ctx.max_res = max_res
        ctx.levels = levels
        ctx.scale = scale

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

        def _launch(divider: int):
            assert 1 <= divider <= ctx.levels, f"Divider must be between 1 and {ctx.levels}"
            assert divider & (divider - 1) == 0, "Divider must be a power of 2"
            for i in range(ctx.levels // divider):
                _bwd_kernel[grid](  # type: ignore
                    # fmt: off
                    coords, pi, do, de,
                    coords.stride(-2), coords.stride(-1),
                    do.stride(-2), do.stride(-1),
                    de.stride(-2),
                    L, D_in, D_embed,
                    ctx.max_entries_per_level, ctx.levels, ctx.min_res, ctx.max_res,
                    i*divider, (i+1)*divider,
                    SCALE=ctx.scale,
                    # fmt: on
                )

        divider = ctx.bwd_divider
        if divider is None:
            key = (L, D_in, D_embed, ctx.max_entries_per_level, ctx.levels, ctx.min_res, coords.dtype, embeddings.dtype)
            if key in BWD_DIVIDER_CONFIGS:
                divider = BWD_DIVIDER_CONFIGS[key]
            else:
                dividers = HashEncoding._possible_dividers(ctx.levels)
                runtimes = [triton.testing.do_bench(partial(_launch, divider)) for divider in dividers]
                divider = BWD_DIVIDER_CONFIGS[key] = dividers[runtimes.index(min(runtimes))]
                de.fill_(0)

        _launch(divider)

        return None, de, None, None, None, None, None, None, None, None, None

    @staticmethod
    def _possible_dividers(levels: int) -> Tuple[int, ...]:
        return tuple(2**i for i in range(int(math.log2(levels)) + 1))


def hash_encoding(
    coord: Tensor,
    embeddings: Tensor,
    features: Tensor | None = None,
    pi: Tensor | None = None,
    max_entries_per_level: int = 2**14,
    min_res: int = 16,
    max_res: int = 512,
    levels: int = 16,
    divider: int | None = None,
    bwd_divider: int | None = None,
    scale: float = 1.0,
) -> Tensor:
    r"""Implements a multi-resolution hash encoding as defined in Instant NGP.

    Args:
        coord: Coodinate inputs on the range :math:`[0, scale]`.
        embeddings: Embedding table.
        features: Optional features to concatenate to the output.
        pi: Hashing primes. Will be computed for up to ``D=3`` if not provided.
        max_entries_per_level: Maximum number of entries per hash table level.
        min_res: Minimum resolution.
        max_res: Maximum resolution.
        levels: Number of hash table levels.
        divider: Number of levels to process in a single pass. If not provided, will be autotuned.
        bwd_divider: Number of levels to process in a single pass during backward pass. If not provided, will be autotuned.
        scale: Scale factor for the input coordinates.

    Shapes:
        ``coord`` - :math:`(..., D)` where :math:`D` is the number of spatial dimensions.
        ``embeddings`` - :math:`(E, D_e)` where :math:`E` is the number of embeddings and :math:`D_e` is the embedding dimension.
        ``features`` - :math:`(..., D_f)` where :math:`D_f` is the number of features.
        ``pi`` - :math:`(D,)`
        Output - :math:`(..., L \times D_e + D_f)` where :math:`L` is the number of levels.

    Returns:
        Retrieved embeddings concatenated with provided features for each input coordinate.
    """
    return HashEncoding.apply(
        coord,
        embeddings,
        pi,
        features,
        max_entries_per_level,
        min_res,
        max_res,
        levels,
        divider,
        bwd_divider,
        scale,
    )
