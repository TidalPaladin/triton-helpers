import math
from typing import Final

import torch
import triton
import triton.language as tl
from torch import Tensor

from triton_helpers.heuristics import PowerOfTwoHeuristic
from triton_helpers.ops import to_tensor


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
    D: tl.constexpr,
    # Table params
    T_l: int, N_l: int,
    # Block sizes
    BLOCK_D: tl.constexpr,
    # Working dtype
    DTYPE: tl.constexpr = tl.constexpr(tl.uint64),
    # fmt: on
) -> tl.tensor:
    r"""Looks up embedding indices for the hybercube corners about a spatial coordinate.

    Shape:
        (N, 2**D)
    """
    # Scale x by this level's resolution and round down to get the lower corner
    if DTYPE is tl.uint64:
        x = tl.math.float2ull_rd(x * N_l)
    elif DTYPE is tl.int64:
        x = tl.math.float2ll_rd(x * N_l)
    else:
        tl.static_assert(False, f"Hash must be either tl.int64 or tl.uint64")

    # Map x to 2**D vertices for each corner
    corners = x[:, None, :] + create_corner_offsets(BLOCK_D).to(x.dtype)

    # At coarse resolution hashing isn't needed, mapping is 1:1
    if tl.math.pow((N_l + 1).to(tl.float32), D) <= T_l:  # type: ignore
        scale = tl.math.pow(
            tl.full((BLOCK_D,), N_l + 1, tl.float32),
            tl.arange(0, BLOCK_D),
        ).to(corners.dtype)
        h = tl.sum(corners * scale[None, None, :], axis=2)
    # Otherwise compute hash function
    else:
        h = tl.xor_sum(corners * pi[None, None, :].to(corners.dtype), axis=2) % to_tensor(T_l, corners.dtype)

    return h


@triton.heuristics(
    values={
        "BLOCK_N": PowerOfTwoHeuristic("N", max_val=64),
        "BLOCK_D": PowerOfTwoHeuristic("D"),
        "BLOCK_F": PowerOfTwoHeuristic("F"),
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
    # fmt: on
):
    # Set pointers to this program's start
    start = tl.program_id(0) * BLOCK_N
    x_p += start * stride_x_n
    o_p += start * stride_o_n

    # Load input
    X_block_ptr = tl.make_block_ptr(
        x_p,
        (N, D),
        (stride_x_n, stride_x_d),
        (0, 0),
        (BLOCK_N, BLOCK_D),
        (1, 0),
    )
    x = tl.load(X_block_ptr, boundary_check=(0, 1))

    # Load pi
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D
    pi = tl.load(pi_p + offsets, mask=mask, eviction_policy="evict_last")

    O_block_ptr = tl.make_block_ptr(
        o_p,
        (N, F * L),
        (stride_o_n, stride_o_f),
        (0, 0),
        (BLOCK_N, BLOCK_F),
        (1, 0),
    )

    # NOTE: For some reason the B calculation must be written just like this for it to compile
    B: tl.constexpr = math.exp((math.log(MAX_RES) - math.log(MIN_RES)) * (L.value - tl.constexpr(1).value))

    # Iterate over hash table levels
    for l in range(L):
        # Resolution and number of array entries for this level
        # N_l: tl.constexpr = int(MIN_RES * B ** l)
        # T_l: tl.constexpr = min((N_l + 1) ** D, T)
        N_l = tl.math.mul_rd(MIN_RES, tl.math.pow(B, l))
        T_l = tl.minimum(tl.math.pow((N_l + 1), D), T).to(tl.int64)

        # Look up embeddings
        embedding_idx = embedding_lookup(x, pi, D, T_l, N_l, BLOCK_D)
        embedding_idx = (embedding_idx * F)[:, :, None] + tl.arange(0, BLOCK_F)[None, None, :]
        emb_mask = tl.arange(0, BLOCK_F) < F
        e = tl.load(e_p + embedding_idx, mask=emb_mask[None, None, :])

        # Interpolate embeddings
        e = interpolate(x, e, D, BLOCK_D)

        # Write features to output
        tl.store(
            O_block_ptr,
            e,
            boundary_check=(
                0,
                1,
            ),
        )

        # Advance pointers
        O_block_ptr = tl.advance(O_block_ptr, (0, BLOCK_F))
        e_p += T_l * stride_e_t


@triton.heuristics(
    values={
        "BLOCK_N": PowerOfTwoHeuristic("N", max_val=64),
        "BLOCK_D": PowerOfTwoHeuristic("D"),
        "BLOCK_F": PowerOfTwoHeuristic("F"),
        "num_warps": lambda _: 1,
        "num_stages": lambda _: 1,
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
    # fmt: on
):
    # Set pointers to this program's start
    start = tl.program_id(0) * BLOCK_N
    x_p += start * stride_x_n
    do_p += start * stride_do_n

    # Load input
    X_block_ptr = tl.make_block_ptr(
        x_p,
        (N, D),
        (stride_x_n, stride_x_d),
        (0, 0),
        (BLOCK_N, BLOCK_D),
        (1, 0),
    )
    x = tl.load(X_block_ptr, boundary_check=(0, 1))

    # Load pi
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D
    pi = tl.load(pi_p + offsets, mask=mask, eviction_policy="evict_last")

    DO_block_ptr = tl.make_block_ptr(
        do_p,
        (N, F * L),
        (stride_do_n, stride_do_f),
        (0, 0),
        (BLOCK_N, BLOCK_F),
        (1, 0),
    )

    # NOTE: For some reason the B calculation must be written just like this for it to compile
    B: tl.constexpr = math.exp((math.log(MAX_RES) - math.log(MIN_RES)) * (L.value - tl.constexpr(1).value))

    # Iterate over hash table levels
    for l in range(L):
        # Resolution and number of array entries for this level
        N_l = tl.math.mul_rd(MIN_RES, tl.math.pow(B, l))
        T_l = tl.minimum(tl.math.pow((N_l + 1), D), T).to(tl.int64)

        # Look up embedding indices (N, 2**D, F)
        embedding_idx = embedding_lookup(x, pi, D, T_l, N_l, BLOCK_D)
        embedding_idx = (embedding_idx * F)[:, :, None] + tl.arange(0, BLOCK_F)[None, None, :]

        # Get interpolation weights (N, 2**D)
        w = get_interpolation_weights(x, D, BLOCK_D)[:, :, None]

        # Load do for this level (N, F)
        do = tl.load(DO_block_ptr)

        # Compute de (N, 2**D, F)
        de = do[:, None, :] * w

        # Store?
        mask_de = (
            (tl.arange(0, BLOCK_N) < N)[:, None, None]
            + (tl.arange(0, 2**BLOCK_D) < 2**D)[None, :, None]
            + (tl.arange(0, BLOCK_F) < F)[None, None, :]
        )
        tl.atomic_add(de_p + embedding_idx, de, mask=mask_de)

        # Advance pointers
        DO_block_ptr = tl.advance(DO_block_ptr, (0, BLOCK_F))
        de_p += T_l * stride_de_t


def compute_resolutions(num_levels: int, min_res: int, max_res: int) -> Tensor:
    b = torch.tensor(math.exp((math.log(max_res) - math.log(min_res)) / (num_levels - 1)))
    l = torch.arange(0, num_levels)
    return b.pow(l).mul(min_res).floor().long()


def compute_embedding_counts(num_levels: int, max_entries_per_level: int, min_res: int, max_res: int) -> Tensor:
    resolutions = compute_resolutions(num_levels, min_res, max_res)
    t = (resolutions + 1) ** 2
    return torch.min(t, t.new_tensor(max_entries_per_level))


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
