import math

import pytest
import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.testing import assert_close

from triton_helpers.layers.hash_encoding import HashEncoding, hash_encoding
from triton_helpers.layers.hash_encoding.kernel import (
    PI_1,
    PI_2,
    PI_3,
    _cpu_embedding_lookup,
    _cpu_hash_encoding,
    _cpu_interpolate,
    compute_embedding_counts,
    create_corner_offsets,
    embedding_lookup,
    interpolate,
)


@pytest.fixture(params=[torch.float32, torch.float16])
def float_dtype(request):
    return request.param


@pytest.fixture(scope="module")
def corner_factory():
    def func(d, **kwargs):
        kwargs.setdefault("dtype", torch.int32)
        match d:
            # Specify these two cases manually
            case 2:
                return torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], **kwargs)
            case 3:
                return torch.tensor(
                    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], **kwargs
                )
            # Generic case
            case _:
                result = (
                    torch.arange(2**d, dtype=torch.int32).view(-1, 1) >> torch.arange(d, dtype=torch.int32).view(1, -1)
                ) & 1
                return result.to(**kwargs)

    return func


@pytest.mark.cuda
class TestCreateCornerOffsets:

    @pytest.fixture
    def kernel(self):
        @triton.jit
        def kernel(o_p, D: tl.constexpr, BLOCK_D: tl.constexpr):
            corner_offsets = create_corner_offsets(BLOCK_D)
            Ptr = tl.make_block_ptr(o_p, (2**D, D), (D, 1), (0, 0), (2**BLOCK_D, BLOCK_D), (1, 0))
            tl.store(Ptr, corner_offsets, boundary_check=(0, 1))

        return kernel

    @pytest.mark.parametrize("d", [2, 3])
    def test_offsets(self, kernel, corner_factory, d):
        exp = corner_factory(d, dtype=torch.int32, device="cuda")
        BLOCK = triton.next_power_of_2(d)
        o = torch.empty(2**d, d, dtype=torch.int32, device="cuda")
        kernel[(1,)](o, d, BLOCK)
        assert_close(o, exp, check_device=False, check_dtype=False)


@pytest.mark.cuda
class TestInterpolate:

    @pytest.fixture
    def kernel(self):
        @triton.jit
        def kernel(
            # fmt: off
            x_p, e_p, o_p, 
            stride_x_l: int, stride_e_l: int, stride_o_l: int,
            L: tl.constexpr, D: tl.constexpr, F: tl.constexpr,
            BLOCK_L: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_F: tl.constexpr,
            # fmt: on
        ):
            start = tl.program_id(0) * BLOCK_L
            x_p += start * stride_x_l
            e_p += start * stride_e_l
            o_p += start * stride_o_l

            X_ptr = tl.make_block_ptr(x_p, (L, D), (D, 1), (0, 0), (BLOCK_L, BLOCK_D), (1, 0))
            E_ptr = tl.make_block_ptr(
                e_p, (L, 2**D, F), (2**D * F, F, 1), (0, 0, 0), (BLOCK_L, 2**BLOCK_D, BLOCK_F), (2, 1, 0)
            )
            x = tl.load(X_ptr, boundary_check=(0, 1))
            e = tl.load(E_ptr, boundary_check=(0, 1, 2))
            o = interpolate(x, e, D, BLOCK_D)
            O_ptr = tl.make_block_ptr(o_p, (L, F), (F, 1), (0, 0), (BLOCK_L, BLOCK_F), (1, 0))
            tl.store(O_ptr, o, boundary_check=(0, 1))

        return kernel

    @pytest.mark.parametrize(
        "coord, idx",
        [
            ((0, 0, 0), 0),
            ((1, 1, 1), 0),
            # NOTE: Must be just under 1, otherwise we're in next hypercube
            ((0.9999, 0.9999, 0.9999), -1),
            ((0.9999, 0.9999, 0), 3),
        ],
    )
    def test_corners(self, kernel, float_dtype, coord, idx):
        torch.random.manual_seed(0)
        L, D, F = 10, 3, 2
        x = torch.tensor(coord, dtype=torch.float32, device="cuda").repeat(L, 1)
        e = torch.randn(L, 2**D, F, dtype=float_dtype, device="cuda")
        o = torch.empty(L, F, dtype=float_dtype, device="cuda")

        baseline = e[:, idx, :]

        # Triton
        BLOCK_L = triton.next_power_of_2(L)
        BLOCK_D = triton.next_power_of_2(D)
        BLOCK_F = triton.next_power_of_2(F)
        kernel[(1,)](x, e, o, x.stride(0), e.stride(0), o.stride(0), L, D, F, BLOCK_L, BLOCK_D, BLOCK_F)

        assert_close(o, baseline, rtol=0, atol=1e-3)

    @pytest.mark.cuda
    def test_mean(self, kernel):
        torch.random.manual_seed(0)
        L, D, F = 10, 3, 2
        x = torch.full((L, D), 0.5, device="cuda")
        e = torch.randn(L, 2**D, F, device="cuda")
        o = torch.empty(L, F, device="cuda")

        # Because we set x=0.5, the interpolation should be the mean of the corners
        baseline = e.mean(1)

        # Triton
        BLOCK_L = triton.next_power_of_2(L)
        BLOCK_D = triton.next_power_of_2(D)
        BLOCK_F = triton.next_power_of_2(F)
        kernel[(1,)](x, e, o, x.stride(0), e.stride(0), o.stride(0), L, D, F, BLOCK_L, BLOCK_D, BLOCK_F)

        assert_close(o, baseline)

    @pytest.mark.cuda
    def test_torch(self, kernel):
        torch.random.manual_seed(0)
        L, D, F = 40, 3, 2
        x = torch.rand(L, D, device="cuda")
        e = torch.randn(L, 2**D, F, device="cuda")
        o = torch.empty(L, F, device="cuda")

        # Because we set x=0.5, the interpolation should be the mean of the corners
        baseline = _cpu_interpolate(x, e, D, 1)

        # Triton
        BLOCK_L = triton.next_power_of_2(L)
        BLOCK_D = triton.next_power_of_2(D)
        BLOCK_F = triton.next_power_of_2(F)
        kernel[(1,)](x, e, o, x.stride(0), e.stride(0), o.stride(0), L, D, F, BLOCK_L, BLOCK_D, BLOCK_F)

        assert_close(o, baseline)


@pytest.mark.cuda
class TestEmbeddingLookup:

    @pytest.fixture
    def kernel(self):
        @triton.heuristics({"num_warps": lambda _: 1})
        @triton.jit
        def kernel(
            # fmt: off
            x_p, pi_p, o_p,
            stride_x_l, stride_o_l,
            L: tl.constexpr, D: tl.constexpr,
            T_l, N_l,
            BLOCK_D: tl.constexpr, BLOCK_L: tl.constexpr, NEEDS_HASH: tl.constexpr,
            T_POW_2: tl.constexpr = False,
            # fmt: on
        ):
            start = tl.program_id(0) * BLOCK_L
            x_p += start * stride_x_l
            o_p += start * stride_o_l

            # Load x
            X_ptr = tl.make_block_ptr(x_p, (L, D), (D, 1), (0, 0), (BLOCK_L, BLOCK_D), (1, 0))
            x = tl.load(X_ptr, boundary_check=(0, 1))

            # Load pi
            offset_pi = tl.arange(0, BLOCK_D)
            mask_pi = offset_pi < D
            pi = tl.load(pi_p + offset_pi, mask=mask_pi).to(tl.uint32)

            # Hash
            x = x * N_l
            o = embedding_lookup(x, pi, D, T_l.to(tl.uint32), N_l.to(tl.uint32), BLOCK_D, NEEDS_HASH, T_POW_2).to(
                o_p.dtype.element_ty
            )

            # Store
            O_ptr = tl.make_block_ptr(o_p, (L, 2**D), (2**D, 1), (0, 0), (BLOCK_L, 2**BLOCK_D), (1, 0))
            tl.store(O_ptr, o, boundary_check=(0, 1))

        return kernel

    def test_quadrants_2d_unique(self, kernel):
        torch.random.manual_seed(0)
        pi = torch.tensor([PI_1, PI_2, PI_3], device="cuda", dtype=torch.int64)

        # Inputs
        x = torch.tensor(
            [
                [0, 0],  # Q1
                [0.55, 0.55],  # Q4
                [0.45, 0.45],  # Q1
                [0.9999, 0.9999],  # Q4
                [0.55, 0.45],  # Q2
                [0.45, 0.55],  # Q3
            ],
            dtype=torch.float32,
            device="cuda",
        )
        L, D = x.shape
        N_level = 2
        # Condition of T for results to be unique
        T = (N_level + 1) ** D

        # Baseline
        baseline = torch.tensor(
            [
                [0, 1, 2, 3],
                [3, 4, 5, 6],
                [0, 1, 2, 3],
                [3, 4, 5, 6],
                [1, 2, 3, 4],
                [2, 3, 4, 5],
            ],
            dtype=torch.int64,
            device="cuda",
        )

        # Triton
        o = torch.zeros(L, 2**D, dtype=torch.int64, device="cuda")
        BLOCK_D = triton.next_power_of_2(D)
        BLOCK_L = triton.next_power_of_2(L)
        NEEDS_HASH = False
        kernel[(1,)](x, pi, o, x.stride(0), o.stride(0), L, D, T, N_level, BLOCK_D, BLOCK_L, NEEDS_HASH)

        assert (o == baseline).all()

    def test_quadrants_2d_nonunique(self, kernel):
        torch.random.manual_seed(0)
        pi = torch.tensor([PI_1, PI_2, PI_3], device="cuda", dtype=torch.int64)

        # Inputs
        x = torch.tensor(
            [
                [0, 0],  # Q1
                [0.55, 0.55],  # Q4
                [0.45, 0.45],  # Q1
                [0.9999, 0.9999],  # Q4
                [0.55, 0.45],  # Q2
                [0.45, 0.55],  # Q3
            ],
            dtype=torch.float32,
            device="cuda",
        )
        L, D = x.shape
        N_level = 2
        # Condition of T for results to not be unique
        T = (N_level + 1) ** D - 2

        # Baseline
        baseline = torch.tensor(
            [
                [0, 1, 5, 4],
                [4, 0, 4, 1],
                [0, 1, 5, 4],
                [4, 0, 4, 1],
                [1, 2, 4, 0],
                [5, 4, 3, 4],
            ],
            dtype=torch.int64,
            device="cuda",
        )

        # Triton
        o = torch.zeros(L, 2**D, dtype=torch.int64, device="cuda")
        BLOCK_D = triton.next_power_of_2(D)
        BLOCK_L = triton.next_power_of_2(L)
        NEEDS_HASH = True
        kernel[(1,)](x, pi, o, x.stride(0), o.stride(0), L, D, T, N_level, BLOCK_D, BLOCK_L, NEEDS_HASH)

        assert (o == baseline).all()

    @pytest.mark.parametrize("unique", [False, True])
    @pytest.mark.parametrize("power_2", [False, True])
    def test_torch_baseline(self, kernel, corner_factory, unique, power_2):
        torch.random.manual_seed(0)
        L, D, N_level = 4, 3, 8
        # Condition of T for results to not be unique
        T = (N_level + 1) ** D - (2 if not unique else 0)

        if power_2:
            T = triton.next_power_of_2(T) if unique else triton.next_power_of_2(T // 2)
            assert math.log2(T).is_integer()

        # Inputs
        x = torch.rand(L, D, device="cuda")
        pi = torch.tensor([PI_1, PI_2, PI_3], device="cuda", dtype=torch.int64)[:D]

        # Baseline
        baseline = _cpu_embedding_lookup(x, pi, D, T, N_level)

        # Triton
        o = torch.zeros(L, 2**D, dtype=torch.int64, device="cuda")
        BLOCK_D = triton.next_power_of_2(D)
        BLOCK_L = triton.next_power_of_2(L)
        NEEDS_HASH = not unique
        kernel[(1,)](
            x, pi, o, x.stride(0), o.stride(0), L, D, T, N_level, BLOCK_D, BLOCK_L, NEEDS_HASH, T_POW_2=power_2
        )

        assert_close(o, baseline, check_device=False, check_dtype=False)

    @pytest.mark.parametrize(
        "D, N_level, T",
        [
            (2, 16, (16 + 1) ** 2),
            (3, 16, (16 + 1) ** 2),
            (3, 16, (16 + 1) ** 2 - 2),
            (3, 128, (128 + 1) ** 2),
            (3, 128, (128 + 1) ** 2 - 1),
            (3, 100, (128 + 1) ** 2),
            (3, 100, (128 + 1) ** 2 - 1),
        ],
    )
    def test_torch_baseline_manyvals(self, kernel, D, N_level, T):
        torch.random.manual_seed(0)
        L = 16

        # Inputs
        x = torch.rand(L, D, device="cuda")
        pi = torch.tensor([PI_1, PI_2, PI_3], device="cuda", dtype=torch.int64)[:D]

        # Baseline
        baseline = _cpu_embedding_lookup(x, pi, D, T, N_level)

        # Triton
        o = torch.zeros(L, 2**D, dtype=torch.int64, device="cuda")
        BLOCK_D = triton.next_power_of_2(D)
        BLOCK_L = triton.next_power_of_2(L)
        NEEDS_HASH = (N_level + 1) ** D > T
        kernel[(1,)](
            x, pi, o, x.stride(0), o.stride(0), L, D, T, N_level, BLOCK_D, BLOCK_L, NEEDS_HASH, T_POW_2=T & (T - 1) == 0
        )

        assert_close(o, baseline, check_device=False, check_dtype=False)

    @pytest.mark.parametrize("N_level", [16, 32, 512])
    @pytest.mark.parametrize("unique", [False, True])
    def test_overflow(self, kernel, N_level, unique):
        torch.random.manual_seed(0)
        L, D = 499, 3
        T = (N_level + 1) ** D - (2 if not unique else 0)

        # Inputs
        x = torch.rand(L, D, device="cuda")
        pi = torch.tensor([PI_1, PI_2, PI_3], device="cuda", dtype=torch.int64)

        # Triton
        o = torch.zeros(L, 2**D, dtype=torch.int64, device="cuda")
        BLOCK_D = triton.next_power_of_2(D)
        BLOCK_L = 64
        NEEDS_HASH = not unique
        grid = (triton.cdiv(L, BLOCK_L),)
        kernel[grid](x, pi, o, x.stride(0), o.stride(0), L, D, T, N_level, BLOCK_D, BLOCK_L, NEEDS_HASH)

        assert (o < T).all()
        assert (o >= 0).all()


@pytest.mark.cuda
class TestHashEncoding:

    def test_forward_basic(self, float_dtype):
        torch.random.manual_seed(0)
        L, D, F = 4, 2, 2
        N_min, N_max = 2, 4
        T, NUM_LEVELS = (N_max + 1) ** D - 1, 2

        # Inputs
        t = compute_embedding_counts(NUM_LEVELS, T, D, N_min, N_max)
        e = torch.randn(sum(t), F, device="cuda", dtype=float_dtype)
        x = torch.rand(L, D, device="cuda")

        # Baseline (x was set at corners so we can easily check edge features)
        pi = torch.tensor([PI_1, PI_2, PI_3], device="cuda", dtype=torch.int64)[:D]
        baseline = torch.zeros(L, F * NUM_LEVELS, device="cuda", dtype=float_dtype)
        _cpu_hash_encoding(x, e, pi, baseline, D, F, T, NUM_LEVELS, N_min, N_max)

        # Triton
        o = hash_encoding(x, e, None, None, T, N_min, N_max, NUM_LEVELS)

        assert_close(o, baseline, atol=1e-3, rtol=0)

    @pytest.mark.parametrize(
        "L, D, F, T, NUM_LEVELS, N_min, N_max",
        [
            (100, 3, 2, 2**14, 16, 16, 512),
            (100, 3, 2, 2**20, 16, 16, 512),
        ],
    )
    def test_forward(self, L, D, F, T, NUM_LEVELS, N_min, N_max):
        torch.random.manual_seed(0)

        # Inputs
        t = compute_embedding_counts(NUM_LEVELS, T, D, N_min, N_max)
        e = torch.randn(sum(t), F, device="cuda", dtype=torch.float16)
        x = torch.rand(L, D, device="cuda")

        # Baseline (x was set at corners so we can easily check edge features)
        pi = torch.tensor([PI_1, PI_2, PI_3], device="cuda", dtype=torch.int64)[:D]
        baseline = torch.zeros(L, F * NUM_LEVELS, device="cuda", dtype=torch.float16)
        _cpu_hash_encoding(x, e, pi, baseline, D, F, T, NUM_LEVELS, N_min, N_max)

        # Triton
        hash_encoding(x, e, None, None, T, N_min, N_max, NUM_LEVELS)

    def test_backward_basic(self, float_dtype):
        torch.random.manual_seed(0)
        L, D, F = 4, 2, 2
        N_min, N_max = 2, 4
        T, NUM_LEVELS = (N_max + 1) ** D - 1, 2

        # Inputs
        t = compute_embedding_counts(NUM_LEVELS, T, D, N_min, N_max)
        e = torch.randn(sum(t), F, device="cuda", dtype=float_dtype, requires_grad=True)
        x = torch.rand(L, D, device="cuda")

        # Baseline (x was set at corners so we can easily check edge features)
        pi = torch.tensor([PI_1, PI_2, PI_3], device="cuda", dtype=torch.int64)[:D]
        o = torch.zeros(L, F * NUM_LEVELS, device="cuda", dtype=float_dtype)
        _cpu_hash_encoding(x, e, pi, o, D, F, T, NUM_LEVELS, N_min, N_max)
        o.sum().backward()
        baseline_de = e.grad
        e.grad = None

        # Triton
        o = hash_encoding(x, e, None, None, T, N_min, N_max, NUM_LEVELS)
        o.sum().backward()
        de = e.grad

        assert_close(de, baseline_de, atol=0, rtol=1e-2)

    @pytest.mark.parametrize(
        "L, D, F, T, NUM_LEVELS, N_min, N_max",
        [
            (8, 3, 2, 30, 2, 2, 4),
            (10, 3, 2, 30, 2, 2, 4),
            (100, 3, 2, 2**14, 16, 16, 512),
        ],
    )
    def test_backward_hash(self, L, D, F, T, NUM_LEVELS, N_min, N_max):
        torch.random.manual_seed(0)

        # Inputs
        t = compute_embedding_counts(NUM_LEVELS, T, D, N_min, N_max)
        e = torch.randn(sum(t), F, device="cuda", dtype=torch.float16, requires_grad=True)
        x = torch.rand(L, D, device="cuda")
        x = x.fill_(0)

        # Baseline (x was set at corners so we can easily check edge features)
        pi = torch.tensor([PI_1, PI_2, PI_3], device="cuda", dtype=torch.int64)[:D]
        o = torch.zeros(L, F * NUM_LEVELS, device="cuda", dtype=torch.float16)
        _cpu_hash_encoding(x, e, pi, o, D, F, T, NUM_LEVELS, N_min, N_max)
        o.sum().backward()
        baseline_de = e.grad
        e.grad = None

        # Triton
        o = hash_encoding(x, e, None, None, T, N_min, N_max, NUM_LEVELS)
        o.sum().backward()
        de = e.grad

        assert_close(de, baseline_de, rtol=1e-2, atol=0)

    @pytest.mark.skip
    def test_forward_module(self):
        # Shapes
        B, L, D, F = 16, 40, 3, 2

        layer = HashEncoding(2**4, 4, F, 16, 512).cuda()
        x = torch.rand(B, L, D, device="cuda")

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            o = layer(x)
        assert isinstance(o, Tensor)
        assert o.dtype == torch.float16
        assert False

    @pytest.mark.skip
    def test_backward_module(self):
        # Shapes
        B, L, D, F = 16, 40, 3, 2

        layer = HashEncoding(2**8, 4, F, 16, 512).cuda()
        x = torch.rand(B, L, D, device="cuda", requires_grad=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            o = layer(x)

        o.sum().backward()
        assert layer.embeddings.grad is not None

    @pytest.mark.skip
    def test_backward_large(self):
        # Shapes
        B, L, D, F = 1, 32, 3, 2

        layer = HashEncoding(2**4, 16, F, 16, 512).cuda()
        x = torch.rand(B, L, D, device="cuda", requires_grad=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            o = layer(x)

        o.sum().backward()
        assert layer.embeddings.grad is not None
        assert False