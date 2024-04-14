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
            L: tl.constexpr, D: tl.constexpr, F: tl.constexpr,
            BLOCK_L: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_F: tl.constexpr,
            # fmt: on
        ):
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
        kernel[(1,)](x, e, o, L, D, F, BLOCK_L, BLOCK_D, BLOCK_F)

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
        kernel[(1,)](x, e, o, L, D, F, BLOCK_L, BLOCK_D, BLOCK_F)

        assert_close(o, baseline)


@pytest.mark.cuda
class TestEmbeddingLookup:

    @pytest.fixture
    def kernel(self):
        @triton.jit
        def kernel(
            # fmt: off
            x_p, pi_p, o_p,
            L: tl.constexpr, D: tl.constexpr,
            T_l: int, N_l: int,
            BLOCK_D: tl.constexpr, BLOCK_L: tl.constexpr, NEEDS_HASH: tl.constexpr,
            # fmt: on
        ):
            start = tl.program_id(0) * BLOCK_L
            x_p += start
            o_p += start

            # Load x
            X_ptr = tl.make_block_ptr(x_p, (L, D), (D, 1), (0, 0), (BLOCK_L, BLOCK_D), (1, 0))
            x = tl.load(X_ptr, boundary_check=(0, 1))

            # Load pi
            offset_pi = tl.arange(0, BLOCK_D)
            mask_pi = offset_pi < D
            pi = tl.load(pi_p + offset_pi, mask=mask_pi)

            # Hash
            o = embedding_lookup(x, pi, D, T_l, N_l, BLOCK_D, NEEDS_HASH).to(o_p.dtype.element_ty)

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
                [0, 1, 3, 4],
                [4, 5, 7, 8],
                [0, 1, 3, 4],
                [4, 5, 7, 8],
                [1, 2, 4, 5],
                [3, 4, 6, 7],
            ],
            dtype=torch.int64,
            device="cuda",
        )

        # Triton
        o = torch.zeros(L, 2**D, dtype=torch.int64, device="cuda")
        BLOCK_D = triton.next_power_of_2(D)
        BLOCK_L = triton.next_power_of_2(L)
        NEEDS_HASH = False
        kernel[(1,)](x, pi, o, L, D, T, N_level, BLOCK_D, BLOCK_L, NEEDS_HASH)

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
        kernel[(1,)](x, pi, o, L, D, T, N_level, BLOCK_D, BLOCK_L, NEEDS_HASH)

        assert (o == baseline).all()

    @pytest.mark.parametrize("unique", [False, True])
    def test_torch_baseline(self, kernel, corner_factory, unique):
        torch.random.manual_seed(0)
        L, D, N_level = 16, 3, 8
        # Condition of T for results to not be unique
        T = (N_level + 1) ** D - (2 if not unique else 0)

        # Inputs
        x = torch.rand(L, D, device="cuda")
        pi = torch.tensor([PI_1, PI_2, PI_3], device="cuda", dtype=torch.int64)

        # Baseline
        x_rd = torch.floor(x * N_level).to(torch.int64)
        offsets = corner_factory(D, device=x_rd.device, dtype=torch.int64)
        corners = x_rd[:, None, :] + offsets[None, :, :]
        if unique:
            h = corners * (pi.new_full((D,), N_level + 1) ** torch.arange(D, device=pi.device))
            baseline = h.sum(-1)
        else:
            t = corners * pi.view(1, 1, -1)
            h = torch.bitwise_xor(t[..., 0], t[..., 1])
            for i in range(2, D):
                h = torch.bitwise_xor(h, t[..., i])
            baseline = h % T

        # Triton
        o = torch.zeros(L, 2**D, dtype=torch.int64, device="cuda")
        BLOCK_D = triton.next_power_of_2(D)
        BLOCK_L = triton.next_power_of_2(L)
        NEEDS_HASH = not unique
        kernel[(1,)](x, pi, o, L, D, T, N_level, BLOCK_D, BLOCK_L, NEEDS_HASH)

        assert_close(o, baseline, check_device=False, check_dtype=False)

    @pytest.mark.parametrize("N_level", [16, 32, 512])
    @pytest.mark.parametrize("unique", [False, True])
    def test_overflow(self, kernel, N_level, unique):
        torch.random.manual_seed(0)
        L, D = 16384, 3
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
        kernel[grid](x, pi, o, L, D, T, N_level, BLOCK_D, BLOCK_L, NEEDS_HASH)

        assert (o < T).all()
        assert (o >= 0).all()


@pytest.mark.cuda
class TestHashEncoding:

    def test_forward(self, float_dtype):
        # Shapes
        L, D, F = 4, 3, 2
        N_min, N_max = 2, 4
        T, NUM_LEVELS = (N_max + 1) ** D, 2
        t = compute_embedding_counts(NUM_LEVELS, T, N_min, N_max)

        # Inputs
        e = torch.randn(int(t.sum().item()), F, device="cuda", dtype=float_dtype)
        x = torch.tensor(
            [
                [0, 0],
                [0.9999, 0.9999],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        L, D = x.shape

        # Baseline (x was set at corners so we can easily check edge features)
        baseline = torch.zeros(L, F * NUM_LEVELS, device="cuda", dtype=float_dtype)
        for i in range(NUM_LEVELS):
            start_t = 0 if i == 0 else t[:i].sum().item()
            end_t = start_t + t[i].item()
            _e = e[start_t:end_t]
            level = torch.stack([_e[0, :], _e[-1, :]])
            baseline[..., i * F : i * F + F] = level

        # Triton
        o = hash_encoding(x, e, None, None, T, N_min, N_max, NUM_LEVELS)

        assert_close(o, baseline, atol=1e-3, rtol=0)

    def test_backward(self, float_dtype):
        # Shapes
        D, F = 3, 2
        N_min, N_max = 2, 4
        T, NUM_LEVELS = (N_max + 1) ** D, 2
        t = compute_embedding_counts(NUM_LEVELS, T, N_min, N_max)

        # Inputs
        e = torch.randn(int(t.sum().item()), F, device="cuda", requires_grad=True, dtype=float_dtype)
        x = torch.tensor(
            [
                [0, 0],
                [0.9999, 0.9999],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        L, D = x.shape

        # Baseline (x was set at corners so we can easily check edge features)
        o = torch.zeros(L, F * NUM_LEVELS, device="cuda", dtype=float_dtype)
        for i in range(NUM_LEVELS):
            start_t = 0 if i == 0 else t[:i].sum().item()
            end_t = start_t + t[i].item()
            _e = e[start_t:end_t]
            level = torch.stack([_e[0, :], _e[-1, :]])
            o[..., i * F : i * F + F] = level
        o.sum().backward()
        baseline_de = e.grad
        e.grad = None

        # Triton
        o = hash_encoding(x, e, None, None, T, N_min, N_max, NUM_LEVELS)
        o.sum().backward()
        de = e.grad

        assert_close(de, baseline_de, atol=1e-3, rtol=0)

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

    def test_backward_module(self):
        # Shapes
        B, L, D, F = 16, 40, 3, 2

        layer = HashEncoding(2**8, 4, F, 16, 512).cuda()
        x = torch.rand(B, L, D, device="cuda", requires_grad=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            o = layer(x)

        o.sum().backward()
        assert layer.embeddings.grad is not None

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
