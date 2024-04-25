import sys

import pytest
import torch
import triton
import triton.language as tl
from torch.testing import assert_close

from triton_helpers.heuristics import (
    BoundaryCheckHeuristic,
    DivisorHeuristic,
    IsBlockMultiple,
    PowerOfTwoHeuristic,
    SelectHeuristic,
    SMHeuristic,
)


@pytest.mark.parametrize(
    "dim, block_dim, override_val, exp",
    [
        (32, 16, None, True),
        (40, 16, None, False),
        (128, 16, None, True),
        (128, 128, None, True),
        (16, 128, None, False),
        (16, 128, True, True),
        (32, 16, False, False),
    ],
)
def test_is_block_multiple(dim, block_dim, override_val, exp):
    heuristic = IsBlockMultiple("dim", "block_dim", override_val)
    meta = {"dim": dim, "block_dim": block_dim}
    assert heuristic(meta) == exp


class TestBoundaryCheckHeuristic:

    @pytest.fixture
    def kernel(self):
        @triton.jit
        def kernel(
            x_p, o_p, M: int, N: int, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BOUNDARY_CHECK: tl.constexpr
        ):
            tl.static_print(BOUNDARY_CHECK)
            tl.static_print(BOUNDARY_CHECK.value)
            ptr = tl.make_block_ptr(x_p, (M, N), (N, 1), (0, 0), (BLOCK_M, BLOCK_N), (1, 0))
            x = tl.load(ptr, boundary_check=BOUNDARY_CHECK.value)
            x += tl.sum(tl.sum(x, 0), 0)
            ptr = tl.make_block_ptr(o_p, (M, N), (N, 1), (0, 0), (BLOCK_M, BLOCK_N), (1, 0))
            tl.store(ptr, x, boundary_check=BOUNDARY_CHECK.value)

        return kernel

    @pytest.mark.parametrize(
        "M, N, BLOCK_M, BLOCK_N, exp",
        [
            (32, 32, 32, 32, tuple()),
            (30, 32, 32, 32, (0,)),
            (32, 30, 32, 32, (1,)),
            (30, 30, 32, 32, (0, 1)),
        ],
    )
    def test_conditions(self, M, N, BLOCK_M, BLOCK_N, exp):
        heuristic = BoundaryCheckHeuristic(["M", "N"], ["BLOCK_M", "BLOCK_N"])
        meta = {"M": M, "N": N, "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N}
        assert heuristic(meta) == exp

    @pytest.mark.cuda
    @pytest.mark.parametrize(
        "M, N, BLOCK_M, BLOCK_N",
        [
            (32, 32, 32, 32),
            (30, 32, 32, 32),
            (32, 30, 32, 32),
            (30, 30, 32, 32),
        ],
    )
    def test_in_kernel(self, kernel, M, N, BLOCK_M, BLOCK_N):
        torch.random.manual_seed(0)
        x = torch.randn(M, N, device="cuda")
        o = torch.empty_like(x)
        heuristic = BoundaryCheckHeuristic(["M", "N"], ["BLOCK_M", "BLOCK_N"])
        kernel = triton.heuristics({"BOUNDARY_CHECK": heuristic})(kernel)
        kernel[(1,)](x, o, M, N, BLOCK_M, BLOCK_N)
        assert_close(o, x + x.sum(), rtol=0, atol=1e-3)


@pytest.mark.parametrize(
    "dim, min_val, max_val, previous, exp",
    [
        (4, 1, sys.maxsize, False, 4),
        (3, 1, sys.maxsize, False, 4),
        (4, 16, sys.maxsize, False, 16),
        (32, 4, 16, False, 16),
        (7, 1, sys.maxsize, True, 4),
        (8, 1, sys.maxsize, True, 8),
    ],
)
def test_power_of_two_heuristic(dim, min_val, max_val, previous, exp):
    heuristic = PowerOfTwoHeuristic("dim", min_val, max_val, previous)
    meta = {"dim": dim}
    assert heuristic(meta) == exp


@pytest.mark.parametrize(
    "dim, min_val, max_val, exp",
    [
        (4, 1, sys.maxsize, 4),
        (128, 1, sys.maxsize, 128),
        (128, 16, 64, 64),
        (100, 16, 64, 16),
        (32, 16, 64, 32),
    ],
)
def test_divisor_heuristic(dim, min_val, max_val, exp):
    heuristic = DivisorHeuristic("dim", min_val, max_val)
    meta = {"dim": dim}
    assert heuristic(meta) == exp


def test_divisor_heuristic_error():
    heuristic = DivisorHeuristic("dim", min_val=16, error_on_non_divisor=True)
    meta = {"dim": 100}
    with pytest.raises(ValueError):
        heuristic(meta)


@pytest.mark.parametrize(
    "func, when_true, when_false, exp",
    [
        (lambda *_: True, lambda *_: 1, lambda *_: 2, 1),
        (lambda *_: False, lambda *_: 1, lambda *_: 2, 2),
    ],
)
def test_select_heuristic(func, when_true, when_false, exp):
    heuristic = SelectHeuristic(func, when_true, when_false)
    assert heuristic({}) == exp


@pytest.mark.parametrize(
    "sm_count, size, min_size, max_size, exp",
    [
        (50, 100, 1, sys.maxsize, 2),
        (80, 100, 1, sys.maxsize, 2),
        (80, 100, 4, sys.maxsize, 4),
        (10, 100, 1, 4, 4),
    ],
)
def test_sm_heuristic(mocker, sm_count, size, min_size, max_size, exp):
    # Mock SM count
    m = mocker.MagicMock()
    m.multi_processor_count = sm_count
    mocker.patch("torch.cuda.get_device_properties", return_value=m)

    heuristic = SMHeuristic("t", "L", min_size, max_size)
    assert heuristic({"t": torch.empty(1), "L": size}) == exp
