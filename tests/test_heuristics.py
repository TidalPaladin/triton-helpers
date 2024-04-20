import sys

import pytest
import torch

from triton_helpers.heuristics import (
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
