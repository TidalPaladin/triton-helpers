import math
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence, Tuple

import torch
import triton
from torch import Tensor


@dataclass
class IsBlockMultiple:
    r"""Heuristic to determine if a dimension is a multiple of a block dimension.

    Args:
        dim: Input dimension name
        block_dim: Block dimension name
        override_val: If set the heuristic will always return this value

    Returns:
        True if the dimension is a multiple of the block dimension, False otherwise.
        If the `override_val` is set, it will be returned instead.
    """

    dim: str
    block_dim: str
    override_val: bool | None = None

    def __call__(self, args: Dict[str, Any]) -> bool:
        if self.override_val is not None:
            return self.override_val
        return args[self.dim] % args[self.block_dim] == 0


@dataclass
class BoundaryCheckHeuristic:
    r"""Heuristic to produce a block pointer boundary check tuple given dimensions and block sizes.

    The output of this heuristic should be used in a kernel as `boundary_check=BOUNDARY_CHECK.value`.

    Args:
        dim: Input dimension name or sequence of such
        block_dim: Block dimension name or sequence of such

    Returns:
        Tuple of integers indicating which dimensions require boundary checks.
    """

    dim: str | Sequence[str]
    block_dim: str | Sequence[str]

    def __post_init__(self):
        if isinstance(self.dim, str):
            self.dim = [self.dim]
        if isinstance(self.block_dim, str):
            self.block_dim = [self.block_dim]
        if len(self.dim) != len(self.block_dim):
            raise ValueError("`dim` and `block_dim` must have the same length")

    def __len__(self):
        return len(self.dim)

    def __call__(self, args: Dict[str, Any]) -> Tuple[int, ...]:
        assert len(self.dim) == len(self.block_dim)
        return tuple(
            i for i, (dim, block_dim) in enumerate(zip(self.dim, self.block_dim)) if args[dim] % args[block_dim] != 0
        )


@dataclass
class PowerOfTwoHeuristic:
    r"""Heuristic to select the next power of two for a given dimension.

    Args:
        dim: Input dimension name
        min_val: Minimum value for the output
        max_val: Maximum value for the output
        previous: If True, the previous power of two is returned if the next one is greater than the input.

    Returns:
        The next power of two for the given dimension.

    Example:
        >>> PowerOfTwoHeuristic("dim", 16, 64)({"dim": 128})
        64
        >>> PowerOfTwoHeuristic("dim", 16, 64)({"dim": 100})
        64
        >>> PowerOfTwoHeuristic("dim", 16)({"dim": 100})
        128
        >>> PowerOfTwoHeuristic("dim", 16, 64)({"dim": 32})
        32
    """

    dim: str
    min_val: int = 1
    max_val: int = sys.maxsize
    previous: bool = False

    def __call__(self, args: Dict[str, Any]) -> int:
        dim = args[self.dim]
        pow_2 = triton.next_power_of_2(dim)
        if self.previous and pow_2 > dim:
            pow_2 //= 2
        return max(self.min_val, min(self.max_val, pow_2))


@dataclass
class DivisorHeuristic:
    r"""Heuristic to select the largest power of two that is a divisor of a given dimension.

    Args:
        dim: Input dimension name
        min_val: Minimum value for the output
        max_val: Maximum value for the output
        error_on_non_divisor: If True, an error is raised if the dimension is not a power of two.

    Returns:
        The next power of two for the given dimension.

    Example:
        >>> DivisorHeuristic("dim", 16, 64)({"dim": 128})
        64
        >>> DivisorHeuristic("dim", 16, 64)({"dim": 100})
        16
        >>> DivisorHeuristic("dim", 16, 64)({"dim": 32})
        32
    """

    dim: str
    min_val: int = 1
    max_val: int = sys.maxsize
    error_on_non_divisor: bool = False

    def __call__(self, args: Dict[str, Any]) -> int:
        dim = args[self.dim]
        largest_divisor_pow_2 = self.min_val
        assert dim > 0
        while dim % (largest_divisor_pow_2 * 2) == 0:
            largest_divisor_pow_2 *= 2

        result = min(self.max_val, largest_divisor_pow_2)
        if self.error_on_non_divisor and dim % result != 0:
            raise ValueError(
                f"Cannot find a divisor for {self.dim} of size {dim} within the range "
                f"[{self.min_val}, {self.max_val}] that is a power of two. "
            )

        return result


@dataclass
class SelectHeuristic:
    r"""Selects between two heuristics based on a condition.

    Args:
        func: Condition to select the heuristic. Should accept `args` dict as input.
        when_true: Minimum value for the output
        when_false: Maximum value for the output

    Returns:
        Selected heuristic based on the condition.
    """

    func: Callable[[Dict[str, Any]], bool]
    when_true: Callable[[Dict[str, Any]], Any]
    when_false: Callable[[Dict[str, Any]], Any]

    def __call__(self, args: Dict[str, Any]) -> Any:
        return self.when_true(args) if self.func(args) else self.when_false(args)


@dataclass
class SMHeuristic:
    r"""Derive a block size from the number of streaming multiprocessors (SMs) in the device.

    Args:
        device_from: Key of a tensor in the config from which to derive the device
        size_dims: List of keys in the config to use as the size dimensions.
            The product of these dimensions will be computed to determine the total size.
        min_size: Minimum block size.
        max_size: Maximum block size.

    Returns:
        Block size that best distributes the total size of the tensor across the number of
        SMs in the device.
    """

    device_from: str
    size_dims: str | Sequence[str]
    min_size: int = 1
    max_size: int = sys.maxsize
    occupancy_threshold: float = 2.0

    def __call__(self, args: Dict[str, Any]) -> int:
        # Get the SM count and the total size of what we're parallelizing over
        sm_count = self.get_sm_count_from_meta(args, self.device_from)
        total_size = self.get_total_size(args, self.size_dims)

        # There should be at least `sm_count` blocks to fully utilize the device
        needed_block_size = triton.cdiv(total_size, sm_count)

        # If the needed block size is a power of two, return it
        if needed_block_size & (needed_block_size - 1) == 0:
            return min(max(needed_block_size, self.min_size), self.max_size)
        assert needed_block_size > 1

        # Otherwise we choose between next and previous power of two
        next_pow_2 = triton.next_power_of_2(needed_block_size)
        prev_pow_2 = next_pow_2 // 2

        # Compute occupancy for both next and previous power of two
        sm_occupancy_high = ((next_pow_2 * sm_count) % total_size) / next_pow_2
        sm_occupancy_low = (total_size % (prev_pow_2 * sm_count)) / prev_pow_2

        # Choose the previous power of two if it has better occupancy by a threshold
        selection = prev_pow_2 if sm_occupancy_low > self.occupancy_threshold * sm_occupancy_high else next_pow_2

        return min(max(selection, self.min_size), self.max_size)

    @classmethod
    def get_device(cls, args: Dict[str, Any], key: str) -> torch.device:
        t = args[key]
        if not isinstance(t, Tensor):
            raise TypeError(f"Expected a tensor, got {type(t)}")
        return t.device

    @classmethod
    def get_sm_count_from_meta(cls, args: Dict[str, Any], key: str) -> int:
        r"""Get the number of streaming multiprocessors (SMs) from a tensor in the config.

        Args:
            args: Config dictionary
            key: Key of a tensor in the config from which to derive the device
        """
        device = cls.get_device(args, key)
        return cls.get_sm_count_from_device(device)

    @classmethod
    def get_sm_count_from_device(cls, device: torch.device) -> int:
        r"""Get the number of streaming multiprocessors (SMs) from a device."""
        return int(torch.cuda.get_device_properties(device).multi_processor_count)

    @classmethod
    def get_total_size(cls, args: Dict[str, Any], size_dims: str | Sequence[str]) -> int:
        size_dims = [size_dims] if isinstance(size_dims, str) else size_dims
        return math.prod(args[dim] for dim in size_dims)
