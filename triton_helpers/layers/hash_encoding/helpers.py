import torch
import math
from typing import List
from torch import Tensor


def compute_b(N_min: int, N_max: int, L: int) -> int:
    r"""Computes the base of the geometric progression of resolutions.

    Args:
        N_min: Resolution of the coarsest table level.
        N_max: Resolution of the finest table level.
        L: Number of table levels.

    Returns:
        The base of the geometric progression of resolutions.
    """
    return int(math.exp((math.log(N_max) - math.log(N_min)) / (L - 1)))


def compute_resolutions(num_levels: int, N_min: int, N_max: int) -> List[int]:
    r"""Computes the resolutions of the table levels.

    Args:
        num_levels: Number of table levels.
        N_min: Resolution of the coarsest table level.
        N_max: Resolution of the finest table level.
    
    Returns:
        List of resolutions of the table levels.
    """
    b = torch.tensor(math.exp((math.log(N_max) - math.log(N_min)) / (num_levels - 1)))
    l = torch.arange(0, num_levels)
    return b.pow(l).mul(N_min).floor().long().tolist()


def compute_embedding_counts(L: int, T: int, N_min: int, N_max: int) -> List[int]:
    r"""Computes the number of embeddings per table level.

    Args:
        L: Number of table levels.
        T: Maximum number of entries per table level.
        N_min: Resolution of the coarsest table level.
        N_max: Resolution of the finest table level.

    Returns:
        List of the number of embeddings per table level.
    """
    resolutions = torch.tensor(compute_resolutions(L, N_min, N_max))
    t = (resolutions + 1) ** 2
    return torch.min(t, t.new_tensor(T)).tolist()


def get_first_hash_level(N_min: int, N_max: int, L: int, T: int) -> int:
    r"""Computes the first table level where the number of embeddings exceeds the size of the table.

    This is the first level at which a hash function will be needed.
    
    Returns:
        The first table level where the number of embeddings exceeds the size of the table,
        satisfying ``0 <= result < L``.
    """
    t_i = torch.tensor(compute_embedding_counts(L, T, N_min, N_max))
    return int((t_i > T).int().argmax())


def seek_to_level_embeddings(e: Tensor, l_i: int, L: int, T: int, N_min: int, N_max: int) -> Tensor:
    r"""Given a dense set of embeddings, returns a view of the embeddings at a specific table level.

    Args:
        e: Dense set of embeddings.
        l_i: Table level index.
        L: Number of table levels.
        T: Maximum number of entries per table level.
        N_min: Resolution of the coarsest table level.
        N_max: Resolution of the finest table level.

    Shapes:
        - e: :math:`(*, E, F)`
        - Output: :math:`(*, E_i, F)`

    Returns:
        View of the embeddings at the specified table level.
    """
    size = compute_embedding_counts(L, T, N_min, N_max)
    start = [0] + torch.cumsum(torch.tensor(size), 0).tolist()
    end = start[1:]
    return e[..., start[l_i]:end[l_i], :]
    