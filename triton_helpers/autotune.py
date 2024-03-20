import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import triton


@dataclass
class PruneConfigs:
    r"""Prune autotuner configs based on a condition.

    Args:
        key: Key to check in the config
        low: Either an integer indicating the minimum allowed value of ``key`` or a string indicating the key to use
            as the minimum value
        high: Either an integer indicating the maximum allowed value of ``key`` or a string indicating the key to use
            as the maximum value
    """

    key: str
    low: int | str = 0
    high: int | str = sys.maxsize

    def __call__(self, configs: List[triton.Config], args: Dict[str, Any]) -> List[triton.Config]:
        out: List[triton.Config] = []
        for config in configs:
            val = config.kwargs[self.key]
            low = self.low if isinstance(self.low, int) else args[self.low]
            high = self.high if isinstance(self.high, int) else args[self.high]
            if low <= val <= high:
                out.append(config)

        if not out:
            raise ValueError("All configurations were pruned")
        return out

    @classmethod
    def compose(cls, *pruners: "PruneConfigs") -> Callable[[List[triton.Config], Dict[str, Any]], List[triton.Config]]:
        r"""Compose multiple pruners into a single pruner."""

        def _composed(configs: List[triton.Config], args: Dict[str, Any]) -> List[triton.Config]:
            for pruner in pruners:
                configs = pruner(configs, args)
            return configs

        return _composed
