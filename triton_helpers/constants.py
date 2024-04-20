from typing import Final


# https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators
CACHE_ALL: Final = ".ca"
CACHE_GLOBAL: Final = ".cg"
CACHE_STREAMING: Final = ".cs"
CACHE_LAST_USE: Final = ".lu"
CACHE_NONE: Final = ".cv"
WRITE_BACK: Final = ".wb"
WRITE_THROUGH: Final = ".wt"
