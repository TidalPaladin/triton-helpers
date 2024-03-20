#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib.metadata
from typing import Final


# In the case of K=16 we will perform the following operation in each tensor core
# (16x16) * (16x8) = (16x8)
# BFloat16 will only support a FP32 accumulator
TENSOR_CORE_K: Final = 16

__version__ = importlib.metadata.version("triton-helpers")
