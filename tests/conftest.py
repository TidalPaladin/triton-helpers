import os

import pytest
import torch


def cuda_available():
    r"""Checks if CUDA is available and device is ready"""
    if not torch.cuda.is_available():
        return False

    capability = torch.cuda.get_device_capability()
    arch_list = torch.cuda.get_arch_list()
    if isinstance(capability, tuple):
        capability = f"sm_{''.join(str(x) for x in capability)}"

    if capability not in arch_list:
        return False

    return True


def handle_cuda_mark(item):  # pragma: no cover
    has_cuda_mark = any(item.iter_markers(name="cuda"))
    if has_cuda_mark and not cuda_available():
        import pytest

        pytest.skip("Test requires CUDA and device is not ready")


def pytest_runtest_setup(item):
    handle_cuda_mark(item)


@pytest.fixture(autouse=True)
def triton_cache(tmp_path):
    # Uses a fresh temporary cache directory for each test
    path = tmp_path / ".triton"
    os.environ["TRITON_CACHE_DIR"] = str(path)
    return path


@pytest.fixture(autouse=True, scope="session")
def triton_debug():
    os.environ["TRITON_DEBUG"] = str(1)
