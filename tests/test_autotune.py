import pytest
from triton import Config

from triton_helpers.autotune import PruneConfigs


@pytest.mark.parametrize(
    "configs, args, prune, exp",
    [
        (
            [Config({"B": 4}), Config({"B": 8}), Config({"B": 16})],
            {"D": 8},
            PruneConfigs("B", low="D"),
            [Config({"B": 8}), Config({"B": 16})],
        ),
        (
            [Config({"B": 4}), Config({"B": 8}), Config({"B": 16})],
            {"D": 8},
            PruneConfigs("B", high="D"),
            [Config({"B": 4}), Config({"B": 8})],
        ),
        pytest.param(
            [Config({"B": 4}), Config({"B": 8}), Config({"B": 16})],
            {"D": 8},
            PruneConfigs("B", high=0),
            [],
            marks=pytest.mark.xfail(raises=ValueError, strict=True),
        ),
    ],
)
def test_prune_configs(configs, args, prune, exp):
    act = prune(configs, args)
    # Default __eq__ is unique object comparison so check kwargs
    act = [c.kwargs for c in act]
    exp = [c.kwargs for c in exp]
    assert act == exp
