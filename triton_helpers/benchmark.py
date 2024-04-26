import tempfile
import warnings
from abc import abstractmethod
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import StrEnum
from functools import partial
from itertools import product
from math import prod
from pathlib import Path
from typing import Any, Callable, Dict, Final, Iterator, Self, Sequence, Tuple, Type, cast

import matplotlib.pyplot as plt
import pandas as pd
import torch
import triton
from torch import Tensor
from tqdm import tqdm


MAX_LINE_WARNING: Final = 10
MAX_PLOT_WARNING: Final = 10


class Dtype(StrEnum):
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"

    @property
    def torch(self) -> torch.dtype:
        match self:
            case Dtype.FP16:
                return torch.float16
            case Dtype.BF16:
                return torch.bfloat16
            case Dtype.FP32:
                return torch.float32
            case _:
                raise TypeError(f"Unsupported dtype {self}")  # pragma: no cover


class Endpoint(StrEnum):
    PROFILE = "profile"
    BENCHMARK = "benchmark"


class Mode(StrEnum):
    FORWARD = "fwd"
    BACKWARD = "bwd"
    FORWARD_BACKWARD = "fwd-bwd"

    @property
    def long_name(self) -> str:
        match self:
            case Mode.FORWARD:
                return "Forward"
            case Mode.BACKWARD:
                return "Backward"
            case Mode.FORWARD_BACKWARD:
                return "Forward-Backward"


@dataclass
class DimConfig:
    name: str
    values: Sequence[int]

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> Iterator[int]:
        yield from self.values

    @classmethod
    def from_range(cls: Type[Self], name: str, start: int, stop: int, step: int, logspace: bool = False) -> Self:
        if logspace:
            vals = torch.logspace(start, stop, step, base=2).int().tolist()
        else:
            vals = list(range(start, stop + 1, step))
        return cls(name, vals)

    @classmethod
    def broadcast(cls: Type[Self], config1: Self, config2: Self) -> Tuple[Self, Self]:
        if len(config1) == len(config2):
            return config1, config2

        combined_values = list(product(config1.values, config2.values))
        config1.values = [v[0] for v in combined_values]
        config2.values = [v[1] for v in combined_values]
        return config1, config2

    @classmethod
    def to_kwargs(cls: Type[Self], *configs: Self) -> Dict[str, Self]:
        return {c.name: c for c in configs}

    @staticmethod
    def add_argument(
        parser: ArgumentParser,
        name: str,
        default_val: int | Sequence[int],
        default_mode: str = "values",
    ) -> None:
        key = f"-{name}" if len(name) == 1 else f"--{name}"
        parser.add_argument(
            key,
            type=int,
            nargs="+",
            default=[default_val] if isinstance(default_val, int) else default_val,
            help=f"Value(s) for {name} dimension.",
        )
        parser.add_argument(
            f"--{name}-mode",
            type=str,
            choices=["values", "linspace", "logspace"],
            default=default_mode,
            help=(
                f"How to generate values for {key}.\n"
                "If 'values', the provided values will be used.\n"
                "If 'linspace', the provided start, stop, and step will be used with linear spacing.\n"
                "If 'logspace', the provided start, stop, and num_steps will be used with base 2 log spacing."
            ),
        )

    @classmethod
    def from_namespace(cls: Type[Self], args: Namespace, name: str) -> Self:
        if name in args:
            values = getattr(args, name)
            assert all(isinstance(v, int) for v in values)
            match (mode := getattr(args, f"{name}_mode")):
                case "values":
                    return cls(name, values)
                case "linspace":
                    return cls.from_range(name, *values)
                case "logspace":
                    return cls.from_range(name, *values, logspace=True)
                case _:
                    raise ValueError(f"Unsupported mode {mode}")  # pragma: no cover
        else:
            raise ValueError(f"Cannot find values for {name} dimension.")  # pragma: no cover


@dataclass
class KernelExecutor:
    provider: str

    callback: Callable | None = None

    # Internal params
    warmup: int = 20
    rep: int = 100
    quantiles: Tuple[float, float, float] = field(default_factory=lambda: (0.5, 0.2, 0.8))

    def __str__(self) -> str:
        return str.title(self.provider)

    def profile(self, mode: Mode, dtype: Dtype, *args, **kwargs) -> None:
        func = self._get_closure(mode, *args, dtype=dtype.torch, **kwargs)

        # Run warmup
        for _ in range(self.warmup):
            func()

        # Run the function to be profiled
        torch.cuda.cudart().cudaProfilerStart()  # type: ignore
        torch.cuda.nvtx.range_push(f"{self.provider}-{mode}-{dtype}")
        func()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()  # type: ignore
        if self.callback:
            self.callback()

    def benchmark(self, mode: Mode, dtype: Dtype, *args, **kwargs) -> Tuple[float, float, float]:
        func = self._get_closure(mode, *args, dtype=dtype.torch, **kwargs)

        # Do the benchmark (no warmup needed, Triton handles it)
        ms, min_ms, max_ms = triton.testing.do_bench(
            func,
            quantiles=self.quantiles,
            warmup=self.warmup,
            rep=self.rep,
        )
        if self.callback:
            self.callback()
        return ms, min_ms, max_ms

    def _get_closure(self, mode: Mode, *args, **kwargs) -> Callable:
        match mode:
            case Mode.FORWARD:
                _kwargs = self.prepare_inputs(*args, **kwargs, requires_grad=False)
                return lambda: self.forward(**_kwargs)

            case Mode.BACKWARD:
                _kwargs = self.prepare_inputs(*args, **kwargs, requires_grad=True)
                output = self.forward(**_kwargs)
                output = self.prepare_backward(output)
                do = torch.randn_like(output)
                return lambda: self.backward(output, do)

            case Mode.FORWARD_BACKWARD:
                _kwargs = self.prepare_inputs(*args, **kwargs, requires_grad=True)

                def func():
                    output = self.forward(**_kwargs)
                    output = self.prepare_backward(output)
                    do = torch.randn_like(output)
                    self.backward(output, do)
                    return output

                return func

            case _:
                raise ValueError(f"Unsupported mode {mode}")  # pragma: no cover

    @abstractmethod
    def prepare_inputs(self, *args, requires_grad: bool = False, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError  # pragma: no cover

    def prepare_backward(self, forward_output: Any) -> Tensor:
        if isinstance(forward_output, Tensor):
            return forward_output
        else:
            raise TypeError(
                f"Unsupported forward output {type(forward_output)}. "
                "Please override `prepare_backward` method to extract and return a tensor."
            )  # pragma: no cover

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError  # pragma: no cover

    def backward(self, output: Tensor, do: Tensor) -> None:
        output.backward(do, retain_graph=True)

    def randn(self, *args, **kwargs) -> Tensor:
        defaults = dict(requires_grad=True)
        defaults.update(kwargs)
        result = torch.randn(*args, **kwargs)
        return result

    def rand(self, *args, **kwargs) -> Tensor:
        defaults = dict(requires_grad=True)
        defaults.update(kwargs)
        result = torch.rand(*args, **kwargs)
        return result


@dataclass
class CLI:
    r"""CLI interface to benchmarking or profiling Triton kernels.

    See :meth:`entrypoint` for usage.
    """

    task: str
    kernels: Sequence["KernelExecutor"]
    dims: Sequence[DimConfig]
    modes: Sequence[Mode] = field(default_factory=lambda: (Mode.FORWARD,))
    dtypes: Sequence[str] = field(default_factory=lambda: (Dtype.FP16,))
    x_label: str = "Input size"
    y_label: str = "Latency (ms)"
    output: Path | None = None
    line_arg: str = "kernel"
    x_dim: str | None = None
    device: torch.device = torch.device("cuda")

    def __post_init__(self):
        if self.output is not None and not self.output.is_dir():
            raise NotADirectoryError(f"Output path {self.output} is not a directory")  # pragma: no cover
        if not self.kernels:
            raise ValueError("At least one kernel must be provided")  # pragma: no cover
        if not self.dims:
            raise ValueError("At least one dimension must be provided")  # pragma: no cover
        if not self.modes:
            raise ValueError("At least one mode must be provided")  # pragma: no cover
        if not self.dtypes:
            raise ValueError("At least one dtype must be provided")  # pragma: no cover
        if self.x_dim is None:
            self.x_dim = self.dims[0].name

    def __call__(self, endpoint: Endpoint) -> None:
        # Separate out the dim that will be the chart x-axis from the other dims
        x_dim = next(d for d in self.dims if d.name == self.x_dim)
        other_dims = [d for d in self.dims if d.name != self.x_dim]

        # Determine which attribute will be the line arg / legend. The other non-X axis
        # values will be used to create separate plots.
        charts = {
            "kernel": self.kernels,
            "dtype": self.dtypes,
            "mode": self.modes,
            "dim": other_dims,
        }
        charts.pop(self.line_arg, None)
        match self.line_arg:
            case "kernel":
                line_vals = self.kernels
                line_names = [str(k) for k in self.kernels]
            case "dtype":
                line_vals = self.dtypes
                line_names = [str(d).upper() for d in self.dtypes]
            case "mode":
                line_vals = self.modes
                line_names = [mode.long_name for mode in self.modes]
            case x if x in {d.name for d in other_dims}:
                idx = other_dims.index(next(d for d in other_dims if d.name == self.line_arg))
                # Ensure we remove it from charts
                dim = other_dims.pop(idx)
                line_vals = list(dim)
                line_names = [f"{dim.name}={v}" for v in line_vals]
                if len(line_vals) > MAX_LINE_WARNING:
                    warnings.warn(
                        f"Using {len(line_vals)} different values for line arg {self.line_arg}. "
                        f"This may result in a large number lines on each plot. Consider using fewer values."
                    )
                assert len(other_dims) + 2 == len(self.dims), "removed x and line dims"
            case _:
                raise ValueError(f"Unsupported line arg {self.line_arg}")  # pragma: no cover

        # Prepare the progress bar
        total_dims = prod(len(d) for d in charts.get("dim", []))
        total_tests = len(x_dim) * len(line_vals) * prod(len(d) for k, d in charts.items() if k != "dim") * total_dims
        bar = tqdm(total=total_tests, desc="Working")
        for kernel in self.kernels:
            kernel.callback = lambda: bar.update(1)

        # Create wrapper that Triton will call
        def func(
            kernel: KernelExecutor,
            dtype: Dtype,
            mode: Mode,
            **kwargs,
        ):
            if endpoint == Endpoint.PROFILE:
                kernel.profile(mode, dtype, device=self.device, **kwargs)
            elif endpoint == Endpoint.BENCHMARK:
                return kernel.benchmark(mode, dtype, device=self.device, **kwargs)
            else:
                raise ValueError(f"Unsupported endpoint {endpoint}")  # pragma: no cover

        with tempfile.TemporaryDirectory() if self.output is None else nullcontext(self.output) as output:
            for config in self.benchmark_configs(
                self.task,
                x_dim,
                self.line_arg,
                line_vals,
                line_names,
                charts,
                self.x_label,
            ):
                wrapped = triton.testing.perf_report(config)(func)
                df = cast(
                    pd.DataFrame,
                    wrapped.run(
                        show_plots=False,
                        print_data=False,
                        return_df=True,
                        save_path=str(output),
                    ),
                )
                if endpoint == Endpoint.BENCHMARK:
                    with tqdm.external_write_mode():
                        print(df.to_string(index=False))
                    plt.close()

        bar.close()

    @staticmethod
    def benchmark_configs(
        task: str,
        x_dim: DimConfig,
        line_arg: str,
        line_vals: Sequence[Any],
        line_names: Sequence[str],
        charts: Dict[str, Sequence[Any]],
        x_label: str = "Input size",
    ) -> Iterator[triton.testing.Benchmark]:
        # Bind static args
        Benchmark = partial(
            triton.testing.Benchmark,
            xlabel=x_label,
            line_arg=line_arg,
            line_vals=list(line_vals),
            line_names=list(line_names),
            x_names=[x_dim.name],
            x_vals=list(x_dim),
        )

        # Separate out the dim key so we can do a proper product.
        # Compute the cartesian product of dimension values and create a dictionary
        # for each product entry.
        if (dims := charts.pop("dim", None)) is not None:
            product_entries = list(product(*[[(d.name, value) for value in d] for d in dims]))
            out_dims: Sequence[Dict[str, int]] = [dict(entry) for entry in product_entries]
            if len(out_dims) > MAX_PLOT_WARNING:
                warnings.warn(
                    f"Creating {len(out_dims)} different plots from the provided dimensions. "
                    f"Consider using fewer values."
                )
            charts["dim"] = out_dims

        # Fill missing keys so we can do a product
        charts = {k: charts.get(k, [None]) for k in ("kernel", "mode", "dtype", "dim")}

        # Build the benchmarks to run
        for mode, dtype, kernel, dim in product(charts["mode"], charts["dtype"], charts["kernel"], charts["dim"]):
            dim: Dict[str, int]
            dim_name = "-".join(f"{k}={v}" for k, v in dim.items())
            kernel_name = str(kernel) if kernel else None
            plot_name = "-".join(filter(None, (task, kernel_name, mode, dim_name, dtype)))
            if kernel is None:
                y_label = f"{mode.long_name} latency @ {dtype.upper()} (ms)"
            elif dtype is None:
                y_label = f"{mode.long_name} latency for {kernel} (ms)"
            elif mode is None:
                y_label = f"{kernel} latency @ {dtype.upper()} (ms)"
            elif not dim:
                y_label = f"{kernel} {mode.long_name} latency @ {dtype.upper()} (ms)"
            else:
                raise AssertionError("This should never happen")  # pragma: no cover

            args = {"mode": mode, "dtype": dtype, "kernel": kernel, **dim}
            args.pop(line_arg, None)
            yield Benchmark(
                plot_name=plot_name,
                ylabel=y_label,
                args=args,
            )

    @classmethod
    def entrypoint(
        cls,
        task: str,
        kernels: Sequence["KernelExecutor"],
        dims: Dict[str, Tuple[int | Sequence[int], str]],
        argv: Sequence[str] | None = None,
    ) -> None:
        """
        Entrypoint for the CLI interface to benchmark or profile Triton kernels.

        This method sets up the command-line interface for benchmarking or profiling,
        allowing the user to specify various parameters such as providers, modes, data types,
        and warmup repetitions. It also dynamically adds arguments for each dimension
        specified in the `dims` attribute.

        Args:
            task: Identifying name for the task being benchmarked or profiled.
            kernels: A sequence of kernel executors to benchmark or profile.
            dims: A dictionary where keys are dimension names and values are tuples containing a
                sequence of dimension values and a mode ('linear' or 'log').
            argv: Optional list of command-line arguments to parse. If not provided, `sys.argv` will be used.

        Raises:
            NotADirectoryError: If the output path is not a directory.
            ValueError: If no kernels or dimensions are provided, or if no modes or data types are specified.

        Example:
            >>> CLI.entrypoint(
                    "LayerNorm",
                    [Baseline("torch"), Triton("triton")],
                    dims={
                        "D": ((6, 15, 10), "logspace"), # 2**6 -> 2**15, 10 steps
                        "L": (256, "values"), # 256 (can manually one or more more discrete values)
                    },
                )
        """
        parser = ArgumentParser(
            description="Benchmark Triton kernels.",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        subparsers = parser.add_subparsers(help="Sub-commands")
        benchmark = subparsers.add_parser("benchmark", help="Benchmark kernels")
        benchmark.set_defaults(endpoint=Endpoint.BENCHMARK)
        profile = subparsers.add_parser("profile", help="Profile kernels")
        profile.set_defaults(endpoint=Endpoint.PROFILE)

        for subparser in (benchmark, profile):
            subparser.add_argument(
                "-p",
                "--providers",
                type=str,
                nargs="+",
                default=[p.provider for p in kernels],
                choices=[p.provider for p in kernels],
                help="Providers to evaluate.",
            )
            subparser.add_argument(
                "-m",
                "--modes",
                type=Mode,
                nargs="+",
                default=[Mode.FORWARD],
                choices=Mode,
                help="Modes to evaluate.",
            )
            subparser.add_argument(
                "-d",
                "--dtypes",
                type=Dtype,
                nargs="+",
                default=[Dtype.FP16],
                choices=Dtype,
                help="Data types to evaluate.",
            )
            subparser.add_argument(
                "-w",
                "--warmup",
                type=int,
                default=20,
                help="Warmup repetitions.",
            )
            subparser.add_argument(
                "--device",
                type=str,
                default="cuda",
                help="Device to use for benchmarking or profiling.",
            )
            for name, (values, mode) in dims.items():
                DimConfig.add_argument(subparser, name, values, mode)

        benchmark.add_argument(
            "-l",
            "--line-arg",
            type=str,
            default="kernel",
            choices=["kernel", "dtype", "mode"] + list(dims.keys()),
            help="Line argument",
        )
        benchmark.add_argument(
            "-x",
            "--x-dim",
            type=str,
            default=(default_dim := next(iter(dims.keys()))),
            choices=dims.keys(),
            help="Dimension to use as plot X axis",
        )
        benchmark.add_argument(
            "--x-label",
            type=str,
            default="Input Size",
            help="Label for plot x axis.",
        )
        benchmark.add_argument(
            "-r",
            "--rep",
            type=int,
            default=100,
            help="Number of repetitions.",
        )
        benchmark.add_argument("-o", "--output", type=Path, default=None, help="Output directory")
        benchmark.add_argument("--plot-dim", type=str, help="Output directory")

        args = parser.parse_args(argv)

        parsed_dims = [DimConfig.from_namespace(args, name) for name in dims.keys()]
        parsed_modes = [Mode(m) for m in args.modes]
        parsed_dtypes = [Dtype(d) for d in args.dtypes]
        kernels = [k for k in kernels if k.provider in args.providers]

        reps = getattr(args, "rep", 100)
        warmup = args.warmup
        for kernel in kernels:
            kernel.rep = reps
            kernel.warmup = warmup

        cli = cls(
            task,
            kernels,
            parsed_dims,
            parsed_modes,
            parsed_dtypes,
            getattr(args, "x_label", "Input size"),
            output=getattr(args, "output", None),
            line_arg=getattr(args, "line_arg", "kernel"),
            x_dim=getattr(args, "x_dim", default_dim),
            device=args.device,
        )
        cli(args.endpoint)
