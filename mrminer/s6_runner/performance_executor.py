"""One-process PyTorch latency measurement for hierarchical performance runs."""

from __future__ import annotations

import copy
import math
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from mrminer.common.schema import (
    PerformanceObservation,
    ProcessRep,
    RunKind,
    RunSpec,
)
from mrminer.s6_runner.paired_executor import ExecutionFailure

PERFORMANCE_SAMPLES_PROBE = "performance.samples/v1"


@dataclass(frozen=True)
class PerformanceProcessExecutionResult:
    process_rep: ProcessRep
    clock: Literal["cuda_events", "perf_counter"]
    compile_counters: dict[str, Any]
    framework_version: str
    device: str
    seed: int


@dataclass(frozen=True)
class PerformanceExecutionResult:
    performance: PerformanceObservation
    framework_version: str
    device: str
    seed: int


def _torch():
    try:
        import torch
    except ImportError as error:
        raise ExecutionFailure("initialization", error) from error
    return torch


def _seed_all(seed: int) -> None:
    torch = _torch()
    random.seed(seed)
    try:
        import numpy
    except ImportError:
        numpy = None
    if numpy is not None:
        numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _call_program(program, materialized_input):
    if isinstance(materialized_input, tuple):
        return program(*materialized_input)
    if isinstance(materialized_input, dict):
        return program(**materialized_input)
    return program(materialized_input)


def _first_tensor_device(value: Any) -> str | None:
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return str(value.device)
    if isinstance(value, (tuple, list)):
        for item in value:
            device = _first_tensor_device(item)
            if device is not None:
                return device
    if isinstance(value, dict):
        for key in sorted(value):
            device = _first_tensor_device(value[key])
            if device is not None:
                return device
    return None


def _fresh_input(materialized_input: Any) -> Any:
    try:
        return copy.deepcopy(materialized_input)
    except Exception as error:
        raise ExecutionFailure("initialization", error) from error


def _module_versions(program: Any) -> dict[str, int]:
    if not hasattr(program, "state_dict"):
        return {}
    try:
        state = program.state_dict()
    except Exception as error:
        raise ExecutionFailure("initialization", error) from error
    return {
        name: int(value._version)
        for name, value in sorted(state.items())
        if hasattr(value, "_version")
    }


def _configure_runtime(run: RunSpec) -> None:
    supported = {"threads", "interop_threads"}
    unknown = set(run.runtime_config) - supported
    if unknown:
        raise ExecutionFailure(
            "initialization",
            exception_type="UnsupportedPerformanceRuntimeConfig",
            message=f"unsupported performance runtime keys: {sorted(unknown)}",
        )
    torch = _torch()
    for name, setter in (
        ("threads", torch.set_num_threads),
        ("interop_threads", torch.set_num_interop_threads),
    ):
        if name not in run.runtime_config:
            continue
        value = run.runtime_config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ExecutionFailure(
                "initialization",
                exception_type="InvalidPerformanceRuntimeConfig",
                message=f"{name} must be a positive integer",
            )
        try:
            setter(value)
        except Exception as error:
            raise ExecutionFailure("initialization", error) from error


def _synchronize(device_class: str, device: str) -> None:
    if device_class == "cuda":
        _torch().cuda.synchronize(device)


class TorchPerformanceExecutor:
    def __init__(
        self,
        *,
        compiler: Callable[[Any, Mapping[str, Any]], Callable[..., Any]],
        compile_counter_probe: Callable[[], Mapping[str, Any]] | None = None,
    ):
        self.compiler = compiler
        self.compile_counter_probe = compile_counter_probe

    def execute(
        self,
        *,
        run: RunSpec,
        program_factory: Callable[[], Any],
        materialized_input: Any,
        seed: int,
        process_index: int,
        rtol: float,
        atol: float,
        phase_callback: Callable[[str], None] | None = None,
    ) -> PerformanceProcessExecutionResult:
        del rtol, atol

        def phase(name: str) -> None:
            if phase_callback is not None:
                phase_callback(name)

        phase("initialization")
        if run.run_kind is not RunKind.PERFORMANCE:
            raise ExecutionFailure(
                "initialization",
                exception_type="InvalidRunKind",
                message="TorchPerformanceExecutor requires run_kind=performance",
            )
        repetitions = run.execution_policy.perf_reps
        if repetitions is None:
            raise ExecutionFailure(
                "initialization",
                exception_type="MissingPerformanceRepetitions",
                message="performance RunSpec has no repetitions",
            )
        if not 1 <= process_index <= repetitions.processes:
            raise ExecutionFailure(
                "initialization",
                exception_type="InvalidPerformanceProcess",
                message="process index is outside the RunSpec repetition range",
            )
        try:
            _seed_all(seed)
            _configure_runtime(run)
            program = program_factory()
        except ExecutionFailure:
            raise
        except Exception as error:
            raise ExecutionFailure("initialization", error) from error

        device = _first_tensor_device(materialized_input) or "cpu"
        device_class = device.split(":", maxsplit=1)[0]
        if device_class != run.device_class:
            raise ExecutionFailure(
                "initialization",
                exception_type="DeviceClassMismatch",
                message=(
                    f"RunSpec device_class={run.device_class!r} does not match "
                    f"materialized input device class={device_class!r}"
                ),
            )
        if device_class not in {"cpu", "cuda"}:
            raise ExecutionFailure(
                "initialization",
                exception_type="UnsupportedPerformanceDevice",
                message=f"unsupported performance device: {device_class}",
            )

        phase("compile")
        try:
            compiled = self.compiler(program, dict(run.compile_config))
        except Exception as error:
            raise ExecutionFailure("compile", error) from error
        try:
            compile_counters = (
                dict(self.compile_counter_probe())
                if self.compile_counter_probe is not None
                else {}
            )
        except Exception as error:
            raise ExecutionFailure("probe", error) from error
        initial_versions = _module_versions(program)

        phase("warmup")
        try:
            for _ in range(repetitions.warmup):
                output = _call_program(compiled, _fresh_input(materialized_input))
                _synchronize(device_class, device)
                del output
        except ExecutionFailure:
            raise
        except Exception as error:
            raise ExecutionFailure("warmup", error) from error
        if _module_versions(program) != initial_versions:
            raise ExecutionFailure(
                "warmup",
                exception_type="MutablePerformanceProgram",
                message="program state changed during performance warmup",
            )

        phase("measurement")
        samples_us = []
        torch = _torch()
        try:
            for _ in range(repetitions.in_proc_samples):
                sample_input = _fresh_input(materialized_input)
                _synchronize(device_class, device)
                if device_class == "cuda":
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    output = _call_program(compiled, sample_input)
                    end.record()
                    end.synchronize()
                    elapsed_us = float(start.elapsed_time(end)) * 1_000
                else:
                    started_ns = time.perf_counter_ns()
                    output = _call_program(compiled, sample_input)
                    elapsed_us = (time.perf_counter_ns() - started_ns) / 1_000
                del output
                if not math.isfinite(elapsed_us) or elapsed_us <= 0:
                    raise ValueError("latency sample must be finite and positive")
                samples_us.append(elapsed_us)
        except ExecutionFailure:
            raise
        except Exception as error:
            raise ExecutionFailure("measurement", error) from error
        if _module_versions(program) != initial_versions:
            raise ExecutionFailure(
                "measurement",
                exception_type="MutablePerformanceProgram",
                message="program state changed during performance measurement",
            )

        return PerformanceProcessExecutionResult(
            process_rep=ProcessRep(
                proc=process_index,
                warmup_discarded=repetitions.warmup,
                samples_us=tuple(samples_us),
            ),
            clock=(
                "cuda_events" if device_class == "cuda" else "perf_counter"
            ),
            compile_counters=compile_counters,
            framework_version=str(torch.__version__),
            device=device,
            seed=seed,
        )
