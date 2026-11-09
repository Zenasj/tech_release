"""Single-run decision execution with structured phase failures."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from mrminer.common.schema import DecisionObservation, RunKind, RunSpec
from mrminer.s6_runner.effect_probe import DecisionProbe
from mrminer.s6_runner.paired_executor import ExecutionFailure


@dataclass(frozen=True)
class SingleDecisionExecutionResult:
    decision: DecisionObservation
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


class SingleDecisionExecutor:
    def __init__(self, *, compiler, decision_probe: DecisionProbe):
        if decision_probe is None:
            raise ValueError("single decision execution requires a decision probe")
        self.compiler = compiler
        self.decision_probe = decision_probe

    def execute(
        self,
        *,
        run: RunSpec,
        program_factory,
        materialized_input: Any,
        seed: int,
        rtol: float,
        atol: float,
        phase_callback=None,
    ) -> SingleDecisionExecutionResult:
        del rtol, atol
        if run.run_kind is not RunKind.SINGLE:
            raise ExecutionFailure(
                "initialization",
                exception_type="InvalidRunKind",
                message="SingleDecisionExecutor requires run_kind=single",
            )

        def phase(name: str) -> None:
            if phase_callback is not None:
                phase_callback(name)

        phase("initialization")
        try:
            _seed_all(seed)
            program = program_factory()
        except ExecutionFailure:
            raise
        except Exception as error:
            raise ExecutionFailure("initialization", error) from error
        input_device = _first_tensor_device(materialized_input) or "cpu"
        input_device_class = input_device.split(":", maxsplit=1)[0]
        if input_device_class != run.device_class:
            raise ExecutionFailure(
                "initialization",
                exception_type="DeviceClassMismatch",
                message=(
                    f"RunSpec device_class={run.device_class!r} does not match "
                    f"materialized input device class={input_device_class!r}"
                ),
            )

        phase("pre_trigger_probe")
        try:
            self.decision_probe.before_compile(program, materialized_input)
        except Exception as error:
            raise ExecutionFailure("pre_trigger_probe", error) from error

        phase("compile")
        try:
            compiled = self.compiler(program, dict(run.compile_config))
        except Exception as error:
            raise ExecutionFailure("compile", error) from error

        phase("runtime")
        try:
            output = _call_program(compiled, materialized_input)
        except Exception as error:
            raise ExecutionFailure("runtime", error) from error

        phase("decision_probe")
        try:
            decision = self.decision_probe.after_runtime(
                program,
                materialized_input,
                output,
            )
        except Exception as error:
            raise ExecutionFailure("decision_probe", error) from error
        if not isinstance(decision, DecisionObservation):
            raise ExecutionFailure(
                "decision_probe",
                exception_type="InvalidProbeResult",
                message="decision probe did not return DecisionObservation",
            )

        device = input_device or _first_tensor_device(output) or "cpu"
        return SingleDecisionExecutionResult(
            decision=decision,
            framework_version=str(_torch().__version__),
            device=device,
            seed=seed,
        )
