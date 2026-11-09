"""Convert paired execution results/failures into immutable observations."""

from __future__ import annotations

import platform
import socket
import time
from pathlib import Path
from typing import Any, Mapping

from mrminer.common.canonical import fs_safe, sha256_digest
from mrminer.common.schema import (
    ObservationError,
    ObservationIdentity,
    ObservationProvenance,
    ObservationSpec,
    ObservationStatus,
    PerformanceObservation,
    ProvisionedEnvironment,
    RunSpec,
)
from mrminer.s6_runner.paired_executor import (
    ExecutionFailure,
    PairedExecutionResult,
    TorchPairedExecutor,
)
from mrminer.s6_runner.single_executor import (
    SingleDecisionExecutionResult,
    SingleDecisionExecutor,
)
from mrminer.s6_runner.performance_executor import (
    PERFORMANCE_SAMPLES_PROBE,
    PerformanceExecutionResult,
    PerformanceProcessExecutionResult,
    TorchPerformanceExecutor,
)
from mrminer.s6_runner.subprocess_executor import (
    ProcessProvenance,
    SubprocessSupervisor,
)


_ERROR_STAGE = {
    "eager": "eager",
    "reference_replay": "eager",
    "compile": "compile",
    "runtime": "runtime",
    "probe": "probe",
    "pre_trigger_probe": "probe",
    "effect_probe": "probe",
    "decision_probe": "probe",
    "warmup": "runtime",
    "measurement": "runtime",
    "compare": "probe",
    "backend_reset": "harness",
    "initialization": "harness",
    "materialization": "harness",
    "startup": "harness",
    "harness": "harness",
    "timeout": "harness",
}

_ERROR_STATUS = {
    "eager": ObservationStatus.EAGER_FAIL,
    "reference_replay": ObservationStatus.EAGER_FAIL,
    "compile": ObservationStatus.COMPILE_FAIL,
    "runtime": ObservationStatus.CRASH,
    "probe": ObservationStatus.CRASH,
    "pre_trigger_probe": ObservationStatus.CRASH,
    "effect_probe": ObservationStatus.CRASH,
    "decision_probe": ObservationStatus.CRASH,
    "warmup": ObservationStatus.CRASH,
    "measurement": ObservationStatus.CRASH,
    "compare": ObservationStatus.CRASH,
    "backend_reset": ObservationStatus.CRASH,
    "initialization": ObservationStatus.CRASH,
    "materialization": ObservationStatus.CRASH,
    "startup": ObservationStatus.CRASH,
    "harness": ObservationStatus.CRASH,
    "timeout": ObservationStatus.TIMEOUT,
}


def _environment(
    run: RunSpec,
    *,
    framework_version: str,
    device: str,
    seed: int,
    provisioned_environment: ProvisionedEnvironment | None = None,
    environment_fingerprint: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str, str]:
    hostname_hash = sha256_digest(socket.gethostname())
    runtime_fingerprint = {
        "python": platform.python_version(),
        "torch": framework_version,
        "device": device,
        "device_class": run.device_class,
        "seed": seed,
        "hostname_hash": hostname_hash,
    }
    fingerprint = dict(environment_fingerprint or {})
    conflicts = {
        key
        for key, value in runtime_fingerprint.items()
        if key in fingerprint and fingerprint[key] != value
    }
    if conflicts:
        raise ValueError(
            "runtime environment conflicts with supplied fingerprint: "
            + ", ".join(sorted(conflicts))
        )
    fingerprint.update(runtime_fingerprint)
    if provisioned_environment is not None:
        if provisioned_environment.artifact_id != run.artifact_id:
            raise ValueError(
                "provisioned environment artifact does not match RunSpec"
            )
        fingerprint["provisioned_environment"] = (
            provisioned_environment.to_record()
        )
        return (
            fingerprint,
            provisioned_environment.functional_env_hash,
            provisioned_environment.performance_env_hash,
        )
    functional = sha256_digest(
        {
            "python": runtime_fingerprint["python"],
            "torch": framework_version,
            "device_class": run.device_class,
        }
    )
    performance = sha256_digest(
        {
            **fingerprint,
            "runtime_config": run.runtime_config,
        }
    )
    return fingerprint, functional, performance


def _identity(
    run: RunSpec,
    *,
    framework_version: str,
    device: str,
    seed: int,
    probe_semantics_versions: Mapping[str, str],
    provisioned_environment: ProvisionedEnvironment | None = None,
    environment_fingerprint: Mapping[str, Any] | None = None,
) -> tuple[ObservationIdentity, dict[str, Any]]:
    if set(probe_semantics_versions) != set(run.probe_bundle.probes):
        raise ValueError(
            "probe semantics versions must cover the RunSpec probe bundle"
        )
    fingerprint, functional, performance = _environment(
        run,
        framework_version=framework_version,
        device=device,
        seed=seed,
        provisioned_environment=provisioned_environment,
        environment_fingerprint=environment_fingerprint,
    )
    identity = ObservationIdentity(
        program_id=run.program_id,
        input_id=run.input_id,
        artifact_id=run.artifact_id,
        compile_config_hash=sha256_digest(run.compile_config),
        runtime_config_hash=sha256_digest(run.runtime_config),
        device_class=run.device_class,
        phase=run.phase,
        run_kind=run.run_kind,
        probe_bundle_version=run.probe_bundle.version,
        probes=run.probe_bundle.probes,
        probe_semantics_versions=dict(probe_semantics_versions),
        functional_env_hash=functional,
        performance_env_hash=performance,
    )
    return identity, fingerprint


def build_success_observation(
    *,
    run: RunSpec,
    result: PairedExecutionResult,
    attempt: int,
    probe_semantics_versions: Mapping[str, str],
    process: ProcessProvenance | None = None,
    provisioned_environment: ProvisionedEnvironment | None = None,
    environment_fingerprint: Mapping[str, Any] | None = None,
) -> ObservationSpec:
    if result.decision is not None:
        for probe, version in result.decision.probe_versions.items():
            if probe_semantics_versions.get(probe) != version:
                raise ValueError(
                    f"decision probe version disagrees with identity: {probe}"
                )
    identity, fingerprint = _identity(
        run,
        framework_version=result.framework_version,
        device=result.device,
        seed=result.seed,
        probe_semantics_versions=probe_semantics_versions,
        provisioned_environment=provisioned_environment,
        environment_fingerprint=environment_fingerprint,
    )
    if process is not None and process.worker_environment:
        fingerprint["worker_environment"] = dict(
            process.worker_environment
        )
    return ObservationSpec.create(
        run_id=run.run_id,
        attempt=attempt,
        identity=identity,
        provenance=ObservationProvenance(
            env_fingerprint=fingerprint,
            stdout_ref=process.stdout_ref if process else None,
            stderr_ref=process.stderr_ref if process else None,
            exit_code=process.exit_code if process else 0,
            timeout_phase=process.timeout_phase if process else None,
            native_signal=process.native_signal if process else None,
        ),
        status=ObservationStatus.OK,
        error=None,
        semantic=result.semantic,
        decision=result.decision,
        performance=None,
    )


def build_single_decision_success_observation(
    *,
    run: RunSpec,
    result: SingleDecisionExecutionResult,
    attempt: int,
    probe_semantics_versions: Mapping[str, str],
    process: ProcessProvenance | None = None,
    provisioned_environment: ProvisionedEnvironment | None = None,
    environment_fingerprint: Mapping[str, Any] | None = None,
) -> ObservationSpec:
    for probe, version in result.decision.probe_versions.items():
        if probe_semantics_versions.get(probe) != version:
            raise ValueError(
                f"decision probe version disagrees with identity: {probe}"
            )
    if set(result.decision.probe_versions) != set(run.probe_bundle.probes):
        raise ValueError(
            "single decision result must cover exactly the RunSpec probe bundle"
        )
    identity, fingerprint = _identity(
        run,
        framework_version=result.framework_version,
        device=result.device,
        seed=result.seed,
        probe_semantics_versions=probe_semantics_versions,
        provisioned_environment=provisioned_environment,
        environment_fingerprint=environment_fingerprint,
    )
    if process is not None and process.worker_environment:
        fingerprint["worker_environment"] = dict(
            process.worker_environment
        )
    return ObservationSpec.create(
        run_id=run.run_id,
        attempt=attempt,
        identity=identity,
        provenance=ObservationProvenance(
            env_fingerprint=fingerprint,
            stdout_ref=process.stdout_ref if process else None,
            stderr_ref=process.stderr_ref if process else None,
            exit_code=process.exit_code if process else 0,
            timeout_phase=process.timeout_phase if process else None,
            native_signal=process.native_signal if process else None,
        ),
        status=ObservationStatus.OK,
        error=None,
        semantic=None,
        decision=result.decision,
        performance=None,
    )


def _process_audit(
    processes: tuple[ProcessProvenance, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "proc": index,
            "stdout_ref": process.stdout_ref,
            "stderr_ref": process.stderr_ref,
            "exit_code": process.exit_code,
            "timeout_phase": process.timeout_phase,
            "native_signal": process.native_signal,
            "worker_environment": dict(process.worker_environment),
        }
        for index, process in enumerate(processes, start=1)
    ]


def _write_process_log_indexes(
    log_dir: Path,
    processes: tuple[ProcessProvenance, ...],
) -> tuple[str, str]:
    stdout_index = log_dir / "stdout.index.txt"
    stderr_index = log_dir / "stderr.index.txt"
    stdout_index.write_text(
        "".join(
            f"proc={index}\t{process.stdout_ref}\n"
            for index, process in enumerate(processes, start=1)
        ),
        encoding="utf-8",
    )
    stderr_index.write_text(
        "".join(
            f"proc={index}\t{process.stderr_ref}\n"
            for index, process in enumerate(processes, start=1)
        ),
        encoding="utf-8",
    )
    return str(stdout_index), str(stderr_index)


def build_performance_success_observation(
    *,
    run: RunSpec,
    result: PerformanceExecutionResult,
    attempt: int,
    probe_semantics_versions: Mapping[str, str],
    processes: tuple[ProcessProvenance, ...],
    log_dir: Path,
    provisioned_environment: ProvisionedEnvironment | None = None,
    environment_fingerprint: Mapping[str, Any] | None = None,
) -> ObservationSpec:
    repetitions = run.execution_policy.perf_reps
    if repetitions is None:
        raise ValueError("performance run has no repetition policy")
    if len(result.performance.process_reps) != repetitions.processes:
        raise ValueError("performance result process count does not match RunSpec")
    for rep in result.performance.process_reps:
        if (
            rep.warmup_discarded != repetitions.warmup
            or len(rep.samples_us) != repetitions.in_proc_samples
        ):
            raise ValueError(
                "performance result samples do not match RunSpec repetitions"
            )
    if len(processes) != repetitions.processes:
        raise ValueError("performance process provenance is incomplete")
    identity, fingerprint = _identity(
        run,
        framework_version=result.framework_version,
        device=result.device,
        seed=result.seed,
        probe_semantics_versions=probe_semantics_versions,
        provisioned_environment=provisioned_environment,
        environment_fingerprint=environment_fingerprint,
    )
    fingerprint["performance_processes"] = _process_audit(processes)
    stdout_ref, stderr_ref = _write_process_log_indexes(log_dir, processes)
    return ObservationSpec.create(
        run_id=run.run_id,
        attempt=attempt,
        identity=identity,
        provenance=ObservationProvenance(
            env_fingerprint=fingerprint,
            stdout_ref=stdout_ref,
            stderr_ref=stderr_ref,
            exit_code=0,
            timeout_phase=None,
            native_signal=None,
        ),
        status=ObservationStatus.OK,
        error=None,
        semantic=None,
        decision=None,
        performance=result.performance,
    )


def build_failure_observation(
    *,
    run: RunSpec,
    failure: ExecutionFailure,
    attempt: int,
    framework_version: str,
    device: str,
    seed: int,
    probe_semantics_versions: Mapping[str, str],
    log_root: Path,
    process: ProcessProvenance | None = None,
    provisioned_environment: ProvisionedEnvironment | None = None,
    environment_fingerprint: Mapping[str, Any] | None = None,
    provenance_details: Mapping[str, Any] | None = None,
) -> ObservationSpec:
    identity, fingerprint = _identity(
        run,
        framework_version=framework_version,
        device=device,
        seed=seed,
        probe_semantics_versions=probe_semantics_versions,
        provisioned_environment=provisioned_environment,
        environment_fingerprint=environment_fingerprint,
    )
    if process is not None and process.worker_environment:
        fingerprint["worker_environment"] = dict(
            process.worker_environment
        )
    if provenance_details:
        fingerprint.update(dict(provenance_details))
    stage = _ERROR_STAGE.get(failure.stage, "harness")
    status = (
        ObservationStatus.CRASH
        if failure.native_signal is not None
        else _ERROR_STATUS.get(failure.stage, ObservationStatus.CRASH)
    )
    stack_path = (
        log_root
        / fs_safe(run.run_id)
        / f"attempt-{attempt}"
        / "exception.stack.txt"
    )
    stack_path.parent.mkdir(parents=True, exist_ok=True)
    stack_path.write_text(failure.stack, encoding="utf-8")
    return ObservationSpec.create(
        run_id=run.run_id,
        attempt=attempt,
        identity=identity,
        provenance=ObservationProvenance(
            env_fingerprint=fingerprint,
            stdout_ref=process.stdout_ref if process else None,
            stderr_ref=process.stderr_ref if process else None,
            exit_code=(
                process.exit_code
                if process
                else failure.exit_code if failure.exit_code is not None else 1
            ),
            timeout_phase=(
                process.timeout_phase if process else failure.timeout_phase
            ),
            native_signal=(
                process.native_signal if process else failure.native_signal
            ),
        ),
        status=status,
        error=ObservationError(
            stage=stage,
            exception_type=failure.exception_type,
            stack_ref=str(stack_path),
        ),
        semantic=None,
        decision=None,
        performance=None,
    )


def execute_observation(
    *,
    executor: TorchPairedExecutor,
    run: RunSpec,
    program_factory,
    materialized_input: Any,
    seed: int,
    rtol: float,
    atol: float,
    attempt: int,
    probe_semantics_versions: Mapping[str, str],
    log_root: Path,
    supervisor: SubprocessSupervisor | None = None,
    input_factory=None,
    provisioned_environment: ProvisionedEnvironment | None = None,
    environment_fingerprint: Mapping[str, Any] | None = None,
) -> ObservationSpec:
    if (
        provisioned_environment is not None
        and provisioned_environment.artifact_id != run.artifact_id
    ):
        raise ValueError(
            "provisioned environment artifact does not match RunSpec"
        )
    log_dir = log_root / fs_safe(run.run_id) / f"attempt-{attempt}"
    supervisor = supervisor or SubprocessSupervisor()
    try:
        supervised = supervisor.execute(
            executor=executor,
            run=run,
            program_factory=program_factory,
            materialized_input=materialized_input,
            input_factory=input_factory,
            seed=seed,
            rtol=rtol,
            atol=atol,
            log_dir=log_dir,
        )
    except Exception as error:
        failure = ExecutionFailure("harness", error)
        process = None
    else:
        failure = supervised.failure
        process = supervised.provenance
    if failure is not None:
        import torch

        device = _materialized_device(materialized_input, run.device_class)
        return build_failure_observation(
            run=run,
            failure=failure,
            attempt=attempt,
            framework_version=torch.__version__,
            device=device,
            seed=seed,
            probe_semantics_versions=probe_semantics_versions,
            log_root=log_root,
            process=process,
            provisioned_environment=provisioned_environment,
            environment_fingerprint=environment_fingerprint,
        )
    result = supervised.result
    if result is None:
        raise AssertionError("successful supervised execution has no result")
    try:
        return build_success_observation(
            run=run,
            result=result,
            attempt=attempt,
            probe_semantics_versions=probe_semantics_versions,
            process=supervised.provenance,
            provisioned_environment=provisioned_environment,
            environment_fingerprint=environment_fingerprint,
        )
    except Exception as error:
        return build_failure_observation(
            run=run,
            failure=ExecutionFailure("harness", error),
            attempt=attempt,
            framework_version=result.framework_version,
            device=result.device,
            seed=result.seed,
            probe_semantics_versions=probe_semantics_versions,
            log_root=log_root,
            process=supervised.provenance,
            provisioned_environment=provisioned_environment,
            environment_fingerprint=environment_fingerprint,
        )


def execute_single_decision_observation(
    *,
    executor: SingleDecisionExecutor,
    run: RunSpec,
    program_factory,
    materialized_input: Any,
    seed: int,
    attempt: int,
    probe_semantics_versions: Mapping[str, str],
    log_root: Path,
    supervisor: SubprocessSupervisor | None = None,
    input_factory=None,
    provisioned_environment: ProvisionedEnvironment | None = None,
    environment_fingerprint: Mapping[str, Any] | None = None,
) -> ObservationSpec:
    if (
        provisioned_environment is not None
        and provisioned_environment.artifact_id != run.artifact_id
    ):
        raise ValueError(
            "provisioned environment artifact does not match RunSpec"
        )
    log_dir = log_root / fs_safe(run.run_id) / f"attempt-{attempt}"
    supervisor = supervisor or SubprocessSupervisor()
    try:
        supervised = supervisor.execute(
            executor=executor,
            run=run,
            program_factory=program_factory,
            materialized_input=materialized_input,
            input_factory=input_factory,
            seed=seed,
            rtol=0,
            atol=0,
            log_dir=log_dir,
        )
    except Exception as error:
        failure = ExecutionFailure("harness", error)
        process = None
    else:
        failure = supervised.failure
        process = supervised.provenance
    if failure is not None:
        import torch

        return build_failure_observation(
            run=run,
            failure=failure,
            attempt=attempt,
            framework_version=torch.__version__,
            device=_materialized_device(materialized_input, run.device_class),
            seed=seed,
            probe_semantics_versions=probe_semantics_versions,
            log_root=log_root,
            process=process,
            provisioned_environment=provisioned_environment,
            environment_fingerprint=environment_fingerprint,
        )
    result = supervised.result
    if result is None:
        raise AssertionError("successful supervised execution has no result")
    if not isinstance(result, SingleDecisionExecutionResult):
        return build_failure_observation(
            run=run,
            failure=ExecutionFailure(
                "harness",
                exception_type="InvalidExecutionResult",
                message="single decision executor returned an incompatible result",
            ),
            attempt=attempt,
            framework_version=result.framework_version,
            device=result.device,
            seed=result.seed,
            probe_semantics_versions=probe_semantics_versions,
            log_root=log_root,
            process=supervised.provenance,
            provisioned_environment=provisioned_environment,
            environment_fingerprint=environment_fingerprint,
        )
    try:
        return build_single_decision_success_observation(
            run=run,
            result=result,
            attempt=attempt,
            probe_semantics_versions=probe_semantics_versions,
            process=supervised.provenance,
            provisioned_environment=provisioned_environment,
            environment_fingerprint=environment_fingerprint,
        )
    except Exception as error:
        return build_failure_observation(
            run=run,
            failure=ExecutionFailure("harness", error),
            attempt=attempt,
            framework_version=result.framework_version,
            device=result.device,
            seed=result.seed,
            probe_semantics_versions=probe_semantics_versions,
            log_root=log_root,
            process=supervised.provenance,
            provisioned_environment=provisioned_environment,
            environment_fingerprint=environment_fingerprint,
        )


def execute_performance_observation(
    *,
    executor: TorchPerformanceExecutor,
    run: RunSpec,
    program_factory,
    materialized_input: Any,
    seed: int,
    attempt: int,
    probe_semantics_versions: Mapping[str, str],
    log_root: Path,
    metric: str,
    direction: str,
    supervisor: SubprocessSupervisor | None = None,
    input_factory=None,
    provisioned_environment: ProvisionedEnvironment | None = None,
    environment_fingerprint: Mapping[str, Any] | None = None,
) -> ObservationSpec:
    if run.run_kind.value != "performance":
        raise ValueError("performance execution requires run_kind=performance")
    if metric != "latency_us" or direction not in {
        "lower_better",
        "lower_is_better",
    }:
        raise ValueError(
            "v1 performance runner supports only latency_us/lower_better"
        )
    if run.probe_bundle.probes != (PERFORMANCE_SAMPLES_PROBE,):
        raise ValueError(
            "v1 performance runner supports only performance.samples/v1"
        )
    if (
        provisioned_environment is not None
        and provisioned_environment.artifact_id != run.artifact_id
    ):
        raise ValueError(
            "provisioned environment artifact does not match RunSpec"
        )
    if run.device_class == "cuda" and input_factory is None:
        raise ValueError(
            "CUDA performance execution requires worker-side input_factory"
        )
    if input_factory is not None and materialized_input is not None:
        raise ValueError(
            "provide either materialized_input or input_factory, not both"
        )
    repetitions = run.execution_policy.perf_reps
    if repetitions is None:
        raise ValueError("performance RunSpec has no repetition policy")
    protocol = {
        "metric": "latency_us",
        "direction": "lower_better",
        "clock": (
            "cuda_events" if run.device_class == "cuda" else "perf_counter"
        ),
        "processes": repetitions.processes,
        "in_proc_samples": repetitions.in_proc_samples,
        "warmup": repetitions.warmup,
        "runtime_config": dict(run.runtime_config),
    }
    fingerprint = dict(environment_fingerprint or {})
    if (
        "performance_protocol" in fingerprint
        and fingerprint["performance_protocol"] != protocol
    ):
        raise ValueError("supplied performance protocol fingerprint conflicts")
    fingerprint["performance_protocol"] = protocol

    log_dir = log_root / fs_safe(run.run_id) / f"attempt-{attempt}"
    log_dir.mkdir(parents=True, exist_ok=True)
    supervisor = supervisor or SubprocessSupervisor()
    deadline = time.monotonic() + run.execution_policy.timeout_s
    results: list[PerformanceProcessExecutionResult] = []
    processes: list[ProcessProvenance] = []
    failure: ExecutionFailure | None = None
    failed_process: ProcessProvenance | None = None
    for process_index in range(1, repetitions.processes + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            failure = ExecutionFailure(
                "timeout",
                exception_type="TimeoutExpired",
                message=(
                    "performance run exhausted its total timeout before "
                    f"process {process_index}"
                ),
                timeout_phase=f"process-{process_index}-startup",
            )
            break
        try:
            supervised = supervisor.execute(
                executor=executor,
                run=run,
                program_factory=program_factory,
                materialized_input=materialized_input,
                input_factory=input_factory,
                seed=seed,
                rtol=0,
                atol=0,
                log_dir=log_dir / f"process-{process_index}",
                process_index=process_index,
                timeout_s=remaining,
            )
        except Exception as error:
            failure = ExecutionFailure("harness", error)
            break
        processes.append(supervised.provenance)
        if supervised.failure is not None:
            failure = supervised.failure
            failed_process = supervised.provenance
            break
        if not isinstance(
            supervised.result,
            PerformanceProcessExecutionResult,
        ):
            failure = ExecutionFailure(
                "harness",
                exception_type="InvalidExecutionResult",
                message="performance executor returned an incompatible result",
            )
            failed_process = supervised.provenance
            break
        results.append(supervised.result)

    if failure is not None:
        import torch

        result = results[-1] if results else None
        return build_failure_observation(
            run=run,
            failure=failure,
            attempt=attempt,
            framework_version=(
                result.framework_version if result else torch.__version__
            ),
            device=(
                result.device
                if result
                else _materialized_device(materialized_input, run.device_class)
            ),
            seed=seed,
            probe_semantics_versions=probe_semantics_versions,
            log_root=log_root,
            process=failed_process,
            provisioned_environment=provisioned_environment,
            environment_fingerprint=fingerprint,
            provenance_details={
                "performance_processes": _process_audit(tuple(processes))
            },
        )

    framework_versions = {result.framework_version for result in results}
    devices = {result.device for result in results}
    clocks = {result.clock for result in results}
    seeds = {result.seed for result in results}
    if (
        len(results) != repetitions.processes
        or len(framework_versions) != 1
        or len(devices) != 1
        or len(clocks) != 1
        or seeds != {seed}
    ):
        return build_failure_observation(
            run=run,
            failure=ExecutionFailure(
                "harness",
                exception_type="InconsistentPerformanceProcesses",
                message="performance process results disagree",
            ),
            attempt=attempt,
            framework_version=(
                results[0].framework_version if results else "unknown"
            ),
            device=(results[0].device if results else run.device_class),
            seed=seed,
            probe_semantics_versions=probe_semantics_versions,
            log_root=log_root,
            provisioned_environment=provisioned_environment,
            environment_fingerprint=fingerprint,
            provenance_details={
                "performance_processes": _process_audit(tuple(processes))
            },
        )
    compile_counters = {
        str(result.process_rep.proc): dict(result.compile_counters)
        for result in results
        if result.compile_counters
    }
    aggregate = PerformanceExecutionResult(
        performance=PerformanceObservation(
            process_reps=tuple(
                result.process_rep
                for result in sorted(
                    results,
                    key=lambda item: item.process_rep.proc,
                )
            ),
            compile_counters=compile_counters,
            clock=results[0].clock,
        ),
        framework_version=results[0].framework_version,
        device=results[0].device,
        seed=seed,
    )
    try:
        return build_performance_success_observation(
            run=run,
            result=aggregate,
            attempt=attempt,
            probe_semantics_versions=probe_semantics_versions,
            processes=tuple(processes),
            log_dir=log_dir,
            provisioned_environment=provisioned_environment,
            environment_fingerprint=fingerprint,
        )
    except Exception as error:
        return build_failure_observation(
            run=run,
            failure=ExecutionFailure("harness", error),
            attempt=attempt,
            framework_version=aggregate.framework_version,
            device=aggregate.device,
            seed=seed,
            probe_semantics_versions=probe_semantics_versions,
            log_root=log_root,
            provisioned_environment=provisioned_environment,
            environment_fingerprint=fingerprint,
            provenance_details={
                "performance_processes": _process_audit(tuple(processes))
            },
        )


def _materialized_device(value: Any, fallback: str) -> str:
    try:
        import torch
    except ImportError:
        return fallback
    if isinstance(value, torch.Tensor):
        return str(value.device)
    if isinstance(value, (tuple, list)):
        for item in value:
            device = _materialized_device(item, "")
            if device:
                return device
    if isinstance(value, dict):
        for key in sorted(value):
            device = _materialized_device(value[key], "")
            if device:
                return device
    return fallback
