"""Spawn-based supervision for paired and single execution."""

from __future__ import annotations

import multiprocessing
import os
import resource
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cloudpickle

from mrminer.common.schema import RunKind, RunSpec
from mrminer.s6_runner.paired_executor import (
    ExecutionFailure,
    PairedExecutionResult,
    TorchPairedExecutor,
)
from mrminer.s6_runner.performance_executor import (
    PerformanceProcessExecutionResult,
    TorchPerformanceExecutor,
)
from mrminer.s6_runner.single_executor import (
    SingleDecisionExecutionResult,
    SingleDecisionExecutor,
)


@dataclass(frozen=True)
class ProcessProvenance:
    stdout_ref: str
    stderr_ref: str
    exit_code: int
    timeout_phase: str | None
    native_signal: str | None
    worker_environment: dict[str, str]


@dataclass(frozen=True)
class SupervisedExecution:
    result: (
        PairedExecutionResult
        | SingleDecisionExecutionResult
        | PerformanceProcessExecutionResult
        | None
    )
    failure: ExecutionFailure | None
    provenance: ProcessProvenance

    def __post_init__(self) -> None:
        if (self.result is None) == (self.failure is None):
            raise ValueError("supervised execution requires exactly one result or failure")


def _send(connection, message: dict[str, Any]) -> None:
    connection.send_bytes(cloudpickle.dumps(message))


def _disable_core_dumps() -> None:
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (OSError, ValueError):
        pass


def _redirect_process_output(stdout_path: str, stderr_path: str) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd = os.open(
        stdout_path,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o600,
    )
    stderr_fd = os.open(
        stderr_path,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o600,
    )
    try:
        os.dup2(stdout_fd, sys.stdout.fileno())
        os.dup2(stderr_fd, sys.stderr.fileno())
    finally:
        os.close(stdout_fd)
        os.close(stderr_fd)


def _worker_main(
    connection,
    serialized_job: bytes,
    materialized_input: Any,
    stdout_path: str,
    stderr_path: str,
    worker_environment: dict[str, str],
) -> None:
    exit_code = 0
    try:
        os.setsid()
        _disable_core_dumps()
        _redirect_process_output(stdout_path, stderr_path)
        for name, value in worker_environment.items():
            os.environ[name] = value
        _send(connection, {"kind": "ready"})
        job = cloudpickle.loads(serialized_job)
        executor: (
            TorchPairedExecutor
            | SingleDecisionExecutor
            | TorchPerformanceExecutor
        ) = job["executor"]

        def report_phase(phase: str) -> None:
            _send(connection, {"kind": "phase", "phase": phase})

        input_factory = job["input_factory"]
        if input_factory is not None:
            report_phase("materialization")
            try:
                materialized_input = input_factory()
            except Exception as error:
                raise ExecutionFailure("materialization", error) from error
        execute_kwargs = {
            "run": job["run"],
            "program_factory": job["program_factory"],
            "materialized_input": materialized_input,
            "seed": job["seed"],
            "rtol": job["rtol"],
            "atol": job["atol"],
            "phase_callback": report_phase,
        }
        if job["process_index"] is not None:
            execute_kwargs["process_index"] = job["process_index"]
        result = executor.execute(
            **execute_kwargs,
        )
        _send(connection, {"kind": "result", "result": result})
    except ExecutionFailure as failure:
        exit_code = 1
        _send(
            connection,
            {
                "kind": "failure",
                "stage": failure.stage,
                "exception_type": failure.exception_type,
                "message": str(failure),
                "stack": failure.stack,
                "timeout_phase": failure.timeout_phase,
            },
        )
    except BaseException as error:
        exit_code = 1
        _send(
            connection,
            {
                "kind": "failure",
                "stage": "harness",
                "exception_type": type(error).__name__,
                "message": str(error),
                "stack": traceback.format_exc(),
                "timeout_phase": None,
            },
        )
    finally:
        connection.close()
    raise SystemExit(exit_code)


def _signal_name(exit_code: int) -> str | None:
    if exit_code >= 0:
        return None
    number = -exit_code
    try:
        return signal.Signals(number).name
    except ValueError:
        return f"SIG{number}"


def _kill_process_group(
    process: multiprocessing.Process,
    *,
    grace_s: float,
    session_ready: bool,
) -> None:
    if not process.is_alive():
        return
    if session_ready and process.pid is not None:
        try:
            if os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            return
    else:
        process.terminate()
    process.join(grace_s)
    if not process.is_alive():
        return
    if session_ready and process.pid is not None:
        try:
            if os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            return
    else:
        process.kill()
    process.join(grace_s)


class SubprocessSupervisor:
    def __init__(
        self,
        *,
        start_method: str = "spawn",
        termination_grace_s: float = 1.0,
        poll_interval_s: float = 0.02,
    ):
        if start_method != "spawn":
            raise ValueError("only spawn is supported to preserve CUDA safety")
        if termination_grace_s <= 0 or poll_interval_s <= 0:
            raise ValueError("supervisor timing values must be positive")
        self._context = multiprocessing.get_context(start_method)
        self.termination_grace_s = termination_grace_s
        self.poll_interval_s = poll_interval_s

    def execute(
        self,
        *,
        executor: (
            TorchPairedExecutor
            | SingleDecisionExecutor
            | TorchPerformanceExecutor
        ),
        run: RunSpec,
        program_factory,
        materialized_input: Any,
        input_factory=None,
        seed: int,
        rtol: float,
        atol: float,
        log_dir: Path,
        process_index: int | None = None,
        timeout_s: float | None = None,
    ) -> SupervisedExecution:
        if run.execution_policy.isolation != "subprocess":
            raise ValueError("SubprocessSupervisor requires subprocess isolation")
        if input_factory is not None and materialized_input is not None:
            raise ValueError(
                "provide either materialized_input or input_factory, not both"
            )
        if (run.run_kind is RunKind.PERFORMANCE) != (
            process_index is not None
        ):
            raise ValueError(
                "performance supervision requires exactly one process index"
            )
        effective_timeout = (
            float(run.execution_policy.timeout_s)
            if timeout_s is None
            else float(timeout_s)
        )
        if (
            not 0 < effective_timeout <= run.execution_policy.timeout_s
        ):
            raise ValueError(
                "timeout override must be positive and no greater than RunSpec"
            )
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "stdout.txt"
        stderr_path = log_dir / "stderr.txt"
        worker_environment = self._worker_environment(run, log_dir)
        job = cloudpickle.dumps(
            {
                "executor": executor,
                "run": run,
                "program_factory": program_factory,
                "input_factory": input_factory,
                "seed": seed,
                "rtol": rtol,
                "atol": atol,
                "process_index": process_index,
            }
        )
        parent_connection, child_connection = self._context.Pipe(duplex=False)
        process = self._context.Process(
            target=_worker_main,
            args=(
                child_connection,
                job,
                materialized_input,
                str(stdout_path),
                str(stderr_path),
                worker_environment,
            ),
            daemon=False,
        )
        process.start()
        child_connection.close()

        deadline = time.monotonic() + effective_timeout
        last_phase = "startup"
        session_ready = False
        final_message: dict[str, Any] | None = None
        timed_out = False
        try:
            while True:
                while parent_connection.poll():
                    try:
                        message = cloudpickle.loads(parent_connection.recv_bytes())
                    except EOFError:
                        break
                    kind = message.get("kind")
                    if kind == "ready":
                        session_ready = True
                    elif kind == "phase":
                        last_phase = str(message["phase"])
                    elif kind in {"result", "failure"}:
                        final_message = message
                if not process.is_alive():
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    _kill_process_group(
                        process,
                        grace_s=self.termination_grace_s,
                        session_ready=session_ready,
                    )
                    break
                time.sleep(self.poll_interval_s)
        finally:
            process.join(self.termination_grace_s)
            while parent_connection.poll():
                try:
                    message = cloudpickle.loads(parent_connection.recv_bytes())
                except EOFError:
                    break
                kind = message.get("kind")
                if kind == "phase":
                    last_phase = str(message["phase"])
                elif kind in {"result", "failure"}:
                    final_message = message
            parent_connection.close()

        exit_code = process.exitcode
        if exit_code is None:
            _kill_process_group(
                process,
                grace_s=self.termination_grace_s,
                session_ready=session_ready,
            )
            exit_code = process.exitcode if process.exitcode is not None else -1
        native_signal = _signal_name(exit_code)
        provenance = ProcessProvenance(
            stdout_ref=str(stdout_path),
            stderr_ref=str(stderr_path),
            exit_code=exit_code,
            timeout_phase=last_phase if timed_out else None,
            native_signal=native_signal if not timed_out else None,
            worker_environment=worker_environment,
        )
        if not process.is_alive():
            process.close()

        if timed_out:
            failure = ExecutionFailure(
                "timeout",
                exception_type="TimeoutExpired",
                message=(
                    f"execution subprocess exceeded "
                    f"{effective_timeout:g}s during {last_phase}"
                ),
                native_signal=None,
                exit_code=exit_code,
                timeout_phase=last_phase,
            )
            return SupervisedExecution(
                result=None,
                failure=failure,
                provenance=provenance,
            )
        if native_signal is not None:
            failure = ExecutionFailure(
                last_phase,
                exception_type="NativeSignal",
                message=f"execution subprocess terminated by {native_signal}",
                native_signal=native_signal,
                exit_code=exit_code,
                timeout_phase=None,
            )
            return SupervisedExecution(
                result=None,
                failure=failure,
                provenance=provenance,
            )
        if final_message is None:
            failure = ExecutionFailure(
                "harness",
                exception_type="ChildProcessExit",
                message=f"execution subprocess exited with code {exit_code}",
                exit_code=exit_code,
            )
            return SupervisedExecution(
                result=None,
                failure=failure,
                provenance=provenance,
            )
        if final_message["kind"] == "result" and exit_code != 0:
            failure = ExecutionFailure(
                "harness",
                exception_type="ChildProcessExit",
                message=(
                    "execution subprocess emitted a result but exited with "
                    f"code {exit_code}"
                ),
                exit_code=exit_code,
            )
            return SupervisedExecution(
                result=None,
                failure=failure,
                provenance=provenance,
            )
        if final_message["kind"] == "failure":
            failure = ExecutionFailure(
                str(final_message["stage"]),
                exception_type=str(final_message["exception_type"]),
                message=str(final_message["message"]),
                stack=str(final_message["stack"]),
                exit_code=exit_code,
                timeout_phase=final_message.get("timeout_phase"),
            )
            return SupervisedExecution(
                result=None,
                failure=failure,
                provenance=provenance,
            )
        return SupervisedExecution(
            result=final_message["result"],
            failure=None,
            provenance=provenance,
        )

    @staticmethod
    def _worker_environment(
        run: RunSpec,
        log_dir: Path,
    ) -> dict[str, str]:
        policy = run.paired_semantic
        paired_fresh = (
            policy is not None
            and policy.state_isolation.backend_cache.inductor_cache == "fresh"
        )
        if (
            not paired_fresh
            and run.run_kind not in {RunKind.SINGLE, RunKind.PERFORMANCE}
        ):
            return {}
        root = log_dir / "compiler-cache"
        paths = {
            "TORCHINDUCTOR_CACHE_DIR": root / "inductor",
            "TRITON_CACHE_DIR": root / "triton",
        }
        for path in paths.values():
            if path.exists() and any(path.iterdir()):
                raise ValueError(
                    f"fresh compiler cache path is not empty: {path}"
                )
            path.mkdir(parents=True, exist_ok=True)
        return {name: str(path) for name, path in paths.items()}
