"""PyTorch paired-semantic execution core with explicit state isolation."""

from __future__ import annotations

import copy
import hashlib
import random
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from mrminer.common.canonical import canonical_json
from mrminer.common.schema import (
    DecisionObservation,
    ModulePolicy,
    PairwiseCompare,
    RunKind,
    RunSpec,
    SemanticObservation,
    SemanticSide,
    SemanticStability,
    TensorSummary,
)
from mrminer.s6_runner.effect_probe import DecisionProbe


class UnsupportedIsolation(RuntimeError):
    pass


class ExecutionFailure(RuntimeError):
    def __init__(
        self,
        stage: str,
        error: BaseException | None = None,
        *,
        exception_type: str | None = None,
        message: str | None = None,
        stack: str | None = None,
        native_signal: str | None = None,
        exit_code: int | None = None,
        timeout_phase: str | None = None,
    ):
        if error is not None:
            exception_type = type(error).__name__
            message = str(error)
            stack = traceback.format_exc()
        if exception_type is None:
            raise ValueError("ExecutionFailure requires an exception type")
        message = message or exception_type
        super().__init__(f"{stage}: {exception_type}: {message}")
        self.stage = stage
        self.exception_type = exception_type
        self.stack = stack or f"{stage}: {exception_type}: {message}\n"
        self.native_signal = native_signal
        self.exit_code = exit_code
        self.timeout_phase = timeout_phase


@dataclass(frozen=True)
class PairedExecutionResult:
    semantic: SemanticObservation
    framework_version: str
    device: str
    seed: int
    backend_reset_performed: bool
    decision: DecisionObservation | None = None


@dataclass(frozen=True)
class _RngSnapshot:
    torch_cpu: Any
    torch_cuda: tuple[Any, ...]
    python: object
    numpy: object | None


@dataclass(frozen=True)
class _GlobalSnapshot:
    grad_enabled: bool
    deterministic_algorithms: bool
    autocast_cpu: bool | None
    autocast_cuda: bool | None


@dataclass(frozen=True)
class _PhaseResult:
    side: SemanticSide
    output: Any


def _torch():
    try:
        import torch
    except ImportError as error:
        raise UnsupportedIsolation("PyTorch is required for paired execution") from error
    return torch


def _numpy():
    try:
        import numpy
    except ImportError:
        return None
    return numpy


def _walk(value: Any, path: str = "$"):
    torch = _torch()
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, tuple):
        for index, item in enumerate(value):
            yield from _walk(item, f"{path}[{index}]")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from _walk(value[key], f"{path}.{key}")


def _tree_spec(value: Any) -> str:
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return "tensor"
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, tuple):
        return canonical_json(
            {"tuple": [_tree_spec(item) for item in value]}
        )
    if isinstance(value, list):
        return canonical_json(
            {"list": [_tree_spec(item) for item in value]}
        )
    if isinstance(value, dict):
        return canonical_json(
            {
                "dict": {
                    str(key): _tree_spec(value[key])
                    for key in sorted(value)
                }
            }
        )
    raise UnsupportedIsolation(
        f"unsupported runtime pytree leaf: {type(value).__name__}"
    )


def _tensor_bytes(tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    if value.numel() == 0:
        return b""
    return value.reshape(-1).view(_torch().uint8).numpy().tobytes()


def _tensor_digest(tensor) -> str:
    return f"sha256:{hashlib.sha256(_tensor_bytes(tensor)).hexdigest()}"


def _storage_key(tensor) -> int:
    return int(tensor.untyped_storage()._cdata)


def _alias_summary(value: Any) -> dict[str, Any]:
    groups: dict[int, list[str]] = {}
    layouts: dict[str, dict[str, Any]] = {}
    for path, tensor in _walk(value):
        groups.setdefault(_storage_key(tensor), []).append(path)
        layouts[path] = {
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "storage_offset": tensor.storage_offset(),
        }
    normalized_groups = sorted(
        sorted(paths) for paths in groups.values() if len(paths) > 1
    )
    return {
        "alias_groups": normalized_groups,
        "layouts": layouts,
    }


def _clone_input(value: Any) -> Any:
    before_spec = _tree_spec(value)
    before_alias = _alias_summary(value)
    try:
        cloned = copy.deepcopy(value)
    except Exception as error:
        raise UnsupportedIsolation(
            f"input deep clone failed: {type(error).__name__}: {error}"
        ) from error
    if _tree_spec(cloned) != before_spec or _alias_summary(cloned) != before_alias:
        raise UnsupportedIsolation(
            "input clone did not preserve pytree/layout/alias structure"
        )
    return cloned


def _snapshot_rng() -> _RngSnapshot:
    torch = _torch()
    cuda_states = (
        tuple(torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else ()
    )
    numpy = _numpy()
    return _RngSnapshot(
        torch_cpu=torch.get_rng_state().clone(),
        torch_cuda=cuda_states,
        python=random.getstate(),
        numpy=numpy.random.get_state() if numpy is not None else None,
    )


def _restore_rng(snapshot: _RngSnapshot) -> None:
    torch = _torch()
    torch.set_rng_state(snapshot.torch_cpu)
    if snapshot.torch_cuda:
        torch.cuda.set_rng_state_all(list(snapshot.torch_cuda))
    random.setstate(snapshot.python)
    numpy = _numpy()
    if numpy is not None and snapshot.numpy is not None:
        numpy.random.set_state(snapshot.numpy)


def _autocast_enabled(device: str) -> bool | None:
    torch = _torch()
    try:
        return bool(torch.is_autocast_enabled(device))
    except (TypeError, RuntimeError):
        return None


def _set_autocast(device: str, enabled: bool | None) -> None:
    if enabled is None:
        return
    torch = _torch()
    try:
        torch.set_autocast_enabled(device, enabled)
    except (TypeError, RuntimeError):
        pass


def _snapshot_global() -> _GlobalSnapshot:
    torch = _torch()
    return _GlobalSnapshot(
        grad_enabled=torch.is_grad_enabled(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        autocast_cpu=_autocast_enabled("cpu"),
        autocast_cuda=_autocast_enabled("cuda"),
    )


def _restore_global(snapshot: _GlobalSnapshot, *, force_deterministic: bool) -> None:
    torch = _torch()
    torch.set_grad_enabled(snapshot.grad_enabled)
    torch.use_deterministic_algorithms(
        True if force_deterministic else snapshot.deterministic_algorithms
    )
    _set_autocast("cpu", snapshot.autocast_cpu)
    _set_autocast("cuda", snapshot.autocast_cuda)


def _state_digest(program: Any) -> dict[str, str]:
    if not hasattr(program, "state_dict"):
        return {}
    state = program.state_dict()
    return {
        name: _tensor_digest(value)
        for name, value in sorted(state.items())
        if hasattr(value, "untyped_storage")
    }


def _mutation_snapshot(program: Any, inputs: Any) -> dict[str, Any]:
    return {
        "input_values": {
            path: _tensor_digest(tensor) for path, tensor in _walk(inputs)
        },
        "input_versions": {
            path: int(tensor._version) for path, tensor in _walk(inputs)
        },
        "module_state": _state_digest(program),
    }


def _mutation_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    changed_inputs = sorted(
        path
        for path, digest in before["input_values"].items()
        if after["input_values"].get(path) != digest
    )
    changed_state = sorted(
        name
        for name, digest in before["module_state"].items()
        if after["module_state"].get(name) != digest
    )
    return {
        "mutated_inputs": changed_inputs,
        "mutated_module_state": changed_state,
        "input_version_delta": {
            path: after["input_versions"].get(path, 0) - version
            for path, version in before["input_versions"].items()
        },
    }


def _output_alias_summary(output: Any, inputs: Any) -> dict[str, Any]:
    input_storage = {
        _storage_key(tensor): path for path, tensor in _walk(inputs)
    }
    aliases = []
    for output_path, tensor in _walk(output):
        input_path = input_storage.get(_storage_key(tensor))
        if input_path is not None:
            aliases.append(
                {"output": output_path, "input": input_path}
            )
    return {
        "output_aliases_input": sorted(
            aliases,
            key=lambda item: (item["output"], item["input"]),
        ),
        "input_alias": _alias_summary(inputs),
    }


def _side(program: Any, inputs: Any, output: Any, before: dict[str, Any]) -> SemanticSide:
    after = _mutation_snapshot(program, inputs)
    summaries = tuple(
        TensorSummary(
            shape=tuple(int(dim) for dim in tensor.shape),
            dtype=str(tensor.dtype).removeprefix("torch."),
            stride=tuple(int(item) for item in tensor.stride()),
            value_digest=_tensor_digest(tensor),
            nan_count=(
                int(_torch().isnan(tensor).sum().item())
                if tensor.is_floating_point() or tensor.is_complex()
                else 0
            ),
            inf_count=(
                int(_torch().isinf(tensor).sum().item())
                if tensor.is_floating_point() or tensor.is_complex()
                else 0
            ),
        )
        for _, tensor in _walk(output)
    )
    return SemanticSide(
        output_tree_spec=_tree_spec(output),
        tensor_summaries=summaries,
        mutation_summary=_mutation_diff(before, after),
        alias_summary=_output_alias_summary(output, inputs),
    )


def _flatten_non_tensors(value: Any) -> list[Any]:
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return []
    if isinstance(value, (tuple, list)):
        return [
            leaf
            for item in value
            for leaf in _flatten_non_tensors(item)
        ]
    if isinstance(value, dict):
        return [
            leaf
            for key in sorted(value)
            for leaf in _flatten_non_tensors(value[key])
        ]
    return [value]


def _compare(
    left: _PhaseResult,
    right: _PhaseResult,
    *,
    rtol: float,
    atol: float,
) -> PairwiseCompare:
    torch = _torch()
    left_tensors = [tensor for _, tensor in _walk(left.output)]
    right_tensors = [tensor for _, tensor in _walk(right.output)]
    structure_equal = left.side.output_tree_spec == right.side.output_tree_spec
    allclose = (
        structure_equal
        and len(left_tensors) == len(right_tensors)
        and _flatten_non_tensors(left.output) == _flatten_non_tensors(right.output)
    )
    max_abs = 0.0
    max_rel = 0.0
    mismatches = 0
    elements = 0
    for lhs, rhs in zip(left_tensors, right_tensors, strict=False):
        if lhs.shape != rhs.shape:
            allclose = False
            continue
        lhs_value = lhs.detach().to(dtype=torch.float64, device="cpu")
        rhs_value = rhs.detach().to(dtype=torch.float64, device="cpu")
        diff = (lhs_value - rhs_value).abs()
        if diff.numel():
            max_abs = max(max_abs, float(diff.max().item()))
            denom = lhs_value.abs().clamp_min(atol)
            max_rel = max(
                max_rel,
                float((diff / denom).max().item()),
            )
            close = torch.isclose(
                lhs_value,
                rhs_value,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )
            mismatches += int((~close).sum().item())
            elements += close.numel()
            allclose = allclose and bool(close.all().item())
    mutation_equal = left.side.mutation_summary == right.side.mutation_summary
    alias_equal = left.side.alias_summary == right.side.alias_summary
    return PairwiseCompare(
        allclose=allclose and structure_equal,
        max_abs_err=max_abs,
        max_rel_err=max_rel,
        rtol_used=rtol,
        atol_used=atol,
        mismatch_ratio=(mismatches / elements if elements else 0.0),
        structure_equal=structure_equal,
        mutation_equal=mutation_equal,
        alias_equal=alias_equal,
    )


def _invoke(callable_program: Callable[..., Any], inputs: Any) -> Any:
    if isinstance(inputs, tuple):
        return callable_program(*inputs)
    if isinstance(inputs, dict):
        return callable_program(**inputs)
    return callable_program(inputs)


class _ModuleController:
    def __init__(self, factory: Callable[[], Any], policy: ModulePolicy):
        self.factory = factory
        self.policy = policy
        self.base = None
        self.state = None
        if policy is not ModulePolicy.RECONSTRUCT_TWICE:
            self.base = factory()
        if policy is ModulePolicy.STATE_DICT_RESTORE:
            if not hasattr(self.base, "state_dict") or not hasattr(
                self.base, "load_state_dict"
            ):
                raise UnsupportedIsolation(
                    "state_dict_restore requires state_dict/load_state_dict"
                )
            self.state = copy.deepcopy(self.base.state_dict())

    def prepare(self) -> Any:
        if self.policy is ModulePolicy.RECONSTRUCT_TWICE:
            return self.factory()
        if self.policy is ModulePolicy.DEEPCOPY:
            try:
                return copy.deepcopy(self.base)
            except Exception as error:
                raise UnsupportedIsolation(
                    f"module deepcopy failed: {type(error).__name__}: {error}"
                ) from error
        self.base.load_state_dict(copy.deepcopy(self.state))
        return self.base


class TorchPairedExecutor:
    def __init__(
        self,
        *,
        compiler: Callable[[Any, Mapping[str, Any]], Callable[..., Any]],
        backend_reset: Callable[[], None] | None = None,
        decision_probe: DecisionProbe | None = None,
    ):
        self.compiler = compiler
        self.backend_reset = backend_reset
        self.decision_probe = decision_probe

    def execute(
        self,
        *,
        run: RunSpec,
        program_factory: Callable[[], Any],
        materialized_input: Any,
        seed: int,
        rtol: float,
        atol: float,
        phase_callback: Callable[[str], None] | None = None,
    ) -> PairedExecutionResult:
        def phase(name: str) -> None:
            if phase_callback is not None:
                phase_callback(name)

        phase("initialization")
        if run.run_kind is not RunKind.PAIRED_SEMANTIC:
            raise UnsupportedIsolation("paired executor requires paired_semantic RunSpec")
        policy = run.paired_semantic
        if policy is None:
            raise UnsupportedIsolation("paired RunSpec has no paired policy")

        torch = _torch()
        input_tensors = list(_walk(materialized_input))
        device_types = {tensor.device.type for _, tensor in input_tensors}
        if len(device_types) > 1:
            raise UnsupportedIsolation("paired inputs span multiple device classes")
        actual_device_class = next(iter(device_types), "cpu")
        if actual_device_class != run.device_class:
            raise UnsupportedIsolation(
                f"RunSpec device_class={run.device_class!r} does not match "
                f"materialized input device class={actual_device_class!r}"
            )
        global_snapshot = _snapshot_global()
        rng_snapshot = _snapshot_rng()
        try:
            controller = _ModuleController(
                program_factory,
                policy.state_isolation.module_policy,
            )
            if (
                policy.state_isolation.module_policy
                is not ModulePolicy.RECONSTRUCT_TWICE
            ):
                rng_snapshot = _snapshot_rng()

            def prepare() -> tuple[Any, Any]:
                _restore_rng(rng_snapshot)
                _restore_global(global_snapshot, force_deterministic=True)
                program = controller.prepare()
                inputs = _clone_input(materialized_input)
                return program, inputs

            def reference(stage: str) -> _PhaseResult:
                phase(stage)
                program, inputs = prepare()
                before = _mutation_snapshot(program, inputs)
                try:
                    output = _invoke(program, inputs)
                except Exception as error:
                    raise ExecutionFailure(stage, error) from error
                phase("probe")
                try:
                    return _PhaseResult(
                        side=_side(program, inputs, output, before),
                        output=output,
                    )
                except Exception as error:
                    raise ExecutionFailure("probe", error) from error

            first_reference = reference("eager")
            candidate_program, candidate_inputs = prepare()
            if self.decision_probe is not None:
                phase("pre_trigger_probe")
                try:
                    self.decision_probe.before_compile(
                        candidate_program,
                        candidate_inputs,
                    )
                except Exception as error:
                    raise ExecutionFailure("probe", error) from error
            reset_performed = False
            if (
                policy.state_isolation.backend_cache.dynamo_reset
                and self.backend_reset is not None
            ):
                phase("backend_reset")
                try:
                    self.backend_reset()
                except Exception as error:
                    raise ExecutionFailure("backend_reset", error) from error
                reset_performed = True
            phase("compile")
            try:
                compiled = self.compiler(
                    candidate_program,
                    run.compile_config,
                )
            except Exception as error:
                raise ExecutionFailure("compile", error) from error
            candidate_before = _mutation_snapshot(
                candidate_program,
                candidate_inputs,
            )
            phase("runtime")
            try:
                candidate_output = _invoke(compiled, candidate_inputs)
            except Exception as error:
                raise ExecutionFailure("runtime", error) from error
            decision = None
            if self.decision_probe is not None:
                phase("effect_probe")
                try:
                    decision = self.decision_probe.after_runtime(
                        candidate_program,
                        candidate_inputs,
                        candidate_output,
                    )
                except Exception as error:
                    raise ExecutionFailure("probe", error) from error
            phase("probe")
            try:
                candidate = _PhaseResult(
                    side=_side(
                        candidate_program,
                        candidate_inputs,
                        candidate_output,
                        candidate_before,
                    ),
                    output=candidate_output,
                )
            except Exception as error:
                raise ExecutionFailure("probe", error) from error

            stable = True
            for _ in range(policy.state_isolation.reference_replay - 1):
                replay = reference("reference_replay")
                phase("compare")
                replay_compare = _compare(
                    first_reference,
                    replay,
                    rtol=rtol,
                    atol=atol,
                )
                stable = stable and all(
                    (
                        replay_compare.allclose,
                        replay_compare.structure_equal,
                        replay_compare.mutation_equal,
                        replay_compare.alias_equal,
                    )
                )
            phase("compare")
            pairwise = _compare(
                first_reference,
                candidate,
                rtol=rtol,
                atol=atol,
            )
            semantic = SemanticObservation(
                reference=first_reference.side,
                candidate=candidate.side,
                pairwise_compare=pairwise,
                stability=(
                    SemanticStability.STABLE
                    if stable
                    else SemanticStability.REFERENCE_UNSTABLE
                ),
            )
            device = (
                str(input_tensors[0][1].device)
                if input_tensors
                else "cpu"
            )
            return PairedExecutionResult(
                semantic=semantic,
                framework_version=torch.__version__,
                device=device,
                seed=seed,
                backend_reset_performed=reset_performed,
                decision=decision,
            )
        finally:
            _restore_rng(rng_snapshot)
            _restore_global(global_snapshot, force_deterministic=False)
