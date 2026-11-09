"""M7.2: validate a (program, input) pair on a target environment.

Materializes an InputManifest, reconstructs and runs the program forward twice,
and only records a pair as validated when both runs succeed and produce an
identical output digest. Non-determinism, execution failure, or an unavailable
device fail closed with a typed reason, so induction is only ever fed pairs
that actually run reproducibly (open questions doc S3 item 3).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from mrminer.common.schema import InputManifest
from mrminer.s6_runner.input_materializer import (
    TorchInputMaterializer,
    UnsupportedInputMaterialization,
)

# A program factory reconstructs the runnable each call (fresh module/state),
# matching the paired executor's reconstruct semantics.
ProgramFactory = Callable[[], Callable[..., Any]]


class ProgramInputStatus(str, Enum):
    VALIDATED = "validated"
    DEVICE_UNAVAILABLE = "device_unavailable"
    MATERIALIZATION_FAILED = "materialization_failed"
    EXECUTION_FAILED = "execution_failed"
    NONDETERMINISTIC = "nondeterministic"


@dataclass(frozen=True, kw_only=True)
class ValidatedProgramInput:
    program_id: str
    input_id: str
    device: str
    status: ProgramInputStatus
    output_digest: str | None = None
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is ProgramInputStatus.VALIDATED

    def to_record(self) -> dict:
        return {
            "program_id": self.program_id,
            "input_id": self.input_id,
            "device": self.device,
            "status": self.status.value,
            "output_digest": self.output_digest,
            "reason": self.reason,
        }

    @classmethod
    def from_record(cls, record: dict) -> "ValidatedProgramInput":
        return cls(
            program_id=record["program_id"],
            input_id=record["input_id"],
            device=record["device"],
            status=ProgramInputStatus(record["status"]),
            output_digest=record.get("output_digest"),
            reason=record.get("reason"),
        )


def _cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def _digest_output(value: Any) -> str:
    import torch

    hasher = hashlib.sha256()

    def walk(node: Any) -> None:
        if isinstance(node, torch.Tensor):
            tensor = node.detach().to("cpu").contiguous()
            hasher.update(b"T")
            hasher.update(str(tuple(tensor.shape)).encode())
            hasher.update(str(tensor.dtype).encode())
            hasher.update(tensor.numpy().tobytes())
        elif isinstance(node, (tuple, list)):
            hasher.update(b"S")
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            hasher.update(b"D")
            for key in sorted(node):
                hasher.update(str(key).encode())
                walk(node[key])
        elif node is None:
            hasher.update(b"N")
        else:
            hasher.update(b"V")
            hasher.update(repr(node).encode())

    walk(value)
    return "sha256:" + hasher.hexdigest()


def _run_once(
    manifest: InputManifest,
    program_factory: ProgramFactory,
    device: str,
) -> tuple[str | None, ProgramInputStatus, str | None]:
    try:
        inputs = TorchInputMaterializer(manifest, device=device)()
    except UnsupportedInputMaterialization as error:
        return None, ProgramInputStatus.MATERIALIZATION_FAILED, str(error)
    try:
        program = program_factory()
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        output = program(*inputs)
    except Exception as error:  # noqa: BLE001 - typed failure surface
        return None, ProgramInputStatus.EXECUTION_FAILED, (
            f"{type(error).__name__}: {error}"
        )
    return _digest_output(output), ProgramInputStatus.VALIDATED, None


def validate_program_input(
    *,
    program_id: str,
    input_id: str,
    manifest: InputManifest,
    program_factory: ProgramFactory,
    device: str,
) -> ValidatedProgramInput:
    def result(
        status: ProgramInputStatus,
        *,
        output_digest: str | None = None,
        reason: str | None = None,
    ) -> ValidatedProgramInput:
        return ValidatedProgramInput(
            program_id=program_id,
            input_id=input_id,
            device=device,
            status=status,
            output_digest=output_digest,
            reason=reason,
        )

    if device.split(":", 1)[0] == "cuda" and not _cuda_available():
        return result(
            ProgramInputStatus.DEVICE_UNAVAILABLE,
            reason="CUDA requested but torch reports it unavailable",
        )

    first_digest, first_status, first_reason = _run_once(
        manifest, program_factory, device
    )
    if first_status is not ProgramInputStatus.VALIDATED:
        return result(first_status, reason=first_reason)

    second_digest, second_status, second_reason = _run_once(
        manifest, program_factory, device
    )
    if second_status is not ProgramInputStatus.VALIDATED:
        return result(second_status, reason=second_reason)

    if first_digest != second_digest:
        return result(
            ProgramInputStatus.NONDETERMINISTIC,
            reason="two reconstructed runs produced different outputs",
        )
    return result(ProgramInputStatus.VALIDATED, output_digest=first_digest)
