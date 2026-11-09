"""Single, auditable boundary for structured LLM calls.

This module intentionally contains no SDK imports, token loading, retries, or network
policy. A concrete provider can be added later without changing claim extraction or
oracle induction code.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from jsonschema import exceptions as jsonschema_exceptions
from jsonschema.validators import validator_for

from mrminer.common.canonical import canonical_bytes, sha256_digest

JsonObject = dict[str, Any]
FixtureHandler = Callable[["StructuredLlmRequest"], Mapping[str, Any]]


class LlmNotConfigured(RuntimeError):
    """Raised when a workflow reaches the intentionally unconfigured LLM boundary."""


class LlmOutputValidationError(ValueError):
    """Raised when a provider payload violates the caller's output schema."""


@dataclass(frozen=True, kw_only=True)
class StructuredLlmRequest:
    """Canonical request passed to every concrete provider."""

    task: str
    system_prompt: str
    input_payload: JsonObject
    output_schema: JsonObject
    request_version: str = "structured_llm_request_v1"

    def __post_init__(self) -> None:
        if not self.task or not self.system_prompt or not self.request_version:
            raise ValueError("task, system_prompt, and request_version must be non-empty")
        canonical_bytes(self.input_payload)
        canonical_bytes(self.output_schema)

    def to_wire_payload(self) -> JsonObject:
        return {
            "request_version": self.request_version,
            "task": self.task,
            "system_prompt": self.system_prompt,
            "input_payload": copy.deepcopy(self.input_payload),
            "output_schema": copy.deepcopy(self.output_schema),
        }

    @property
    def prompt_sha(self) -> str:
        return sha256_digest(self.to_wire_payload())


@dataclass(frozen=True, kw_only=True)
class ProviderResult:
    """Raw structured result returned by a provider adapter."""

    payload: JsonObject
    model: str
    request_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class StructuredLlmResult:
    """Validated result plus the audit identity computed by this boundary."""

    payload: JsonObject
    model: str
    prompt_sha: str
    request_id: str | None


@runtime_checkable
class StructuredLlmProvider(Protocol):
    def complete(self, request: StructuredLlmRequest) -> ProviderResult:
        """Return one JSON object conforming to the supplied output schema."""


class UnconfiguredLlmProvider:
    def complete(self, request: StructuredLlmRequest) -> ProviderResult:
        raise LlmNotConfigured(
            f"no LLM provider configured for task {request.task!r}; "
            "inject a StructuredLlmProvider"
        )


@dataclass
class FixtureLlmProvider:
    """Deterministic in-memory provider for offline tests and golden fixtures."""

    responses: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    handler: FixtureHandler | None = None
    model: str = "fixture-llm"
    requests: list[StructuredLlmRequest] = field(default_factory=list, init=False)

    def complete(self, request: StructuredLlmRequest) -> ProviderResult:
        self.requests.append(copy.deepcopy(request))
        if self.handler is not None:
            payload = self.handler(request)
        else:
            if request.task not in self.responses:
                raise KeyError(f"no fixture response for task {request.task!r}")
            payload = self.responses[request.task]
        if not isinstance(payload, Mapping):
            raise TypeError("fixture LLM response must be a JSON object")
        copied = copy.deepcopy(dict(payload))
        return ProviderResult(
            payload=copied,
            model=self.model,
            request_id=f"fixture:{len(self.requests)}",
        )


def invoke_structured(
    provider: StructuredLlmProvider | None,
    *,
    task: str,
    system_prompt: str,
    input_payload: Mapping[str, Any],
    output_schema: Mapping[str, Any],
    request_version: str = "structured_llm_request_v1",
) -> StructuredLlmResult:
    """Invoke a provider and attach a stable prompt identity.

    Callers must pass an explicit provider. Passing ``None`` fails closed and never
    attempts to discover credentials or a network service.
    """

    request = StructuredLlmRequest(
        task=task,
        system_prompt=system_prompt,
        input_payload=copy.deepcopy(dict(input_payload)),
        output_schema=copy.deepcopy(dict(output_schema)),
        request_version=request_version,
    )
    raw = (provider or UnconfiguredLlmProvider()).complete(request)
    if not isinstance(raw, ProviderResult):
        raise TypeError("LLM provider must return ProviderResult")
    if type(raw.payload) is not dict:
        raise TypeError("LLM provider payload must be a JSON object")
    if not raw.model:
        raise ValueError("LLM provider model identifier must be non-empty")
    canonical_bytes(raw.payload)
    try:
        validator_type = validator_for(request.output_schema)
        validator_type.check_schema(request.output_schema)
        validator_type(request.output_schema).validate(raw.payload)
    except jsonschema_exceptions.SchemaError as error:
        raise ValueError(f"invalid caller output schema: {error.message}") from error
    except jsonschema_exceptions.ValidationError as error:
        location = ".".join(str(item) for item in error.absolute_path) or "<root>"
        raise LlmOutputValidationError(
            f"LLM payload violates output schema at {location}: {error.message}"
        ) from error
    return StructuredLlmResult(
        payload=copy.deepcopy(raw.payload),
        model=raw.model,
        prompt_sha=request.prompt_sha,
        request_id=raw.request_id,
    )
