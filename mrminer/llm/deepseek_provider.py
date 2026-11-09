"""Audited DeepSeek adapter for the structured LLM boundary."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlparse

import requests
from jsonschema import exceptions as jsonschema_exceptions
from jsonschema.validators import validator_for

from mrminer.llm.provider import (
    LlmNotConfigured,
    ProviderResult,
    StructuredLlmRequest,
)

DEFAULT_DEEPSEEK_BASE_URL = ""
DEFAULT_DEEPSEEK_MODEL = ""


class DeepSeekProviderError(RuntimeError):
    """A request failed without exposing credentials or source text."""


class _RetryableDeepSeekError(DeepSeekProviderError):
    pass


@dataclass(frozen=True, kw_only=True)
class DeepSeekConfig:
    api_key: str | None = field(default=None, repr=False)
    base_url: str = field(default_factory=lambda: os.environ.get("MRMINER_API_BASE", ""))
    model: str = field(default_factory=lambda: os.environ.get("MRMINER_MODEL", ""))
    connect_timeout_s: float = 10.0
    read_timeout_s: float = 120.0
    max_retries: int = 2
    retry_backoff_s: float = 0.5
    max_tokens: int = 8_192
    temperature: float = 0.0
    thinking: bool = False

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("DeepSeek base_url must be an HTTP(S) URL")
        if not self.model:
            raise ValueError("DeepSeek model must be non-empty")
        if self.connect_timeout_s <= 0 or self.read_timeout_s <= 0:
            raise ValueError("DeepSeek timeouts must be positive")
        if self.max_retries < 0 or self.retry_backoff_s < 0:
            raise ValueError("DeepSeek retry settings must be non-negative")
        if self.max_tokens <= 0:
            raise ValueError("DeepSeek max_tokens must be positive")
        if not 0 <= self.temperature <= 2:
            raise ValueError("DeepSeek temperature must be between 0 and 2")


@dataclass(frozen=True, kw_only=True)
class DeepSeekCallAudit:
    prompt_sha: str
    status: str
    request_id: str | None
    model: str
    attempts: int
    duration_s: float
    usage: dict[str, int]
    error_kind: str | None

    def to_record(self) -> dict[str, Any]:
        return {
            "prompt_sha": self.prompt_sha,
            "status": self.status,
            "request_id": self.request_id,
            "model": self.model,
            "attempts": self.attempts,
            "duration_s": self.duration_s,
            "usage": dict(sorted(self.usage.items())),
            "error_kind": self.error_kind,
        }


def resolve_deepseek_api_key(explicit: str | None = None) -> str:
    key = (
        explicit.strip()
        if isinstance(explicit, str) and explicit.strip()
        else (os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("MRMINER_API_KEY", "")).strip()
    )
    if not key:
        raise LlmNotConfigured(
            "API key is missing; set DEEPSEEK_API_KEY or MRMINER_API_KEY"
        )
    return key


def _usage(record: object) -> dict[str, int]:
    if type(record) is not dict:
        return {}
    return {
        str(key): value
        for key, value in record.items()
        if type(value) is int and value >= 0
    }


class DeepSeekProvider:
    """OpenAI-compatible DeepSeek JSON provider with bounded retries."""

    def __init__(
        self,
        config: DeepSeekConfig | None = None,
        *,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.config = config or DeepSeekConfig()
        self._api_key = resolve_deepseek_api_key(self.config.api_key)
        self._session = session or requests.Session()
        self._sleep = sleep
        self.call_audits: list[DeepSeekCallAudit] = []

    @property
    def endpoint(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/chat/completions"

    def _request_body(self, request: StructuredLlmRequest) -> dict[str, Any]:
        user_payload = {
            "task": request.task,
            "input_payload": request.input_payload,
            "output_schema": request.output_schema,
        }
        body = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        request.system_prompt.rstrip()
                        + "\nReturn exactly one valid JSON object conforming to "
                        "the supplied JSON schema. Do not use Markdown fences."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        user_payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
            "thinking": {
                "type": "enabled" if self.config.thinking else "disabled"
            },
            "stream": False,
            "max_tokens": self.config.max_tokens,
        }
        if not self.config.thinking:
            body["temperature"] = self.config.temperature
        return body

    def _validate_model_payload(
        self,
        payload: dict[str, Any],
        output_schema: dict[str, Any],
    ) -> None:
        validator_type = validator_for(output_schema)
        try:
            validator_type.check_schema(output_schema)
        except jsonschema_exceptions.SchemaError as error:
            raise ValueError(
                f"invalid caller output schema: {error.message}"
            ) from error
        try:
            validator_type(output_schema).validate(payload)
        except jsonschema_exceptions.ValidationError as error:
            location = ".".join(
                str(item) for item in error.absolute_path
            ) or "<root>"
            raise _RetryableDeepSeekError(
                f"DeepSeek response violates output schema at {location}"
            ) from error

    def _call_once(
        self,
        request: StructuredLlmRequest,
    ) -> tuple[ProviderResult, dict[str, int]]:
        try:
            response = self._session.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=self._request_body(request),
                timeout=(
                    self.config.connect_timeout_s,
                    self.config.read_timeout_s,
                ),
            )
        except (requests.Timeout, requests.ConnectionError) as error:
            raise _RetryableDeepSeekError(type(error).__name__) from error
        if response.status_code == 429 or response.status_code >= 500:
            raise _RetryableDeepSeekError(
                f"DeepSeek API returned HTTP {response.status_code}"
            )
        if response.status_code >= 400:
            raise DeepSeekProviderError(
                f"DeepSeek API returned HTTP {response.status_code}"
            )
        try:
            response_payload = response.json()
        except ValueError as error:
            raise _RetryableDeepSeekError(
                "DeepSeek API returned a non-JSON response"
            ) from error
        if type(response_payload) is not dict:
            raise _RetryableDeepSeekError(
                "DeepSeek API response must be a JSON object"
            )
        choices = response_payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise _RetryableDeepSeekError("DeepSeek response has no choices")
        choice = choices[0]
        if type(choice) is not dict:
            raise _RetryableDeepSeekError("DeepSeek choice must be an object")
        if choice.get("finish_reason") == "length":
            raise DeepSeekProviderError(
                "DeepSeek response was truncated at max_tokens"
            )
        message = choice.get("message")
        content = message.get("content") if type(message) is dict else None
        if not isinstance(content, str) or not content.strip():
            raise _RetryableDeepSeekError("DeepSeek response content is empty")
        try:
            model_payload = json.loads(content)
        except json.JSONDecodeError as error:
            raise _RetryableDeepSeekError(
                "DeepSeek content is not valid JSON"
            ) from error
        if type(model_payload) is not dict:
            raise _RetryableDeepSeekError(
                "DeepSeek content must decode to one JSON object"
            )
        self._validate_model_payload(model_payload, request.output_schema)
        model = response_payload.get("model")
        request_id = response_payload.get("id")
        return (
            ProviderResult(
                payload=model_payload,
                model=(model if isinstance(model, str) and model else self.config.model),
                request_id=(request_id if isinstance(request_id, str) else None),
            ),
            _usage(response_payload.get("usage")),
        )

    def complete(self, request: StructuredLlmRequest) -> ProviderResult:
        validator_type = validator_for(request.output_schema)
        try:
            validator_type.check_schema(request.output_schema)
        except jsonschema_exceptions.SchemaError as error:
            raise ValueError(
                f"invalid caller output schema: {error.message}"
            ) from error
        started = time.monotonic()
        attempts = 0
        last_error: BaseException | None = None
        for attempt in range(self.config.max_retries + 1):
            attempts = attempt + 1
            try:
                result, usage = self._call_once(request)
            except _RetryableDeepSeekError as error:
                last_error = error
                if attempt >= self.config.max_retries:
                    break
                self._sleep(self.config.retry_backoff_s * (2**attempt))
                continue
            except DeepSeekProviderError as error:
                last_error = error
                break
            self.call_audits.append(
                DeepSeekCallAudit(
                    prompt_sha=request.prompt_sha,
                    status="ok",
                    request_id=result.request_id,
                    model=result.model,
                    attempts=attempts,
                    duration_s=round(time.monotonic() - started, 6),
                    usage=usage,
                    error_kind=None,
                )
            )
            return result
        assert last_error is not None
        self.call_audits.append(
            DeepSeekCallAudit(
                prompt_sha=request.prompt_sha,
                status="failed",
                request_id=None,
                model=self.config.model,
                attempts=attempts,
                duration_s=round(time.monotonic() - started, 6),
                usage={},
                error_kind=type(last_error).__name__,
            )
        )
        raise DeepSeekProviderError(str(last_error)) from last_error
