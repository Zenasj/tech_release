"""Structured LLM adapters for property mining and test generation.

The OpenAI-compatible adapter reads its endpoint, model, and optional API key
from environment variables. An offline fixture provider supports unit tests.
"""
from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable

from mrminer.llm.provider import (
    FixtureLlmProvider,
    ProviderResult,
    StructuredLlmProvider,
    StructuredLlmRequest,
)


def _balanced_end(text: str, start: int) -> int:
    """Index of the `}` closing the object at `start`, or -1 if truncated. String-aware."""
    depth = 0
    in_str = esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _repair_truncated(frag: str) -> str | None:
    """Salvage a JSON object that was cut off at max_tokens: trim back to the last
    complete token boundary and close the brackets that were open there. Recovers the
    already-complete tests from a `{"tests":[{..},{..},{..<cut>` reply. String-aware."""
    stack: list[str] = []
    in_str = esc = False
    safe = -1
    safe_stack: list[str] | None = None
    for i, c in enumerate(frag):
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c in "{[":
            stack.append("}" if c == "{" else "]")
        elif c in "}]":
            if stack:
                stack.pop()
            safe, safe_stack = i, list(stack)        # cut only at a complete container
    if safe < 0 or safe_stack is None:
        return None
    return frag[: safe + 1] + "".join(reversed(safe_stack))


def _extract_json(text: str) -> dict:
    """Pull the JSON object out of a chat reply, tolerating a QwQ <think> block,
    ```json fences, and truncation at max_tokens."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]   # keep only the post-reasoning tail
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found in LLM reply")
    end = _balanced_end(text, start)
    if end > start:                              # a complete object is present
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    repaired = _repair_truncated(text[start:])   # truncated -> salvage complete elements
    if repaired:
        try:
            return json.loads(repaired)
        except Exception:
            pass
    raise ValueError("no JSON object found in LLM reply")


class LocalOpenAIProvider:
    """Minimal OpenAI-compatible chat provider for a local vLLM server.

    Bypasses the machine HTTP proxy (localhost must not be proxied) and strips a
    reasoning-model <think> block before parsing the JSON payload.
    """

    # hard ceiling for the output cap: keep prompt+output under the server's
    # max_model_len (32768) with headroom for even the large real-model prompts.
    _MAX_TOKENS_CEILING = 22528

    def __init__(
        self,
        *,
        base_url: str = "",
        model: str = "",
        max_tokens: int = 12288,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.0,
        timeout: int = 1200,
        max_retries: int = 2,
        api_key: str | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("MRMINER_API_BASE", "")).rstrip("/")
        self.model = model or os.environ.get("MRMINER_MODEL", "")
        if not self.base_url or not self.model:
            raise ValueError("set MRMINER_API_BASE and MRMINER_MODEL")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_key = api_key if api_key is not None else os.environ.get("MRMINER_API_KEY", "")
        self.use_vllm_extensions = os.environ.get("MRMINER_VLLM_EXTENSIONS", "0") == "1"
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _post(self, body: dict) -> tuple[str, str | None]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
        )
        with self._opener.open(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read())
        return payload["choices"][0]["message"]["content"], payload.get("id")

    def complete(self, request: StructuredLlmRequest) -> ProviderResult:
        base_user = (
            json.dumps(request.input_payload, ensure_ascii=False)
            + "\n\nReturn ONLY a JSON object conforming to this schema "
            "(no prose, no markdown fences):\n"
            + json.dumps(request.output_schema)
        )
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            # escalate the output cap each retry; a truncated reply re-asked at the same
            # cap would just truncate again.
            mt = min(self.max_tokens + attempt * 5120, self._MAX_TOKENS_CEILING)
            user = base_user
            if attempt:
                # nudge QwQ to spend fewer tokens on reasoning and emit the JSON.
                user += (
                    "\n\nThe previous reply did not contain a complete JSON object. "
                    "Keep any reasoning brief and output the JSON object only."
                )
            body = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": user},
                ],
                "temperature": self.temperature,
                "top_p": self.top_p,
                # top_k / min_p are vLLM sampling extensions (not stock OpenAI); the raw
                # endpoint accepts them as top-level fields. Keep them with top_p per the
                # QwQ-32B recommendation.
                "max_tokens": mt,
            }
            if self.use_vllm_extensions:
                body["top_k"] = self.top_k
                body["min_p"] = self.min_p
            try:
                content, req_id = self._post(body)
                return ProviderResult(
                    payload=_extract_json(content),
                    model=self.model,
                    request_id=req_id,
                )
            except Exception as e:  # noqa: BLE001 - re-ask on both transport and parse failures
                last_err = e
                continue
        raise ValueError(f"no JSON object found in LLM reply (after {self.max_retries + 1} attempts): {last_err}")


def provider_from_env(fixture_handler: Callable | None = None) -> StructuredLlmProvider:
    kind = os.environ.get("MRMINER_FUZZ_LLM", "fixture").lower()
    if kind in {"qwq", "openai"}:
        return LocalOpenAIProvider(
            base_url=os.environ.get("MRMINER_API_BASE", ""),
            model=os.environ.get("MRMINER_MODEL", ""),
        )
    if kind == "deepseek":
        from mrminer.llm import DeepSeekConfig, DeepSeekProvider

        model = os.environ.get("MRMINER_MODEL", "")
        return DeepSeekProvider(DeepSeekConfig(model=model))
    return FixtureLlmProvider(handler=fixture_handler)
