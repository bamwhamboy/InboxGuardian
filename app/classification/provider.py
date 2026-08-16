"""LLM provider abstraction for the email classifier.

`LLMClient` is a small protocol so the classifier logic (validation, retry,
guardrail-preserving behavior) can be unit tested with a mocked client that
never touches the network or requires an API key. Two concrete providers are
implemented: `GeminiLLMClient` (the default) and `AnthropicLLMClient` (kept
available, selectable via `INBOXGUARDIAN_LLM_PROVIDER`). Both read their
credentials solely from an environment variable — no API keys are ever
hard-coded, printed, logged, or committed.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

from dotenv import load_dotenv

load_dotenv()

PROVIDER_ENV_VAR = "INBOXGUARDIAN_LLM_PROVIDER"
DEFAULT_PROVIDER = "gemini"
MODEL_ENV_VAR = "INBOXGUARDIAN_LLM_MODEL"
ANTHROPIC_API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"
API_KEY_ENV_VAR = ANTHROPIC_API_KEY_ENV_VAR
ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_MODEL = ANTHROPIC_DEFAULT_MODEL
GEMINI_API_KEY_ENV_VAR = "GEMINI_API_KEY"
GEMINI_DEFAULT_MODEL = "gemini-3.5-flash"
CLASSIFICATION_TOOL_NAME = "emit_classification"


def _classification_schema_object(category_values: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": category_values},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "rationale": {"type": "string"},
        },
        "required": ["category", "confidence", "rationale"],
    }


def _classification_tool_schema(category_values: list[str]) -> dict[str, Any]:
    schema = dict(_classification_schema_object(category_values))
    schema["additionalProperties"] = False
    return {
        "name": CLASSIFICATION_TOOL_NAME,
        "description": "Emit the structured classification result for one email.",
        "input_schema": schema,
    }


class LLMClient(Protocol):
    def classify_raw(
        self, system_prompt: str, user_prompt: str, category_values: list[str]
    ) -> dict[str, Any]: ...


class LLMConfigurationError(RuntimeError):
    """Raised when the LLM provider is missing required configuration."""


class AnthropicLLMClient:
    def __init__(self, model: str | None = None, max_tokens: int = 512) -> None:
        self.model = model or os.environ.get(MODEL_ENV_VAR, ANTHROPIC_DEFAULT_MODEL)
        self.max_tokens = max_tokens
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = os.environ.get(API_KEY_ENV_VAR)
        if not api_key:
            raise LLMConfigurationError(f"Missing {API_KEY_ENV_VAR} environment variable.")
        try:
            import anthropic
        except ImportError as exc:
            raise LLMConfigurationError(
                "The 'anthropic' package is required to use AnthropicLLMClient."
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def classify_raw(
        self, system_prompt: str, user_prompt: str, category_values: list[str]
    ) -> dict[str, Any]:
        response = self._get_client().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            tools=[_classification_tool_schema(category_values)],
            tool_choice={"type": "tool", "name": CLASSIFICATION_TOOL_NAME},
            messages=[{"role": "user", "content": user_prompt}],
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == CLASSIFICATION_TOOL_NAME:
                return dict(block.input)
        raise LLMConfigurationError(
            "Model response did not include the expected emit_classification tool call."
        )


class GeminiLLMClient:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get(MODEL_ENV_VAR, GEMINI_DEFAULT_MODEL)
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = os.environ.get(GEMINI_API_KEY_ENV_VAR)
        if not api_key:
            raise LLMConfigurationError(f"Missing {GEMINI_API_KEY_ENV_VAR} environment variable.")
        try:
            from google import genai
        except ImportError as exc:
            raise LLMConfigurationError(
                "The 'google-genai' package is required to use GeminiLLMClient."
            ) from exc
        self._client = genai.Client(api_key=api_key)
        return self._client

    def classify_raw(
        self, system_prompt: str, user_prompt: str, category_values: list[str]
    ) -> dict[str, Any]:
        client = self._get_client()
        response_schema = _classification_schema_object(category_values)
        from google.genai import types
        response = client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=response_schema,
                # Structured JSON output uses response_schema rather than Gemini
                # function-calling tools. Disable SDK automatic function calling
                # explicitly to avoid the generate_content AFC warning and keep
                # each classification request stateless across retries.
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        text = getattr(response, "text", None)
        if not text:
            raise LLMConfigurationError(
                "Gemini response did not include structured JSON output as expected."
            )
        import json
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMConfigurationError(f"Gemini response was not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise LLMConfigurationError("Gemini response JSON was not an object.")
        return data


def default_llm_client() -> LLMClient:
    provider = os.environ.get(PROVIDER_ENV_VAR, DEFAULT_PROVIDER).strip().lower()
    if provider == "gemini":
        return GeminiLLMClient()
    if provider == "anthropic":
        return AnthropicLLMClient()
    raise LLMConfigurationError(
        f"Unknown {PROVIDER_ENV_VAR}={provider!r}. Supported providers: 'gemini', 'anthropic'."
    )
