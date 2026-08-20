"""LLM provider abstraction for the email classifier.

`LLMClient` is a small protocol so the classifier logic (validation, retry,
guardrail-preserving behavior) can be unit tested with a mocked client that
never touches the network or requires an API key. Four concrete providers
are implemented: `GroqLLMClient` (the default), `GeminiLLMClient`,
`AnthropicLLMClient`, and `OllamaLLMClient` (all kept available, selectable
via `INBOXGUARDIAN_LLM_PROVIDER`). All but Ollama read their credentials
solely from an environment variable — no API keys are ever hard-coded,
printed, logged, or committed. Ollama requires no API key at all: it talks
to a local Ollama service instead.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

from dotenv import load_dotenv

load_dotenv()

PROVIDER_ENV_VAR = "INBOXGUARDIAN_LLM_PROVIDER"
DEFAULT_PROVIDER = "groq"
MODEL_ENV_VAR = "INBOXGUARDIAN_LLM_MODEL"
ANTHROPIC_API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"
API_KEY_ENV_VAR = ANTHROPIC_API_KEY_ENV_VAR
ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_MODEL = ANTHROPIC_DEFAULT_MODEL
GEMINI_API_KEY_ENV_VAR = "GEMINI_API_KEY"
GEMINI_DEFAULT_MODEL = "gemini-3.5-flash"
GROQ_API_KEY_ENV_VAR = "GROQ_API_KEY"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"
GROQ_MODEL_ENV_VAR = "INBOXGUARDIAN_GROQ_MODEL"
# Ollama needs no API key -- it talks to a local service. Model selection
# follows the same provider-specific pattern as Groq (never falls back to
# the generic, cross-provider MODEL_ENV_VAR, so a stale value left over
# from a different provider can't silently become Ollama's model). The host
# var name matches Ollama's own upstream convention (OLLAMA_HOST) rather
# than inventing a new one, consistent with how provider API key vars
# already reuse each provider's own upstream naming.
OLLAMA_DEFAULT_MODEL = "qwen3:8b"
OLLAMA_MODEL_ENV_VAR = "INBOXGUARDIAN_OLLAMA_MODEL"
OLLAMA_HOST_ENV_VAR = "OLLAMA_HOST"
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


def _with_additional_properties_false(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy with additionalProperties=false on every object node."""
    import copy

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            walked = {key: _walk(value) for key, value in node.items()}
            if walked.get("type") == "object":
                walked["additionalProperties"] = False
            return walked
        if isinstance(node, list):
            return [_walk(item) for item in node]
        return node

    return _walk(copy.deepcopy(schema))


def _classification_schema_object_for_groq(category_values: list[str]) -> dict[str, Any]:
    """Build the shared classification schema with Groq's strict object rule."""
    return _with_additional_properties_false(_classification_schema_object(category_values))


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


class GroqLLMClient:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get(GROQ_MODEL_ENV_VAR) or GROQ_DEFAULT_MODEL
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = os.environ.get(GROQ_API_KEY_ENV_VAR)
        if not api_key:
            raise LLMConfigurationError(f"Missing {GROQ_API_KEY_ENV_VAR} environment variable.")
        try:
            from groq import Groq
        except ImportError as exc:
            raise LLMConfigurationError(
                "The 'groq' package is required to use GroqLLMClient."
            ) from exc
        self._client = Groq(api_key=api_key)
        return self._client

    def classify_raw(
        self, system_prompt: str, user_prompt: str, category_values: list[str]
    ) -> dict[str, Any]:
        client = self._get_client()
        response_schema = _classification_schema_object_for_groq(category_values)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "classification",
                    "schema": response_schema,
                    "strict": True,
                },
            },
            reasoning_effort="low",
            include_reasoning=False,
        )
        choice = response.choices[0] if getattr(response, "choices", None) else None
        text = choice.message.content if choice is not None and choice.message is not None else None
        if not text:
            raise LLMConfigurationError(
                "Groq response did not include structured JSON output as expected."
            )
        import json
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMConfigurationError(f"Groq response was not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise LLMConfigurationError("Groq response JSON was not an object.")
        return data


class OllamaLLMClient:
    """Local provider: talks to a locally-running Ollama service. Requires
    no API key -- only that the Ollama service is running and the model has
    been pulled (e.g. `ollama pull qwen3:8b`)."""

    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        self.model = model or os.environ.get(OLLAMA_MODEL_ENV_VAR) or OLLAMA_DEFAULT_MODEL
        # None is intentional here (not empty string): passing host=None to
        # ollama.Client() lets the client fall back to its own default
        # (http://localhost:11434), rather than us hard-coding that URL.
        self.host = host or os.environ.get(OLLAMA_HOST_ENV_VAR)
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import ollama
        except ImportError as exc:
            raise LLMConfigurationError(
                "The 'ollama' package is required to use OllamaLLMClient. "
                "Install it with `pip install ollama`."
            ) from exc
        self._client = ollama.Client(host=self.host) if self.host else ollama.Client()
        return self._client

    def classify_raw(
        self, system_prompt: str, user_prompt: str, category_values: list[str]
    ) -> dict[str, Any]:
        client = self._get_client()
        response_schema = _classification_schema_object(category_values)

        try:
            response = client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                # Ollama's structured-output support: pass the JSON schema
                # directly via `format`, reusing the same shared schema/
                # taxonomy every other provider uses -- never redefined here.
                format=response_schema,
                # qwen3:8b supports an extended-thinking mode; disable it so
                # reasoning tokens never end up mixed into the classification
                # response, mirroring the Groq provider's include_reasoning=False.
                think=False,
                options={"temperature": 0},
            )
        except Exception as exc:
            # Covers a missing/unreachable local Ollama service, a model
            # that hasn't been pulled, or any other request failure.
            raise LLMConfigurationError(
                f"Ollama request failed (is the local Ollama service running at "
                f"{self.host or 'the default host'}? is {self.model!r} pulled?): {exc}"
            ) from exc

        message = getattr(response, "message", None)
        text = getattr(message, "content", None) if message is not None else None
        if not text:
            raise LLMConfigurationError(
                "Ollama response did not include structured JSON output as expected."
            )
        import json
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMConfigurationError(f"Ollama response was not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise LLMConfigurationError("Ollama response JSON was not an object.")
        return data


def default_llm_client() -> LLMClient:
    provider = os.environ.get(PROVIDER_ENV_VAR, DEFAULT_PROVIDER).strip().lower()
    if provider == "groq":
        return GroqLLMClient()
    if provider == "gemini":
        return GeminiLLMClient()
    if provider == "anthropic":
        return AnthropicLLMClient()
    if provider == "ollama":
        return OllamaLLMClient()
    raise LLMConfigurationError(
        f"Unknown {PROVIDER_ENV_VAR}={provider!r}. Supported providers: "
        "'groq', 'gemini', 'anthropic', 'ollama'."
    )
