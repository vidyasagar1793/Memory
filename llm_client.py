"""Shared entry point for all language-model requests in this project."""

import os
from collections.abc import Sequence
from typing import Any

from dotenv import load_dotenv


class LLMClientError(RuntimeError):
    """Raised when a configured LLM provider cannot complete a request."""


def _normalise_messages(
    prompt: str | None, messages: Sequence[dict[str, Any]] | None
) -> list[dict[str, str]]:
    if messages is not None:
        if not messages:
            raise ValueError("messages must not be empty")
        return [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in messages
        ]
    if prompt is None or not prompt.strip():
        raise ValueError("Provide either a non-empty prompt or messages.")
    return [{"role": "user", "content": prompt}]


def _gemini_contents(messages: Sequence[dict[str, str]]) -> str:
    """Preserve chat context for Gemini without provider-specific types."""
    return "\n\n".join(
        f"{message['role'].upper()}:\n{message['content']}" for message in messages
    )


def call_llm(
    prompt: str | None = None,
    *,
    messages: Sequence[dict[str, Any]] | None = None,
    provider: str = "gemini",
    model: str | None = None,
    max_tokens: int = 1500,
    temperature: float = 0.1,
) -> str:
    """Return a text response from Gemini or Hugging Face.

    Credentials are loaded from ``.env`` once per call: ``GEMINI_API_KEY`` for
    Gemini and ``HF_TOKEN`` for Hugging Face. No caller should create a vendor
    client or read an API key directly.
    """
    load_dotenv()
    normalised_messages = _normalise_messages(prompt, messages)
    provider_key = provider.lower()

    if provider_key == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise LLMClientError("GEMINI_API_KEY is not configured.")

        from google import genai
        from google.genai import types

        selected_model = model or "gemini-3.1-flash-lite"
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=selected_model,
                contents=_gemini_contents(normalised_messages),
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            )
            text = (response.text or "").strip()
        except Exception as error:
            raise LLMClientError(
                f"Gemini request failed for model '{selected_model}': {error}"
            ) from error
    elif provider_key in {"huggingface", "hf", "llama"}:
        api_key = os.getenv("HF_TOKEN")
        if not api_key:
            raise LLMClientError("HF_TOKEN is not configured.")

        from huggingface_hub import InferenceClient

        selected_model = model or "meta-llama/Llama-3.3-70B-Instruct"
        try:
            client = InferenceClient(api_key=api_key)
            response = client.chat.completions.create(
                model=selected_model,
                messages=normalised_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = (response.choices[0].message.content or "").strip()
        except Exception as error:
            raise LLMClientError(
                f"Hugging Face request failed for model '{selected_model}': {error}"
            ) from error
    elif provider_key == "groq":
        print("Using GROQ provider for LLM calls.")
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise LLMClientError("GROQ_API_KEY is not configured.")

        from groq import Groq
        client = Groq(
         api_key=os.environ.get("GROQ_API_KEY"),
)    
        response = client.chat.completions.create(
                 model="llama-3.3-70b-versatile",
                messages=normalised_messages,
                max_tokens=max_tokens,
                temperature=temperature,
)
        text = (response.choices[0].message.content or "").strip()
 
    else:
        raise ValueError(f"Unsupported LLM provider: {provider!r}")

    if not text:
        raise LLMClientError(f"{provider_key} returned an empty response.")
    return text


def call_llama(prompt: str, json_mode: bool = False, provider: str = "llama") -> str:
    """Backward-compatible alias for older callers."""
    return call_llm(prompt, provider=provider)
