import os
import json
from huggingface_hub import InferenceClient

def call_llama(prompt: str, json_mode: bool = False, provider: str = "llama") -> str:
    """Centralized LLM caller that supports a provider hint for model selection."""
    provider_key = provider.lower()
    default_model = "meta-llama/Llama-3.3-70B-Instruct"
    if provider_key == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("HF_TOKEN")
        model = "google/gemini-pro"
    else:
        api_key = os.environ.get("HF_TOKEN")
        model = default_model
    client = InferenceClient(api_key=api_key)
    
    # We use chat completion mode
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = client.chat_completion(
            model=model,
            messages=messages,
            max_tokens=1500,
            temperature=0.1 # Low temp for deterministic reasoning
        )
        text = response.choices[0].message.content
        return text
    except Exception as e:
        if provider_key == "gemini" and model != default_model:
            try:
                response = client.chat_completion(
                    model=default_model,
                    messages=messages,
                    max_tokens=1500,
                    temperature=0.1
                )
            except Exception:
                return f"Error calling Llama: {e}"
        else:
            return f"Error calling Llama: {e}"

    text = response.choices[0].message.content
    return text