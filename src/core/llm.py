"""Provider-independent LLM port with a LiteLLM-backed default adapter."""
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class LLMClient(ABC):
    @abstractmethod
    def complete(self, messages: List[Dict[str, str]], *, model: Optional[str] = None, **kwargs) -> str: ...

    def complete_json(self, messages, schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        raw = self.complete(messages, response_format={"type": "json_schema", "json_schema": {"name": "output", "strict": True, "schema": schema}}, **kwargs)
        return json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())

class LiteLLMClient(LLMClient):
    def __init__(self, model: str, **defaults):
        self.model, self.defaults = model, defaults

    def complete(self, messages, *, model=None, **kwargs):
        import litellm
        response = litellm.completion(model=model or self.model, messages=messages, **self.defaults, **kwargs)
        return response.choices[0].message.content.strip()

class LLMClientFactory:
    @staticmethod
    def create(provider: str = "litellm", model: str = "vertex_ai/gemini-2.5-flash", **kwargs) -> LLMClient:
        if provider == "litellm":
            return LiteLLMClient(model, **kwargs)
        raise ValueError(f"Unsupported LLM provider: {provider}")
