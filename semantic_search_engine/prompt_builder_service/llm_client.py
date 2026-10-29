from typing import Protocol


class LLMClientError(RuntimeError):
    """Base error for optional inference providers that cannot return an answer."""


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str:
        """Return a generated explanation or raise LLMClientError."""
