from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models.output import GenerationOutput


class FuncLLMError(Exception):
    pass


class ModelNotFoundError(FuncLLMError):
    pass


class DeploymentNotFoundError(FuncLLMError):
    pass


class AuthError(FuncLLMError):
    pass


class ProviderError(FuncLLMError):
    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"Provider returned {status_code}: {body}")


class SerializationError(FuncLLMError):
    pass


class MediaResolutionError(FuncLLMError):
    def __init__(self, failed_ids: list[str] | None = None, msg: str = "") -> None:
        self.failed_ids = failed_ids
        if not msg:
            msg = "Media resolution failed"
            if failed_ids:
                msg += f" for IDs: {', '.join(failed_ids)}"
        super().__init__(msg)


class StructuredOutputError(FuncLLMError):
    def __init__(self, output: GenerationOutput, msg: str) -> None:
        self.output = output
        super().__init__(msg)
