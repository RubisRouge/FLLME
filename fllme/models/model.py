from enum import StrEnum

from pydantic import BaseModel, Field

from .input import ThinkingLevel


class Provider(StrEnum):
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    MISTRAL = "mistral"
    OPENAI = "openai"


class ThinkingStyle(StrEnum):
    NONE = "none"  # no reasoning mode: send nothing
    BUDGET = "budget"  # token budget (Claude <= 4.6, Gemini 2.5)
    ADAPTIVE = (
        "adaptive"  # recent Claude: thinking.type=adaptive + output_config.effort
    )
    LEVEL = "level"  # Gemini 3+: thinkingConfig.thinkingLevel
    EFFORT = "effort"  # OpenAI / Mistral: reasoning_effort


class ModelCapabilities(BaseModel):
    """Technical capabilities of a model, used by adapters to shape the payload.

    Defaults reproduce the behaviour from before capabilities existed, so a model
    registered without capabilities is serialized exactly as it used to be.
    """

    thinking_style: ThinkingStyle = ThinkingStyle.BUDGET
    thinking_budget_min: int | None = None
    thinking_budget_max: int | None = None
    # Accepted thinking levels (e.g. Mistral: high only). None = no restriction.
    thinking_efforts: list[ThinkingLevel] | None = None
    supports_parallel_tool_calls: bool = True
    search_with_function_tools: bool = True


class LLMModel(BaseModel):
    id: str
    name: str
    provider: Provider
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
