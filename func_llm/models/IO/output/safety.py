from pydantic import BaseModel
from enum import StrEnum


class SafetyCategory(StrEnum):
    HARASSMENT = "harassment"
    HATE = "hate"
    SEXUAL = "sexual"
    VIOLENCE = "violence"
    SELF_HARM = "self_harm"
    DANGEROUS = "dangerous"
    PROFANITY = "profanity"
    OTHER = "other"


class SafetySeverity(StrEnum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SafetyRating(BaseModel):
    category : SafetyCategory
    severity : SafetySeverity = SafetySeverity.SAFE
    filtered : bool = False


class SafetyResult(BaseModel):
    ratings : list[SafetyRating] = []
    blocked : bool = False
    refusal_message : str | None = None
    refusal_category : str | None = None
