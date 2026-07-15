from pydantic import BaseModel
from pydantic import Field


class AuthPrinciple(BaseModel):
    id: str
    name: str
    resolver_id: str

    required_env_vars: list[str] = Field(default_factory=list)
    config: dict[str, str] = Field(default_factory=dict)


GOOGLE_ADC_PRINCIPLE = AuthPrinciple(
    id="google_adc",
    name="Google ADC",
    resolver_id="google_adc",
)

API_KEY_PRINCIPLE = AuthPrinciple(
    id="api_key",
    name="API Key",
    resolver_id="api_key",
    config={"header_name": "api-key"},
)

BUILTIN_PRINCIPLES: list[AuthPrinciple] = [GOOGLE_ADC_PRINCIPLE, API_KEY_PRINCIPLE]
