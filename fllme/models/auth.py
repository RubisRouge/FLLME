from pydantic import BaseModel


class AuthPrinciple(BaseModel):
    id: str
    name: str
    resolver_id: str
    required_env_vars: list[str] = []
    config: dict[str, str] = {}


GOOGLE_ADC_PRINCIPLE = AuthPrinciple(
    id="google_adc",
    name="Google ADC",
    resolver_id="google_adc",
    required_env_vars=[],
    config={},
)

API_KEY_PRINCIPLE = AuthPrinciple(
    id="api_key",
    name="API Key",
    resolver_id="api_key",
    required_env_vars=[],
    config={"header_name": "api-key"},
)

BUILTIN_PRINCIPLES: list[AuthPrinciple] = [GOOGLE_ADC_PRINCIPLE, API_KEY_PRINCIPLE]
