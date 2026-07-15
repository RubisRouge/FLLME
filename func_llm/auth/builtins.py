from .types import AuthPrinciple


GOOGLE_ADC_PRINCIPLE = AuthPrinciple(
    id="google_adc",
    name="Google ADC",
)

API_KEY_PRINCIPLE = AuthPrinciple(
    id="api_key",
    name="API Key",
    config={"header_name": "api-key"},
)

BUILTIN_PRINCIPLES: list[AuthPrinciple] = [GOOGLE_ADC_PRINCIPLE, API_KEY_PRINCIPLE]
