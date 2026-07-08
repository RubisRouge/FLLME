from pydantic import BaseModel


class LLMModel(BaseModel):
    id: str
    name: str
