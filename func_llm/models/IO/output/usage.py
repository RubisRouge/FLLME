from pydantic import BaseModel


class CacheUsage(BaseModel):
    read_tokens : int = 0
    creation_tokens : int = 0


class Usage(BaseModel):
    input_tokens : int
    output_tokens : int
    thinking_tokens : int = 0
    cache : CacheUsage = CacheUsage()
    total_tokens : int | None = None
