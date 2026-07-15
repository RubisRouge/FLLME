from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Generic
from typing import TypeVar

AuthResolverT = TypeVar("AuthResolverT")


@dataclass
class AuthPrinciple:
    id: str
    name: str

    config: dict[str, str] = field(default_factory=dict)
    required_env_vars: list[str] = field(default_factory=list)


class AuthResolver(ABC, Generic[AuthResolverT]):
    @abstractmethod
    async def resolve(
        self,
        principle: AuthPrinciple,
    ) -> AuthResolverT:
        pass
