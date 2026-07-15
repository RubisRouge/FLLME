import importlib
from typing import Any

import pytest

from fllm.generate import configure, get_service
from fllm.service import DeploymentService


def _gen_mod() -> Any:
    return importlib.import_module("func_llm.generate")


class TestConfigureAndGetService:
    def test_get_service_unconfigured(self) -> None:
        mod = _gen_mod()
        original = mod.DEFAULT_SERVICE
        mod.DEFAULT_SERVICE = None
        try:
            with pytest.raises(RuntimeError, match="No service configured"):
                get_service()
        finally:
            mod.DEFAULT_SERVICE = original

    @pytest.mark.asyncio
    async def test_configure(self) -> None:
        mod = _gen_mod()
        original = mod.DEFAULT_SERVICE
        try:
            svc = await DeploymentService.from_sqlite(":memory:")
            configure(svc)
            assert get_service() is svc
        finally:
            mod.DEFAULT_SERVICE = original
