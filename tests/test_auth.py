from unittest.mock import MagicMock, patch

import pytest

from func_llm.auth import RESOLVERS, get_resolver, register_resolver
from func_llm.auth.deprecated.api_key import ApiKeyResolver
from func_llm.auth.deprecated.google_adc import GoogleADCResolver
from func_llm.errors import AuthError
from func_llm.models.auth import AuthPrinciple


class TestResolverRegistry:
    def test_builtins_registered(self) -> None:
        assert "google_adc" in RESOLVERS
        assert "api_key" in RESOLVERS

    def test_get_resolver(self) -> None:
        resolver = get_resolver("google_adc")
        assert isinstance(resolver, GoogleADCResolver)

    def test_get_unknown_resolver(self) -> None:
        with pytest.raises(AuthError, match="Unknown auth resolver"):
            get_resolver("nonexistent")

    def test_register_custom(self) -> None:
        mock_resolver = MagicMock()
        register_resolver("custom", mock_resolver)
        assert get_resolver("custom") is mock_resolver
        del RESOLVERS["custom"]


class TestApiKeyResolver:
    @pytest.mark.asyncio
    async def test_get_headers(self) -> None:
        resolver = ApiKeyResolver()
        principle = AuthPrinciple(
            id="test",
            name="Test",
            resolver_id="api_key",
            required_env_vars=["TEST_API_KEY"],
            config={"header_name": "x-api-key"},
        )
        with patch.dict("os.environ", {"TEST_API_KEY": "secret123"}):
            headers = await resolver.get_headers(principle)
        assert headers == {"x-api-key": "secret123"}

    @pytest.mark.asyncio
    async def test_missing_env_var(self) -> None:
        resolver = ApiKeyResolver()
        principle = AuthPrinciple(
            id="test",
            name="Test",
            resolver_id="api_key",
            required_env_vars=["MISSING_VAR"],
            config={"header_name": "api-key"},
        )
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(AuthError, match="Missing required environment variable"),
        ):
            await resolver.get_headers(principle)

    @pytest.mark.asyncio
    async def test_no_env_vars_configured(self) -> None:
        resolver = ApiKeyResolver()
        principle = AuthPrinciple(
            id="test",
            name="Test",
            resolver_id="api_key",
        )
        with pytest.raises(AuthError, match="requires at least one env var"):
            await resolver.get_headers(principle)

    def test_check_env_missing(self) -> None:
        resolver = ApiKeyResolver()
        principle = AuthPrinciple(
            id="test",
            name="Test",
            resolver_id="api_key",
            required_env_vars=["VAR_A", "VAR_B"],
        )
        with patch.dict("os.environ", {"VAR_A": "val"}, clear=True):
            missing = resolver.check_env(principle)
        assert missing == ["VAR_B"]

    def test_check_env_all_present(self) -> None:
        resolver = ApiKeyResolver()
        principle = AuthPrinciple(
            id="test",
            name="Test",
            resolver_id="api_key",
            required_env_vars=["VAR_A"],
        )
        with patch.dict("os.environ", {"VAR_A": "val"}):
            missing = resolver.check_env(principle)
        assert missing == []

    @pytest.mark.asyncio
    async def test_default_header_name(self) -> None:
        resolver = ApiKeyResolver()
        principle = AuthPrinciple(
            id="test",
            name="Test",
            resolver_id="api_key",
            required_env_vars=["MY_KEY"],
        )
        with patch.dict("os.environ", {"MY_KEY": "val"}):
            headers = await resolver.get_headers(principle)
        assert "api-key" in headers


class TestGoogleADCResolver:
    @pytest.mark.asyncio
    async def test_get_headers(self) -> None:
        resolver = GoogleADCResolver()
        principle = AuthPrinciple(
            id="google_adc",
            name="Google ADC",
            resolver_id="google_adc",
        )

        mock_creds = MagicMock()
        mock_creds.token = "fake-token-123"
        mock_creds.valid = False

        with patch("func_llm.auth.google_adc.asyncio.to_thread") as mock_to_thread:
            mock_to_thread.side_effect = [
                (mock_creds, "project-id"),
                None,
            ]
            headers = await resolver.get_headers(principle)

        assert headers == {"Authorization": "Bearer fake-token-123"}

    @pytest.mark.asyncio
    async def test_cached_credentials_skip_default(self) -> None:
        resolver = GoogleADCResolver()
        principle = AuthPrinciple(
            id="google_adc",
            name="Google ADC",
            resolver_id="google_adc",
        )

        mock_creds = MagicMock()
        mock_creds.token = "token-1"
        mock_creds.valid = False

        with patch("func_llm.auth.google_adc.asyncio.to_thread") as mock_to_thread:
            mock_to_thread.side_effect = [
                (mock_creds, "project-id"),
                None,
            ]
            await resolver.get_headers(principle)

        mock_creds.valid = True
        mock_creds.token = "token-1"

        with patch("func_llm.auth.google_adc.asyncio.to_thread") as mock_to_thread:
            headers = await resolver.get_headers(principle)

        assert headers == {"Authorization": "Bearer token-1"}
        mock_to_thread.assert_not_called()

    @pytest.mark.asyncio
    async def test_refresh_when_expired(self) -> None:
        resolver = GoogleADCResolver()
        principle = AuthPrinciple(
            id="google_adc",
            name="Google ADC",
            resolver_id="google_adc",
        )

        mock_creds = MagicMock()
        mock_creds.token = "token-1"
        mock_creds.valid = False

        with patch("func_llm.auth.google_adc.asyncio.to_thread") as mock_to_thread:
            mock_to_thread.side_effect = [
                (mock_creds, "project-id"),
                None,
            ]
            await resolver.get_headers(principle)

        mock_creds.valid = False
        mock_creds.token = "token-2"

        with patch("func_llm.auth.google_adc.asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = None
            headers = await resolver.get_headers(principle)

        assert headers == {"Authorization": "Bearer token-2"}
        mock_to_thread.assert_called_once()

    def test_check_env(self) -> None:
        resolver = GoogleADCResolver()
        principle = AuthPrinciple(
            id="google_adc",
            name="Google ADC",
            resolver_id="google_adc",
        )
        assert resolver.check_env(principle) == []
