from __future__ import annotations

import json

import pytest

from fllme.errors import ProviderError, ProviderErrorKind

K = ProviderErrorKind


class TestBackwardCompatibility:
    def test_str_and_raw_attributes_unchanged(self) -> None:
        err = ProviderError(400, '{"error": {"message": "bad"}}')
        assert str(err) == 'Provider returned 400: {"error": {"message": "bad"}}'
        assert err.status_code == 400
        assert err.body == '{"error": {"message": "bad"}}'

    def test_positional_construction_still_works(self) -> None:
        err = ProviderError(0, "Stream ended without a final GenerationOutput")
        assert err.kind == K.UNKNOWN
        assert err.retryable is False
        assert err.provider_message is None

    def test_explicit_values_win(self) -> None:
        err = ProviderError(
            400, "x", kind=K.POLICY, retryable=True, provider_message="m"
        )
        assert (err.kind, err.retryable, err.provider_message) == (K.POLICY, True, "m")


class TestStatusClassification:
    @pytest.mark.parametrize(
        ("status", "kind", "retryable"),
        [
            (400, K.INVALID_REQUEST, False),
            (422, K.INVALID_REQUEST, False),
            (401, K.AUTHENTICATION, False),
            (403, K.PERMISSION, False),
            (404, K.NOT_FOUND, False),
            (429, K.RATE_LIMITED, True),
            (500, K.UNAVAILABLE, True),
            (503, K.UNAVAILABLE, True),
            (529, K.UNAVAILABLE, True),
            (418, K.UNKNOWN, False),
        ],
    )
    def test_plain_text_body(self, status: int, kind: K, retryable: bool) -> None:
        err = ProviderError(status, "not json")
        assert err.kind == kind
        assert err.retryable is retryable
        assert err.provider_message is None


class TestAnthropic:
    def test_invalid_request(self) -> None:
        body = json.dumps(
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "max_tokens must be greater than thinking.budget_tokens.",
                },
            }
        )
        err = ProviderError(400, body)
        assert err.kind == K.INVALID_REQUEST
        assert err.provider_message == (
            "max_tokens must be greater than thinking.budget_tokens."
        )

    def test_overloaded(self) -> None:
        body = json.dumps(
            {
                "type": "error",
                "error": {"type": "overloaded_error", "message": "Overloaded"},
            }
        )
        err = ProviderError(529, body)
        assert err.kind == K.UNAVAILABLE
        assert err.retryable is True
        assert err.provider_message == "Overloaded"


class TestOpenAIAzure:
    def test_unsupported_parameter(self) -> None:
        body = json.dumps(
            {
                "error": {
                    "message": "Unsupported parameter: 'parallel_tool_calls' is not "
                    "supported with this model.",
                    "type": "invalid_request_error",
                    "param": "parallel_tool_calls",
                    "code": "unsupported_parameter",
                }
            }
        )
        err = ProviderError(400, body)
        assert err.kind == K.INVALID_REQUEST
        assert err.provider_message is not None
        assert err.provider_message.startswith("Unsupported parameter")

    def test_content_filter_is_policy(self) -> None:
        body = json.dumps(
            {
                "error": {
                    "message": "The response was filtered due to the prompt "
                    "triggering Azure OpenAI's content management policy.",
                    "type": None,
                    "param": "prompt",
                    "code": "content_filter",
                }
            }
        )
        assert ProviderError(400, body).kind == K.POLICY

    def test_rate_limit(self) -> None:
        body = json.dumps(
            {"error": {"code": "429", "message": "Rate limit exceeded, retry later."}}
        )
        err = ProviderError(429, body)
        assert err.kind == K.RATE_LIMITED
        assert err.retryable is True
        assert err.provider_message == "Rate limit exceeded, retry later."


class TestGeminiVertex:
    def test_invalid_argument(self) -> None:
        body = json.dumps(
            {
                "error": {
                    "code": 400,
                    "message": "Multiple tools are supported only when they are all "
                    "search tools.",
                    "status": "INVALID_ARGUMENT",
                }
            }
        )
        err = ProviderError(400, body)
        assert err.kind == K.INVALID_REQUEST
        assert err.provider_message == (
            "Multiple tools are supported only when they are all search tools."
        )

    def test_error_wrapped_in_list(self) -> None:
        body = json.dumps(
            [
                {
                    "error": {
                        "code": 429,
                        "message": "Quota exceeded",
                        "status": "RESOURCE_EXHAUSTED",
                    }
                }
            ]
        )
        err = ProviderError(429, body)
        assert err.kind == K.RATE_LIMITED
        assert err.retryable is True
        assert err.provider_message == "Quota exceeded"

    def test_google_status_refines_kind(self) -> None:
        body = json.dumps(
            {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}}
        )
        assert ProviderError(503, body).kind == K.UNAVAILABLE

    def test_permission_denied(self) -> None:
        body = json.dumps(
            {"error": {"code": 403, "message": "denied", "status": "PERMISSION_DENIED"}}
        )
        assert ProviderError(403, body).kind == K.PERMISSION


class TestAnthropicVertexPolicy:
    def test_gcp_organization_policy_is_policy(self) -> None:
        body = json.dumps(
            {
                "error": {
                    "code": 400,
                    "message": "Request is prohibited by organization's policy. "
                    "constraints/vertexai.allowedPartnerModelFeatures: "
                    "structured_outputs is not allowed.",
                    "status": "FAILED_PRECONDITION",
                }
            }
        )
        err = ProviderError(400, body)
        assert err.kind == K.POLICY
        assert err.retryable is False
        assert err.provider_message is not None
        assert "structured_outputs" in err.provider_message


class TestMistral:
    def test_native_error_shape(self) -> None:
        body = json.dumps(
            {
                "object": "error",
                "message": "reasoning_effort is not supported for this model",
                "type": "invalid_request_error",
                "code": "3051",
            }
        )
        err = ProviderError(400, body)
        assert err.kind == K.INVALID_REQUEST
        assert err.provider_message == (
            "reasoning_effort is not supported for this model"
        )

    def test_validation_error_detail_list(self) -> None:
        body = json.dumps(
            {
                "detail": [
                    {
                        "type": "literal_error",
                        "loc": ["body", "reasoning_effort"],
                        "msg": "Input should be 'none' or 'high'",
                    }
                ]
            }
        )
        err = ProviderError(422, body)
        assert err.kind == K.INVALID_REQUEST
        assert err.provider_message == (
            "reasoning_effort: Input should be 'none' or 'high'"
        )

    def test_unauthorized(self) -> None:
        body = json.dumps({"message": "Unauthorized", "request_id": "abc"})
        err = ProviderError(401, body)
        assert err.kind == K.AUTHENTICATION
        assert err.provider_message == "Unauthorized"


class TestMalformedBodies:
    @pytest.mark.parametrize("body", ["", "[]", "null", '"text"', "{}", '{"error": 3}'])
    def test_never_raises(self, body: str) -> None:
        err = ProviderError(400, body)
        assert err.kind == K.INVALID_REQUEST
        assert err.provider_message is None
