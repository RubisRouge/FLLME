"""OpenAI Responses API adapter (``/responses``) for Azure OpenAI.

The GPT-5.6 family only accepts function tools together with
``reasoning_effort`` on the Responses API, not on ``/chat/completions``.

Not supported by this API, so not sent: ``stop``, ``presence_penalty``,
``frequency_penalty``. ``candidates > 1`` raises ``SerializationError``.
"""

from __future__ import annotations

import dataclasses
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from ...errors import ProviderError, SerializationError
from ...models.input import (
    GenerationInput,
    OutputType,
    ThinkingLevel,
    Tool,
    ToolsCallingMode,
    WebSearchTool,
)
from ...models.message import (
    Base64Source,
    Content,
    MediaContent,
    Message,
    MessageSource,
    TextContent,
    ToolCallContent,
    ToolResponseContent,
    UrlSource,
    ThinkingContent,
)
from ...models.model import LLMModel, ThinkingStyle
from ...models.output import (
    FinishReason,
    GenerationOutput,
    StreamDelta,
    TextDelta,
    ThinkingDelta,
    Usage,
)
from ...models.output.usage import CacheUsage
from ..base import (
    accumulate_content,
    get_capabilities,
    is_structured,
    parse_structured,
)
from .azure_v1 import _extract_text, _strict_schema

_TOOL_MODE_MAP: dict[ToolsCallingMode, str] = {
    ToolsCallingMode.AUTO: "auto",
    ToolsCallingMode.ANY: "required",
    ToolsCallingMode.NONE: "none",
}

_INCOMPLETE_REASON_MAP: dict[str, FinishReason] = {
    "max_output_tokens": FinishReason.MAX_TOKENS,
    "max_tokens": FinishReason.MAX_TOKENS,
    "content_filter": FinishReason.CONTENT_FILTER,
}


@dataclasses.dataclass
class _FunctionCallAccumulator:
    call_id: str
    name: str
    arguments: str


def _serialize_user_content(msg: Message) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for content in msg.contents:
        match content:
            case TextContent(text=text):
                parts.append({"type": "input_text", "text": text})
            case MediaContent(source=Base64Source(data=data), media_type=mt):
                parts.append(
                    {"type": "input_image", "image_url": f"data:{mt};base64,{data}"}
                )
            case MediaContent(source=UrlSource(url=url)):
                parts.append({"type": "input_image", "image_url": url})
    return parts


def _serialize_model_items(msg: Message) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    text_parts: list[str] = []

    def flush_text() -> None:
        if text_parts:
            items.append({"role": "assistant", "content": "\n".join(text_parts)})
            text_parts.clear()

    for content in msg.contents:
        match content:
            case TextContent(text=text):
                text_parts.append(text)
            case ToolCallContent(id=call_id, name=name, arguments=args):
                flush_text()
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": json.dumps(args),
                    }
                )
    flush_text()
    return items


def _error_body(payload: dict[str, Any]) -> str:
    error = payload.get("error")
    if not isinstance(error, dict):
        error = payload
    return json.dumps({"error": error})


class OpenAIAzureResponsesV1:
    def serialize(
        self,
        gen_input: GenerationInput,
        model: LLMModel | None = None,
    ) -> dict[str, Any]:
        cfg = gen_input.llm_config
        if cfg.candidates > 1:
            msg = "The OpenAI Responses API does not support candidates > 1"
            raise SerializationError(msg)

        payload: dict[str, Any] = {"model": gen_input.model}

        if gen_input.system_prompt:
            payload["instructions"] = gen_input.system_prompt

        items: list[dict[str, Any]] = []
        for message in gen_input.conversation:
            match message.source:
                case MessageSource.SYSTEM:
                    items.append({"role": "system", "content": _extract_text(message)})
                case MessageSource.USER:
                    items.append(
                        {"role": "user", "content": _serialize_user_content(message)}
                    )
                case MessageSource.MODEL:
                    items.extend(_serialize_model_items(message))
                case MessageSource.TOOL:
                    for content in message.contents:
                        match content:
                            case ToolResponseContent(
                                tool_call_id=call_id, content=text
                            ):
                                items.append(
                                    {
                                        "type": "function_call_output",
                                        "call_id": call_id,
                                        "output": text,
                                    }
                                )
        payload["input"] = items

        if cfg.temperature is not None:
            payload["temperature"] = cfg.temperature
        if cfg.top_p is not None:
            payload["top_p"] = cfg.top_p
        payload["max_output_tokens"] = cfg.max_tokens

        if (
            cfg.thinking != ThinkingLevel.NO
            and get_capabilities(model).thinking_style != ThinkingStyle.NONE
        ):
            # "summary" is what makes the API stream the reasoning text.
            payload["reasoning"] = {"effort": cfg.thinking.value, "summary": "auto"}

        if gen_input.tool_config and gen_input.tool_config.tools:
            tools: list[dict[str, Any]] = []
            for tool in gen_input.tool_config.tools:
                match tool:
                    case Tool(
                        name=name, description=description, parameters=parameters
                    ):
                        tools.append(
                            {
                                "type": "function",
                                "name": name,
                                "description": description,
                                "parameters": parameters,
                                # Strict by default on this API, unlike chat
                                # completions: keep the same semantics.
                                "strict": False,
                            }
                        )
                    case WebSearchTool():
                        error_msg = (
                            "Web search tool is not supported by the OpenAI "
                            "Azure Responses adapter"
                        )
                        raise SerializationError(error_msg)
            payload["tools"] = tools
            if get_capabilities(model).supports_parallel_tool_calls:
                payload["parallel_tool_calls"] = gen_input.tool_config.parallel_calling
            payload["tool_choice"] = _TOOL_MODE_MAP[gen_input.tool_config.mode]

        output_type = gen_input.output_type
        if is_structured(output_type):
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": output_type.__name__,
                    "schema": _strict_schema(output_type.model_json_schema()),
                    "strict": True,
                }
            }

        payload["stream"] = True
        # Do not retain the conversation on the provider side.
        payload["store"] = False

        return payload

    async def parse_stream(
        self,
        output_type: OutputType,
        lines: AsyncIterator[str],
    ) -> AsyncIterator[StreamDelta | GenerationOutput]:
        accumulated: list[Content] = []
        finish_reason = FinishReason.STOP
        usage_meta: dict[str, Any] = {}
        model_name = ""
        response_id = ""
        calls: dict[str, _FunctionCallAccumulator] = {}
        has_thinking = False

        async for line in lines:
            if not line.startswith("data: "):
                continue
            raw = line.removeprefix("data: ").strip()
            if not raw or raw == "[DONE]":
                continue

            event: dict[str, Any] = json.loads(raw)

            match event.get("type", ""):
                case "response.created" | "response.in_progress":
                    response = event.get("response", {})
                    response_id = response.get("id", response_id)
                    model_name = response.get("model", model_name)

                case "response.output_text.delta":
                    if text := event.get("delta"):
                        yield StreamDelta(delta=TextDelta(text=text))
                        accumulate_content(accumulated, TextContent(text=text))

                case "response.reasoning_summary_part.added":
                    # Separate successive summary parts.
                    if has_thinking:
                        sep = "\n\n"
                        yield StreamDelta(delta=ThinkingDelta(thinking=sep))
                        accumulate_content(accumulated, ThinkingContent(thinking=sep))

                case "response.reasoning_summary_text.delta" | (
                    "response.reasoning_text.delta"
                ):
                    if thinking := event.get("delta"):
                        has_thinking = True
                        yield StreamDelta(delta=ThinkingDelta(thinking=thinking))
                        accumulate_content(
                            accumulated, ThinkingContent(thinking=thinking)
                        )

                case "response.output_item.added":
                    item = event.get("item", {})
                    if item.get("type") == "function_call":
                        calls[_item_key(item, event)] = _FunctionCallAccumulator(
                            call_id=item.get("call_id", ""),
                            name=item.get("name", ""),
                            arguments=item.get("arguments", ""),
                        )

                case "response.function_call_arguments.delta":
                    acc = calls.get(_item_key({"id": event.get("item_id")}, event))
                    if acc is not None:
                        acc.arguments += event.get("delta", "")

                case "response.output_item.done":
                    item = event.get("item", {})
                    if item.get("type") == "function_call":
                        # The completed item is the source of truth.
                        key = _item_key(item, event)
                        acc = calls.setdefault(
                            key, _FunctionCallAccumulator("", "", "")
                        )
                        acc.call_id = item.get("call_id") or acc.call_id
                        acc.name = item.get("name") or acc.name
                        acc.arguments = item.get("arguments") or acc.arguments

                case "response.completed" | "response.incomplete":
                    response = event.get("response", {})
                    response_id = response.get("id", response_id)
                    model_name = response.get("model", model_name)
                    usage_meta = response.get("usage") or usage_meta
                    reason = (response.get("incomplete_details") or {}).get("reason")
                    if reason:
                        finish_reason = _INCOMPLETE_REASON_MAP.get(
                            reason, FinishReason.ERROR
                        )

                case "response.failed":
                    raise ProviderError(0, _error_body(event.get("response", {})))

                case "error":
                    raise ProviderError(0, _error_body(event))

        for acc in calls.values():
            try:
                arguments = json.loads(acc.arguments) if acc.arguments else {}
            except json.JSONDecodeError:
                arguments = {"raw": acc.arguments}
            accumulated.append(
                ToolCallContent(
                    id=acc.call_id or uuid.uuid4().hex[:12],
                    name=acc.name,
                    arguments=arguments,
                )
            )

        if calls and finish_reason == FinishReason.STOP:
            finish_reason = FinishReason.TOOL_USE

        input_details = usage_meta.get("input_tokens_details") or {}
        output_details = usage_meta.get("output_tokens_details") or {}

        output = GenerationOutput(
            id=response_id or uuid.uuid4().hex,
            model=model_name,
            message=Message(source=MessageSource.MODEL, contents=accumulated),
            finish_reason=finish_reason,
            usage=Usage(
                input_tokens=usage_meta.get("input_tokens", 0),
                output_tokens=usage_meta.get("output_tokens", 0),
                thinking_tokens=output_details.get("reasoning_tokens", 0),
                cache=CacheUsage(read_tokens=input_details.get("cached_tokens", 0)),
                total_tokens=usage_meta.get("total_tokens", 0),
            ),
        )
        if is_structured(output_type):
            parse_structured(output, output_type)
        yield output


def _item_key(item: dict[str, Any], event: dict[str, Any]) -> str:
    """Identify a streamed output item: its id, else its output index."""
    return str(item.get("id") or event.get("output_index", 0))
