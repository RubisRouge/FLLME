from __future__ import annotations

import dataclasses
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from ...models.input import GenerationInput, ThinkingLevel, ToolsCallingMode
from ...models.message import (
    Base64Source,
    Content,
    MediaContent,
    Message,
    MessageSource,
    TextContent,
    ThinkingContent,
    ToolCallContent,
    ToolResponseContent,
    UrlSource,
)
from ...models.output import (
    FinishReason,
    GenerationOutput,
    StreamDelta,
    TextDelta,
    ThinkingDelta,
    Usage,
)
from ...models.output.usage import CacheUsage

_FINISH_REASON_MAP: dict[str, FinishReason] = {
    "stop": FinishReason.STOP,
    "length": FinishReason.MAX_TOKENS,
    "tool_calls": FinishReason.TOOL_USE,
    "content_filter": FinishReason.CONTENT_FILTER,
}

_TOOL_MODE_MAP: dict[ToolsCallingMode, str] = {
    ToolsCallingMode.AUTO: "auto",
    ToolsCallingMode.ANY: "required",
    ToolsCallingMode.NONE: "none",
}

_THINKING_MAP: dict[ThinkingLevel, str] = {
    ThinkingLevel.LOW: "low",
    ThinkingLevel.MEDIUM: "medium",
    ThinkingLevel.HIGH: "high",
}


@dataclasses.dataclass
class _ToolCallAccumulator:
    id: str
    name: str
    arguments: str


def _extract_text(msg: Message) -> str:
    parts: list[str] = []
    for content in msg.contents:
        match content:
            case TextContent(text=text):
                parts.append(text)
    return "\n".join(parts)


def _serialize_user_content(
    msg: Message,
) -> str | list[dict[str, Any]]:
    has_media = any(isinstance(c, MediaContent) for c in msg.contents)
    if not has_media:
        return _extract_text(msg)

    parts: list[dict[str, Any]] = []
    for content in msg.contents:
        match content:
            case TextContent(text=text):
                parts.append({"type": "text", "text": text})
            case MediaContent(source=Base64Source(data=data), media_type=mt):
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mt};base64,{data}"},
                })
            case MediaContent(source=UrlSource(url=url)):
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": url},
                })
    return parts


def _serialize_model_messages(msg: Message) -> list[dict[str, Any]]:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for content in msg.contents:
        match content:
            case TextContent(text=text):
                text_parts.append(text)
            case ToolCallContent(id=tc_id, name=name, arguments=args):
                tool_calls.append({
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args),
                    },
                })

    result: dict[str, Any] = {"role": "assistant"}
    result["content"] = "\n".join(text_parts) if text_parts else None
    if tool_calls:
        result["tool_calls"] = tool_calls
    return [result]


class OpenAIAzureV1:
    def serialize(self, gen_input: GenerationInput) -> dict[str, Any]:
        payload: dict[str, Any] = {}

        messages: list[dict[str, Any]] = []

        if gen_input.system_prompt:
            messages.append({"role": "system", "content": gen_input.system_prompt})

        for msg in gen_input.conversation:
            if msg.source == MessageSource.SYSTEM:
                messages.append(
                    {"role": "system", "content": _extract_text(msg)}
                )
            elif msg.source == MessageSource.USER:
                messages.append(
                    {"role": "user", "content": _serialize_user_content(msg)}
                )
            elif msg.source == MessageSource.MODEL:
                messages.extend(_serialize_model_messages(msg))
            elif msg.source == MessageSource.TOOL:
                for content in msg.contents:
                    match content:
                        case ToolResponseContent(
                            tool_call_id=call_id, content=text
                        ):
                            messages.append({
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": text,
                            })
        payload["messages"] = messages

        cfg = gen_input.llm_config
        if cfg.temperature is not None:
            payload["temperature"] = cfg.temperature
        if cfg.top_p is not None:
            payload["top_p"] = cfg.top_p
        payload["max_completion_tokens"] = cfg.max_tokens
        if cfg.stop:
            payload["stop"] = cfg.stop
        if cfg.presence_penalty is not None:
            payload["presence_penalty"] = cfg.presence_penalty
        if cfg.frequency_penalty is not None:
            payload["frequency_penalty"] = cfg.frequency_penalty
        if cfg.candidates > 1:
            payload["n"] = cfg.candidates

        if cfg.thinking != ThinkingLevel.NO:
            payload["reasoning_effort"] = _THINKING_MAP.get(
                cfg.thinking, "medium"
            )

        if gen_input.tool_config and gen_input.tool_config.tools:
            tools: list[dict[str, Any]] = []
            for tool in gen_input.tool_config.tools:
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                        },
                    }
                )
            payload["tools"] = tools
            payload["parallel_tool_calls"] = gen_input.tool_config.parallel_calling
            payload["tool_choice"] = _TOOL_MODE_MAP[gen_input.tool_config.mode]

        output_type = gen_input.output_type
        if isinstance(output_type, type) and hasattr(output_type, "model_json_schema"):
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": output_type.__name__,
                    "schema": output_type.model_json_schema(),
                    "strict": True,
                },
            }

        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}

        return payload

    async def parse_stream(
        self,
        lines: AsyncIterator[str],
    ) -> AsyncIterator[StreamDelta | GenerationOutput]:
        accumulated: list[Content] = []
        finish_reason = FinishReason.STOP
        usage_meta: dict[str, Any] = {}
        model_name = ""
        completion_id = ""

        tool_calls_acc: dict[int, _ToolCallAccumulator] = {}

        async for line in lines:
            if not line.startswith("data: "):
                continue
            raw = line.removeprefix("data: ").strip()
            if not raw or raw == "[DONE]":
                continue

            chunk: dict[str, Any] = json.loads(raw)

            if cid := chunk.get("id"):
                completion_id = cid
            if model := chunk.get("model"):
                model_name = model

            for choice in chunk.get("choices", []):
                if fr := choice.get("finish_reason"):
                    finish_reason = _FINISH_REASON_MAP.get(
                        fr, FinishReason.ERROR
                    )

                delta = choice.get("delta", {})

                if reasoning := delta.get("reasoning_content"):
                    yield ThinkingDelta(thinking=reasoning)
                    accumulated.append(ThinkingContent(thinking=reasoning))

                if text := delta.get("content"):
                    yield TextDelta(text=text)
                    accumulated.append(TextContent(text=text))

                for tc in delta.get("tool_calls", []):
                    idx = tc.get("index", 0)
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = _ToolCallAccumulator(
                            id=tc.get("id", ""),
                            name=tc.get("function", {}).get("name", ""),
                            arguments="",
                        )
                    acc = tool_calls_acc[idx]
                    if tc_id := tc.get("id"):
                        acc.id = tc_id
                    if fn := tc.get("function"):
                        if fn_name := fn.get("name"):
                            acc.name = fn_name
                        acc.arguments += fn.get("arguments", "")

            if usage := chunk.get("usage"):
                usage_meta = usage

        for acc in tool_calls_acc.values():
            try:
                arguments = json.loads(acc.arguments) if acc.arguments else {}
            except json.JSONDecodeError:
                arguments = {"raw": acc.arguments}
            accumulated.append(
                ToolCallContent(
                    id=acc.id or uuid.uuid4().hex[:12],
                    name=acc.name,
                    arguments=arguments,
                )
            )

        if tool_calls_acc:
            finish_reason = FinishReason.TOOL_USE

        message = Message(source=MessageSource.MODEL, contents=accumulated)

        prompt_details = usage_meta.get("prompt_tokens_details", {})
        completion_details = usage_meta.get("completion_tokens_details", {})

        parsed_usage = Usage(
            input_tokens=usage_meta.get("prompt_tokens", 0),
            output_tokens=usage_meta.get("completion_tokens", 0),
            thinking_tokens=completion_details.get("reasoning_tokens", 0),
            cache=CacheUsage(
                read_tokens=prompt_details.get("cached_tokens", 0),
            ),
            total_tokens=usage_meta.get("total_tokens", 0),
        )

        yield GenerationOutput(
            id=completion_id or uuid.uuid4().hex,
            model=model_name,
            message=message,
            finish_reason=finish_reason,
            usage=parsed_usage,
        )
