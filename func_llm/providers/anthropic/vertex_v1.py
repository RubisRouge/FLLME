from __future__ import annotations

import dataclasses
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from ...models.input import GenerationInput, ThinkingLevel, ToolsCallingMode
from ..base import accumulate_content
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
    "end_turn": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "max_tokens": FinishReason.MAX_TOKENS,
    "tool_use": FinishReason.TOOL_USE,
    "refusal": FinishReason.CONTENT_FILTER,
    "pause_turn": FinishReason.STOP,
}

_THINKING_BUDGET: dict[ThinkingLevel, int] = {
    ThinkingLevel.LOW: 1024,
    ThinkingLevel.MEDIUM: 8192,
    ThinkingLevel.HIGH: 32768,
}


@dataclasses.dataclass
class _ToolInputAccumulator:
    id: str
    name: str
    partial_json: str


def _serialize_content_blocks(
    msg: Message,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for content in msg.contents:
        match content:
            case TextContent(text=text):
                blocks.append({"type": "text", "text": text})
            case MediaContent(source=Base64Source(data=data), media_type=mt):
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mt,
                        "data": data,
                    },
                })
            case ThinkingContent(thinking=thinking, signature=sig):
                if sig:
                    blocks.append({
                        "type": "thinking",
                        "thinking": thinking,
                        "signature": sig,
                    })
            case ToolCallContent(id=tc_id, name=name, arguments=args):
                blocks.append({
                    "type": "tool_use",
                    "id": tc_id,
                    "name": name,
                    "input": args,
                })
    return blocks


def _serialize_tool_results(msg: Message) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for content in msg.contents:
        match content:
            case ToolResponseContent(tool_call_id=call_id, content=text):
                blocks.append({
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": text,
                })
    return blocks


class AnthropicVertexV1:
    def serialize(self, gen_input: GenerationInput) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "anthropic_version": "vertex-2023-10-16",
        }

        messages: list[dict[str, Any]] = []

        for msg in gen_input.conversation:
            match msg.source:
                case MessageSource.SYSTEM:
                    continue
                case MessageSource.USER:
                    has_media = any(
                        isinstance(c, MediaContent) for c in msg.contents
                    )
                    if has_media:
                        messages.append({
                            "role": "user",
                            "content": _serialize_content_blocks(msg),
                        })
                    else:
                        texts = [
                            c.text
                            for c in msg.contents
                            if isinstance(c, TextContent)
                        ]
                        messages.append({
                            "role": "user",
                            "content": "\n".join(texts),
                        })
                case MessageSource.MODEL:
                    blocks = _serialize_content_blocks(msg)
                    if blocks:
                        messages.append({
                            "role": "assistant",
                            "content": blocks,
                        })
                case MessageSource.TOOL:
                    tool_results = _serialize_tool_results(msg)
                    if tool_results:
                        messages.append({
                            "role": "user",
                            "content": tool_results,
                        })

        payload["messages"] = messages

        if gen_input.system_prompt:
            payload["system"] = gen_input.system_prompt

        cfg = gen_input.llm_config
        if cfg.temperature is not None:
            payload["temperature"] = cfg.temperature
        if cfg.top_p is not None:
            payload["top_p"] = cfg.top_p
        if cfg.top_k is not None:
            payload["top_k"] = cfg.top_k
        payload["max_tokens"] = cfg.max_tokens
        if cfg.stop:
            payload["stop_sequences"] = cfg.stop

        if cfg.thinking != ThinkingLevel.NO:
            budget = _THINKING_BUDGET.get(cfg.thinking)
            if budget is not None:
                payload["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget,
                }
            else:
                payload["thinking"] = {"type": "adaptive"}

        if gen_input.tool_config and gen_input.tool_config.tools:
            tools: list[dict[str, Any]] = []
            for tool in gen_input.tool_config.tools:
                tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                })
            payload["tools"] = tools

            match gen_input.tool_config.mode:
                case ToolsCallingMode.AUTO:
                    payload["tool_choice"] = {"type": "auto"}
                case ToolsCallingMode.ANY:
                    payload["tool_choice"] = {"type": "any"}
                case ToolsCallingMode.NONE:
                    payload["tool_choice"] = {"type": "none"}

        output_type = gen_input.output_type
        if isinstance(output_type, type) and hasattr(
            output_type, "model_json_schema"
        ):
            payload["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": output_type.model_json_schema(),
                },
            }

        payload["stream"] = True

        return payload

    async def parse_stream(
        self,
        lines: AsyncIterator[str],
    ) -> AsyncIterator[StreamDelta | GenerationOutput]:
        accumulated: list[Content] = []
        finish_reason = FinishReason.STOP
        input_usage: dict[str, Any] = {}
        output_tokens = 0
        model_name = ""
        message_id = ""

        current_block_type = ""
        tool_acc: _ToolInputAccumulator | None = None

        event_type = ""

        async for line in lines:
            if line.startswith("event: "):
                event_type = line.removeprefix("event: ").strip()
                continue
            if not line.startswith("data: "):
                continue
            raw = line.removeprefix("data: ").strip()
            if not raw:
                continue

            chunk: dict[str, Any] = json.loads(raw)

            match event_type:
                case "message_start":
                    msg_meta = chunk.get("message", {})
                    message_id = msg_meta.get("id", "")
                    model_name = msg_meta.get("model", "")
                    input_usage = msg_meta.get("usage", {})

                case "content_block_start":
                    block = chunk.get("content_block", {})
                    current_block_type = block.get("type", "")
                    if current_block_type == "tool_use":
                        tool_acc = _ToolInputAccumulator(
                            id=block.get("id", ""),
                            name=block.get("name", ""),
                            partial_json="",
                        )

                case "content_block_delta":
                    delta = chunk.get("delta", {})
                    delta_type = delta.get("type", "")

                    match delta_type:
                        case "thinking_delta":
                            thinking = delta.get("thinking", "")
                            if thinking:
                                yield ThinkingDelta(thinking=thinking)
                                accumulate_content(accumulated, ThinkingContent(thinking=thinking))
                        case "signature_delta":
                            sig = delta.get("signature", "")
                            if sig and accumulated:
                                last = accumulated[-1]
                                if isinstance(last, ThinkingContent):
                                    accumulated[-1] = ThinkingContent(
                                        thinking=last.thinking,
                                        signature=sig,
                                    )
                        case "text_delta":
                            text = delta.get("text", "")
                            if text:
                                yield TextDelta(text=text)
                                accumulate_content(accumulated, TextContent(text=text))
                        case "input_json_delta":
                            if tool_acc is not None:
                                tool_acc.partial_json += delta.get(
                                    "partial_json", ""
                                )

                case "content_block_stop":
                    if (
                        current_block_type == "tool_use"
                        and tool_acc is not None
                    ):
                        try:
                            arguments = (
                                json.loads(tool_acc.partial_json)
                                if tool_acc.partial_json
                                else {}
                            )
                        except json.JSONDecodeError:
                            arguments = {"raw": tool_acc.partial_json}
                        accumulated.append(
                            ToolCallContent(
                                id=tool_acc.id
                                or uuid.uuid4().hex[:12],
                                name=tool_acc.name,
                                arguments=arguments,
                            )
                        )
                        tool_acc = None
                    current_block_type = ""

                case "message_delta":
                    delta = chunk.get("delta", {})
                    if sr := delta.get("stop_reason"):
                        finish_reason = _FINISH_REASON_MAP.get(
                            sr, FinishReason.ERROR
                        )
                    if usage := chunk.get("usage"):
                        output_tokens = usage.get("output_tokens", 0)

                case "ping" | "message_stop":
                    pass

        message = Message(source=MessageSource.MODEL, contents=accumulated)

        thinking_tokens = (
            input_usage.get("output_tokens_details", {}).get(
                "thinking_tokens", 0
            )
        )

        parsed_usage = Usage(
            input_tokens=input_usage.get("input_tokens", 0),
            output_tokens=output_tokens
            or input_usage.get("output_tokens", 0),
            thinking_tokens=thinking_tokens,
            cache=CacheUsage(
                read_tokens=input_usage.get(
                    "cache_read_input_tokens", 0
                ),
                creation_tokens=input_usage.get(
                    "cache_creation_input_tokens", 0
                ),
            ),
        )

        yield GenerationOutput(
            id=message_id or uuid.uuid4().hex,
            model=model_name,
            message=message,
            finish_reason=finish_reason,
            usage=parsed_usage,
        )
