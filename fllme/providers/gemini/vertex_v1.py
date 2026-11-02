from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from ...models.input import (
    BasicOutputType,
    GenerationInput,
    OutputType,
    ThinkingLevel,
    Tool,
    ToolsCallingMode,
    WebSearchTool,
)
from ..base import accumulate_content, is_structured, parse_structured
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
    WebSearchQueryContent,
    UrlSource,
)
from ...models.output import (
    Citation,
    CitationType,
    FinishReason,
    GenerationOutput,
    MediaDelta,
    SafetyCategory,
    SafetyRating,
    SafetyResult,
    SafetySeverity,
    StreamDelta,
    TextDelta,
    TextSpan,
    ThinkingDelta,
    Usage,
)

_THINKING_BUDGET: dict[ThinkingLevel, int] = {
    ThinkingLevel.LOW: 1024,
    ThinkingLevel.MEDIUM: 8192,
    ThinkingLevel.HIGH: 32768,
}

_SAFETY_CATEGORY_MAP: dict[str, SafetyCategory] = {
    "HARM_CATEGORY_HARASSMENT": SafetyCategory.HARASSMENT,
    "HARM_CATEGORY_HATE_SPEECH": SafetyCategory.HATE,
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": SafetyCategory.SEXUAL,
    "HARM_CATEGORY_DANGEROUS_CONTENT": SafetyCategory.DANGEROUS,
    "HARM_CATEGORY_CIVIC_INTEGRITY": SafetyCategory.OTHER,
}

_SAFETY_SEVERITY_MAP: dict[str, SafetySeverity] = {
    "NEGLIGIBLE": SafetySeverity.SAFE,
    "LOW": SafetySeverity.LOW,
    "MEDIUM": SafetySeverity.MEDIUM,
    "HIGH": SafetySeverity.HIGH,
}

_FINISH_REASON_MAP: dict[str, FinishReason] = {
    "STOP": FinishReason.STOP,
    "MAX_TOKENS": FinishReason.MAX_TOKENS,
    "SAFETY": FinishReason.CONTENT_FILTER,
    "RECITATION": FinishReason.CONTENT_FILTER,
    "BLOCKLIST": FinishReason.CONTENT_FILTER,
    "PROHIBITED_CONTENT": FinishReason.CONTENT_FILTER,
    "SPII": FinishReason.CONTENT_FILTER,
}

_TOOL_MODE_MAP: dict[ToolsCallingMode, str] = {
    ToolsCallingMode.AUTO: "AUTO",
    ToolsCallingMode.ANY: "ANY",
    ToolsCallingMode.NONE: "NONE",
}


class GeminiVertexV1:
    def serialize(self, gen_input: GenerationInput) -> dict[str, Any]:
        payload: dict[str, Any] = {}

        tool_name_map = _build_tool_name_map(gen_input.conversation)

        contents: list[dict[str, Any]] = []
        for msg in gen_input.conversation:
            if msg.source == MessageSource.SYSTEM:
                continue
            parts = _serialize_parts(msg, tool_name_map)
            if parts:
                role = "model" if msg.source == MessageSource.MODEL else "user"
                contents.append({"role": role, "parts": parts})
        payload["contents"] = contents

        if gen_input.system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": gen_input.system_prompt}],
            }

        gen_config = _build_generation_config(gen_input)
        if gen_config:
            payload["generationConfig"] = gen_config

        if gen_input.tool_config and gen_input.tool_config.tools:
            func_decls: list[dict[str, Any]] = []
            tool_entries: list[dict[str, Any]] = []
            for tool in gen_input.tool_config.tools:
                match tool:
                    case WebSearchTool(exclude_domains=exclude_domains):
                        web_search: dict[str, Any] = {}
                        if exclude_domains:
                            web_search["excludeDomains"] = exclude_domains
                        tool_entries.append({"enterpriseWebSearch": web_search})
                    case Tool(
                        name=name, description=description, parameters=parameters
                    ):
                        decl: dict[str, Any] = {
                            "name": name,
                            "description": description,
                        }
                        if parameters:
                            decl["parameters"] = parameters
                        func_decls.append(decl)

            if func_decls:
                tool_entries.insert(0, {"functionDeclarations": func_decls})
            payload["tools"] = tool_entries

            if func_decls:
                mode = _TOOL_MODE_MAP.get(gen_input.tool_config.mode)
                if mode:
                    payload["toolConfig"] = {
                        "functionCallingConfig": {"mode": mode},
                    }

        return payload

    async def parse_stream(
        self,
        output_type: OutputType,
        lines: AsyncIterator[str],
    ) -> AsyncIterator[StreamDelta | GenerationOutput]:
        accumulated: list[Content] = []
        finish_reason = FinishReason.STOP
        usage_meta: dict[str, Any] = {}
        safety_ratings: list[dict[str, Any]] = []
        grounding_meta: dict[str, Any] = {}
        model_name = ""
        has_tool_calls = False
        search_queries: list[str] = []

        async for line in lines:
            if not line.startswith("data: "):
                continue
            raw = line.removeprefix("data: ").strip()
            if not raw or raw == "[DONE]":
                continue

            chunk: dict[str, Any] = json.loads(raw)

            if model_version := chunk.get("modelVersion"):
                model_name = model_version

            for candidate in chunk.get("candidates", []):
                if fr := candidate.get("finishReason"):
                    finish_reason = _FINISH_REASON_MAP.get(fr, FinishReason.ERROR)

                if ratings := candidate.get("safetyRatings"):
                    safety_ratings = ratings

                if gm := candidate.get("groundingMetadata"):
                    grounding_meta = gm
                    for query in gm.get("webSearchQueries") or []:
                        if query and query not in search_queries:
                            search_queries.append(query)

                for part in candidate.get("content", {}).get("parts", []):
                    if part.get("thought") and (text := part.get("text")):
                        sig = part.get("thoughtSignature") or part.get(
                            "thought_signature"
                        )
                        yield StreamDelta(delta=ThinkingDelta(thinking=text))
                        accumulate_content(
                            accumulated, ThinkingContent(thinking=text, signature=sig)
                        )
                    elif text := part.get("text"):
                        yield StreamDelta(delta=TextDelta(text=text))
                        accumulate_content(accumulated, TextContent(text=text))
                    elif inline := part.get("inlineData"):
                        mime = inline.get("mimeType", "application/octet-stream")
                        data = inline.get("data", "")
                        yield StreamDelta(delta=MediaDelta(media_type=mime, data=data))
                        accumulated.append(
                            MediaContent(
                                media_type=mime,
                                source=Base64Source(data=data),
                            )
                        )
                    elif fc := part.get("functionCall"):
                        has_tool_calls = True
                        sig = part.get("thoughtSignature") or part.get(
                            "thought_signature"
                        )
                        accumulated.append(
                            ToolCallContent(
                                id=fc.get("id", uuid.uuid4().hex[:12]),
                                name=fc["name"],
                                arguments=fc.get("args", {}),
                                thought_signature=sig,
                            )
                        )

            if um := chunk.get("usageMetadata"):
                usage_meta = um

        if has_tool_calls:
            finish_reason = FinishReason.TOOL_USE

        accumulated.extend(
            WebSearchQueryContent(query=query) for query in search_queries
        )

        message = Message(source=MessageSource.MODEL, contents=accumulated)

        usage = Usage(
            input_tokens=usage_meta.get("promptTokenCount", 0),
            output_tokens=usage_meta.get("candidatesTokenCount", 0),
            thinking_tokens=usage_meta.get("thoughtsTokenCount", 0),
            total_tokens=usage_meta.get("totalTokenCount", 0),
        )

        safety = _parse_safety_ratings(safety_ratings) if safety_ratings else None
        citations = _parse_grounding_metadata(grounding_meta) if grounding_meta else []

        output = GenerationOutput(
            id=uuid.uuid4().hex,
            model=model_name,
            message=message,
            finish_reason=finish_reason,
            usage=usage,
            safety=safety,
            citations=citations,
        )
        if is_structured(output_type):
            parse_structured(output, output_type)
        yield output


def _build_tool_name_map(conversation: list[Message]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for msg in conversation:
        for content in msg.contents:
            match content:
                case ToolCallContent(id=tc_id, name=name):
                    mapping[tc_id] = name
    return mapping


def _serialize_parts(
    msg: Message,
    tool_name_map: dict[str, str],
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for content in msg.contents:
        match content:
            case ThinkingContent(thinking=text, signature=sig):
                thought_part: dict[str, Any] = {"text": text, "thought": True}
                if sig:
                    thought_part["thoughtSignature"] = sig
                parts.append(thought_part)
            case TextContent(text=text):
                parts.append({"text": text})
            case MediaContent(source=Base64Source(data=data), media_type=mt):
                parts.append({"inlineData": {"mimeType": mt, "data": data}})
            case MediaContent(source=UrlSource(url=url), media_type=mt):
                parts.append({"fileData": {"mimeType": mt, "fileUri": url}})
            case ToolCallContent(name=name, arguments=args, thought_signature=sig):
                fc_part: dict[str, Any] = {"name": name, "args": args}
                part_dict: dict[str, Any] = {"functionCall": fc_part}
                if sig:
                    part_dict["thoughtSignature"] = sig
                parts.append(part_dict)
            case ToolResponseContent(tool_call_id=call_id, content=text):
                fn_name = tool_name_map.get(call_id, call_id)
                try:
                    response = json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    response = {"result": text}
                parts.append(
                    {"functionResponse": {"name": fn_name, "response": response}}
                )
    return parts


def _build_generation_config(gen_input: GenerationInput) -> dict[str, Any]:
    cfg = gen_input.llm_config
    gen_config: dict[str, Any] = {}

    if cfg.temperature is not None:
        gen_config["temperature"] = cfg.temperature
    if cfg.top_p is not None:
        gen_config["topP"] = cfg.top_p
    if cfg.top_k is not None:
        gen_config["topK"] = cfg.top_k
    gen_config["maxOutputTokens"] = cfg.max_tokens
    if cfg.stop:
        gen_config["stopSequences"] = cfg.stop
    if cfg.candidates > 1:
        gen_config["candidateCount"] = cfg.candidates
    if cfg.presence_penalty is not None:
        gen_config["presencePenalty"] = cfg.presence_penalty
    if cfg.frequency_penalty is not None:
        gen_config["frequencyPenalty"] = cfg.frequency_penalty

    if cfg.thinking != ThinkingLevel.NO:
        budget = _THINKING_BUDGET.get(cfg.thinking)
        if budget is not None:
            gen_config["thinkingConfig"] = {"thinkingBudget": budget}

    output_type = gen_input.output_type
    if isinstance(output_type, type) and hasattr(output_type, "model_json_schema"):
        gen_config["responseMimeType"] = "application/json"
        gen_config["responseJsonSchema"] = output_type.model_json_schema()
    elif output_type == BasicOutputType.IMAGE:
        gen_config["responseModalities"] = ["IMAGE"]
    elif output_type == BasicOutputType.HYBRID:
        gen_config["responseModalities"] = ["TEXT", "IMAGE"]

    if gen_input.image_config is not None:
        img = gen_input.image_config
        image_cfg: dict[str, Any] = {
            "aspectRatio": img.ratio.value if ":" in img.ratio.value else "auto",
            "imageSize": img.resolution.value,
            "personGeneration": img.person_generation.value,
            "imageOutputOptions": {"mimeType": img.mime_type.value},
        }
        gen_config["imageConfig"] = image_cfg

    return gen_config


def _parse_safety_ratings(ratings: list[dict[str, Any]]) -> SafetyResult:
    parsed: list[SafetyRating] = []
    blocked = False
    for rating in ratings:
        category = _SAFETY_CATEGORY_MAP.get(
            rating.get("category", ""), SafetyCategory.OTHER
        )
        probability = rating.get("probability", "NEGLIGIBLE")
        severity = _SAFETY_SEVERITY_MAP.get(probability, SafetySeverity.SAFE)
        is_blocked = rating.get("blocked", False)
        if is_blocked:
            blocked = True
        parsed.append(
            SafetyRating(
                category=category,
                severity=severity,
                filtered=is_blocked,
            )
        )
    return SafetyResult(ratings=parsed, blocked=blocked)


def _parse_grounding_metadata(grounding: dict[str, Any]) -> list[Citation]:
    chunks = grounding.get("groundingChunks", [])
    citations: list[Citation] = []
    for support in grounding.get("groundingSupports", []):
        segment = support.get("segment", {})
        span = TextSpan(
            start=segment.get("startIndex", 0),
            end=segment.get("endIndex", 0),
            text=segment.get("text"),
        )
        for idx in support.get("groundingChunkIndices", []):
            if idx >= len(chunks):
                continue
            web = chunks[idx].get("web", {})
            citations.append(
                Citation(
                    type=CitationType.SEARCH,
                    url=web.get("uri"),
                    title=web.get("title"),
                    span=span,
                )
            )
    return citations
