# Func-LLM

A functional-oriented Python library for LLM calling with multi-provider and multi-model support.

Provides a unified, provider-agnostic interface over **OpenAI** (Azure), **Anthropic**, **Gemini**, and **Mistral** — all accessed through Vertex AI or Azure endpoints.

## Project Structure

```
func_llm/
  models/
    IO/
      message.py              # Shared message model (text, media, tool calls, thinking, errors)
      input/
        main.py               # GenerationInput, LLMConfig, OutputType
        provider.py            # Provider enum, Region, LLMModel
        tools.py               # Tool, ToolsConfig, ToolsCallingMode
        image.py               # ImageConfig (ratio, resolution, person generation)
      output/
        main.py               # GenerationOutput, FinishReason
        usage.py              # Usage, CacheUsage (token counts)
        citation.py           # Citation, CitationType, TextSpan (grounding)
        safety.py             # SafetyResult, SafetyRating (content filtering)
        stream.py             # TextDelta, ThinkingDelta (streaming chunks)
```

## IO Models

### Input

`GenerationInput` is the unified request object. It carries:

- **model** — provider, endpoint, and model identification (`LLMModel`)
- **conversation** — list of `Message` with typed content blocks (text, media, tool calls, tool responses, thinking, errors)
- **llm_config** — generation parameters (temperature, top_p, top_k, max_tokens, stop sequences, thinking level)
- **tool_config** — function calling tools, mode (auto/any/none), parallel calling
- **image_config** — image generation settings (ratio, resolution, person generation, mime type)
- **output_type** — text, image, hybrid, or a Pydantic `BaseModel` for structured output
- **system_prompt** — extracted from provider-specific locations into a dedicated field
- **stream** — whether to stream the response

### Output

`GenerationOutput` is the unified response object. It carries:

- **message** — the model's response as a `Message` (directly appendable to conversation history)
- **finish_reason** — why generation stopped (stop, max_tokens, tool_use, content_filter, error)
- **usage** — token breakdown (input, output, thinking, cache read/creation)
- **citations** — grounding annotations (URL, document, search) with text spans and confidence
- **safety** — content filtering results with per-category ratings and refusal details

### Streaming

During streaming, the library yields unified `StreamDelta` events (`TextDelta` or `ThinkingDelta`) on the fly, then returns the aggregated `GenerationOutput` at the end.

## Supported Providers

| Provider | Enum | Endpoint |
|----------|------|----------|
| OpenAI | `openai` | Azure OpenAI |
| Anthropic | `anthropic` | Vertex AI |
| Gemini | `gemini` | Vertex AI |
| Mistral | `mistral` | Vertex AI |

## Requirements

- Python >= 3.13
- pydantic >= 2
- httpx
