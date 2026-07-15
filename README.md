# Func-LLM

A functional-oriented Python library for LLM calling with multi-provider and multi-model support.

Provides a unified, provider-agnostic interface over **Anthropic**, **Gemini**, **Mistral** (Vertex AI), and **OpenAI** (Azure) through a single data-driven configuration layer.

## Quick Start

```python
import asyncio
import fllm

from fflm.auth.google import GoogleADCAuthResolver


async def main():
    # 1. Create the service (default SQLite backend)
    google_adc = GoogleADCAuthResolver()
    service = await create_default_deployment_service()
    # service = await fllm.DeploymentService.from_sqlite("deployments.db")
    # 2. Register a model and its deployment
    await service.add_model(fllm.LLMModel(
        id="claude-sonnet-4",
        name="Claude Sonnet 4",
        provider=fllm.Provider.ANTHROPIC,
    ))
    await service.add_deployment(
      fllm.Deployment(
        id="claude-vertex-euw1",
        url="https://europe-west1-aiplatform.googleapis.com/v1/projects/my-project/...",
        model_id="claude-sonnet-4",
        adapter=fllm.AdapterType.ANTHROPIC_VERTEX_V1,
        auth_id="google_adc",
      )
    )
    # 3. Generate
    gen_input = fllm.GenerationInput(
        model="claude-sonnet-4",
        conversation=[fllm.Message(source="user", contents=[...])],
    )
    output = await service.generate(
      gen_input,
      auth_resolver=google_adc,
    )

asyncio.run(main())
```

## Architecture

The design is **data-driven**: models, deployments, and auth are stored as data (not coded as classes). Users register entries in a database; the library resolves them at generation time.

```
LLMModel ──┐
            ├── DeploymentService ── generate()
Deployment ─┤        │
            │        ├── resolves model → deployment → adapter
AuthPrinciple ┘      └── resolves auth → headers
```

The flow: `GenerationInput` → `DeploymentService` resolves the deployment → `Adapter` serializes → HTTP call → `Adapter` deserializes → `GenerationOutput`.

## Project Structure

```
func_llm/
  models/
    model.py              # LLMModel (id, name)
    deployment.py         # AdapterType enum, Deployment (url, model_id, adapter, auth_id)
    auth.py               # AuthPrinciple, built-in principles (google_adc, api_key)
    message.py            # Message, Content blocks (text, media, tool calls, thinking, errors)
    input/
      main.py             # ThinkingLevel, LLMConfig, BasicOutputType, GenerationInput
      tools.py            # ToolsCallingMode, Tool, ToolsConfig
      image.py            # Ratio, Resolution, PersonGeneration, MimeType, ImageConfig
    output/
      main.py             # FinishReason, GenerationOutput
      usage.py            # CacheUsage, Usage
      citation.py         # CitationType, TextSpan, Citation
      safety.py           # SafetyCategory, SafetySeverity, SafetyRating, SafetyResult
      stream.py           # StreamEventType, TextDelta, ThinkingDelta, StreamDelta
  auth/
    protocol.py           # AuthResolver protocol
    google_adc.py         # Google ADC resolver (asyncio.to_thread wrapping google.auth)
    api_key.py            # API key resolver (reads env var, returns header)
  store/
    deployments/
      protocol.py         # ModelRepository, DeploymentRepository, AuthRepository protocols
      sqlite.py           # SQLiteStore — default async SQLite implementation
  providers/
    base.py               # Adapter protocol (serialize, parse_stream)
    anthropic/vertex_v1.py
    gemini/vertex_v1.py
    mistral/vertex_v1.py
    openai/azure_v1.py
  service.py              # DeploymentService — composes repos + auth resolution
  generate.py             # configure(), generate() — main entry point
  media.py                # MediaResolver protocol, resolve_references(), store_media()
  http.py                 # Shared httpx async client
  errors.py               # FuncLLMError hierarchy
```

## Domain Models

### LLMModel

Simple identity — an ID, a display name, and the provider:

```python
LLMModel(id="claude-sonnet-4", name="Claude Sonnet 4", provider=Provider.ANTHROPIC)
```

### Deployment

Links a model to a concrete cloud endpoint, an adapter for serialization, and an auth method:

```python
Deployment(
    id="claude-vertex-euw1",
    url="https://europe-west1-aiplatform.googleapis.com/v1/...",
    model_id="claude-sonnet-4",
    adapter=AdapterType.ANTHROPIC_VERTEX_V1,
    auth_id="google_adc",
)
```

### AuthPrinciple

Data-driven auth configuration. Declares which resolver to use, which env vars are required, and resolver-specific config:

```python
AuthPrinciple(
    id="azure_openai_key",
    name="Azure OpenAI API Key",
    resolver_id="api_key",
    required_env_vars=["AZURE_OPENAI_KEY"],
    config={"header_name": "api-key"},
)
```

Built-in principles (`google_adc`, `api_key`) are pre-seeded in the default SQLite store.

## IO Models

### Input

`GenerationInput` is the unified request object. It carries:

- **model** — model ID string (resolved to a `Deployment` by `DeploymentService`)
- **conversation** — list of `Message` with typed content blocks (text, media, tool calls, tool responses, thinking, errors). Each message has a `source` field (`user | model | system | tool`)
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

### Media Resolution

`MediaContent` blocks can carry three source types: `Base64Source`, `UrlSource`, or `ReferenceSource`. References are opaque user-domain IDs (e.g. a document-management key) that providers cannot consume directly.

The `MediaResolver` protocol bridges this gap:

```python
class MediaResolver(Protocol):
    async def resolve(self, references: list[ReferenceSource]) -> list[Base64Source | UrlSource]: ...
    async def store(self, media: list[Base64Source | UrlSource]) -> list[ReferenceSource]: ...
```

- **`resolve`** — outbound: converts user-domain IDs into provider-sendable sources before serialization
- **`store`** — inbound: uploads AI-generated media and returns user-domain IDs after deserialization

Pass an implementation to `generate()` via the `media_resolver` keyword argument. Resolution and storage are batched and applied transparently. Resolver failures are wrapped in `MediaResolutionError`.

## Deployment & Storage

### Repository Protocols

Three separate async protocols define data access — users can swap the storage backend by implementing them:

- `ModelRepository` — CRUD for `LLMModel`
- `DeploymentRepository` — CRUD for `Deployment`, plus `get_for_model()`
- `AuthRepository` — CRUD for `AuthPrinciple`

### SQLiteStore (default)

`SQLiteStore` is the built-in implementation using `aiosqlite`. It manages three tables with foreign keys (deployments cascade-delete when a model is removed) and pre-seeds built-in auth principles.

```python
store = await SQLiteStore.create("deployments.db")  # or ":memory:"
```

### DeploymentService

Composes the three repositories and auth resolution into a high-level API:

```python
service = await DeploymentService.from_sqlite("deployments.db")

await service.add_model(model)
await service.add_deployment(deployment)  # validates model_id + auth_id exist
deployment = await service.resolve_deployment("claude-sonnet-4")
headers = await service.get_auth_headers(deployment)
issues = await service.check_deployment_ready(deployment)  # missing env vars, etc.
```

## Auth System

Auth is data-driven and extensible. Each `AuthPrinciple` references a `resolver_id` that maps to a registered `AuthResolver`:

```python
class AuthResolver(Protocol):
    async def get_headers(self, principle: AuthPrinciple) -> dict[str, str]: ...
    def check_env(self, principle: AuthPrinciple) -> list[str]: ...
```

Built-in resolvers:

| Resolver ID | Resolver | Description |
|---|---|---|
| `google_adc` | `GoogleADCResolver` | Google Application Default Credentials via `google.auth` |
| `api_key` | `ApiKeyResolver` | Reads an env var, returns it as a header |

Register custom resolvers:

```python
func_llm.register_resolver("my_oauth", MyOAuthResolver())
```

## Adapters

Each provider/cloud/version combination has its own adapter implementing the `Adapter` protocol (`serialize`, `parse_stream`).

| AdapterType | Provider | Cloud |
|---|---|---|
| `anthropic_vertex_v1` | Anthropic | Vertex AI |
| `gemini_vertex_v1` | Gemini | Vertex AI |
| `mistral_vertex_v1` | Mistral | Vertex AI |
| `openai_azure_v1` | OpenAI | Azure |

New API versions get new enum values (e.g., `anthropic_vertex_v2`). Adapters are looked up via `get_adapter(adapter_type)`.

## Requirements

- Python >= 3.13
- pydantic >= 2.13
- httpx >= 0.28
- aiosqlite >= 0.20
- google-auth >= 2.55
- requests >= 2.32
