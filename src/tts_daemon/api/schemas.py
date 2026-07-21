"""Pydantic request/response models for the HTTP and WebSocket APIs.

These DTOs are the public wire contract (documented in docs/api.md and the
generated OpenAPI schema). Keep changes backwards-compatible: add optional
fields, never repurpose existing ones.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SpeakRequest(BaseModel):
    """Body of ``POST /v1/speak``."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="Text to speak.", min_length=1)
    provider: str | None = Field(
        default=None,
        description="Provider name; defaults to the configured default (or 'auto').",
    )
    voice: str | None = Field(default=None, description="Provider-specific voice id.")
    speed: float = Field(default=1.0, gt=0, le=10, description="Rate multiplier; 1.0 is normal.")
    options: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific options passed through untouched."
    )
    interrupt: bool = Field(
        default=False, description="Cancel queued and current speech before this utterance."
    )
    wait: bool = Field(
        default=False,
        description="Do not return until the utterance finished (or failed/was cancelled).",
    )


class SynthesizeRequest(BaseModel):
    """Body of ``POST /v1/synthesize`` (returns audio bytes, plays nothing)."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="Text to synthesize.", min_length=1)
    provider: str | None = None
    voice: str | None = None
    speed: float = Field(default=1.0, gt=0, le=10)
    options: dict[str, Any] = Field(default_factory=dict)


class OpenAISpeechRequest(BaseModel):
    """Body of ``POST /v1/audio/speech`` (OpenAI ``audio.speech.create`` schema).

    Unknown fields (e.g. ``instructions``) are ignored so real OpenAI clients
    work unchanged; only ``input`` is required.
    """

    model_config = ConfigDict(extra="ignore")

    input: str = Field(description="Text to synthesize.", min_length=1)
    model: str = Field(
        default="",
        description="A registered provider name is honored; 'tts-1'/'tts-1-hd'/"
        "unknown fall back to the gateway default provider.",
    )
    voice: str = Field(
        default="",
        description="OpenAI voice name (mapped via openai_compat.voice_aliases) "
        "or a real provider voice id; empty uses the default voice.",
    )
    response_format: str | None = Field(
        default=None, description="Only 'wav' is supported for now; others return 422."
    )
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Rate multiplier (0.25-4.0).")


class UtteranceModel(BaseModel):
    """Snapshot of an utterance's state."""

    id: str
    text: str
    provider: str
    voice: str | None = None
    speed: float = 1.0
    state: str
    error: str | None = None
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None


class SpeakResponse(BaseModel):
    utterance: UtteranceModel


class StopResponse(BaseModel):
    cancelled: int = Field(description="Utterances cancelled (queued + interrupted).")


class VoiceModel(BaseModel):
    id: str
    name: str
    language: str | None = None
    description: str | None = None
    provider: str


class VoicesResponse(BaseModel):
    voices: list[VoiceModel]


class ProviderModel(BaseModel):
    name: str
    available: bool
    reason: str | None = Field(default=None, description="Why the provider is unavailable.")
    default: bool = Field(description="Whether this provider is the resolved default.")


class ProvidersResponse(BaseModel):
    providers: list[ProviderModel]


class QueueModel(BaseModel):
    current: UtteranceModel | None = None
    queued: list[UtteranceModel]
    history: list[UtteranceModel]
    size: int
    max_size: int


class CacheModel(BaseModel):
    entries: int
    size_mb: float
    hits: int
    misses: int


class StatusResponse(BaseModel):
    queue: QueueModel
    default_provider: str | None
    default_provider_error: str | None = None
    playback_available: bool
    cache: CacheModel | None = Field(
        default=None, description="Synthesis cache stats, or null when the cache is disabled."
    )


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str


class ErrorResponse(BaseModel):
    detail: str
