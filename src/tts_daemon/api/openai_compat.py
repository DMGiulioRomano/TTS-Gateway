"""OpenAI-compatible speech endpoint: ``POST /v1/audio/speech``.

Any tool that already speaks the OpenAI text-to-speech API gets free, private,
local voices by pointing its ``base_url`` at the gateway::

    from openai import OpenAI
    client = OpenAI(base_url="http://127.0.0.1:5111/v1", api_key="unused")
    client.audio.speech.create(model="tts-1", input="Hello", voice="alloy")

The route is a thin translation layer over
:meth:`~tts_daemon.core.service.SpeechService.synthesize`; the gateway core is
unchanged. Mapping rules are documented in ``docs/api.md``.
"""

from __future__ import annotations

import anyio.to_thread
from fastapi import APIRouter, Request, Response

from tts_daemon.api.schemas import ErrorResponse, OpenAISpeechRequest
from tts_daemon.config import GatewayConfig
from tts_daemon.core.service import SpeechService

router = APIRouter()

#: OpenAI's built-in voice names. When one of these is requested but not mapped
#: in ``openai_compat.voice_aliases``, we fall back to the gateway default voice
#: rather than passing a name no local engine knows.
_OPENAI_VOICES = frozenset(
    {"alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"}
)


def _resolve_provider(service: SpeechService, model: str) -> str | None:
    """A registered provider name is honored; anything else uses the default."""
    return model if model and model in service.registry else None


def _resolve_voice(voice: str, aliases: dict[str, str]) -> str | None:
    """Map an OpenAI voice name to a real voice id (or the default)."""
    if not voice:
        return None
    if voice in aliases:
        return aliases[voice] or None
    if voice in _OPENAI_VOICES:
        return None  # an OpenAI name with no alias -> gateway default voice
    return voice  # assume a real provider voice id, pass it through


@router.post(
    "/audio/speech",
    responses={
        200: {"content": {"audio/wav": {}}, "description": "Synthesized audio bytes"},
        422: {"model": ErrorResponse, "description": "Invalid request / unsupported format"},
        502: {"model": ErrorResponse, "description": "Synthesis failed"},
        503: {"model": ErrorResponse, "description": "Provider unavailable"},
    },
    summary="OpenAI-compatible speech synthesis",
    tags=["speech"],
)
async def audio_speech(body: OpenAISpeechRequest, request: Request) -> Response:
    service: SpeechService = request.app.state.service
    config: GatewayConfig = request.app.state.config

    response_format = (body.response_format or "wav").lower()
    if response_format != "wav":
        # Raised as ValueError -> 422 by the app-level handler.
        raise ValueError(
            f"response_format {response_format!r} is not supported yet "
            "(only 'wav'); omit it or set 'wav'."
        )

    provider = _resolve_provider(service, body.model)
    voice = _resolve_voice(body.voice, config.openai_compat.voice_aliases)

    clip = await anyio.to_thread.run_sync(
        lambda: service.synthesize(body.input, provider=provider, voice=voice, speed=body.speed)
    )
    return Response(content=clip.data, media_type=clip.media_type)
