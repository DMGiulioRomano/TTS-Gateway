"""Kokoro provider — trending high-quality local neural TTS.

`Kokoro <https://github.com/thewh1teagle/kokoro-onnx>`_ (~82M params) runs
through ONNX Runtime on CPU with noticeably better prosody than Piper. It is an
optional extra (``pip install 'tts-daemon[kokoro]'``) and needs two downloadable
artifacts — the model (``kokoro-v1.0.onnx``) and the voices pack
(``voices-v1.0.bin``) — from the kokoro-onnx releases page.

Settings (``providers.kokoro`` in the config file):

``model_path``
    Path to the ``.onnx`` model (default
    ``$XDG_DATA_HOME/tts-daemon/kokoro/kokoro-v1.0.onnx``).
``voices_path``
    Path to the voices ``.bin`` (default alongside the model as
    ``voices-v1.0.bin``).
``default_voice``
    Voice used when a request names none (default ``af_sarah``).
``lang``
    Language passed to the engine (default ``en-us``).
"""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path

from tts_daemon.core.audio import float_pcm16_wav
from tts_daemon.core.errors import SynthesisError
from tts_daemon.core.interfaces import TTSProvider
from tts_daemon.core.models import AudioClip, AudioFormat, Availability, SynthesisRequest, Voice

logger = logging.getLogger(__name__)

_MODEL_NAME = "kokoro-v1.0.onnx"
_VOICES_NAME = "voices-v1.0.bin"
_DEFAULT_VOICE = "af_sarah"
_RELEASES = "https://github.com/thewh1teagle/kokoro-onnx/releases"
_INSTALL_HINT = "install with: pip install 'tts-daemon[kokoro]'"


def default_kokoro_dir() -> Path:
    """``$XDG_DATA_HOME/tts-daemon/kokoro`` with the ``~/.local/share`` fallback."""
    xdg = os.environ.get("XDG_DATA_HOME", "").strip()
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "tts-daemon" / "kokoro"


class KokoroProvider(TTSProvider):
    """Synthesize speech with the local Kokoro ONNX model."""

    name = "kokoro"

    def __init__(self, settings: dict | None = None) -> None:
        super().__init__(settings)
        directory = default_kokoro_dir()
        model = self.settings.get("model_path")
        voices = self.settings.get("voices_path")
        self._model_path = Path(model).expanduser() if model else directory / _MODEL_NAME
        self._voices_path = Path(voices).expanduser() if voices else directory / _VOICES_NAME
        self._default_voice = str(self.settings.get("default_voice") or _DEFAULT_VOICE)
        self._lang = str(self.settings.get("lang") or "en-us")
        self._engine = None  # lazily constructed on first use

    def synthesize(self, request: SynthesisRequest) -> AudioClip:
        if request.speed <= 0:
            raise SynthesisError(f"speed must be positive, got {request.speed}")
        if request.options:
            unknown = ", ".join(sorted(request.options))
            raise SynthesisError(f"The kokoro provider accepts no options (got: {unknown})")

        engine = self._get_engine()
        voice = request.voice or self._default_voice
        try:
            samples, sample_rate = engine.create(
                request.text, voice=voice, speed=request.speed, lang=self._lang
            )
        except Exception as exc:
            raise SynthesisError(f"kokoro synthesis failed for voice {voice!r}: {exc}") from exc
        return AudioClip(data=float_pcm16_wav(samples, int(sample_rate)), format=AudioFormat.WAV)

    def voices(self) -> list[Voice]:
        if not self.availability().available:
            return []
        try:
            names = self._get_engine().get_voices()
        except Exception:
            logger.exception("Listing kokoro voices failed")
            return []
        return [Voice(id=name, name=name, language=self._lang) for name in names]

    def availability(self) -> Availability:
        if importlib.util.find_spec("kokoro_onnx") is None:
            return Availability.unavailable(f"kokoro-onnx not installed ({_INSTALL_HINT})")
        if not self._model_path.is_file():
            return Availability.unavailable(
                f"kokoro model not found at {self._model_path} "
                f"(download {_MODEL_NAME} from {_RELEASES}, or set providers.kokoro.model_path)"
            )
        if not self._voices_path.is_file():
            return Availability.unavailable(
                f"kokoro voices file not found at {self._voices_path} "
                f"(download {_VOICES_NAME} from {_RELEASES}, or set providers.kokoro.voices_path)"
            )
        return Availability.ok()

    def close(self) -> None:
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            try:
                from kokoro_onnx import Kokoro
            except ImportError as exc:
                raise SynthesisError(f"kokoro-onnx is not installed; {_INSTALL_HINT}") from exc
            if not self._model_path.is_file():
                raise SynthesisError(f"kokoro model not found at {self._model_path}")
            if not self._voices_path.is_file():
                raise SynthesisError(f"kokoro voices file not found at {self._voices_path}")
            self._engine = Kokoro(str(self._model_path), str(self._voices_path))
        return self._engine
