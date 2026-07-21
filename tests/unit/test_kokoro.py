"""Kokoro provider — hermetic (kokoro-onnx is never really imported).

A fake ``kokoro_onnx`` module is injected into ``sys.modules`` and the model /
voices files are empty temp files, so availability, voice listing, speed
passthrough, and WAV packing are tested with no ONNX runtime or downloads.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
import wave
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

import pytest

from tts_daemon.core.errors import SynthesisError
from tts_daemon.core.models import AudioFormat, SynthesisRequest
from tts_daemon.providers.kokoro import KokoroProvider, default_kokoro_dir


class _FakeKokoro:
    last: _FakeKokoro | None = None

    def __init__(self, model_path: str, voices_path: str) -> None:
        self.model_path = model_path
        self.voices_path = voices_path
        self.create_calls: list[dict] = []
        _FakeKokoro.last = self

    def create(self, text: str, voice: str, speed: float = 1.0, lang: str = "en-us"):
        self.create_calls.append({"text": text, "voice": voice, "speed": speed, "lang": lang})
        return [0.0, 0.5, -0.5, 1.0, -1.0], 24000  # samples, sample_rate

    def get_voices(self) -> list[str]:
        return ["af_sarah", "am_adam"]


def _make_fake_module() -> types.ModuleType:
    module = types.ModuleType("kokoro_onnx")
    module.__spec__ = importlib.machinery.ModuleSpec("kokoro_onnx", loader=None)
    module.Kokoro = _FakeKokoro
    return module


@pytest.fixture()
def kokoro_files(tmp_path: Path) -> tuple[Path, Path]:
    model = tmp_path / "kokoro-v1.0.onnx"
    voices = tmp_path / "voices-v1.0.bin"
    model.write_bytes(b"onnx")
    voices.write_bytes(b"voices")
    return model, voices


@pytest.fixture()
def provider(
    monkeypatch: pytest.MonkeyPatch, kokoro_files: tuple[Path, Path]
) -> Iterator[KokoroProvider]:
    _FakeKokoro.last = None
    monkeypatch.setitem(sys.modules, "kokoro_onnx", _make_fake_module())
    model, voices = kokoro_files
    yield KokoroProvider({"model_path": str(model), "voices_path": str(voices)})


def test_default_dir_respects_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert default_kokoro_dir() == tmp_path / "tts-daemon" / "kokoro"


def test_synthesize_returns_wav(provider: KokoroProvider) -> None:
    clip = provider.synthesize(SynthesisRequest("hello"))
    assert clip.format is AudioFormat.WAV
    with wave.open(BytesIO(clip.data)) as wav:  # a valid, readable WAV
        assert wav.getframerate() == 24000
        assert wav.getnframes() == 5


def test_speed_and_voice_passed_through(provider: KokoroProvider) -> None:
    provider.synthesize(SynthesisRequest("hi", voice="am_adam", speed=1.3))
    call = _FakeKokoro.last.create_calls[-1]
    assert call["voice"] == "am_adam"
    assert call["speed"] == 1.3


def test_default_voice_used_when_unspecified(provider: KokoroProvider) -> None:
    provider.synthesize(SynthesisRequest("hi"))
    assert _FakeKokoro.last.create_calls[-1]["voice"] == "af_sarah"


def test_options_are_rejected(provider: KokoroProvider) -> None:
    with pytest.raises(SynthesisError, match="accepts no options"):
        provider.synthesize(SynthesisRequest("hi", options={"pitch": 1}))


def test_engine_is_reused(provider: KokoroProvider) -> None:
    provider.synthesize(SynthesisRequest("one"))
    first = _FakeKokoro.last
    provider.synthesize(SynthesisRequest("two"))
    assert _FakeKokoro.last is first  # constructed once, cached


def test_availability_ok_with_files(provider: KokoroProvider) -> None:
    assert provider.availability().available is True


def test_voices_listed(provider: KokoroProvider) -> None:
    assert [voice.id for voice in provider.voices()] == ["af_sarah", "am_adam"]


# --------------------------------------------------- unavailable states


def test_availability_missing_package(kokoro_files: tuple[Path, Path]) -> None:
    model, voices = kokoro_files
    provider = KokoroProvider({"model_path": str(model), "voices_path": str(voices)})
    availability = provider.availability()
    assert availability.available is False
    assert "tts-daemon[kokoro]" in availability.reason


def test_availability_missing_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "kokoro_onnx", _make_fake_module())
    voices = tmp_path / "voices-v1.0.bin"
    voices.write_bytes(b"voices")
    provider = KokoroProvider(
        {"model_path": str(tmp_path / "missing.onnx"), "voices_path": str(voices)}
    )
    reason = provider.availability().reason
    assert "model not found" in reason
    assert provider.voices() == []


def test_availability_missing_voices(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "kokoro_onnx", _make_fake_module())
    model = tmp_path / "kokoro-v1.0.onnx"
    model.write_bytes(b"onnx")
    provider = KokoroProvider(
        {"model_path": str(model), "voices_path": str(tmp_path / "missing.bin")}
    )
    assert "voices file not found" in provider.availability().reason
