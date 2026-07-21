"""SpeechService <-> synthesis cache integration."""

from __future__ import annotations

from pathlib import Path

from tests.conftest import make_config
from tts_daemon.core.errors import SynthesisError
from tts_daemon.core.events import EventBus
from tts_daemon.core.interfaces import TTSProvider
from tts_daemon.core.models import AudioClip, AudioFormat, SynthesisRequest, Voice
from tts_daemon.core.service import SpeechService
from tts_daemon.players.null import NullPlayer
from tts_daemon.providers.registry import ProviderRegistry


class CountingProvider(TTSProvider):
    """Counts synthesize() calls and echoes the text; rejects unknown options."""

    name = "counting"

    def __init__(self, settings: dict | None = None) -> None:
        super().__init__(settings)
        self.calls = 0
        self.seen_options: list[dict] = []

    def synthesize(self, request: SynthesisRequest) -> AudioClip:
        if request.options:
            raise SynthesisError(f"unknown options: {sorted(request.options)}")
        self.calls += 1
        self.seen_options.append(dict(request.options))
        return AudioClip(data=b"clip-" + request.text.encode(), format=AudioFormat.WAV)

    def voices(self) -> list[Voice]:
        return []


def _service(tmp_path: Path, *, enabled: bool = True) -> tuple[SpeechService, CountingProvider]:
    config = make_config(cache={"enabled": enabled, "dir": str(tmp_path), "max_mb": 10})
    registry = ProviderRegistry(config)
    registry.register(CountingProvider)
    config.speech.default_provider = "counting"
    service = SpeechService(config, registry, NullPlayer(), EventBus())
    return service, registry.get("counting")  # type: ignore[return-value]


class TestServiceCache:
    def test_second_identical_synthesis_is_a_cache_hit(self, tmp_path: Path) -> None:
        service, provider = _service(tmp_path)
        try:
            first = service.synthesize("build finished", provider="counting")
            second = service.synthesize("build finished", provider="counting")
            assert first.data == second.data
            assert provider.calls == 1  # second call served from cache
            assert service.status()["cache"]["hits"] == 1
        finally:
            service.close()

    def test_different_text_is_a_miss(self, tmp_path: Path) -> None:
        service, provider = _service(tmp_path)
        try:
            service.synthesize("one", provider="counting")
            service.synthesize("two", provider="counting")
            assert provider.calls == 2
        finally:
            service.close()

    def test_no_cache_option_bypasses_and_is_stripped(self, tmp_path: Path) -> None:
        service, provider = _service(tmp_path)
        try:
            service.synthesize("hi", provider="counting", options={"no_cache": True})
            service.synthesize("hi", provider="counting", options={"no_cache": True})
            # Both bypass the cache -> two real syntheses...
            assert provider.calls == 2
            # ...and the provider never saw the gateway-only no_cache option.
            assert provider.seen_options == [{}, {}]
        finally:
            service.close()

    def test_disabled_cache_reports_none_and_always_synthesizes(self, tmp_path: Path) -> None:
        service, provider = _service(tmp_path, enabled=False)
        try:
            service.synthesize("x", provider="counting")
            service.synthesize("x", provider="counting")
            assert provider.calls == 2
            assert service.status()["cache"] is None
        finally:
            service.close()

    def test_speak_path_populates_cache(self, tmp_path: Path) -> None:
        service, provider = _service(tmp_path)
        try:
            utterance = service.speak("spoken", provider="counting")
            assert service.wait_for(utterance, timeout=2)
            # synthesize the same text: served from the cache the queue filled.
            service.synthesize("spoken", provider="counting")
            assert provider.calls == 1
        finally:
            service.close()


def test_no_cache_stripped_even_when_cache_disabled(tmp_path: Path) -> None:
    # Regression: the gateway must strip no_cache so providers that reject
    # unknown options don't 502, regardless of whether the cache is on.
    service, provider = _service(tmp_path, enabled=False)
    try:
        clip = service.synthesize("hi", provider="counting", options={"no_cache": True})
        assert clip.data == b"clip-hi"
        assert provider.seen_options == [{}]
    finally:
        service.close()
