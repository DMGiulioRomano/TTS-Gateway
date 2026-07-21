"""Synthesis cache: hashing, hit/miss, LRU eviction, bypass, corruption."""

from __future__ import annotations

from pathlib import Path

import pytest

from tts_daemon.core.cache import SynthesisCache, cache_key, default_cache_dir
from tts_daemon.core.models import AudioClip, AudioFormat, SynthesisRequest


def _clip(data: bytes = b"RIFFdata", fmt: AudioFormat = AudioFormat.WAV) -> AudioClip:
    return AudioClip(data=data, format=fmt)


def _req(text: str = "hello", **kw) -> SynthesisRequest:
    return SynthesisRequest(text=text, **kw)


class TestCacheKey:
    def test_stable_across_option_ordering(self) -> None:
        a = cache_key(provider="piper", fingerprint="f", request=_req(options={"x": 1, "y": 2}))
        b = cache_key(provider="piper", fingerprint="f", request=_req(options={"y": 2, "x": 1}))
        assert a == b

    def test_varies_with_text_voice_speed_provider_fingerprint(self) -> None:
        base = cache_key(provider="piper", fingerprint="f", request=_req())
        assert base != cache_key(provider="piper", fingerprint="f", request=_req(text="other"))
        assert base != cache_key(provider="piper", fingerprint="f", request=_req(voice="v"))
        assert base != cache_key(provider="piper", fingerprint="f", request=_req(speed=2.0))
        assert base != cache_key(provider="tone", fingerprint="f", request=_req())
        assert base != cache_key(provider="piper", fingerprint="g", request=_req())


class TestSynthesisCache:
    def test_miss_then_hit(self, tmp_path: Path) -> None:
        cache = SynthesisCache(tmp_path, max_bytes=1_000_000)
        assert cache.get("k") is None
        cache.put("k", _clip(b"audio-bytes"))
        hit = cache.get("k")
        assert hit is not None
        assert hit.data == b"audio-bytes"
        assert hit.format is AudioFormat.WAV
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entries"] == 1

    def test_format_roundtrips(self, tmp_path: Path) -> None:
        cache = SynthesisCache(tmp_path, max_bytes=1_000_000)
        cache.put("mp3", _clip(b"\xff\xfbmp3", AudioFormat.MP3))
        hit = cache.get("mp3")
        assert hit is not None and hit.format is AudioFormat.MP3

    def test_persists_across_instances(self, tmp_path: Path) -> None:
        SynthesisCache(tmp_path, max_bytes=1_000_000).put("k", _clip(b"persisted"))
        reopened = SynthesisCache(tmp_path, max_bytes=1_000_000)
        hit = reopened.get("k")
        assert hit is not None and hit.data == b"persisted"

    def test_lru_eviction_by_size(self, tmp_path: Path) -> None:
        # Budget fits two 100-byte clips but not three.
        cache = SynthesisCache(tmp_path, max_bytes=250)
        cache.put("a", _clip(b"a" * 100))
        cache.put("b", _clip(b"b" * 100))
        cache.get("a")  # touch a so b becomes least-recently-used
        cache.put("c", _clip(b"c" * 100))  # eviction: b (LRU) is dropped
        assert cache.get("b") is None
        assert cache.get("a") is not None
        assert cache.get("c") is not None

    def test_disabled_when_zero_budget(self, tmp_path: Path) -> None:
        cache = SynthesisCache(tmp_path, max_bytes=0)
        cache.put("k", _clip())
        assert cache.get("k") is None
        assert cache.stats()["entries"] == 0

    def test_missing_blob_is_a_miss_and_forgotten(self, tmp_path: Path) -> None:
        cache = SynthesisCache(tmp_path, max_bytes=1_000_000)
        cache.put("k", _clip(b"gone"))
        (tmp_path / "k").unlink()  # corrupt: blob vanished, index still references it
        assert cache.get("k") is None
        assert cache.get("k") is None  # entry cleaned up; still a clean miss
        assert cache.stats()["entries"] == 0

    def test_corrupt_index_is_ignored(self, tmp_path: Path) -> None:
        (tmp_path).mkdir(exist_ok=True)
        (tmp_path / "index.json").write_text("{ not json", encoding="utf-8")
        cache = SynthesisCache(tmp_path, max_bytes=1_000_000)  # must not raise
        assert cache.stats()["entries"] == 0
        cache.put("k", _clip(b"ok"))
        assert cache.get("k") is not None


def test_default_cache_dir_uses_xdg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/xdg-cache-test")
    assert default_cache_dir() == Path("/tmp/xdg-cache-test/tts-daemon")
