"""On-disk cache for synthesized audio clips.

Real usage repeats phrases constantly ("Build finished", notification
strings, Claude Code hook messages). Caching turns those into a zero-cost
lookup, which matters most for the heavier engines this project wants to
attract (Kokoro, XTTS) and makes the notification use case feel native.

The cache is a plain directory: one file per clip, named by the SHA-256 of
everything that affects the audio (provider, a provider fingerprint, voice,
speed, options, text), plus a small JSON ``index.json`` recording each entry's
format, size, and last-access time. Eviction is size-based LRU. Writes are
atomic (``.tmp`` + rename) and any corruption is treated as a miss and cleaned
up, never raised — a broken cache must never break synthesis.

Pure standard library (``hashlib``, ``json``, ``os``); no new dependencies.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from tts_daemon.core.models import AudioClip, AudioFormat, SynthesisRequest

logger = logging.getLogger(__name__)

_INDEX_NAME = "index.json"


def default_cache_dir() -> Path:
    """``$XDG_CACHE_HOME/tts-daemon`` with the ``~/.cache`` fallback."""
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "tts-daemon"


def cache_key(
    *,
    provider: str,
    fingerprint: str,
    request: SynthesisRequest,
) -> str:
    """A stable content hash for one synthesis request.

    ``fingerprint`` lets a provider fold in anything that changes its output
    but is not part of the request (a voice-model file's mtime, an engine
    version), so a swapped model invalidates cached clips automatically.
    """
    canonical = json.dumps(
        {
            "provider": provider,
            "fingerprint": fingerprint,
            "voice": request.voice,
            "speed": request.speed,
            "options": _canonical(request.options),
            "text": request.text,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> Any:
    """Recursively sort dict keys so option ordering never changes the key."""
    if isinstance(value, dict):
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


class SynthesisCache:
    """Size-bounded, LRU, on-disk store of synthesized clips.

    Thread-safe: the queue worker (playback path) and API threads
    (``/v1/synthesize``) may use it concurrently. Metadata operations are
    guarded by a lock; the audio blobs are small enough that guarding their
    reads/writes too keeps the implementation simple without hurting the hot
    path (synthesis, the slow part, happens outside the lock).
    """

    def __init__(self, directory: Path, max_bytes: int) -> None:
        self._dir = directory
        self._max_bytes = max(0, max_bytes)
        self._lock = threading.Lock()
        self._index: dict[str, dict[str, Any]] = {}
        self._clock = 0  # monotonic access counter for LRU ordering
        self._hits = 0
        self._misses = 0
        self._load_index()

    # ---------------------------------------------------------------- lookup

    def get(self, key: str) -> AudioClip | None:
        """Return the cached clip for ``key`` or ``None`` on a miss."""
        with self._lock:
            entry = self._index.get(key)
            if entry is None:
                self._misses += 1
                return None
            try:
                data = self._blob_path(key).read_bytes()
                fmt = AudioFormat(entry["format"])
            except (OSError, ValueError, KeyError):
                # Missing or corrupt blob: forget it and report a miss.
                self._forget(key)
                self._misses += 1
                return None
            entry["atime"] = self._tick()
            self._hits += 1
            self._save_index()
            return AudioClip(data=data, format=fmt)

    def put(self, key: str, clip: AudioClip) -> None:
        """Store ``clip`` under ``key`` (best-effort; failures are swallowed)."""
        if self._max_bytes <= 0:
            return
        with self._lock:
            try:
                self._dir.mkdir(parents=True, exist_ok=True)
                self._atomic_write(self._blob_path(key), clip.data)
            except OSError:
                logger.exception("Failed to write cache entry %s", key)
                return
            self._index[key] = {
                "format": clip.format.value,
                "size": len(clip.data),
                "atime": self._tick(),
            }
            self._evict_over_budget()
            self._save_index()

    # -------------------------------------------------------------- reporting

    def stats(self) -> dict[str, Any]:
        with self._lock:
            size = sum(int(entry.get("size", 0)) for entry in self._index.values())
            return {
                "entries": len(self._index),
                "size_mb": round(size / (1024 * 1024), 3),
                "hits": self._hits,
                "misses": self._misses,
            }

    # -------------------------------------------------------------- internals

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def _blob_path(self, key: str) -> Path:
        return self._dir / key

    def _forget(self, key: str) -> None:
        self._index.pop(key, None)
        try:
            self._blob_path(key).unlink(missing_ok=True)
        except OSError:
            logger.debug("Could not remove cache blob %s", key, exc_info=True)

    def _evict_over_budget(self) -> None:
        size = sum(int(entry.get("size", 0)) for entry in self._index.values())
        if size <= self._max_bytes:
            return
        # Least-recently-used first (smallest atime).
        for key in sorted(self._index, key=lambda k: self._index[k].get("atime", 0)):
            if size <= self._max_bytes:
                break
            size -= int(self._index[key].get("size", 0))
            self._forget(key)

    def _atomic_write(self, path: Path, data: bytes) -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def _index_path(self) -> Path:
        return self._dir / _INDEX_NAME

    def _load_index(self) -> None:
        path = self._index_path()
        if not path.is_file():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("Ignoring corrupt cache index at %s", path)
            return
        if not isinstance(raw, dict):
            return
        entries = raw.get("entries")
        if isinstance(entries, dict):
            self._index = {k: v for k, v in entries.items() if isinstance(v, dict)}
        self._clock = max((int(e.get("atime", 0)) for e in self._index.values()), default=0)

    def _save_index(self) -> None:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            self._atomic_write(
                self._index_path(),
                json.dumps({"entries": self._index}, separators=(",", ":")).encode("utf-8"),
            )
        except OSError:
            logger.exception("Failed to persist cache index")
