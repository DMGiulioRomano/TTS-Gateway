"""The ``winsound`` fallback player — Windows-only, run on the Windows CI job.

``winsound`` only imports on Windows, so the whole module is skipped elsewhere
and the code under test is imported inside each test (a top-level import would
break collection on Linux/macOS). Playback is driven through a monkeypatched
``winsound.PlaySound`` so CI stays silent and deterministic — no real audio
device, no brief noise on the runner — while still covering the player's logic
(format gate, ``SND_MEMORY`` flag, error mapping, ``SND_PURGE`` on stop).
"""

from __future__ import annotations

import sys

import pytest

from tts_daemon.core.errors import PlaybackError
from tts_daemon.core.models import AudioClip, AudioFormat

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="winsound is Windows-only")


def _wav_clip() -> AudioClip:
    return AudioClip(data=b"RIFF0000WAVEfmt ", format=AudioFormat.WAV)


def test_import_and_construct() -> None:
    from tts_daemon.players.winsound_player import WinsoundPlayer

    assert WinsoundPlayer() is not None


def test_auto_backend_selects_winsound_without_a_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With no external player on PATH, `auto` must fall back to winsound.
    import tts_daemon.players.command as command_module
    from tts_daemon.config import PlaybackConfig
    from tts_daemon.players import create_player
    from tts_daemon.players.winsound_player import WinsoundPlayer

    monkeypatch.setattr(command_module.shutil, "which", lambda name: None)
    player = create_player(PlaybackConfig(backend="auto"))
    assert isinstance(player, WinsoundPlayer)


def test_rejects_non_wav() -> None:
    from tts_daemon.players.winsound_player import WinsoundPlayer

    player = WinsoundPlayer()
    with pytest.raises(PlaybackError, match="can only play WAV"):
        player.play(AudioClip(data=b"\xff\xfb", format=AudioFormat.MP3))


def test_plays_wav_via_snd_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    import tts_daemon.players.winsound_player as mod

    calls: list[tuple] = []
    monkeypatch.setattr(mod.winsound, "PlaySound", lambda *args: calls.append(args))
    clip = _wav_clip()
    assert mod.WinsoundPlayer().play(clip) is True
    assert calls == [(clip.data, mod.winsound.SND_MEMORY)]


def test_runtime_error_becomes_playback_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import tts_daemon.players.winsound_player as mod

    def boom(*args: object) -> None:
        raise RuntimeError("device busy")

    monkeypatch.setattr(mod.winsound, "PlaySound", boom)
    with pytest.raises(PlaybackError, match="winsound failed to play"):
        mod.WinsoundPlayer().play(_wav_clip())


def test_stop_purges_current_sound(monkeypatch: pytest.MonkeyPatch) -> None:
    import tts_daemon.players.winsound_player as mod

    calls: list[tuple] = []
    monkeypatch.setattr(mod.winsound, "PlaySound", lambda *args: calls.append(args))
    mod.WinsoundPlayer().stop()
    assert calls == [(None, mod.winsound.SND_PURGE)]
