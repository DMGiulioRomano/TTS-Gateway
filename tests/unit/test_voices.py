"""Piper voice catalog + downloader, exercised against a local HTTP server.

No test touches the network: a throwaway ``http.server`` serves a tiny fake
``rhasspy/piper-voices`` tree (manifest + model files) on localhost, exactly the
hermetic pattern the project requires for download code.
"""

from __future__ import annotations

import functools
import hashlib
import http.server
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from tts_daemon.voices import (
    VoiceCatalog,
    VoiceCatalogError,
    closest_matches,
    parse_catalog,
)

EN_ONNX = b"EN-ONNX-MODEL-BYTES" * 64
EN_JSON = json.dumps({"audio": {"sample_rate": 22050}, "language": {"code": "en_US"}}).encode()
IT_ONNX = b"IT-ONNX" * 32
IT_JSON = b"{}"

EN_ONNX_PATH = "en/en_US/lessac/medium/en_US-lessac-medium.onnx"
IT_ONNX_PATH = "it/it_IT/riccardo/x_low/it_IT-riccardo-x_low.onnx"


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args: object) -> None:  # keep test output clean
        pass


def _write(root: Path, rel_path: str, data: bytes) -> None:
    dest = root / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)


def _md5(data: bytes) -> str:
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def _build_repo(
    root: Path, *, en_onnx_declared: int | None = None, en_onnx_md5: str | None = None
) -> None:
    """Write a fake piper-voices tree under ``root`` (files + voices.json)."""
    _write(root, EN_ONNX_PATH, EN_ONNX)
    _write(root, f"{EN_ONNX_PATH}.json", EN_JSON)
    _write(root, IT_ONNX_PATH, IT_ONNX)
    _write(root, f"{IT_ONNX_PATH}.json", IT_JSON)

    en_size = en_onnx_declared if en_onnx_declared is not None else len(EN_ONNX)
    manifest = {
        "en_US-lessac-medium": {
            "key": "en_US-lessac-medium",
            "name": "lessac",
            "language": {"code": "en_US", "family": "en"},
            "quality": "medium",
            "num_speakers": 1,
            "files": {
                EN_ONNX_PATH: {
                    "size_bytes": en_size,
                    "md5_digest": en_onnx_md5 or _md5(EN_ONNX),
                },
                f"{EN_ONNX_PATH}.json": {
                    "size_bytes": len(EN_JSON),
                    "md5_digest": _md5(EN_JSON),
                },
                "en/en_US/lessac/medium/MODEL_CARD": {"size_bytes": 42},
            },
        },
        "it_IT-riccardo-x_low": {
            "key": "it_IT-riccardo-x_low",
            "language": {"code": "it_IT", "family": "it"},
            "quality": "x_low",
            "num_speakers": 1,
            "files": {
                # No md5 here: the manifest is not guaranteed to carry one, and
                # a voice without a digest must still be downloadable.
                IT_ONNX_PATH: {"size_bytes": len(IT_ONNX)},
                f"{IT_ONNX_PATH}.json": {"size_bytes": len(IT_JSON)},
            },
        },
    }
    (root / "voices.json").write_text(json.dumps(manifest))


def _serve(root: Path) -> tuple[str, http.server.HTTPServer, threading.Thread]:
    handler = functools.partial(_QuietHandler, directory=str(root))
    httpd = http.server.HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{httpd.server_address[1]}", httpd, thread


@pytest.fixture()
def server(tmp_path: Path) -> Iterator[SimpleNamespace]:
    root = tmp_path / "repo"
    root.mkdir()
    _build_repo(root)
    base_url, httpd, thread = _serve(root)
    try:
        yield SimpleNamespace(base_url=base_url, root=root)
    finally:
        httpd.shutdown()
        thread.join()


@pytest.fixture()
def catalog(server: SimpleNamespace) -> VoiceCatalog:
    return VoiceCatalog(server.base_url, timeout=5.0)


class TestParseCatalog:
    @pytest.mark.parametrize(
        "digest, expected",
        [
            ("A" * 32, "a" * 32),  # normalized to lower case
            ("abc", None),  # too short
            ("z" * 32, None),  # not hexadecimal
            (12345, None),  # not a string
        ],
    )
    def test_md5_digest_is_validated(self, digest: object, expected: str | None) -> None:
        manifest = {
            "a-b-c": {
                "files": {
                    "x/a-b-c.onnx": {"size_bytes": 10, "md5_digest": digest},
                    "x/a-b-c.onnx.json": {"size_bytes": 3},
                },
            }
        }
        assert parse_catalog(manifest)["a-b-c"].onnx_md5 == expected

    def test_pairs_onnx_and_config(self) -> None:
        manifest = {
            "a-b-c": {
                "key": "a-b-c",
                "language": {"code": "en_GB"},
                "quality": "high",
                "num_speakers": 2,
                "files": {
                    "x/a-b-c.onnx": {"size_bytes": 10},
                    "x/a-b-c.onnx.json": {"size_bytes": 3},
                },
            }
        }
        catalog = parse_catalog(manifest)
        voice = catalog["a-b-c"]
        assert voice.language == "en_GB"
        assert voice.quality == "high"
        assert voice.num_speakers == 2
        assert voice.onnx_path == "x/a-b-c.onnx"
        assert voice.config_path == "x/a-b-c.onnx.json"
        assert voice.size_bytes == 13

    def test_skips_entries_missing_a_file(self) -> None:
        manifest = {"only-onnx": {"files": {"p/only-onnx.onnx": {"size_bytes": 1}}}}
        assert parse_catalog(manifest) == {}

    def test_ignores_non_dict_entries(self) -> None:
        assert parse_catalog({"weird": "not a dict"}) == {}


class TestClosestMatches:
    def test_substring_first_then_fuzzy(self) -> None:
        ids = ["en_US-lessac-medium", "en_US-ryan-high", "it_IT-riccardo-x_low"]
        matches = closest_matches("lessac", ids)
        assert matches[0] == "en_US-lessac-medium"

    def test_typo_is_matched_fuzzily(self) -> None:
        ids = ["en_US-lessac-medium"]
        assert "en_US-lessac-medium" in closest_matches("en_US-lessac-mediun", ids)


class TestBrowse:
    def test_lists_all_voices(self, catalog: VoiceCatalog) -> None:
        ids = {voice.id for voice in catalog.list()}
        assert ids == {"en_US-lessac-medium", "it_IT-riccardo-x_low"}

    def test_language_filter_by_family(self, catalog: VoiceCatalog) -> None:
        assert [v.id for v in catalog.list("it")] == ["it_IT-riccardo-x_low"]

    def test_language_filter_by_full_code(self, catalog: VoiceCatalog) -> None:
        assert [v.id for v in catalog.list("en_US")] == ["en_US-lessac-medium"]

    def test_manifest_fetched_once_and_cached(self, catalog: VoiceCatalog) -> None:
        catalog.voices()
        cached = catalog._voices
        assert catalog.voices() is cached

    def test_unknown_voice_suggests_matches(self, catalog: VoiceCatalog) -> None:
        with pytest.raises(VoiceCatalogError, match=r"Did you mean.*en_US-lessac-medium"):
            catalog.get("en_US-lessac")


class TestDownload:
    def test_downloads_both_files(self, catalog: VoiceCatalog, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        result = catalog.download("en_US-lessac-medium", models_dir)

        onnx = models_dir / "en_US-lessac-medium.onnx"
        config = models_dir / "en_US-lessac-medium.onnx.json"
        assert onnx.read_bytes() == EN_ONNX
        assert config.read_bytes() == EN_JSON
        assert result.skipped is False
        assert set(result.downloaded) == {onnx, config}
        # no leftover partial files
        assert list(models_dir.glob("*.part")) == []

    def test_creates_missing_models_dir(self, catalog: VoiceCatalog, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested" / "models"
        catalog.download("it_IT-riccardo-x_low", nested)
        assert (nested / "it_IT-riccardo-x_low.onnx").is_file()

    def test_idempotent_skips_when_present(self, catalog: VoiceCatalog, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        catalog.download("en_US-lessac-medium", models_dir)
        onnx = models_dir / "en_US-lessac-medium.onnx"
        stamp = onnx.stat().st_mtime_ns

        result = catalog.download("en_US-lessac-medium", models_dir)
        assert result.skipped is True
        assert result.downloaded == []
        assert onnx.stat().st_mtime_ns == stamp  # untouched

    def test_force_refetches(self, catalog: VoiceCatalog, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        catalog.download("en_US-lessac-medium", models_dir)
        result = catalog.download("en_US-lessac-medium", models_dir, force=True)
        assert result.skipped is False
        assert len(result.downloaded) == 2

    def test_progress_callback_receives_totals(self, catalog: VoiceCatalog, tmp_path: Path) -> None:
        seen: list[tuple[str, int, int]] = []
        catalog.download(
            "en_US-lessac-medium", tmp_path / "m", progress=lambda *args: seen.append(args)
        )
        assert seen  # at least one progress tick
        assert all(0 <= done <= total for _, done, total in seen)
        assert seen[-1][1] == seen[-1][2]  # last tick reaches 100%

    def test_unknown_voice_download_errors(self, catalog: VoiceCatalog, tmp_path: Path) -> None:
        with pytest.raises(VoiceCatalogError, match="unknown Piper voice"):
            catalog.download("nope-nope", tmp_path / "m")


class TestVerification:
    def test_size_mismatch_is_rejected(self, tmp_path: Path) -> None:
        # Declare a wrong size in the manifest: the server's Content-Length will
        # disagree, and the download must refuse rather than install a bad model.
        root = tmp_path / "repo"
        root.mkdir()
        _build_repo(root, en_onnx_declared=len(EN_ONNX) + 999)
        base_url, httpd, thread = _serve(root)
        try:
            catalog = VoiceCatalog(base_url, timeout=5.0)
            models_dir = tmp_path / "models"
            with pytest.raises(VoiceCatalogError, match="size mismatch"):
                catalog.download("en_US-lessac-medium", models_dir)
            assert list(models_dir.glob("*.part")) == []  # cleaned up
            assert not (models_dir / "en_US-lessac-medium.onnx").exists()
        finally:
            httpd.shutdown()
            thread.join()

    def test_checksum_mismatch_is_rejected(self, tmp_path: Path) -> None:
        # Right length, wrong bytes: only the digest can catch this, so a
        # corrupted-in-transit model never reaches the models dir.
        root = tmp_path / "repo"
        root.mkdir()
        _build_repo(root, en_onnx_md5="0" * 32)
        base_url, httpd, thread = _serve(root)
        try:
            catalog = VoiceCatalog(base_url, timeout=5.0)
            models_dir = tmp_path / "models"
            with pytest.raises(VoiceCatalogError, match="checksum mismatch"):
                catalog.download("en_US-lessac-medium", models_dir)
            assert list(models_dir.glob("*.part")) == []
            assert not (models_dir / "en_US-lessac-medium.onnx").exists()
        finally:
            httpd.shutdown()
            thread.join()

    def test_voice_without_digest_still_downloads(
        self, catalog: VoiceCatalog, tmp_path: Path
    ) -> None:
        # The italian entry carries no md5_digest; verification must be skipped,
        # not treated as a mismatch.
        models_dir = tmp_path / "models"
        result = catalog.download("it_IT-riccardo-x_low", models_dir)
        assert (models_dir / "it_IT-riccardo-x_low.onnx").read_bytes() == IT_ONNX
        assert len(result.downloaded) == 2


class TestOffline:
    def test_unreachable_catalog_is_actionable(self, tmp_path: Path) -> None:
        # Port 1 is unusable, so the fetch fails fast with a friendly message.
        catalog = VoiceCatalog("http://127.0.0.1:1", timeout=1.0)
        with pytest.raises(VoiceCatalogError, match="cannot reach the Piper voice catalog"):
            catalog.list()


class TestDownloadCli:
    """The `tts-daemon download` command, pointed at the local test server."""

    @pytest.fixture()
    def cli_catalog(self, server: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
        from tts_daemon import cli

        monkeypatch.setattr(
            cli, "VoiceCatalog", functools.partial(VoiceCatalog, server.base_url, timeout=5.0)
        )

    def test_list_prints_catalog(self, cli_catalog: None, capsys: pytest.CaptureFixture) -> None:
        from tts_daemon.cli import main

        assert main(["download", "--list"]) == 0
        out = capsys.readouterr().out
        assert "en_US-lessac-medium" in out
        assert "it_IT-riccardo-x_low" in out

    def test_list_language_filter(self, cli_catalog: None, capsys: pytest.CaptureFixture) -> None:
        from tts_daemon.cli import main

        assert main(["download", "--list", "--language", "it"]) == 0
        out = capsys.readouterr().out
        assert "it_IT-riccardo-x_low" in out
        assert "en_US-lessac-medium" not in out

    def test_download_into_models_dir(
        self, cli_catalog: None, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        from tts_daemon.cli import main

        models = tmp_path / "m"
        assert main(["download", "en_US-lessac-medium", "--models-dir", str(models)]) == 0
        assert (models / "en_US-lessac-medium.onnx").read_bytes() == EN_ONNX
        assert "downloaded en_US-lessac-medium" in capsys.readouterr().out

    def test_missing_voice_and_list_is_usage_error(
        self, cli_catalog: None, capsys: pytest.CaptureFixture
    ) -> None:
        from tts_daemon.cli import main

        assert main(["download"]) == 2
        assert "give a voice id" in capsys.readouterr().err

    def test_unknown_voice_exits_1(
        self, cli_catalog: None, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        from tts_daemon.cli import main

        assert main(["download", "does-not-exist", "--models-dir", str(tmp_path)]) == 1
        assert "unknown Piper voice" in capsys.readouterr().err
