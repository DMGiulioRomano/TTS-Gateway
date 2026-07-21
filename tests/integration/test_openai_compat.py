"""OpenAI-compatible endpoint ``POST /v1/audio/speech``."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_config
from tts_daemon.api.app import create_app


def _client(**config_overrides) -> TestClient:
    config = make_config(**config_overrides)
    return TestClient(create_app(config))


@pytest.fixture()
def client() -> Iterator[TestClient]:
    with _client() as test_client:
        yield test_client


def test_returns_wav_audio(client: TestClient) -> None:
    response = client.post(
        "/v1/audio/speech",
        json={"model": "tts-1", "input": "hello there", "voice": "alloy"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content[:4] == b"RIFF"  # a real WAV from the tone provider


def test_ignores_unknown_openai_fields(client: TestClient) -> None:
    response = client.post(
        "/v1/audio/speech",
        json={"input": "hi", "voice": "nova", "instructions": "speak warmly", "extra": 1},
    )
    assert response.status_code == 200


def test_missing_input_is_422(client: TestClient) -> None:
    assert client.post("/v1/audio/speech", json={"model": "tts-1"}).status_code == 422


def test_unsupported_format_is_422(client: TestClient) -> None:
    response = client.post(
        "/v1/audio/speech",
        json={"input": "hi", "response_format": "mp3"},
    )
    assert response.status_code == 422
    assert "wav" in response.json()["detail"]


def test_speed_out_of_range_is_422(client: TestClient) -> None:
    assert client.post("/v1/audio/speech", json={"input": "hi", "speed": 9.0}).status_code == 422


def test_registered_provider_name_is_honored() -> None:
    # 'tone' is a registered provider, so model='tone' selects it explicitly.
    with _client() as client:
        response = client.post("/v1/audio/speech", json={"model": "tone", "input": "beep"})
        assert response.status_code == 200
        assert response.content[:4] == b"RIFF"


def test_voice_alias_maps_to_real_voice() -> None:
    # Alias an OpenAI voice to a real tone voice; a bad alias would make the
    # tone provider raise (proving the mapping actually took effect).
    with _client(openai_compat={"voice_aliases": {"alloy": "low"}}) as client:
        ok = client.post("/v1/audio/speech", json={"model": "tone", "input": "x", "voice": "alloy"})
        assert ok.status_code == 200
    with _client(openai_compat={"voice_aliases": {"alloy": "does-not-exist"}}) as client:
        bad = client.post(
            "/v1/audio/speech", json={"model": "tone", "input": "x", "voice": "alloy"}
        )
        assert bad.status_code == 502  # synthesis error surfaces as bad gateway


def test_unaliased_openai_voice_falls_back_to_default() -> None:
    # 'shimmer' is an OpenAI name with no alias -> gateway default voice, so the
    # tone provider must not receive the literal 'shimmer'.
    with _client() as client:
        response = client.post(
            "/v1/audio/speech", json={"model": "tone", "input": "x", "voice": "shimmer"}
        )
        assert response.status_code == 200
