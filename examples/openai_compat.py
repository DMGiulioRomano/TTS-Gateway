#!/usr/bin/env python3
"""Use tts-daemon as a drop-in OpenAI text-to-speech backend.

Any code written against OpenAI's `audio.speech` API works unchanged — just
point `base_url` at the gateway and the audio is synthesized locally.

    pip install openai        # example-only dependency (the gateway needs nothing)
    tts-daemon serve          # in another terminal
    python examples/openai_compat.py

The `api_key` is required by the client library but ignored by the gateway
(until you enable bearer-token auth, in which case pass your token here).
"""

from __future__ import annotations

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:5111/v1", api_key="unused")

# `model` may be a real provider name ("piper", "tone") or an OpenAI model id
# ("tts-1"), which falls back to the gateway's default provider. `voice` maps
# through the `openai_compat.voice_aliases` config section.
response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Hello from a local, private text-to-speech server.",
    response_format="wav",
)

response.write_to_file("openai_compat_out.wav")
print("wrote openai_compat_out.wav")
