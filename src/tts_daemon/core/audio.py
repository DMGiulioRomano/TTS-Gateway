"""Small audio helpers shared by providers that synthesize raw samples.

Neural and synthetic engines hand back float samples plus a sample rate;
packing those into a WAV container is identical work, so it lives here instead
of being copied into each provider.
"""

from __future__ import annotations

import io
import struct
import wave
from collections.abc import Iterable


def float_pcm16_wav(samples: Iterable[float], sample_rate: int) -> bytes:
    """Pack an iterable of float samples in ``[-1, 1]`` into mono 16-bit PCM WAV.

    Values outside the range are clamped rather than allowed to wrap, which
    would otherwise turn a loud clip into noise.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frames = bytearray()
        for sample in samples:
            clamped = -1.0 if sample < -1.0 else 1.0 if sample > 1.0 else sample
            frames += struct.pack("<h", int(clamped * 32767))
        wav.writeframes(bytes(frames))
    return buffer.getvalue()
