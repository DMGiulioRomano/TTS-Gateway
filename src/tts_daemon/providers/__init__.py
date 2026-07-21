"""TTS provider implementations and the registry that serves them.

Built-in providers are registered by :func:`create_default_registry`;
third-party packages plug in through the ``tts_daemon.providers`` entry
point group without any change to this package (see docs/providers.md).
"""

from tts_daemon.providers.registry import (
    ENTRY_POINT_GROUP,
    ProviderRegistry,
    create_default_registry,
)

__all__ = ["ENTRY_POINT_GROUP", "ProviderRegistry", "create_default_registry"]
