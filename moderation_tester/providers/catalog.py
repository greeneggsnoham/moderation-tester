"""Supported provider model catalogs used for validation and UI options."""

from __future__ import annotations

SUPPORTED_MODELS: dict[str, tuple[str, ...]] = {
    "openai": ("o3", "o4-mini", "gpt-5", "gpt-5.1", "gpt-5.2"),
    "gemini": (),
    "elm_local": (),
}


def get_supported_models(provider: str) -> tuple[str, ...]:
    """Return supported models for a provider."""
    return SUPPORTED_MODELS.get(provider.lower(), ())
