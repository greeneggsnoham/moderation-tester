"""Provider registry and factory helpers."""

from __future__ import annotations

from moderation_tester.config import AppConfig, ConfigError
from moderation_tester.providers.base import ProviderClient
from moderation_tester.providers.openai_client import OpenAIProviderClient


class ProviderFactory:
    """Create and cache provider integrations using application config."""

    def __init__(self, config: AppConfig) -> None:
        """Store app config for provider construction."""
        self._config = config
        self._providers: dict[str, ProviderClient] = {}

    def get(self, provider: str) -> ProviderClient:
        """Return a provider client instance for the given provider name."""
        name = provider.strip().lower()
        if name in self._providers:
            return self._providers[name]

        if name == "openai":
            key = self._config.require_api_key_for_provider("openai")
            client = OpenAIProviderClient(api_key=key)
            self._providers[name] = client
            return client

        if name in {"gemini", "elm_local"}:
            raise ConfigError(
                f"Provider '{name}' is not implemented yet in Step 3."
            )

        raise ConfigError(f"Unsupported provider: {name}")
