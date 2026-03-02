"""Tests for provider factory behavior and routing."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from moderation_tester.config import AppConfig, ConfigError
from moderation_tester.providers.factory import ProviderFactory
from moderation_tester.providers.openai_client import OpenAIProviderClient


class ProviderFactoryTests(unittest.TestCase):
    """Verify provider registration and API key enforcement behavior."""

    @patch.dict(
        os.environ,
        {
            "DEFAULT_PROVIDER": "openai",
            "DEFAULT_MODEL": "o4-mini",
            "OPENAI_API_KEY": "test-key",
        },
        clear=True,
    )
    def test_openai_provider_is_created(self) -> None:
        config = AppConfig.from_env()
        factory = ProviderFactory(config)
        client = factory.get("openai")
        self.assertIsInstance(client, OpenAIProviderClient)

    @patch.dict(
        os.environ,
        {"DEFAULT_PROVIDER": "openai", "DEFAULT_MODEL": "o4-mini"},
        clear=True,
    )
    def test_missing_openai_key_raises(self) -> None:
        config = AppConfig.from_env()
        factory = ProviderFactory(config)
        with self.assertRaises(ConfigError):
            factory.get("openai")

    @patch.dict(
        os.environ,
        {
            "DEFAULT_PROVIDER": "openai",
            "DEFAULT_MODEL": "o4-mini",
            "OPENAI_API_KEY": "test-key",
        },
        clear=True,
    )
    def test_unimplemented_provider_raises(self) -> None:
        config = AppConfig.from_env()
        factory = ProviderFactory(config)
        with self.assertRaises(ConfigError):
            factory.get("gemini")


if __name__ == "__main__":
    unittest.main()
