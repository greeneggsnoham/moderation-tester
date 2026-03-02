"""Tests for configuration behavior."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from moderation_tester.config import AppConfig, ConfigError


class AppConfigTests(unittest.TestCase):
    """Validate environment parsing and guardrail behavior."""

    @patch.dict(os.environ, {"DEFAULT_PROVIDER": "openai", "DEFAULT_MODEL": "o4-mini"}, clear=True)
    def test_defaults_are_loaded(self) -> None:
        config = AppConfig.from_env()
        config.validate()
        self.assertEqual(config.runtime.default_provider, "openai")
        self.assertEqual(config.runtime.default_model, "o4-mini")
        self.assertTrue(config.runtime.default_moderation_enabled)

    @patch.dict(
        os.environ,
        {
            "DEFAULT_PROVIDER": "invalid-provider",
            "DEFAULT_MODEL": "o4-mini",
        },
        clear=True,
    )
    def test_invalid_provider_raises(self) -> None:
        config = AppConfig.from_env()
        with self.assertRaises(ConfigError):
            config.validate()

    @patch.dict(
        os.environ,
        {
            "DEFAULT_PROVIDER": "openai",
            "DEFAULT_MODEL": "not-a-valid-model",
        },
        clear=True,
    )
    def test_invalid_default_model_raises(self) -> None:
        config = AppConfig.from_env()
        with self.assertRaises(ConfigError):
            config.validate()


if __name__ == "__main__":
    unittest.main()
