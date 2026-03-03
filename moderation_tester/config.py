"""Configuration loading and validation utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass

from moderation_tester.providers.catalog import get_supported_models

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args: object, **kwargs: object) -> bool:
        """Fallback no-op when python-dotenv is not installed yet."""
        return False


class ConfigError(ValueError):
    """Raised when runtime configuration is missing or invalid."""


@dataclass(frozen=True)
class APIKeys:
    """Provider API key values loaded from environment variables."""

    openai_api_key: str | None
    gemini_api_key: str | None
    elm_local_api_key: str | None


@dataclass(frozen=True)
class RuntimeConfig:
    """Default runtime settings used by the app UI and request pipeline."""

    default_provider: str
    default_model: str
    default_moderation_enabled: bool


@dataclass(frozen=True)
class SecurityConfig:
    """Security-related limits for uploaded content and local handling."""

    max_upload_size_mb: int
    allowed_extensions: tuple[str, ...]


@dataclass(frozen=True)
class AppConfig:
    """Top-level immutable application configuration object."""

    api_keys: APIKeys
    runtime: RuntimeConfig
    security: SecurityConfig

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create configuration from environment variables and optional .env file."""
        load_dotenv(override=False)

        provider = os.getenv("DEFAULT_PROVIDER", "openai").strip().lower()
        model = os.getenv("DEFAULT_MODEL", "o4-mini").strip()
        moderation_enabled = (
            os.getenv("DEFAULT_MODERATION_ENABLED", "true").strip().lower()
            in {"1", "true", "yes", "on"}
        )

        upload_size_raw = os.getenv("MAX_UPLOAD_SIZE_MB", "10").strip()
        try:
            upload_size = int(upload_size_raw)
        except ValueError as exc:
            raise ConfigError("MAX_UPLOAD_SIZE_MB must be an integer.") from exc

        allowed_extensions_env = os.getenv(
            "ALLOWED_UPLOAD_EXTENSIONS", "pdf,csv,doc,docx"
        )
        extensions = tuple(
            ext.strip().lower()
            for ext in allowed_extensions_env.split(",")
            if ext.strip()
        )

        return cls(
            api_keys=APIKeys(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                gemini_api_key=os.getenv("GEMINI_API_KEY"),
                elm_local_api_key=os.getenv("ELM_LOCAL_API_KEY"),
            ),
            runtime=RuntimeConfig(
                default_provider=provider,
                default_model=model,
                default_moderation_enabled=moderation_enabled,
            ),
            security=SecurityConfig(
                max_upload_size_mb=upload_size,
                allowed_extensions=extensions,
            ),
        )

    def validate(self) -> None:
        """Validate non-provider-specific configuration constraints."""
        if self.runtime.default_provider not in {"openai", "gemini", "elm_local"}:
            raise ConfigError(
                "DEFAULT_PROVIDER must be one of: openai, gemini, elm_local."
            )

        if not self.runtime.default_model:
            raise ConfigError("DEFAULT_MODEL cannot be empty.")

        provider_models = get_supported_models(self.runtime.default_provider)
        if provider_models and self.runtime.default_model not in provider_models:
            supported = ", ".join(provider_models)
            raise ConfigError(
                f"DEFAULT_MODEL must be one of [{supported}] "
                f"for provider '{self.runtime.default_provider}'."
            )

        if self.security.max_upload_size_mb <= 0:
            raise ConfigError("MAX_UPLOAD_SIZE_MB must be greater than zero.")

        if not self.security.allowed_extensions:
            raise ConfigError("ALLOWED_UPLOAD_EXTENSIONS cannot be empty.")

    def require_api_key_for_provider(self, provider: str) -> str:
        """Return the API key for a provider or raise a configuration error."""
        provider_name = provider.strip().lower()
        provider_map = {
            "openai": self.api_keys.openai_api_key,
            "gemini": self.api_keys.gemini_api_key,
            "elm_local": self.api_keys.elm_local_api_key,
        }

        if provider_name not in provider_map:
            raise ConfigError(f"Unsupported provider: {provider_name}")

        key = provider_map[provider_name]
        if not key:
            raise ConfigError(
                f"Missing API key for provider '{provider_name}'. "
                f"Set the corresponding value in .env."
            )

        return key
