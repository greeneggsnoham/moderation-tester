"""Abstract provider contract for moderation tester integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from moderation_tester.models import ChatRequest, ModerationResult, ProviderResponse


class ProviderClient(ABC):
    """Base class that all provider implementations must satisfy."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return provider identifier used in routing."""

    @abstractmethod
    def available_models(self) -> tuple[str, ...]:
        """Return models supported by this provider implementation."""

    @abstractmethod
    def send_chat(self, request: ChatRequest) -> ProviderResponse:
        """Execute a chat request and return a normalized response."""

    @abstractmethod
    def run_moderation(self, request: ChatRequest) -> ModerationResult | None:
        """Run moderation for a request if provider supports it."""
