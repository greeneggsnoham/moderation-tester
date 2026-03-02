"""Shared typed models for request, moderation, and response payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ChatMessage:
    """A single message in the chat transcript."""

    role: MessageRole
    content: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class ModerationSettings:
    """Moderation controls associated with a request."""

    enabled: bool
    categories: tuple[str, ...] = ()
    level: str | None = None


@dataclass(frozen=True)
class DocumentAttachment:
    """Metadata and extracted text for uploaded files used in prompting."""

    filename: str
    media_type: str
    extracted_text: str


@dataclass(frozen=True)
class ChatRequest:
    """Normalized request object passed into provider integrations."""

    provider: str
    model: str
    messages: tuple[ChatMessage, ...]
    moderation: ModerationSettings
    attachments: tuple[DocumentAttachment, ...] = ()
    request_id: str | None = None


@dataclass(frozen=True)
class ModerationResult:
    """Structured moderation outcome from a provider or moderation model."""

    flagged: bool
    categories: dict[str, float] = field(default_factory=dict)
    reason: str | None = None


@dataclass(frozen=True)
class ProviderResponse:
    """Normalized model response payload."""

    text: str
    model: str
    provider: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineResult:
    """Combined result for one request pipeline execution."""

    request: ChatRequest
    response: ProviderResponse | None
    moderation: ModerationResult | None = None
    error: str | None = None
