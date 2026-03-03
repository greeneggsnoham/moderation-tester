"""Pipeline orchestration stubs used by upcoming implementation steps."""

from __future__ import annotations

from typing import Protocol

from moderation_tester.config import AppConfig
from moderation_tester.models import ChatRequest, PipelineResult
from moderation_tester.models import ProviderResponse
from moderation_tester.providers.factory import ProviderFactory


class ProviderFactoryLike(Protocol):
    """Minimal factory interface needed by pipeline orchestration."""

    def get(self, provider: str) -> object:
        """Return provider client for selected provider."""


def build_not_implemented_result(request: ChatRequest) -> PipelineResult:
    """Return a placeholder result until provider and moderation logic is added."""
    return PipelineResult(
        request=request,
        response=ProviderResponse(
            text="Pipeline not implemented yet.",
            model=request.model,
            provider=request.provider,
        ),
        error=None,
    )


def run_provider_request(config: AppConfig, request: ChatRequest) -> PipelineResult:
    """Send request to selected provider without moderation orchestration."""
    factory = ProviderFactory(config)
    client = factory.get(request.provider)
    response = client.send_chat(request)
    return PipelineResult(request=request, response=response)


def execute_request_pipeline(
    config: AppConfig,
    request: ChatRequest,
    provider_factory: ProviderFactoryLike | None = None,
) -> PipelineResult:
    """Execute provider request with OpenAI moderation ON/OFF orchestration."""
    factory = provider_factory or ProviderFactory(config)
    client = factory.get(request.provider)
    moderation_result = None

    try:
        provider_name = request.provider.strip().lower()
        if provider_name == "openai" and request.moderation.enabled:
            moderation_result = client.run_moderation(request)

        response = client.send_chat(request)
        return PipelineResult(
            request=request,
            response=response,
            moderation=moderation_result,
        )
    except Exception as err:  # pragma: no cover - defensive boundary.
        return PipelineResult(
            request=request,
            response=None,
            moderation=moderation_result,
            error=str(err),
        )
