"""Tests for moderation pipeline ON/OFF orchestration behavior."""

from __future__ import annotations

import unittest

from moderation_tester.config import APIKeys
from moderation_tester.config import AppConfig
from moderation_tester.config import RuntimeConfig
from moderation_tester.config import SecurityConfig
from moderation_tester.models import ChatMessage, ChatRequest, ModerationResult
from moderation_tester.models import ModerationSettings, ProviderResponse
from moderation_tester.services.pipeline import execute_request_pipeline


class _FakeProvider:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def send_chat(self, request: ChatRequest) -> ProviderResponse:
        self.calls.append("send_chat")
        return ProviderResponse(
            text="ok",
            model=request.model,
            provider=request.provider,
        )

    def run_moderation(self, request: ChatRequest) -> ModerationResult:
        self.calls.append("run_moderation")
        return ModerationResult(flagged=False, categories={})


class _FakeFactory:
    def __init__(self, provider: _FakeProvider) -> None:
        self.provider = provider

    def get(self, provider: str) -> _FakeProvider:
        return self.provider


def _build_config() -> AppConfig:
    return AppConfig(
        api_keys=APIKeys(
            openai_api_key="x",
            gemini_api_key=None,
            elm_local_api_key=None,
        ),
        runtime=RuntimeConfig(
            default_provider="openai",
            default_model="o4-mini",
            default_moderation_enabled=True,
        ),
        security=SecurityConfig(
            max_upload_size_mb=10,
            allowed_extensions=("pdf", "csv", "doc", "docx"),
        ),
    )


def _build_request(provider: str, moderation_enabled: bool) -> ChatRequest:
    return ChatRequest(
        provider=provider,
        model="o4-mini",
        messages=(ChatMessage(role="user", content="hello"),),
        moderation=ModerationSettings(enabled=moderation_enabled),
    )


class PipelineTests(unittest.TestCase):
    """Validate routing for moderation toggle behavior."""

    def test_openai_moderation_off_skips_moderation_call(self) -> None:
        provider = _FakeProvider()
        factory = _FakeFactory(provider)
        request = _build_request(provider="openai", moderation_enabled=False)

        result = execute_request_pipeline(
            config=_build_config(),
            request=request,
            provider_factory=factory,
        )

        self.assertIsNone(result.error)
        self.assertEqual(provider.calls, ["send_chat"])
        self.assertIsNone(result.moderation)

    def test_openai_moderation_on_calls_moderation_then_chat(self) -> None:
        provider = _FakeProvider()
        factory = _FakeFactory(provider)
        request = _build_request(provider="openai", moderation_enabled=True)

        result = execute_request_pipeline(
            config=_build_config(),
            request=request,
            provider_factory=factory,
        )

        self.assertIsNone(result.error)
        self.assertEqual(provider.calls, ["run_moderation", "send_chat"])
        self.assertIsNotNone(result.moderation)

    def test_non_openai_moderation_on_skips_moderation_call(self) -> None:
        provider = _FakeProvider()
        factory = _FakeFactory(provider)
        request = _build_request(provider="gemini", moderation_enabled=True)

        result = execute_request_pipeline(
            config=_build_config(),
            request=request,
            provider_factory=factory,
        )

        self.assertIsNone(result.error)
        self.assertEqual(provider.calls, ["send_chat"])
        self.assertIsNone(result.moderation)


if __name__ == "__main__":
    unittest.main()
