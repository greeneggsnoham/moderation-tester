"""Tests for OpenAI provider request and response normalization."""

from __future__ import annotations

import unittest

from moderation_tester.models import ChatMessage, ChatRequest, DocumentAttachment
from moderation_tester.models import ModerationSettings
from moderation_tester.providers.openai_client import OpenAIProviderClient
from moderation_tester.providers.openai_client import ProviderIntegrationError


class _FakeResponses:
    def create(self, **kwargs: object) -> dict[str, object]:
        self.kwargs = kwargs
        return {"output_text": "hello from fake model"}


class _FakeModerations:
    def create(self, **kwargs: object) -> dict[str, object]:
        self.kwargs = kwargs
        return {
            "results": [
                {
                    "flagged": True,
                    "category_scores": {"violence": 0.91, "self-harm": 0.02},
                }
            ]
        }


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.responses = _FakeResponses()
        self.moderations = _FakeModerations()


class OpenAIProviderTests(unittest.TestCase):
    """Validate OpenAI provider model checks and payload normalization."""

    def test_send_chat_returns_normalized_response(self) -> None:
        provider = OpenAIProviderClient(
            api_key="fake", client=_FakeOpenAIClient()
        )
        request = ChatRequest(
            provider="openai",
            model="o4-mini",
            messages=(ChatMessage(role="user", content="Hi"),),
            moderation=ModerationSettings(enabled=False),
        )

        result = provider.send_chat(request)
        self.assertEqual(result.provider, "openai")
        self.assertEqual(result.model, "o4-mini")
        self.assertEqual(result.text, "hello from fake model")

    def test_invalid_model_raises(self) -> None:
        provider = OpenAIProviderClient(
            api_key="fake", client=_FakeOpenAIClient()
        )
        request = ChatRequest(
            provider="openai",
            model="invalid-model",
            messages=(ChatMessage(role="user", content="Hi"),),
            moderation=ModerationSettings(enabled=False),
        )

        with self.assertRaises(ProviderIntegrationError):
            provider.send_chat(request)

    def test_run_moderation_returns_scores(self) -> None:
        provider = OpenAIProviderClient(
            api_key="fake", client=_FakeOpenAIClient()
        )
        request = ChatRequest(
            provider="openai",
            model="o4-mini",
            messages=(ChatMessage(role="user", content="Moderate this"),),
            moderation=ModerationSettings(enabled=True),
        )

        moderation = provider.run_moderation(request)
        self.assertIsNotNone(moderation)
        assert moderation is not None
        self.assertTrue(moderation.flagged)
        self.assertIn("violence", moderation.categories)

    def test_attachments_are_included_in_chat_input(self) -> None:
        fake_client = _FakeOpenAIClient()
        provider = OpenAIProviderClient(api_key="fake", client=fake_client)
        request = ChatRequest(
            provider="openai",
            model="o4-mini",
            messages=(ChatMessage(role="user", content="Use attachment"),),
            moderation=ModerationSettings(enabled=False),
            attachments=(
                DocumentAttachment(
                    filename="example.csv",
                    media_type="text/csv",
                    extracted_text="a,b\n1,2",
                ),
            ),
        )

        provider.send_chat(request)
        payload = fake_client.responses.kwargs["input"]
        assert isinstance(payload, list)
        self.assertEqual(payload[-1]["role"], "user")
        self.assertIn("Attached document context", payload[-1]["content"])
        self.assertIn("example.csv", payload[-1]["content"])


if __name__ == "__main__":
    unittest.main()
