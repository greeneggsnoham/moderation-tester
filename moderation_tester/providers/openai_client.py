"""OpenAI provider integration for chat and moderation requests."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from moderation_tester.models import ChatMessage, ChatRequest, ModerationResult
from moderation_tester.models import ProviderResponse
from moderation_tester.providers.base import ProviderClient
from moderation_tester.providers.catalog import get_supported_models


class ProviderIntegrationError(RuntimeError):
    """Raised when provider SDK integration cannot satisfy a request."""


class OpenAIProviderClient(ProviderClient):
    """OpenAI implementation of the provider client contract."""

    def __init__(
        self,
        api_key: str,
        moderation_model: str = "omni-moderation-latest",
        client: Any | None = None,
    ) -> None:
        """Initialize the OpenAI provider with API key and optional SDK client."""
        self._api_key = api_key
        self._moderation_model = moderation_model
        self._client = client

    @property
    def name(self) -> str:
        """Return provider identifier used in routing."""
        return "openai"

    def available_models(self) -> tuple[str, ...]:
        """Return models currently approved for this integration."""
        return get_supported_models("openai")

    def send_chat(self, request: ChatRequest) -> ProviderResponse:
        """Execute a chat completion through the OpenAI Responses API."""
        self._validate_model(request.model)
        client = self._get_client()
        messages = self._to_openai_messages(request)

        response = client.responses.create(model=request.model, input=messages)
        text = self._extract_text(response)

        return ProviderResponse(
            text=text,
            model=request.model,
            provider=self.name,
            raw=self._safe_raw(response),
        )

    def run_moderation(self, request: ChatRequest) -> ModerationResult | None:
        """Run moderation using OpenAI moderation model for request content."""
        client = self._get_client()
        moderation_input = self._build_moderation_input(request)

        if not moderation_input.strip():
            return ModerationResult(flagged=False, categories={})

        result = client.moderations.create(
            model=self._moderation_model, input=moderation_input
        )
        parsed = self._parse_moderation_result(result)
        return parsed

    def _validate_model(self, model: str) -> None:
        """Validate that the requested model is in the supported model set."""
        supported = self.available_models()
        if model not in supported:
            raise ProviderIntegrationError(
                f"Unsupported OpenAI model '{model}'. "
                f"Supported models: {', '.join(supported)}."
            )

    def _get_client(self) -> Any:
        """Return injected client or lazily construct OpenAI SDK client."""
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise ProviderIntegrationError(
                "OpenAI SDK not installed. Install dependencies from "
                "requirements.txt."
            ) from exc

        self._client = OpenAI(api_key=self._api_key)
        return self._client

    @staticmethod
    def _to_openai_messages(
        request: ChatRequest
    ) -> list[dict[str, str]]:
        """Map request payload to OpenAI Responses API input format."""
        mapped: list[dict[str, str]] = []
        for msg in request.messages:
            mapped.append({"role": msg.role, "content": msg.content})

        if request.attachments:
            attachment_lines = [
                f"[{attachment.filename}]\n{attachment.extracted_text}"
                for attachment in request.attachments
                if attachment.extracted_text
            ]
            if attachment_lines:
                mapped.append(
                    {
                        "role": "user",
                        "content": (
                            "Attached document context:\n\n"
                            + "\n\n".join(attachment_lines)
                        ),
                    }
                )
        return mapped

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract best-effort text from a response object or dict."""
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text:
            return output_text

        if isinstance(response, dict):
            maybe = response.get("output_text")
            if isinstance(maybe, str) and maybe:
                return maybe

        return ""

    @staticmethod
    def _safe_raw(response: Any) -> dict[str, Any]:
        """Build a serializable raw payload for debugging and display."""
        if isinstance(response, dict):
            return response

        if hasattr(response, "model_dump"):
            dumped = response.model_dump()
            if isinstance(dumped, dict):
                return dumped

        if hasattr(response, "__dict__"):
            return dict(response.__dict__)

        return {"repr": repr(response)}

    @staticmethod
    def _build_moderation_input(request: ChatRequest) -> str:
        """Flatten request content and attachments into moderation input text."""
        message_lines = [msg.content for msg in request.messages if msg.content]
        attachment_lines = [
            f"[{att.filename}] {att.extracted_text}"
            for att in request.attachments
            if att.extracted_text
        ]
        return "\n".join((*message_lines, *attachment_lines))

    @staticmethod
    def _parse_moderation_result(result: Any) -> ModerationResult:
        """Parse first moderation result item into normalized structure."""
        payload: dict[str, Any]
        if isinstance(result, dict):
            payload = result
        elif hasattr(result, "model_dump"):
            payload = result.model_dump()
        elif hasattr(result, "__dict__"):
            if hasattr(result, "__dataclass_fields__"):
                payload = asdict(result)
            else:
                payload = dict(result.__dict__)
        else:
            payload = {}

        result_list = payload.get("results") or []
        if not result_list:
            return ModerationResult(flagged=False, categories={})

        first = result_list[0]
        if not isinstance(first, dict):
            return ModerationResult(flagged=False, categories={})

        flagged = bool(first.get("flagged", False))
        categories = first.get("category_scores")
        if not isinstance(categories, dict):
            categories = {}

        return ModerationResult(
            flagged=flagged,
            categories={k: float(v) for k, v in categories.items()},
            reason="Flagged by moderation model" if flagged else None,
        )
