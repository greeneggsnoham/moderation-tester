"""Moderation Tester application entrypoint."""

from __future__ import annotations

from typing import Any

import streamlit as st

from moderation_tester.config import AppConfig, ConfigError
from moderation_tester.models import ChatMessage, ChatRequest, ModerationSettings
from moderation_tester.providers.catalog import get_supported_models
from moderation_tester.services.pipeline import execute_request_pipeline
from moderation_tester.services.uploads import UploadPayload
from moderation_tester.services.uploads import UploadValidationError
from moderation_tester.services.uploads import build_attachments_from_uploads


def _init_chat_state() -> None:
    """Initialize chat transcript state."""
    if "chat_transcript" not in st.session_state:
        st.session_state.chat_transcript = []


def _append_user_message(content: str, attachment_names: list[str]) -> None:
    """Append user message to transcript."""
    display_text = content
    if attachment_names:
        display_text = (
            f"{content}\n\nAttachments: {', '.join(attachment_names)}"
        )
    st.session_state.chat_transcript.append(
        {"role": "user", "content": display_text}
    )


def _append_assistant_message(
    content: str,
    provider: str,
    model: str,
    moderation: dict[str, Any] | None,
    error: str | None,
) -> None:
    """Append assistant response with metadata to transcript."""
    st.session_state.chat_transcript.append(
        {
            "role": "assistant",
            "content": content,
            "provider": provider,
            "model": model,
            "moderation": moderation,
            "error": error,
        }
    )


def _render_transcript() -> None:
    """Render transcript messages in a chat-like interface."""
    for item in st.session_state.chat_transcript:
        role = item.get("role", "assistant")
        content = item.get("content", "")

        with st.chat_message(role):
            st.write(content)
            if role == "assistant":
                st.caption(
                    f"Provider: {item.get('provider')} | "
                    f"Model: {item.get('model')}"
                )
                moderation = item.get("moderation")
                if moderation is not None:
                    st.write("Moderation")
                    st.json(moderation)
                error = item.get("error")
                if error:
                    st.error(error)


def main() -> None:
    """Render a minimal shell UI for early development milestones."""
    st.set_page_config(page_title="Moderation Tester", layout="wide")
    st.title("Moderation Tester")
    st.caption("Internal evaluation tool for moderation behavior across providers.")
    _init_chat_state()

    try:
        config = AppConfig.from_env()
        config.validate()
    except ConfigError as err:
        st.error(f"Configuration error: {err}")
        st.info("Copy .env.example to .env and set required keys before running.")
        return

    st.success("Configuration loaded.")
    with st.sidebar:
        st.subheader("Request Settings")
        provider = st.selectbox(
            "Provider",
            options=[config.runtime.default_provider],
            index=0,
            disabled=True,
        )
    models = get_supported_models(provider)
    if not models:
        st.error(f"No models configured for provider '{provider}'.")
        return

    default_index = 0
    if config.runtime.default_model in models:
        default_index = models.index(config.runtime.default_model)
    with st.sidebar:
        model = st.selectbox(
            "Model",
            options=models,
            index=default_index,
        )
        moderation_enabled = st.checkbox(
            "Moderation enabled",
            value=config.runtime.default_moderation_enabled,
        )
        uploaded_files = st.file_uploader(
            "Attach documents",
            type=list(config.security.allowed_extensions),
            accept_multiple_files=True,
            help="Supported: PDF, CSV, DOC, DOCX",
        )
        if st.button("Clear chat"):
            st.session_state.chat_transcript = []
            st.rerun()
        st.caption("Supported models")
        st.json({"provider": provider, "supported_models": models})

    _render_transcript()
    prompt = st.chat_input("Enter your prompt")

    if prompt:
        uploads: list[UploadPayload] = []
        attachment_names: list[str] = []
        if uploaded_files:
            for file in uploaded_files:
                uploads.append(
                    UploadPayload(
                        filename=file.name,
                        media_type=file.type,
                        content=file.getvalue(),
                    )
                )
            attachment_names = [item.filename for item in uploads]

        try:
            attachments = build_attachments_from_uploads(
                uploads=uploads,
                max_upload_size_mb=config.security.max_upload_size_mb,
                allowed_extensions=config.security.allowed_extensions,
            )
        except UploadValidationError as err:
            st.error(f"Upload error: {err}")
            return

        _append_user_message(prompt, attachment_names)
        request = ChatRequest(
            provider=provider,
            model=model,
            messages=(ChatMessage(role="user", content=prompt.strip()),),
            moderation=ModerationSettings(enabled=moderation_enabled),
            attachments=attachments,
        )
        with st.spinner("Waiting for provider response..."):
            result = execute_request_pipeline(config=config, request=request)
        moderation_payload = (
            None
            if result.moderation is None
            else {
                "flagged": result.moderation.flagged,
                "categories": result.moderation.categories,
                "reason": result.moderation.reason,
            }
        )
        assistant_text = (
            result.response.text
            if result.response is not None and result.response.text
            else "No response text returned."
        )
        _append_assistant_message(
            content=assistant_text,
            provider=provider,
            model=model,
            moderation=moderation_payload,
            error=result.error,
        )
        st.rerun()


if __name__ == "__main__":
    main()
