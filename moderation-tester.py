"""
Moderation Tester: local web app for evaluating moderation settings.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import tempfile
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from flask import Flask, jsonify, render_template_string, request
from werkzeug.utils import secure_filename
import requests

# External dependencies:
# - Flask: web server + request routing + templating for local UI.
# - Werkzeug: secure_filename to avoid unsafe upload paths.

Provider = str

OPENAI_MODELS = ["o3", "o4-mini", "gpt-5", "gpt-5.1", "gpt-5.2"]
GEMINI_MODELS = ["gemini-2.5-pro", "gemini-2.5-flash"]

ALLOWED_UPLOAD_EXTS = {".pdf", ".csv", ".doc", ".docx"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_INLINE_FILE_BYTES = 2 * 1024 * 1024
GEMINI_THRESHOLDS = {
    "BLOCK_NONE",
    "BLOCK_ONLY_HIGH",
    "BLOCK_MEDIUM_AND_ABOVE",
    "BLOCK_LOW_AND_ABOVE",
}


def log_event(message: str) -> None:
    """
    Log a simple timestamped message to the terminal.
    """

    timestamp = datetime.utcnow().isoformat(timespec="seconds")
    print(f"[{timestamp}Z] {message}")


@dataclass(frozen=True)
class AppConfig:
    """
    Configuration for the local web app.
    """

    host: str
    port: int
    upload_dir: Path
    max_upload_bytes: int


def load_config() -> AppConfig:
    """
    Load application configuration from environment with defaults.
    """

    host = os.getenv("MODERATION_TESTER_HOST", "127.0.0.1")
    port = int(os.getenv("MODERATION_TESTER_PORT", "8000"))
    upload_dir = Path(
        os.getenv(
            "MODERATION_TESTER_UPLOAD_DIR",
            Path(tempfile.gettempdir()) / "moderation_tester_uploads",
        )
    )
    max_upload_bytes = int(
        os.getenv("MODERATION_TESTER_MAX_UPLOAD_BYTES", MAX_UPLOAD_BYTES)
    )
    return AppConfig(
        host=host,
        port=port,
        upload_dir=upload_dir,
        max_upload_bytes=max_upload_bytes,
    )


def get_api_key(provider: Provider, hardcoded_key: Optional[str]) -> str:
    """
    Retrieve API key from request override or environment variables.
    """

    env_map = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    if hardcoded_key:
        return hardcoded_key.strip()
    env_var = env_map.get(provider)
    if not env_var:
        raise ValueError(f"Unknown provider '{provider}'.")
    key = os.getenv(env_var, "").strip()
    if not key:
        raise ValueError(
            f"Missing API key. Set {env_var} or pass --api-key."
        )
    return key


def validate_upload(
    filename: str,
    file_size: int,
    allowed_exts: Iterable[str],
    max_bytes: int,
) -> None:
    """
    Validate file extension and size for uploads.
    """

    ext = Path(filename).suffix.lower()
    if ext not in allowed_exts:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Allowed: {', '.join(sorted(allowed_exts))}."
        )
    if file_size > max_bytes:
        raise ValueError(
            f"File too large ({file_size} bytes). Max allowed is "
            f"{max_bytes} bytes."
        )


def get_models_for_provider(provider: Provider) -> List[str]:
    """
    Return the allowed models for the provider.
    """

    if provider == "openai":
        return OPENAI_MODELS
    if provider == "gemini":
        return GEMINI_MODELS
    raise ValueError(f"Unknown provider '{provider}'.")


def gemini_safety_settings(
    moderation_on: bool,
    overrides: Optional[Dict[str, str]] = None,
) -> List[Dict[str, str]]:
    """
    Build Gemini safety settings based on moderation toggle.
    """

    # Map "high" to blocking low and above, and "negligible" to block none.
    threshold = "BLOCK_LOW_AND_ABOVE" if moderation_on else "BLOCK_NONE"
    if overrides:
        threshold = overrides.get("default", threshold)
    return [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": overrides.get("harassment", threshold)
            if overrides
            else threshold,
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": overrides.get("hate_speech", threshold)
            if overrides
            else threshold,
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": overrides.get("sexually_explicit", threshold)
            if overrides
            else threshold,
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": overrides.get("dangerous_content", threshold)
            if overrides
            else threshold,
        },
    ]


def _read_file_base64(path: Path) -> str:
    """
    Read a file and return base64-encoded content.
    """

    return base64.b64encode(path.read_bytes()).decode("ascii")


def _infer_mime_type(path: Path) -> str:
    """
    Infer MIME type for a file path with a safe fallback.
    """

    mime_type, _ = mimetypes.guess_type(path.name)
    return mime_type or "application/octet-stream"


def validate_gemini_overrides(
    overrides: Dict[str, str],
) -> Dict[str, str]:
    """
    Validate Gemini safety threshold overrides.
    """

    for key, value in overrides.items():
        if value not in GEMINI_THRESHOLDS:
            raise ValueError(
                f"Invalid Gemini safety threshold for {key}: {value}"
            )
    return overrides


def run_openai_moderation(prompt: str, api_key: str) -> Dict[str, object]:
    """
    Call OpenAI Moderations API.
    """

    url = "https://api.openai.com/v1/moderations"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "omni-moderation-latest",
        "input": prompt,
    }
    log_event("OpenAI moderation request: sending")
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    if response.status_code >= 400:
        log_event(
            "OpenAI moderation error "
            f"{response.status_code}: {response.text}"
        )
        response.raise_for_status()
    return response.json()


def run_openai_completion(
    prompt: str,
    model: str,
    api_key: str,
    file_path: Optional[Path],
) -> Dict[str, object]:
    """
    Call OpenAI Responses API.
    """

    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    content_parts = [
        {"type": "input_text", "text": prompt},
    ]
    if file_path:
        content_parts.append(
            {
                "type": "input_file",
                "file_data": _read_file_base64(file_path),
                "filename": file_path.name,
            }
        )
    payload = {
        "model": model,
        "store": False,
        "input": [
            {
                "role": "user",
                "content": content_parts,
            }
        ],
    }
    log_event("OpenAI responses request: sending")
    response = requests.post(url, headers=headers, json=payload, timeout=120)
    if response.status_code >= 400:
        log_event(
            "OpenAI responses error "
            f"{response.status_code}: {response.text}"
        )
        response.raise_for_status()
    return response.json()


def run_gemini_completion(
    prompt: str,
    model: str,
    api_key: str,
    safety_settings: List[Dict[str, str]],
    file_path: Optional[Path],
) -> Dict[str, object]:
    """
    Call Gemini generateContent with optional inline file data.
    """

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{model}:generateContent"
    )
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    parts: List[Dict[str, object]] = [{"text": prompt}]
    if file_path:
        file_uri, mime_type = gemini_upload_file(file_path, api_key)
        parts.append(
            {
                "file_data": {
                    "mime_type": mime_type,
                    "file_uri": file_uri,
                }
            }
        )
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "safetySettings": safety_settings,
    }
    log_event("Gemini generateContent request: sending")
    response = requests.post(url, headers=headers, json=payload, timeout=120)
    if response.status_code >= 400:
        log_event(
            "Gemini generateContent error "
            f"{response.status_code}: {response.text}"
        )
        response.raise_for_status()
    return response.json()


def gemini_upload_file(path: Path, api_key: str) -> Tuple[str, str]:
    """
    Upload a file to Gemini Files API and return (file_uri, mime_type).
    """

    mime_type = _infer_mime_type(path)
    num_bytes = path.stat().st_size
    upload_start_url = (
        "https://generativelanguage.googleapis.com/"
        "upload/v1beta/files"
    )
    start_headers = {
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(num_bytes),
        "X-Goog-Upload-Header-Content-Type": mime_type,
        "Content-Type": "application/json",
    }
    log_event(f"Gemini file upload start: {path.name} ({num_bytes} bytes)")
    start_response = requests.post(
        f"{upload_start_url}?key={api_key}",
        headers=start_headers,
        json={"file": {"display_name": path.name}},
        timeout=120,
    )
    if start_response.status_code >= 400:
        log_event(
            "Gemini file upload start error "
            f"{start_response.status_code}: {start_response.text}"
        )
        start_response.raise_for_status()
    upload_url = start_response.headers.get("X-Goog-Upload-URL")
    if not upload_url:
        raise ValueError("Gemini upload URL missing from response.")

    upload_headers = {
        "Content-Length": str(num_bytes),
        "X-Goog-Upload-Offset": "0",
        "X-Goog-Upload-Command": "upload, finalize",
    }
    log_event(f"Gemini file upload bytes: {path.name}")
    upload_response = requests.post(
        upload_url,
        headers=upload_headers,
        data=path.read_bytes(),
        timeout=120,
    )
    if upload_response.status_code >= 400:
        log_event(
            "Gemini file upload bytes error "
            f"{upload_response.status_code}: {upload_response.text}"
        )
        upload_response.raise_for_status()
    payload = upload_response.json()
    file_uri = payload.get("file", {}).get("uri")
    if not file_uri:
        raise ValueError("Gemini upload response missing file.uri.")
    return file_uri, mime_type


def handle_request(
    provider: Provider,
    model: str,
    prompt: str,
    moderation_on: bool,
    api_key: str,
    file_path: Optional[Path],
    safety_overrides: Optional[Dict[str, str]],
) -> Tuple[Dict[str, str], Optional[Dict[str, str]]]:
    """
    Handle a request across providers and moderation settings.
    """

    moderation_result: Optional[Dict[str, str]] = None

    if provider == "openai":
        # Assumption: moderation-on sends to moderation endpoint first.
        if moderation_on:
            moderation_result = run_openai_moderation(prompt, api_key)
        response = run_openai_completion(prompt, model, api_key, file_path)
        return response, moderation_result

    if provider == "gemini":
        settings = gemini_safety_settings(moderation_on, overrides=safety_overrides)
        response = run_gemini_completion(
            prompt, model, api_key, settings, file_path
        )
        return response, None

    raise ValueError(f"Unknown provider '{provider}'.")


def create_app(config: AppConfig) -> Flask:
    """
    Create and configure the Flask application.
    """

    app = Flask(__name__)
    config.upload_dir.mkdir(parents=True, exist_ok=True)

    @app.get("/")
    def index() -> str:
        """
        Render the main UI page.
        """

        return render_template_string(
            INDEX_HTML,
            openai_models=OPENAI_MODELS,
            gemini_models=GEMINI_MODELS,
        )

    @app.post("/api/chat")
    def chat() -> Dict[str, object]:
        """
        Accept a chat request with optional file upload.
        """

        provider = request.form.get("provider", "").strip().lower()
        model = request.form.get("model", "").strip()
        prompt = request.form.get("prompt", "").strip()
        moderation_on = request.form.get("moderation") == "on"
        hardcoded_key = request.form.get("api_key", "").strip()
        safety_overrides = {
            "default": request.form.get("safety_default", "").strip(),
            "harassment": request.form.get("safety_harassment", "").strip(),
            "hate_speech": request.form.get("safety_hate_speech", "").strip(),
            "sexually_explicit": request.form.get(
                "safety_sexually_explicit", ""
            ).strip(),
            "dangerous_content": request.form.get(
                "safety_dangerous_content", ""
            ).strip(),
        }
        safety_overrides = {
            key: value for key, value in safety_overrides.items() if value
        }

        if not provider or not model or not prompt:
            return jsonify(
                {
                    "error": "provider, model, and prompt are required.",
                }
            ), 400

        if model not in get_models_for_provider(provider):
            return jsonify(
                {"error": f"Invalid model '{model}' for {provider}."}
            ), 400

        if provider == "gemini" and safety_overrides:
            try:
                safety_overrides = validate_gemini_overrides(safety_overrides)
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400
        elif provider != "gemini":
            safety_overrides = {}

        try:
            api_key = get_api_key(provider, hardcoded_key)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        upload_info: Optional[Dict[str, str]] = None
        uploaded_path: Optional[Path] = None
        if "document" in request.files:
            uploaded = request.files["document"]
            if uploaded and uploaded.filename:
                filename = secure_filename(uploaded.filename)
                uploaded.seek(0, os.SEEK_END)
                file_size = uploaded.tell()
                uploaded.seek(0)
                try:
                    validate_upload(
                        filename,
                        file_size,
                        ALLOWED_UPLOAD_EXTS,
                        config.max_upload_bytes,
                    )
                except ValueError as exc:
                    return jsonify({"error": str(exc)}), 400
                dest = config.upload_dir / filename
                uploaded.save(dest)
                uploaded_path = dest
                upload_info = {
                    "filename": filename,
                    "stored_at": str(dest),
                }
                log_event(f"Upload saved: {filename}")

        log_event(
            "Request start "
            f"provider={provider} model={model} "
            f"moderation_on={moderation_on} "
            f"file={'yes' if uploaded_path else 'no'}"
        )
        try:
            response, moderation_result = handle_request(
                provider=provider,
                model=model,
                prompt=prompt,
                moderation_on=moderation_on,
                api_key=api_key,
                file_path=uploaded_path,
                safety_overrides=safety_overrides or None,
            )
        except requests.HTTPError as exc:
            log_event(f"Request failed: {exc}")
            return jsonify({"error": str(exc)}), 502
        except Exception as exc:
            log_event(f"Request failed: {exc}")
            return jsonify({"error": str(exc)}), 500
        finally:
            if uploaded_path and uploaded_path.exists():
                uploaded_path.unlink(missing_ok=True)
                log_event(f"Upload cleaned up: {uploaded_path.name}")

        return jsonify(
            {
                "provider": provider,
                "model": model,
                "moderation_on": moderation_on,
                "moderation_result": moderation_result,
                "response": response,
                "upload": upload_info,
            }
        )

    return app


INDEX_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Moderation Tester</title>
    <style>
      body { font-family: "Georgia", serif; margin: 24px; }
      .row { margin-bottom: 12px; }
      label { display: block; font-weight: bold; margin-bottom: 4px; }
      textarea { width: 100%; height: 120px; }
      .chat { border: 1px solid #ccc; padding: 12px; }
      .chat pre { white-space: pre-wrap; margin: 0; }
      .spinner { display: none; margin-top: 8px; }
      .spinner .dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        margin-right: 6px;
        border-radius: 50%;
        background: #333;
        animation: bounce 1s infinite;
      }
      .spinner .dot:nth-child(2) { animation-delay: 0.15s; }
      .spinner .dot:nth-child(3) { animation-delay: 0.3s; }
      @keyframes bounce {
        0%, 80%, 100% { transform: translateY(0); opacity: 0.6; }
        40% { transform: translateY(-6px); opacity: 1; }
      }
      .safety-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
      .hidden { display: none; }
    </style>
  </head>
  <body>
    <h1>Moderation Tester</h1>
    <p><strong>Internal Use Only:</strong> This tool is intended for internal evaluation.</p>
    <form id="chat-form">
      <div class="row">
        <label>Provider</label>
        <select id="provider" name="provider">
          <option value="openai">OpenAI</option>
          <option value="gemini">Gemini</option>
        </select>
      </div>
      <div class="row">
        <label>Model</label>
        <select id="model" name="model"></select>
      </div>
      <div class="row">
        <label>Moderation</label>
        <select id="moderation" name="moderation">
          <option value="on">On</option>
          <option value="off">Off</option>
        </select>
      </div>
      <div class="row hidden" id="gemini-safety">
        <label>Gemini Safety Settings (optional overrides)</label>
        <div class="safety-grid">
          <div>
            <label>Default Threshold</label>
            <select name="safety_default">
              <option value="">Use moderation toggle</option>
              <option value="BLOCK_NONE">BLOCK_NONE</option>
              <option value="BLOCK_ONLY_HIGH">BLOCK_ONLY_HIGH</option>
              <option value="BLOCK_MEDIUM_AND_ABOVE">
                BLOCK_MEDIUM_AND_ABOVE
              </option>
              <option value="BLOCK_LOW_AND_ABOVE">
                BLOCK_LOW_AND_ABOVE
              </option>
            </select>
          </div>
          <div>
            <label>Harassment</label>
            <select name="safety_harassment">
              <option value="">Use default</option>
              <option value="BLOCK_NONE">BLOCK_NONE</option>
              <option value="BLOCK_ONLY_HIGH">BLOCK_ONLY_HIGH</option>
              <option value="BLOCK_MEDIUM_AND_ABOVE">
                BLOCK_MEDIUM_AND_ABOVE
              </option>
              <option value="BLOCK_LOW_AND_ABOVE">
                BLOCK_LOW_AND_ABOVE
              </option>
            </select>
          </div>
          <div>
            <label>Hate Speech</label>
            <select name="safety_hate_speech">
              <option value="">Use default</option>
              <option value="BLOCK_NONE">BLOCK_NONE</option>
              <option value="BLOCK_ONLY_HIGH">BLOCK_ONLY_HIGH</option>
              <option value="BLOCK_MEDIUM_AND_ABOVE">
                BLOCK_MEDIUM_AND_ABOVE
              </option>
              <option value="BLOCK_LOW_AND_ABOVE">
                BLOCK_LOW_AND_ABOVE
              </option>
            </select>
          </div>
          <div>
            <label>Sexually Explicit</label>
            <select name="safety_sexually_explicit">
              <option value="">Use default</option>
              <option value="BLOCK_NONE">BLOCK_NONE</option>
              <option value="BLOCK_ONLY_HIGH">BLOCK_ONLY_HIGH</option>
              <option value="BLOCK_MEDIUM_AND_ABOVE">
                BLOCK_MEDIUM_AND_ABOVE
              </option>
              <option value="BLOCK_LOW_AND_ABOVE">
                BLOCK_LOW_AND_ABOVE
              </option>
            </select>
          </div>
          <div>
            <label>Dangerous Content</label>
            <select name="safety_dangerous_content">
              <option value="">Use default</option>
              <option value="BLOCK_NONE">BLOCK_NONE</option>
              <option value="BLOCK_ONLY_HIGH">BLOCK_ONLY_HIGH</option>
              <option value="BLOCK_MEDIUM_AND_ABOVE">
                BLOCK_MEDIUM_AND_ABOVE
              </option>
              <option value="BLOCK_LOW_AND_ABOVE">
                BLOCK_LOW_AND_ABOVE
              </option>
            </select>
          </div>
        </div>
      </div>
      <div class="row">
        <label>API Key (optional override)</label>
        <input type="password" name="api_key" />
      </div>
      <div class="row">
        <label>Prompt</label>
        <textarea name="prompt" required></textarea>
      </div>
      <div class="row">
        <label>Document Upload</label>
        <input type="file" name="document" />
      </div>
      <button type="submit">Send</button>
      <div class="spinner" id="spinner">
        <span class="dot"></span><span class="dot"></span><span class="dot"></span>
        Loading...
      </div>
    </form>
    <h2>Response</h2>
    <div class="chat" id="chat"></div>
    <script>
      const models = {
        openai: {{ openai_models|tojson }},
        gemini: {{ gemini_models|tojson }},
      };
      const providerEl = document.getElementById("provider");
      const modelEl = document.getElementById("model");
      const chatEl = document.getElementById("chat");
      const spinnerEl = document.getElementById("spinner");
      const geminiSafetyEl = document.getElementById("gemini-safety");

      function refreshModels() {
        const list = models[providerEl.value] || [];
        modelEl.innerHTML = "";
        list.forEach((model) => {
          const opt = document.createElement("option");
          opt.value = model;
          opt.textContent = model;
          modelEl.appendChild(opt);
        });
        if (providerEl.value === "gemini") {
          geminiSafetyEl.classList.remove("hidden");
        } else {
          geminiSafetyEl.classList.add("hidden");
        }
      }
      providerEl.addEventListener("change", refreshModels);
      refreshModels();

      document.getElementById("chat-form").addEventListener(
        "submit",
        async (event) => {
          event.preventDefault();
          const formData = new FormData(event.target);
          spinnerEl.style.display = "block";
          chatEl.textContent = "";
          const response = await fetch("/api/chat", {
            method: "POST",
            body: formData,
          });
          const data = await response.json();
          spinnerEl.style.display = "none";
          chatEl.innerHTML = "";
          const pre = document.createElement("pre");
          pre.textContent = JSON.stringify(data, null, 2);
          chatEl.appendChild(pre);
        }
      );
    </script>
  </body>
</html>
"""


def _test_get_models_for_provider() -> None:
    """
    Unit test for model selection by provider.
    """

    assert "o3" in get_models_for_provider("openai")
    assert "gemini-2.5-pro" in get_models_for_provider("gemini")


def _test_gemini_safety_settings() -> None:
    """
    Unit test for Gemini safety settings mapping.
    """

    on = gemini_safety_settings(True)
    off = gemini_safety_settings(False)
    assert on[0]["threshold"] == "BLOCK_LOW_AND_ABOVE"
    assert off[0]["threshold"] == "BLOCK_NONE"


def run_tests() -> None:
    """
    Run minimal unit tests for critical paths.
    """

    _test_get_models_for_provider()
    _test_gemini_safety_settings()


def main() -> None:
    """
    Entry point for running the local web app.
    """

    config = load_config()
    app = create_app(config)
    app.run(host=config.host, port=config.port, debug=True)


if __name__ == "__main__":
    run_tests()
    main()
