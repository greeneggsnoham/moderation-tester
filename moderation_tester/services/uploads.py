"""Document upload validation and text extraction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from moderation_tester.models import DocumentAttachment


class UploadValidationError(ValueError):
    """Raised when an uploaded file fails validation or parsing."""


@dataclass(frozen=True)
class UploadPayload:
    """Normalized upload payload independent of UI framework types."""

    filename: str
    media_type: str
    content: bytes


def build_attachments_from_uploads(
    uploads: list[UploadPayload],
    max_upload_size_mb: int,
    allowed_extensions: tuple[str, ...],
) -> tuple[DocumentAttachment, ...]:
    """Validate and parse uploaded files into prompt attachment objects."""
    attachments: list[DocumentAttachment] = []
    max_bytes = max_upload_size_mb * 1024 * 1024
    allowed = {ext.lower().lstrip(".") for ext in allowed_extensions}

    for upload in uploads:
        ext = _extract_extension(upload.filename)
        if ext not in allowed:
            raise UploadValidationError(
                f"Unsupported file extension '.{ext}' for {upload.filename}."
            )
        if len(upload.content) > max_bytes:
            raise UploadValidationError(
                f"File {upload.filename} exceeds max size "
                f"{max_upload_size_mb} MB."
            )

        extracted_text = _extract_text(upload.filename, ext, upload.content)
        attachments.append(
            DocumentAttachment(
                filename=upload.filename,
                media_type=upload.media_type or "application/octet-stream",
                extracted_text=extracted_text.strip(),
            )
        )

    return tuple(attachments)


def _extract_extension(filename: str) -> str:
    """Extract a lower-cased file extension without leading dot."""
    parts = filename.rsplit(".", maxsplit=1)
    if len(parts) != 2 or not parts[1]:
        raise UploadValidationError(f"File '{filename}' has no extension.")
    return parts[1].lower()


def _extract_text(filename: str, extension: str, content: bytes) -> str:
    """Extract text content from supported document formats."""
    if extension == "csv":
        return _extract_csv_text(content)
    if extension == "pdf":
        return _extract_pdf_text(content)
    if extension == "docx":
        return _extract_docx_text(content)
    if extension == "doc":
        # Legacy .doc is not reliably parseable without external tooling.
        return _extract_plain_text(content)
    raise UploadValidationError(f"Unsupported file type for {filename}.")


def _extract_csv_text(content: bytes) -> str:
    """Decode CSV content to text with safe fallbacks."""
    return _extract_plain_text(content)


def _extract_pdf_text(content: bytes) -> str:
    """Extract text from PDF content using pypdf."""
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise UploadValidationError(
            "PDF parsing requires pypdf. Install dependencies first."
        ) from exc

    reader = PdfReader(BytesIO(content))
    page_text: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text:
            page_text.append(text)
    return "\n".join(page_text)


def _extract_docx_text(content: bytes) -> str:
    """Extract text from DOCX content using python-docx."""
    try:
        from docx import Document
    except ModuleNotFoundError as exc:
        raise UploadValidationError(
            "DOCX parsing requires python-docx. Install dependencies first."
        ) from exc

    document = Document(BytesIO(content))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def _extract_plain_text(content: bytes) -> str:
    """Decode bytes into text using robust encoding fallbacks."""
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UploadValidationError("Unable to decode file content as text.")
