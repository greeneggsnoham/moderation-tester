"""Tests for document upload parsing and validation behavior."""

from __future__ import annotations

import unittest

from moderation_tester.services.uploads import UploadPayload
from moderation_tester.services.uploads import UploadValidationError
from moderation_tester.services.uploads import build_attachments_from_uploads


class UploadsTests(unittest.TestCase):
    """Validate upload extension, size, and parsing behavior."""

    def test_csv_upload_is_parsed(self) -> None:
        uploads = [
            UploadPayload(
                filename="sample.csv",
                media_type="text/csv",
                content=b"col1,col2\nalpha,beta",
            )
        ]
        attachments = build_attachments_from_uploads(
            uploads=uploads,
            max_upload_size_mb=10,
            allowed_extensions=("pdf", "csv", "doc", "docx"),
        )
        self.assertEqual(len(attachments), 1)
        self.assertIn("alpha", attachments[0].extracted_text)

    def test_invalid_extension_raises(self) -> None:
        uploads = [
            UploadPayload(
                filename="sample.txt",
                media_type="text/plain",
                content=b"hello",
            )
        ]
        with self.assertRaises(UploadValidationError):
            build_attachments_from_uploads(
                uploads=uploads,
                max_upload_size_mb=10,
                allowed_extensions=("pdf", "csv", "doc", "docx"),
            )

    def test_size_limit_raises(self) -> None:
        uploads = [
            UploadPayload(
                filename="sample.csv",
                media_type="text/csv",
                content=(b"a" * 2048),
            )
        ]
        with self.assertRaises(UploadValidationError):
            build_attachments_from_uploads(
                uploads=uploads,
                max_upload_size_mb=0,
                allowed_extensions=("pdf", "csv", "doc", "docx"),
            )


if __name__ == "__main__":
    unittest.main()
