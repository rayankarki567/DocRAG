from __future__ import annotations

import io

from pypdf import PdfReader


def extract_text(filename: str, raw_bytes: bytes) -> str:
    """Extract text from a PDF or TXT file."""

    filename = filename.lower()

    if filename.endswith(".pdf"):
        return _extract_pdf(raw_bytes)

    if filename.endswith(".txt"):
        return _extract_txt(raw_bytes)

    raise ValueError("Only PDF and TXT files are supported.")


def _extract_pdf(raw_bytes: bytes) -> str:
    """Extract text from all pages of a PDF."""
    reader = PdfReader(io.BytesIO(raw_bytes))

    text = []

    for page in reader.pages:
        page_text = page.extract_text() or ""

        if page_text.strip():
            text.append(page_text)

    result = "\n\n".join(text).strip()

    if not result:
        raise ValueError("The PDF contains no extractable text.")

    return result


def _extract_txt(raw_bytes: bytes) -> str:
    """Decode a TXT file."""
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            text = raw_bytes.decode(encoding).strip()

            if text:
                return text

        except UnicodeDecodeError:
            continue

    raise ValueError("Could not decode the TXT file.")