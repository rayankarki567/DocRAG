from __future__ import annotations

import io
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from pypdf import PdfReader


def extract_text(filename: str, raw_bytes: bytes) -> str:
    """Extract text from PDF, TXT, or image files."""

    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return _extract_pdf(raw_bytes)

    if extension == ".txt":
        return _extract_txt(raw_bytes)

    if extension in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}:
        return _extract_image(raw_bytes)

    raise ValueError(
        "Unsupported file type. Supported formats: PDF, TXT, PNG, JPG, JPEG, WEBP, BMP, TIFF."
    )


def _extract_pdf(raw_bytes: bytes) -> str:
    """
    Extract text from a PDF.

    First attempts normal PDF text extraction.
    If the PDF contains no usable text, falls back to OCR.
    """

    # First: normal PDF text extraction
    reader = PdfReader(io.BytesIO(raw_bytes))

    text = []

    for page in reader.pages:
        page_text = page.extract_text() or ""

        if page_text.strip():
            text.append(page_text.strip())

    result = "\n\n".join(text).strip()

    if result:
        return result

    # Fallback: OCR scanned/image PDF
    return _ocr_pdf(raw_bytes)


def _ocr_pdf(raw_bytes: bytes) -> str:
    """OCR every page of a scanned PDF."""

    document = fitz.open(stream=raw_bytes, filetype="pdf")

    text = []

    try:
        for page_number, page in enumerate(document):
            # Render page at 200 DPI
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(200 / 72, 200 / 72),
                alpha=False,
            )

            image = Image.frombytes(
                "RGB",
                [pixmap.width, pixmap.height],
                pixmap.samples,
            )

            page_text = pytesseract.image_to_string(image)

            if page_text.strip():
                text.append(
                    f"[Page {page_number + 1}]\n{page_text.strip()}"
                )

    finally:
        document.close()

    result = "\n\n".join(text).strip()

    if not result:
        raise ValueError(
            "The PDF contains no extractable text and OCR could not detect any text."
        )

    return result


def _extract_image(raw_bytes: bytes) -> str:
    """Extract text from an image using OCR."""

    image = Image.open(io.BytesIO(raw_bytes))

    text = pytesseract.image_to_string(image).strip()

    if not text:
        raise ValueError("No text could be detected in the image.")

    return text


def _extract_txt(raw_bytes: bytes) -> str:
    """Decode a TXT file using several common encodings."""

    # BOM-aware UTF encodings first
    for encoding in (
        "utf-8-sig",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
        "utf-32",
        "utf-32-le",
        "utf-32-be",
        "latin-1",
    ):
        try:
            text = raw_bytes.decode(encoding).strip()

            if text:
                return text

        except (UnicodeDecodeError, UnicodeError):
            continue

    raise ValueError("Could not decode the TXT file.")