from __future__ import annotations

import re

from app.models.schemas import ChunkingStrategy


def fixed_size_chunks(
    text: str,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[str]:
    """Split text into fixed-size character chunks with overlap."""
    text = _normalize_text(text)

    if not text:
        return []

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

        if start + chunk_size >= len(text):
            break

    return chunks


def semantic_chunks(
    text: str,
    target_size: int = 900,
    max_size: int = 1400,
) -> list[str]:
    """
    Split text on paragraph boundaries where possible, then sentence
    boundaries for oversized paragraphs.
    """
    text = _normalize_text(text, preserve_paragraphs=True)

    if not text:
        return []

    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:

        candidate = (
            f"{current}\n\n{paragraph}"
            if current
            else paragraph
        )

        if len(candidate) <= target_size:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
            current = ""

        if len(paragraph) <= max_size:
            current = paragraph
        else:
            chunks.extend(
                _split_large_paragraph(paragraph, max_size)
            )

    if current:
        chunks.append(current.strip())

    return chunks


def _split_large_paragraph(
    paragraph: str,
    max_size: int,
) -> list[str]:
    """Split an oversized paragraph using sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)

    chunks = []
    current = ""

    for sentence in sentences:
        candidate = (
            f"{current} {sentence}"
            if current
            else sentence
        )

        if len(candidate) <= max_size:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())

        if len(sentence) <= max_size:
            current = sentence
        else:
            for i in range(0, len(sentence), max_size):
                chunks.append(sentence[i:i + max_size].strip())

            current = ""

    if current:
        chunks.append(current.strip())

    return chunks


def _normalize_text(
    text: str,
    preserve_paragraphs: bool = False,
) -> str:
    """Clean common whitespace problems from extracted text."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)

    if preserve_paragraphs:
        text = re.sub(r"\n{3,}", "\n\n", text)
    else:
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r" {2,}", " ", text)

    return text.strip()


def chunk_text(
    text: str,
    strategy: ChunkingStrategy,
    *,
    fixed_size: int = 800,
    fixed_overlap: int = 120,
    semantic_target: int = 900,
    semantic_max: int = 1400,
) -> list[str]:
    """Apply the selected chunking strategy."""
    if strategy == ChunkingStrategy.FIXED_SIZE:
        return fixed_size_chunks(
            text,
            chunk_size=fixed_size,
            overlap=fixed_overlap,
        )

    if strategy == ChunkingStrategy.SEMANTIC:
        return semantic_chunks(
            text,
            target_size=semantic_target,
            max_size=semantic_max,
        )

    raise ValueError(f"Unsupported chunking strategy: {strategy}")