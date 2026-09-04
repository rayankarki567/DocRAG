from __future__ import annotations

from datetime import date, time

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db_models import Document, InterviewBooking
from app.models.schemas import ChunkingStrategy, DocumentStatus


async def create_document(
    db: AsyncSession,
    filename: str,
    content_type: str,
    chunking_strategy: ChunkingStrategy,
) -> Document:
    """Create a document metadata record."""

    document = Document(
        filename=filename,
        content_type=content_type,
        chunking_strategy=chunking_strategy.value,
        status=DocumentStatus.PENDING.value,
    )

    db.add(document)
    await db.commit()
    await db.refresh(document)

    return document


async def update_document(
    db: AsyncSession,
    document: Document,
    status: DocumentStatus,
    num_chunks: int = 0,
    char_count: int = 0,
    error_message: str | None = None,
) -> Document:
    """Update document ingestion metadata."""

    document.status = status.value
    document.num_chunks = num_chunks
    document.char_count = char_count
    document.error_message = error_message

    await db.commit()
    await db.refresh(document)

    return document


async def create_booking(
    db: AsyncSession,
    session_id: str,
    name: str,
    email: str,
    interview_date: date,
    interview_time: time,
) -> InterviewBooking:
    """Save a completed interview booking."""

    booking = InterviewBooking(
        session_id=session_id,
        name=name,
        email=email,
        interview_date=interview_date,
        interview_time=interview_time,
    )

    db.add(booking)
    await db.commit()
    await db.refresh(booking)

    return booking


async def get_booking(
    db: AsyncSession,
    booking_id: str,
) -> InterviewBooking | None:
    """Find a booking by ID."""

    result = await db.execute(
        select(InterviewBooking).where(
            InterviewBooking.id == booking_id
        )
    )

    return result.scalar_one_or_none()