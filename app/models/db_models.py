from __future__ import annotations

import uuid
from datetime import date, datetime, time

from sqlalchemy import Date, DateTime, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def generate_id() -> str:
    """Generate a UUID string."""
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


class Document(Base):
    """Metadata for an uploaded document."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_id,
    )

    filename: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
    )

    content_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    chunking_strategy: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        default="completed",
        nullable=False,
    )

    num_chunks: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    char_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class InterviewBooking(Base):
    """A completed interview booking."""

    __tablename__ = "interview_bookings"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_id,
    )

    session_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    email: Mapped[str] = mapped_column(
        String(320),
        nullable=False,
    )

    interview_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    interview_time: Mapped[time] = mapped_column(
        # SQLAlchemy can infer TIME from Python's time type,
        # but this explicit annotation makes the model clearer.
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )