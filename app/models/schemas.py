from __future__ import annotations

from datetime import date, datetime, time
from enum import Enum

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class ChunkingStrategy(str, Enum):
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    content_type: str
    chunking_strategy: ChunkingStrategy
    status: DocumentStatus
    num_chunks: int
    char_count: int
    error_message: str | None = None
    created_at: datetime


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=4, ge=1, le=20)


class RetrievedSource(BaseModel):
    document_id: str
    chunk_index: int
    score: float
    text: str


class BookingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    name: str
    email: EmailStr
    interview_date: date
    interview_time: time
    created_at: datetime


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[RetrievedSource] = Field(default_factory=list)
    booking: BookingResponse | None = None