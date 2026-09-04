from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.dependencies import get_chat_memory, get_rag_pipeline
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    RetrievedSource,
)
from app.services.memory import RedisChatMemory
from app.services.rag import RAGPipeline


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    rag: RAGPipeline = Depends(get_rag_pipeline),
) -> ChatResponse:
    """Send a message and receive a RAG or booking response."""

    result = await rag.chat(
        db=db,
        session_id=request.session_id,
        message=request.message,
        top_k=request.top_k,
    )

    return ChatResponse(
        session_id=request.session_id,
        answer=result.answer,
        sources=[
            RetrievedSource(
                document_id=source.document_id,
                chunk_index=source.chunk_index,
                score=source.score,
                text=source.text,
            )
            for source in result.sources
        ],
    )


@router.get("/{session_id}/history")
async def get_history(
    session_id: str,
    memory: RedisChatMemory = Depends(get_chat_memory),
) -> list[dict[str, str]]:
    """Return conversation history for a session."""

    return await memory.get_history(session_id)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def clear_session(
    session_id: str,
    memory: RedisChatMemory = Depends(get_chat_memory),
) -> Response:
    """Delete conversation history for a session."""

    await memory.clear(session_id)

    return Response(status_code=status.HTTP_204_NO_CONTENT)