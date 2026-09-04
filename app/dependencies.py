from __future__ import annotations

from collections.abc import AsyncGenerator

from fastapi import Depends
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db.session import get_db
from app.services.booking import BookingService
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMClient
from app.services.memory import RedisChatMemory
from app.services.rag import RAGPipeline
from app.services.vector_store import QdrantVectorStore


# Shared application clients
redis_client: Redis | None = None
vector_store: QdrantVectorStore | None = None
embedding_service: EmbeddingService | None = None
llm_client: LLMClient | None = None


def initialize(settings: Settings) -> None:
    """Create shared service clients when the application starts."""
    global redis_client, vector_store, embedding_service, llm_client

    redis_client = Redis.from_url(
        settings.redis_url,
        decode_responses=True,
    )

    vector_store = QdrantVectorStore(settings)
    embedding_service = EmbeddingService(settings)
    llm_client = LLMClient(settings)


async def shutdown() -> None:
    """Close shared connections when the application stops."""
    if redis_client is not None:
        await redis_client.aclose()

    if vector_store is not None:
        await vector_store.client.close()

    if llm_client is not None:
        await llm_client.client.close()


def get_redis() -> Redis:
    if redis_client is None:
        raise RuntimeError("Application has not been initialized.")

    return redis_client


def get_vector_store() -> QdrantVectorStore:
    if vector_store is None:
        raise RuntimeError("Application has not been initialized.")

    return vector_store


def get_embedding_service() -> EmbeddingService:
    if embedding_service is None:
        raise RuntimeError("Application has not been initialized.")

    return embedding_service


def get_llm_client() -> LLMClient:
    if llm_client is None:
        raise RuntimeError("Application has not been initialized.")

    return llm_client


def get_chat_memory(
    redis: Redis = Depends(get_redis),
    settings: Settings = Depends(get_settings),
) -> RedisChatMemory:
    return RedisChatMemory(redis, settings)


def get_booking_service(
    redis: Redis = Depends(get_redis),
    llm: LLMClient = Depends(get_llm_client),
) -> BookingService:
    return BookingService(redis, llm)


def get_rag_pipeline(
    settings: Settings = Depends(get_settings),
    embeddings: EmbeddingService = Depends(get_embedding_service),
    vector_store: QdrantVectorStore = Depends(get_vector_store),
    llm: LLMClient = Depends(get_llm_client),
    memory: RedisChatMemory = Depends(get_chat_memory),
    booking: BookingService = Depends(get_booking_service),
) -> RAGPipeline:
    return RAGPipeline(
        settings=settings,
        embeddings=embeddings,
        vector_store=vector_store,
        llm=llm,
        memory=memory,
        booking=booking,
    )


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide a database session to FastAPI."""
    async for session in get_db():
        yield session