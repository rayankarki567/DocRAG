from __future__ import annotations

import asyncio

from sentence_transformers import SentenceTransformer

from app.config import Settings


class EmbeddingService:
    """Generate local text embeddings using Sentence Transformers."""

    def __init__(self, settings: Settings) -> None:
        self.model = SentenceTransformer(settings.embedding_model)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        embeddings = await asyncio.to_thread(
            self.model.encode,
            texts,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    async def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a single query."""
        embeddings = await self.embed_texts([text])
        return embeddings[0]