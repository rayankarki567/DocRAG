from __future__ import annotations

import uuid
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient, models

from app.config import Settings


@dataclass
class RetrievedChunk:
    """A chunk returned by Qdrant similarity search."""

    document_id: str
    chunk_index: int
    text: str
    score: float


class QdrantVectorStore:
    """Store and retrieve document embeddings using Qdrant."""

    def __init__(self, settings: Settings) -> None:
        self.client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        self.collection = settings.qdrant_collection

    async def create_collection(self, dimensions: int) -> None:
        """Create the collection if it does not already exist."""
        if await self.client.collection_exists(self.collection):
            return

        await self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=dimensions,
                distance=models.Distance.COSINE,
            ),
        )

    async def add_chunks(
        self,
        document_id: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """Store embedded document chunks in Qdrant."""

        points = []

        for index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "document_id": document_id,
                        "chunk_index": index,
                        "text": chunk,
                    },
                )
            )

        if points:
            await self.client.upsert(
                collection_name=self.collection,
                points=points,
            )

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 4,
    ) -> list[RetrievedChunk]:
        """Find the most similar document chunks."""

        results = await self.client.query_points(
            collection_name=self.collection,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        )

        return [
            RetrievedChunk(
                document_id=point.payload["document_id"],
                chunk_index=point.payload["chunk_index"],
                text=point.payload["text"],
                score=point.score,
            )
            for point in results.points
        ]