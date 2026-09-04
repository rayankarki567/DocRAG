from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.services.booking import BookingService
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMClient
from app.services.memory import RedisChatMemory
from app.services.vector_store import QdrantVectorStore, RetrievedChunk


RAG_SYSTEM_PROMPT = """
You are a helpful assistant.

Answer the user's question using the supplied information from the
uploaded documents and the previous conversation.

Rules:
- Do not make up information.
- If the answer is not available in the supplied document information,
  say that you do not have enough information.
- Keep answers clear and concise.
"""


@dataclass
class RAGResult:
    """Result returned by the RAG pipeline."""

    answer: str
    sources: list[RetrievedChunk]


class RAGPipeline:
    """Custom RAG pipeline without LangChain or RetrievalQAChain."""

    def __init__(
        self,
        settings: Settings,
        embeddings: EmbeddingService,
        vector_store: QdrantVectorStore,
        llm: LLMClient,
        memory: RedisChatMemory,
        booking: BookingService,
    ) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.llm = llm
        self.memory = memory
        self.booking = booking

    async def chat(
        self,
        db: AsyncSession,
        session_id: str,
        message: str,
        top_k: int | None = None,
    ) -> RAGResult:
        # 1. Save the user's message to Redis.
        await self.memory.add_message(
            session_id,
            "user",
            message,
        )

        # 2. Check whether this message belongs to a booking flow.
        booking_response = await self.booking.handle(
            db,
            session_id,
            message,
        )

        if booking_response is not None:
            await self.memory.add_message(
                session_id,
                "assistant",
                booking_response,
            )

            return RAGResult(
                answer=booking_response,
                sources=[],
            )

        # 3. Convert the question into an embedding.
        query_embedding = await self.embeddings.embed_query(message)

        # 4. Retrieve relevant chunks from Qdrant.
        chunks = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k or self.settings.retrieval_top_k,
        )

        # 5. Get previous conversation history.
        history = await self.memory.get_history(session_id)

        # The current user message is already stored in Redis.
        # Do not duplicate it in the conversation history sent to the LLM.
        prior_history = history[:-1]

        # 6. Build the document context manually.
        context = "\n\n".join(
            f"Source {i + 1}:\n{chunk.text}"
            for i, chunk in enumerate(chunks)
        )

        if not context:
            context = "No relevant information was found."

        # 7. Build the prompt manually.
        prompt = f"""
Document information:

{context}

User question:

{message}
"""

        # 8. Generate the answer.
        answer = await self.llm.generate(
            system_prompt=RAG_SYSTEM_PROMPT,
            messages=[
                *prior_history,
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        # 9. Save the assistant response.
        await self.memory.add_message(
            session_id,
            "assistant",
            answer,
        )

        return RAGResult(
            answer=answer,
            sources=chunks,
        )