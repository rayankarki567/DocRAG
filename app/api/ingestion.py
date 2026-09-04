from __future__ import annotations

from fastapi import APIRouter, Depends, File, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.core.exceptions import FileTooLargeError
from app.db import crud
from app.db.session import get_db
from app.dependencies import get_embedding_service, get_vector_store
from app.models.schemas import ChunkingStrategy, DocumentResponse, DocumentStatus
from app.services.chunking import chunk_text
from app.services.embeddings import EmbeddingService
from app.services.text_extraction import extract_text
from app.services.vector_store import QdrantVectorStore


router = APIRouter(
    prefix="/documents",
    tags=["Document Ingestion"],
)


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile = File(...),
    chunking_strategy: ChunkingStrategy = Query(
        default=ChunkingStrategy.SEMANTIC,
    ),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
    embeddings: EmbeddingService = Depends(get_embedding_service),
    vector_store: QdrantVectorStore = Depends(get_vector_store),
) -> DocumentResponse:

    # 1. Read uploaded file.
    raw_bytes = await file.read()

    # 2. Check file size.
    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    if len(raw_bytes) > max_bytes:
        raise FileTooLargeError(
            f"File exceeds the {settings.max_upload_size_mb} MB limit."
        )

    filename = file.filename or "unnamed"
    content_type = file.content_type or "application/octet-stream"

    # 3. Create SQL metadata record.
    document = await crud.create_document(
        db,
        filename=filename,
        content_type=content_type,
        chunking_strategy=chunking_strategy,
    )

    try:
        # 4. Extract text.
        text = extract_text(
            filename,
            raw_bytes,
        )

        # 5. Chunk the text.
        chunks = chunk_text(
            text,
            chunking_strategy,
            fixed_size=settings.fixed_chunk_size,
            fixed_overlap=settings.fixed_chunk_overlap,
            semantic_target=settings.semantic_chunk_target_size,
            semantic_max=settings.semantic_chunk_max_size,
        )

        if not chunks:
            raise ValueError("No chunks were generated from the document.")

        # 6. Generate embeddings.
        embeddings_list = await embeddings.embed_texts(chunks)

        # 7. Create Qdrant collection if necessary.
        await vector_store.create_collection(
            settings.embedding_dimensions,
        )

        # 8. Store chunks + vectors in Qdrant.
        await vector_store.add_chunks(
            document_id=document.id,
            chunks=chunks,
            embeddings=embeddings_list,
        )

        # 9. Update SQL metadata.
        document = await crud.update_document(
            db,
            document,
            status=DocumentStatus.COMPLETED,
            num_chunks=len(chunks),
            char_count=len(text),
        )

        return DocumentResponse.model_validate(document)

    except Exception as exc:
        await crud.update_document(
            db,
            document,
            status=DocumentStatus.FAILED,
            error_message=str(exc),
        )
        raise