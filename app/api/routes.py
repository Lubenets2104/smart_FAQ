"""API routes for the SmartTask FAQ service."""

import os
import re
import time
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.models import (
    AskRequest,
    AskResponse,
    SourceInfo,
    DocumentUploadResponse,
    HealthCheckResponse,
    QueryHistoryItem,
    ErrorResponse,
)
from app.db import get_db, QueryHistory, check_db_connection
from app.services import cache_service, rag_service, llm_service
from app.utils import (
    get_logger,
    record_cache_hit,
    record_cache_miss,
    record_llm_usage,
    record_rag_search,
    update_documents_indexed,
    record_document_upload,
    update_service_health,
)
from app.config import get_settings

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter()

# Security constants
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
ALLOWED_FILENAME_PATTERN = re.compile(r'^[\w\-. ]+$')  # alphanumeric, dash, dot, space


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal and other attacks.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename safe for storage
    """
    # Remove any path components
    filename = os.path.basename(filename)
    # Replace potentially dangerous characters
    if not ALLOWED_FILENAME_PATTERN.match(filename):
        # Generate safe filename preserving extension
        name, ext = os.path.splitext(filename)
        safe_name = re.sub(r'[^\w\-.]', '_', name)
        filename = f"{safe_name}{ext}"
    return filename


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Check the health status of all services",
)
async def health_check() -> HealthCheckResponse:
    """Check health of all services."""
    
    # Check PostgreSQL
    try:
        pg_ok = await check_db_connection()
        postgres_status = "healthy" if pg_ok else "unhealthy"
    except Exception as e:
        logger.error("PostgreSQL health check failed", error=str(e))
        postgres_status = "unhealthy"
    
    # Check Redis
    try:
        redis_ok = await cache_service.check_connection()
        redis_status = "healthy" if redis_ok else "unhealthy"
    except Exception as e:
        logger.error("Redis health check failed", error=str(e))
        redis_status = "unhealthy"
    
    # Check ChromaDB
    try:
        chroma_ok = await rag_service.check_connection()
        chromadb_status = "healthy" if chroma_ok else "unhealthy"
    except Exception as e:
        logger.error("ChromaDB health check failed", error=str(e))
        chromadb_status = "unhealthy"
    
    # Overall status
    all_healthy = all([
        postgres_status == "healthy",
        redis_status == "healthy",
        chromadb_status == "healthy"
    ])

    # Update service health metrics
    update_service_health("postgres", postgres_status == "healthy")
    update_service_health("redis", redis_status == "healthy")
    update_service_health("chromadb", chromadb_status == "healthy")

    return HealthCheckResponse(
        status="healthy" if all_healthy else "degraded",
        postgres=postgres_status,
        redis=redis_status,
        chromadb=chromadb_status,
    )


@router.post(
    "/ask",
    response_model=AskResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Ask a Question",
    description="Submit a question and get an answer based on SmartTask knowledge base",
)
async def ask_question(
    request: AskRequest,
    db: AsyncSession = Depends(get_db),
) -> AskResponse:
    """
    Process a user question using RAG pipeline.
    
    1. Check cache for existing answer
    2. If not cached, search relevant documents
    3. Generate answer using LLM
    4. Cache the response
    5. Save to history
    """
    start_time = time.time()
    question = request.question.strip()
    
    logger.info("Received question", question=question[:100])
    
    # Check cache first
    try:
        cached = await cache_service.get_cached_answer(question)
        if cached:
            record_cache_hit()
            return AskResponse(
                answer=cached["answer"],
                sources=[SourceInfo(**s) for s in cached["sources"]],
                tokens_used=cached["tokens_used"],
                response_time_ms=int((time.time() - start_time) * 1000),
                cached=True,
            )
        record_cache_miss()
    except Exception as e:
        logger.warning("Cache check failed, continuing without cache", error=str(e))
        record_cache_miss()
    
    # Get relevant context from RAG
    try:
        search_results = await rag_service.search(question)
        context = await rag_service.get_context(question)
        record_rag_search(success=True)

        sources = [
            SourceInfo(document=source, chunk=chunk[:200] + "..." if len(chunk) > 200 else chunk)
            for source, chunk, score in search_results
        ]
    except Exception as e:
        logger.error("RAG search failed", error=str(e))
        record_rag_search(success=False)
        sources = []
        context = ""
    
    # Generate answer using LLM
    try:
        llm_start = time.time()
        answer, tokens_used, llm_time = await llm_service.generate_answer(question, context)
        llm_duration = time.time() - llm_start
        record_llm_usage(
            provider=settings.llm_provider,
            tokens=tokens_used,
            duration=llm_duration,
            success=True
        )
    except Exception as e:
        logger.error("LLM generation failed", error=str(e))
        record_llm_usage(
            provider=settings.llm_provider,
            tokens=0,
            duration=0,
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate answer. Please try again later."
        )
    
    response_time_ms = int((time.time() - start_time) * 1000)
    
    # Create response
    response = AskResponse(
        answer=answer,
        sources=sources,
        tokens_used=tokens_used,
        response_time_ms=response_time_ms,
        cached=False,
    )
    
    # Cache the response
    try:
        cache_data = {
            "answer": answer,
            "sources": [s.model_dump() for s in sources],
            "tokens_used": tokens_used,
        }
        await cache_service.set_cached_answer(question, cache_data)
    except Exception as e:
        logger.warning("Failed to cache response", error=str(e))
    
    # Save to history
    try:
        history_entry = QueryHistory(
            question=question,
            answer=answer,
            tokens_used=tokens_used,
            response_time_ms=response_time_ms,
            sources=[s.model_dump() for s in sources],
        )
        db.add(history_entry)
        await db.commit()
        logger.info("Saved to history", query_id=str(history_entry.id))
    except Exception as e:
        logger.warning("Failed to save to history", error=str(e))
        await db.rollback()
    
    return response


@router.post(
    "/documents",
    response_model=DocumentUploadResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type"},
        500: {"model": ErrorResponse, "description": "Upload failed"},
    },
    summary="Upload Document",
    description="Upload a document to the knowledge base (txt/md)",
)
async def upload_document(
    file: UploadFile = File(..., description="Document file (txt or md)"),
) -> DocumentUploadResponse:
    """Upload a document to the vector store."""

    # Validate filename exists
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required"
        )

    # Validate file extension
    if not file.filename.endswith(('.txt', '.md')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .txt and .md files are supported"
        )

    # Sanitize filename to prevent path traversal attacks
    safe_filename = sanitize_filename(file.filename)

    logger.info("Uploading document", filename=safe_filename)

    # Read and validate file size
    try:
        content = await file.read()

        # Validate file size
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB"
            )

        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File is empty"
            )

        content_str = content.decode('utf-8')
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be UTF-8 encoded"
        )

    try:
        chunks_created = await rag_service.add_document(safe_filename, content_str)
        record_document_upload(success=True)
    except Exception as e:
        logger.error("Failed to add document", filename=safe_filename, error=str(e))
        record_document_upload(success=False)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process document. Please try again later."
        )

    return DocumentUploadResponse(
        message="Document uploaded successfully",
        filename=safe_filename,
        chunks_created=chunks_created,
    )


@router.get(
    "/history",
    response_model=List[QueryHistoryItem],
    summary="Get Query History",
    description="Get recent query history",
)
async def get_history(
    limit: int = Query(default=10, ge=1, le=100, description="Number of items to return (1-100)"),
    db: AsyncSession = Depends(get_db),
) -> List[QueryHistoryItem]:
    """Get recent query history."""

    result = await db.execute(
        select(QueryHistory)
        .order_by(QueryHistory.created_at.desc())
        .limit(limit)
    )

    items = result.scalars().all()
    return [QueryHistoryItem.model_validate(item) for item in items]


@router.get(
    "/stats",
    summary="Get Statistics",
    description="Get service statistics",
)
async def get_stats(
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Get service statistics."""

    # Get RAG stats
    try:
        rag_stats = await rag_service.get_collection_stats()
        doc_count = rag_stats.get("document_count", 0)
        update_documents_indexed(doc_count)
    except Exception:
        rag_stats = {"document_count": 0}

    # Get query count efficiently using COUNT
    try:
        result = await db.execute(select(func.count(QueryHistory.id)))
        query_count = result.scalar() or 0
    except Exception:
        query_count = 0

    return {
        "total_queries": query_count,
        "documents_indexed": rag_stats.get("document_count", 0),
        "collection": rag_stats.get("collection", "unknown"),
    }
