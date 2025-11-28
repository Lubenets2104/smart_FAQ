"""Pydantic models for API requests and responses."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from uuid import UUID


class AskRequest(BaseModel):
    """Request model for /api/ask endpoint."""
    question: str = Field(..., min_length=1, max_length=1000, description="User question")


class SourceInfo(BaseModel):
    """Information about a source document."""
    document: str = Field(..., description="Document name")
    chunk: str = Field(..., description="Relevant text chunk")


class AskResponse(BaseModel):
    """Response model for /api/ask endpoint."""
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceInfo] = Field(default_factory=list, description="Source documents used")
    tokens_used: int = Field(..., description="Number of tokens used")
    response_time_ms: int = Field(..., description="Response time in milliseconds")
    cached: bool = Field(default=False, description="Whether response was from cache")


class DocumentUploadResponse(BaseModel):
    """Response model for /api/documents endpoint."""
    message: str = Field(..., description="Status message")
    filename: str = Field(..., description="Uploaded filename")
    chunks_created: int = Field(..., description="Number of chunks created")


class HealthCheckResponse(BaseModel):
    """Response model for /api/health endpoint."""
    status: str = Field(..., description="Overall status")
    postgres: str = Field(..., description="PostgreSQL status")
    redis: str = Field(..., description="Redis status")
    chromadb: str = Field(..., description="ChromaDB status")


class QueryHistoryItem(BaseModel):
    """Model for query history item."""
    id: UUID
    question: str
    answer: str
    tokens_used: int
    response_time_ms: int
    sources: List[dict]
    created_at: datetime

    class Config:
        from_attributes = True


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
