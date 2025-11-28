"""Utils module."""

from app.utils.logging import setup_logging, get_logger
from app.utils.metrics import (
    init_app_info,
    record_request,
    record_cache_hit,
    record_cache_miss,
    record_llm_usage,
    record_rag_search,
    update_documents_indexed,
    record_document_upload,
    update_service_health,
)

__all__ = [
    "setup_logging",
    "get_logger",
    "init_app_info",
    "record_request",
    "record_cache_hit",
    "record_cache_miss",
    "record_llm_usage",
    "record_rag_search",
    "update_documents_indexed",
    "record_document_upload",
    "update_service_health",
]
