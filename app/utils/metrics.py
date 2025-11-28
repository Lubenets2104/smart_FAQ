"""Prometheus metrics for the SmartTask FAQ service."""

from prometheus_client import Counter, Histogram, Gauge, Info

# Application info
APP_INFO = Info(
    "smarttask_faq_app",
    "SmartTask FAQ application information"
)

# Request metrics
REQUEST_COUNT = Counter(
    "smarttask_faq_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "smarttask_faq_request_latency_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Cache metrics
CACHE_HITS = Counter(
    "smarttask_faq_cache_hits_total",
    "Total number of cache hits"
)

CACHE_MISSES = Counter(
    "smarttask_faq_cache_misses_total",
    "Total number of cache misses"
)

# LLM metrics
LLM_TOKENS_USED = Counter(
    "smarttask_faq_llm_tokens_used_total",
    "Total number of LLM tokens used",
    ["provider"]
)

LLM_REQUESTS = Counter(
    "smarttask_faq_llm_requests_total",
    "Total number of LLM requests",
    ["provider", "status"]
)

LLM_LATENCY = Histogram(
    "smarttask_faq_llm_latency_seconds",
    "LLM response latency in seconds",
    ["provider"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

# RAG metrics
RAG_SEARCH_COUNT = Counter(
    "smarttask_faq_rag_searches_total",
    "Total number of RAG searches",
    ["status"]
)

RAG_DOCUMENTS_INDEXED = Gauge(
    "smarttask_faq_rag_documents_indexed",
    "Number of documents indexed in the vector store"
)

# Document upload metrics
DOCUMENT_UPLOADS = Counter(
    "smarttask_faq_document_uploads_total",
    "Total number of document uploads",
    ["status"]
)

# Service health metrics
SERVICE_UP = Gauge(
    "smarttask_faq_service_up",
    "Service availability (1 = up, 0 = down)",
    ["service"]
)


def init_app_info(version: str = "1.0.0") -> None:
    """Initialize application info metric."""
    APP_INFO.info({
        "version": version,
        "service": "smarttask-faq"
    })


def record_request(method: str, endpoint: str, status: int, duration: float) -> None:
    """Record a request metric."""
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)


def record_cache_hit() -> None:
    """Record a cache hit."""
    CACHE_HITS.inc()


def record_cache_miss() -> None:
    """Record a cache miss."""
    CACHE_MISSES.inc()


def record_llm_usage(provider: str, tokens: int, duration: float, success: bool = True) -> None:
    """Record LLM usage metrics."""
    LLM_TOKENS_USED.labels(provider=provider).inc(tokens)
    LLM_REQUESTS.labels(provider=provider, status="success" if success else "error").inc()
    LLM_LATENCY.labels(provider=provider).observe(duration)


def record_rag_search(success: bool = True) -> None:
    """Record a RAG search."""
    RAG_SEARCH_COUNT.labels(status="success" if success else "error").inc()


def update_documents_indexed(count: int) -> None:
    """Update the number of indexed documents."""
    RAG_DOCUMENTS_INDEXED.set(count)


def record_document_upload(success: bool = True) -> None:
    """Record a document upload."""
    DOCUMENT_UPLOADS.labels(status="success" if success else "error").inc()


def update_service_health(service: str, is_up: bool) -> None:
    """Update service health status."""
    SERVICE_UP.labels(service=service).set(1 if is_up else 0)
