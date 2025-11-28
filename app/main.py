"""Main FastAPI application module."""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.api import router
from app.config import get_settings
from app.db import init_db
from app.services import cache_service, rag_service
from app.utils import setup_logging, get_logger, init_app_info

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)

    # Initialize metrics
    init_app_info(version="1.0.0")

    logger.info("Starting SmartTask FAQ Service...")
    
    # Initialize database
    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
    
    # Connect to Redis
    try:
        await cache_service.connect()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.warning("Failed to connect to Redis", error=str(e))
    
    # Connect to ChromaDB and load documents
    try:
        await rag_service.connect()
        logger.info("Connected to ChromaDB")
        
        # Load documents from directory
        docs_dir = os.path.join(os.path.dirname(__file__), "..", "documents")
        if os.path.exists(docs_dir):
            chunks = await rag_service.load_documents_from_directory(docs_dir)
            logger.info("Loaded documents", chunks=chunks)
    except Exception as e:
        logger.warning("Failed to initialize RAG service", error=str(e))
    
    logger.info("SmartTask FAQ Service started successfully")
    
    yield
    
    # Cleanup
    logger.info("Shutting down SmartTask FAQ Service...")
    await cache_service.disconnect()


# Create FastAPI application
app = FastAPI(
    title="SmartTask FAQ API",
    description="Умный FAQ сервис для SmartTask с RAG",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api", tags=["FAQ"])

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
async def root():
    """Serve the main page."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "SmartTask FAQ API", "docs": "/docs"}


@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Expose Prometheus metrics."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
    )
