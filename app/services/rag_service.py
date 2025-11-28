"""RAG (Retrieval-Augmented Generation) service using ChromaDB."""

import asyncio
import os
import time
from typing import List, Optional, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import get_settings
from app.utils import get_logger

logger = get_logger(__name__)
settings = get_settings()


class ChromaDBUnavailableError(Exception):
    """Raised when ChromaDB is unavailable and operation cannot proceed."""
    pass


class RAGService:
    """Service for RAG operations with ChromaDB vector store."""

    COLLECTION_NAME = "smarttask_docs"
    MAX_RECONNECT_ATTEMPTS = 3
    RECONNECT_BASE_DELAY = 1.0  # seconds

    def __init__(self):
        self._client: Optional[chromadb.HttpClient] = None
        self._collection = None
        self._available: bool = False
        self._last_error: Optional[str] = None
        self._last_error_time: Optional[float] = None
        self._reconnect_attempts: int = 0

    @property
    def is_available(self) -> bool:
        """Check if ChromaDB is currently available."""
        return self._available and self._client is not None

    @property
    def last_error(self) -> Optional[str]:
        """Get the last error message if any."""
        return self._last_error
    
    async def connect(self) -> None:
        """Connect to ChromaDB."""
        if self._client is not None and self._available:
            return

        try:
            self._client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            self._available = True
            self._last_error = None
            self._reconnect_attempts = 0
            logger.info(
                "Connected to ChromaDB",
                host=settings.chroma_host,
                port=settings.chroma_port,
                collection=self.COLLECTION_NAME
            )
        except Exception as e:
            self._available = False
            self._last_error = str(e)
            self._last_error_time = time.time()
            self._client = None
            self._collection = None
            logger.error("Failed to connect to ChromaDB", error=str(e))
            raise

    async def _try_reconnect(self) -> bool:
        """
        Attempt to reconnect to ChromaDB with exponential backoff.

        Returns:
            True if reconnection successful, False otherwise
        """
        if self._reconnect_attempts >= self.MAX_RECONNECT_ATTEMPTS:
            logger.warning(
                "Max reconnection attempts reached",
                attempts=self._reconnect_attempts
            )
            return False

        delay = self.RECONNECT_BASE_DELAY * (2 ** self._reconnect_attempts)
        self._reconnect_attempts += 1

        logger.info(
            "Attempting ChromaDB reconnection",
            attempt=self._reconnect_attempts,
            delay=delay
        )

        await asyncio.sleep(delay)

        try:
            await self.connect()
            logger.info("ChromaDB reconnection successful")
            return True
        except Exception as e:
            logger.warning(
                "ChromaDB reconnection failed",
                attempt=self._reconnect_attempts,
                error=str(e)
            )
            return False

    async def _ensure_connection(self) -> bool:
        """
        Ensure connection is available, attempting reconnect if needed.

        Returns:
            True if connection is available, False otherwise
        """
        if self._available and self._client is not None:
            return True

        # Try to reconnect
        return await self._try_reconnect()
    
    def _chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to split
            chunk_size: Size of each chunk (default from settings)
            overlap: Overlap between chunks (default from settings)
            
        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or settings.chunk_size
        overlap = overlap or settings.chunk_overlap
        
        chunks = []
        start = 0
        text = text.strip()
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                if break_point > chunk_size // 2:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return [c for c in chunks if c]  # Filter empty chunks
    
    async def add_document(self, filename: str, content: str) -> int:
        """
        Add a document to the vector store.

        Args:
            filename: Name of the document
            content: Document content

        Returns:
            Number of chunks created

        Raises:
            ChromaDBUnavailableError: If ChromaDB is unavailable after reconnection attempts
        """
        if not self._available or not self._client:
            if not await self._ensure_connection():
                raise ChromaDBUnavailableError(
                    f"ChromaDB is unavailable. Last error: {self._last_error}"
                )

        chunks = self._chunk_text(content)

        if not chunks:
            logger.warning("No chunks created from document", filename=filename)
            return 0

        ids = [f"{filename}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

        try:
            # Delete existing chunks for this document
            existing = self._collection.get(where={"source": filename})
            if existing["ids"]:
                self._collection.delete(ids=existing["ids"])
                logger.info("Deleted existing chunks", filename=filename, count=len(existing["ids"]))

            # Add new chunks
            self._collection.add(
                documents=chunks,
                ids=ids,
                metadatas=metadatas,
            )

            logger.info("Added document to vector store", filename=filename, chunks=len(chunks))
            return len(chunks)

        except Exception as e:
            # Mark as unavailable on connection-related errors
            self._available = False
            self._last_error = str(e)
            self._last_error_time = time.time()
            logger.error("Error adding document", filename=filename, error=str(e))
            raise ChromaDBUnavailableError(f"Failed to add document: {e}")
    
    async def search(self, query: str, top_k: int = None) -> List[Tuple[str, str, float]]:
        """
        Search for relevant document chunks.

        This method implements graceful degradation - if ChromaDB is unavailable,
        it returns an empty list instead of raising an exception, allowing the
        application to continue functioning (albeit without RAG context).

        Args:
            query: Search query
            top_k: Number of results to return (default from settings)

        Returns:
            List of tuples (document_name, chunk_text, score), empty if unavailable
        """
        if not self._available or not self._client:
            if not await self._ensure_connection():
                logger.warning(
                    "ChromaDB unavailable for search, returning empty results",
                    query=query[:50]
                )
                return []

        top_k = top_k or settings.top_k_results

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
            )

            if not results["documents"] or not results["documents"][0]:
                logger.debug("No search results found", query=query[:50])
                return []

            output = []
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                # ChromaDB returns distances, convert to similarity score
                score = 1 - distance if distance else 0
                output.append((metadata["source"], doc, score))

            logger.info("Search completed", query=query[:50], results=len(output))
            return output

        except Exception as e:
            # Mark as unavailable and return empty results for graceful degradation
            self._available = False
            self._last_error = str(e)
            self._last_error_time = time.time()
            logger.error(
                "Error searching ChromaDB, returning empty results",
                query=query[:50],
                error=str(e)
            )
            return []
    
    async def get_context(self, query: str, top_k: int = None) -> str:
        """
        Get formatted context from relevant documents for LLM.
        
        Args:
            query: User's question
            top_k: Number of chunks to include
            
        Returns:
            Formatted context string
        """
        results = await self.search(query, top_k)
        
        if not results:
            return ""
        
        context_parts = []
        for source, chunk, score in results:
            context_parts.append(f"[Источник: {source}]\n{chunk}")
        
        return "\n\n---\n\n".join(context_parts)
    
    async def load_documents_from_directory(self, directory: str) -> int:
        """
        Load all documents from a directory.
        
        Args:
            directory: Path to directory with documents
            
        Returns:
            Total number of chunks created
        """
        total_chunks = 0
        
        if not os.path.exists(directory):
            logger.warning("Documents directory not found", directory=directory)
            return 0
        
        for filename in os.listdir(directory):
            if filename.endswith(('.txt', '.md')):
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    chunks = await self.add_document(filename, content)
                    total_chunks += chunks
                except Exception as e:
                    logger.error("Error loading document", filename=filename, error=str(e))
        
        logger.info("Loaded documents from directory", directory=directory, total_chunks=total_chunks)
        return total_chunks
    
    async def check_connection(self) -> bool:
        """
        Check if ChromaDB connection is working.

        This also serves as a health check and will update internal availability state.

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            if not self._client:
                await self.connect()
            self._client.heartbeat()
            self._available = True
            self._reconnect_attempts = 0  # Reset on successful health check
            return True
        except Exception as e:
            self._available = False
            self._last_error = str(e)
            self._last_error_time = time.time()
            logger.error("ChromaDB connection check failed", error=str(e))
            return False

    async def get_collection_stats(self) -> dict:
        """
        Get statistics about the collection.

        Returns stats with availability info even when ChromaDB is unavailable.

        Returns:
            Dictionary with collection stats and availability status
        """
        base_stats = {
            "collection": self.COLLECTION_NAME,
            "available": self._available,
            "last_error": self._last_error,
        }

        if not self._available or not self._client:
            if not await self._ensure_connection():
                return {
                    **base_stats,
                    "document_count": 0,
                    "available": False,
                }

        try:
            count = self._collection.count()
            return {
                **base_stats,
                "document_count": count,
                "available": True,
            }
        except Exception as e:
            self._available = False
            self._last_error = str(e)
            self._last_error_time = time.time()
            logger.error("Error getting collection stats", error=str(e))
            return {
                **base_stats,
                "document_count": 0,
                "available": False,
                "last_error": str(e),
            }

    def reset_reconnect_counter(self) -> None:
        """Reset the reconnection attempt counter to allow fresh reconnection attempts."""
        self._reconnect_attempts = 0
        logger.info("Reconnect counter reset")


# Global RAG service instance
rag_service = RAGService()
