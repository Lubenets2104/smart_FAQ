"""Unit tests for RAG service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestRAGService:
    """Tests for RAG service functionality."""
    
    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        from app.services.rag_service import RAGService
        
        service = RAGService()
        text = "This is a test. " * 100  # ~1600 characters
        
        chunks = service._chunk_text(text, chunk_size=500, overlap=50)
        
        assert len(chunks) > 1
        assert all(len(c) <= 550 for c in chunks)  # Allow some margin
    
    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        from app.services.rag_service import RAGService
        
        service = RAGService()
        chunks = service._chunk_text("")
        
        assert chunks == []
    
    def test_chunk_text_small(self):
        """Test chunking text smaller than chunk size."""
        from app.services.rag_service import RAGService
        
        service = RAGService()
        text = "Short text."
        
        chunks = service._chunk_text(text, chunk_size=500, overlap=50)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_text_preserves_sentence_boundaries(self):
        """Test that chunking tries to preserve sentence boundaries."""
        from app.services.rag_service import RAGService
        
        service = RAGService()
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        
        chunks = service._chunk_text(text, chunk_size=40, overlap=5)
        
        # Check that chunks tend to end with periods
        for chunk in chunks[:-1]:  # Exclude last chunk
            stripped = chunk.strip()
            if len(stripped) > 20:  # Only check if chunk is substantial
                assert stripped.endswith('.') or stripped.endswith('\n')
    
    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        """Test that search returns properly formatted results."""
        from app.services.rag_service import RAGService
        
        service = RAGService()
        
        # Mock the ChromaDB collection
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Test chunk 1", "Test chunk 2"]],
            "metadatas": [[{"source": "doc1.txt"}, {"source": "doc2.txt"}]],
            "distances": [[0.1, 0.2]],
        }
        
        service._collection = mock_collection
        service._client = MagicMock()
        
        results = await service.search("test query", top_k=2)
        
        assert len(results) == 2
        assert results[0][0] == "doc1.txt"  # source
        assert results[0][1] == "Test chunk 1"  # chunk
        assert isinstance(results[0][2], float)  # score
    
    @pytest.mark.asyncio
    async def test_get_context_formats_correctly(self):
        """Test that context is properly formatted."""
        from app.services.rag_service import RAGService
        
        service = RAGService()
        
        # Mock search to return results
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Chunk content here"]],
            "metadatas": [[{"source": "test.txt"}]],
            "distances": [[0.1]],
        }
        
        service._collection = mock_collection
        service._client = MagicMock()
        
        context = await service.get_context("test query")
        
        assert "test.txt" in context
        assert "Chunk content here" in context
        assert "[Источник:" in context


class TestCacheService:
    """Tests for cache service functionality."""
    
    @pytest.mark.asyncio
    async def test_hash_question_consistency(self):
        """Test that same question produces same hash."""
        from app.services.cache_service import CacheService
        
        service = CacheService()
        
        hash1 = service._hash_question("What is SmartTask?")
        hash2 = service._hash_question("What is SmartTask?")
        hash3 = service._hash_question("what is smarttask?")  # Different case
        
        assert hash1 == hash2
        assert hash1 == hash3  # Should be case-insensitive
    
    @pytest.mark.asyncio
    async def test_hash_different_questions(self):
        """Test that different questions produce different hashes."""
        from app.services.cache_service import CacheService
        
        service = CacheService()
        
        hash1 = service._hash_question("What is SmartTask?")
        hash2 = service._hash_question("How to create a task?")
        
        assert hash1 != hash2
    
    @pytest.mark.asyncio
    async def test_cache_miss_returns_none(self):
        """Test that cache miss returns None."""
        from app.services.cache_service import CacheService
        
        service = CacheService()
        
        # Mock Redis client
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        service._client = mock_redis
        
        result = await service.get_cached_answer("Unknown question")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_hit_returns_data(self):
        """Test that cache hit returns cached data."""
        import json
        from app.services.cache_service import CacheService
        
        service = CacheService()
        
        cached_data = {
            "answer": "SmartTask is a project management tool.",
            "sources": [{"document": "test.txt", "chunk": "Test content"}],
            "tokens_used": 100
        }
        
        # Mock Redis client
        mock_redis = AsyncMock()
        mock_redis.get.return_value = json.dumps(cached_data)
        service._client = mock_redis
        
        result = await service.get_cached_answer("What is SmartTask?")
        
        assert result == cached_data
        assert result["answer"] == "SmartTask is a project management tool."
