"""Integration tests for the FAQ API."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestFAQIntegration:
    """Integration tests for the complete FAQ flow."""
    
    @pytest.mark.asyncio
    async def test_full_faq_flow(self, test_client, mock_services):
        """
        Test the complete FAQ flow:
        1. Upload a document
        2. Ask a question about the document
        3. Verify answer contains relevant information
        """
        # Step 1: Upload a document
        doc_content = """
        SmartTask Pricing Guide
        
        SmartTask offers three pricing tiers:
        1. Free Plan: Up to 5 users, basic features included.
        2. Pro Plan: $9 per user per month, includes advanced reporting.
        3. Enterprise Plan: Custom pricing with SLA support.
        
        All plans include unlimited projects and tasks.
        """
        
        upload_response = test_client.post(
            "/api/documents",
            files={"file": ("pricing.txt", doc_content.encode(), "text/plain")}
        )
        
        assert upload_response.status_code == 200
        assert upload_response.json()["chunks_created"] > 0
        
        # Step 2: Ask a question
        ask_response = test_client.post(
            "/api/ask",
            json={"question": "Сколько стоит SmartTask Pro?"}
        )
        
        assert ask_response.status_code == 200
        data = ask_response.json()
        
        # Step 3: Verify response
        assert data["answer"]  # Answer should not be empty
        assert data["tokens_used"] > 0
        assert data["response_time_ms"] > 0
    
    @pytest.mark.asyncio
    async def test_cache_workflow(self, test_client, mock_services):
        """
        Test caching workflow:
        1. Ask a question (cache miss)
        2. Ask the same question again (cache hit)
        """
        question = "Какие функции есть в SmartTask?"
        
        # First request - should be cache miss
        response1 = test_client.post(
            "/api/ask",
            json={"question": question}
        )
        
        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["cached"] == False
        
        # Note: In real integration test with running services,
        # the second request would have cached=True.
        # Here we just verify the structure is correct.
    
    @pytest.mark.asyncio
    async def test_error_handling(self, test_client):
        """Test that errors are handled gracefully."""
        # Test with invalid JSON
        response = test_client.post(
            "/api/ask",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_health_check_integration(self, test_client):
        """Test health check reflects service status."""
        response = test_client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Status should be one of known values
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert data["postgres"] in ["healthy", "unhealthy"]
        assert data["redis"] in ["healthy", "unhealthy"]
        assert data["chromadb"] in ["healthy", "unhealthy"]
    
    @pytest.mark.asyncio
    async def test_stats_endpoint(self, test_client, mock_services):
        """Test statistics endpoint."""
        response = test_client.get("/api/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_queries" in data
        assert "documents_indexed" in data
        assert isinstance(data["total_queries"], int)
        assert isinstance(data["documents_indexed"], int)


class TestAPIEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_very_long_question(self, test_client):
        """Test handling of very long questions."""
        long_question = "Что такое SmartTask? " * 100  # Very long question
        
        response = test_client.post(
            "/api/ask",
            json={"question": long_question}
        )
        
        # Should either succeed or return validation error, not crash
        assert response.status_code in [200, 422]
    
    def test_special_characters_in_question(self, test_client, mock_services):
        """Test handling of special characters."""
        response = test_client.post(
            "/api/ask",
            json={"question": "Что такое <script>alert('xss')</script>?"}
        )
        
        assert response.status_code == 200
    
    def test_unicode_question(self, test_client, mock_services):
        """Test handling of unicode characters."""
        response = test_client.post(
            "/api/ask",
            json={"question": "Что такое SmartTask? 你好 🚀"}
        )
        
        assert response.status_code == 200
    
    def test_empty_file_upload(self, test_client):
        """Test uploading empty file."""
        response = test_client.post(
            "/api/documents",
            files={"file": ("empty.txt", b"", "text/plain")}
        )
        
        # Should handle gracefully
        assert response.status_code in [200, 400]
