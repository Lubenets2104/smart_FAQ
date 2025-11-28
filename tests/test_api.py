"""Unit tests for API endpoints."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient


class TestAskEndpoint:
    """Tests for /api/ask endpoint."""
    
    @pytest.mark.asyncio
    async def test_ask_empty_question_returns_error(self, test_client):
        """Test that empty question returns validation error."""
        response = test_client.post("/api/ask", json={"question": ""})
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_ask_missing_question_returns_error(self, test_client):
        """Test that missing question field returns validation error."""
        response = test_client.post("/api/ask", json={})
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_ask_valid_question_structure(self, test_client, mock_services):
        """Test that valid question returns proper response structure."""
        response = test_client.post(
            "/api/ask", 
            json={"question": "Что такое SmartTask?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "answer" in data
        assert "sources" in data
        assert "tokens_used" in data
        assert "response_time_ms" in data
        assert "cached" in data
        
        # Check types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["tokens_used"], int)
        assert isinstance(data["response_time_ms"], int)
        assert isinstance(data["cached"], bool)


class TestHealthEndpoint:
    """Tests for /api/health endpoint."""
    
    def test_health_returns_status(self, test_client):
        """Test that health endpoint returns status information."""
        response = test_client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "postgres" in data
        assert "redis" in data
        assert "chromadb" in data


class TestDocumentsEndpoint:
    """Tests for /api/documents endpoint."""
    
    def test_upload_invalid_file_type(self, test_client):
        """Test that uploading invalid file type returns error."""
        response = test_client.post(
            "/api/documents",
            files={"file": ("test.pdf", b"content", "application/pdf")}
        )
        
        assert response.status_code == 400
        assert "txt" in response.json()["detail"].lower() or "md" in response.json()["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_upload_valid_file(self, test_client, mock_services):
        """Test that uploading valid file succeeds."""
        content = "Test document content for SmartTask FAQ."
        
        response = test_client.post(
            "/api/documents",
            files={"file": ("test.txt", content.encode(), "text/plain")}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["filename"] == "test.txt"
        assert "chunks_created" in data
        assert isinstance(data["chunks_created"], int)
