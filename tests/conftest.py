"""Pytest configuration and fixtures."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def mock_services():
    """Mock all external services for testing."""
    with patch('app.services.cache_service.cache_service') as mock_cache, \
         patch('app.services.rag_service.rag_service') as mock_rag, \
         patch('app.services.llm_service.llm_service') as mock_llm, \
         patch('app.db.database.check_db_connection') as mock_db_check:
        
        # Mock cache service
        mock_cache.get_cached_answer = AsyncMock(return_value=None)
        mock_cache.set_cached_answer = AsyncMock(return_value=True)
        mock_cache.check_connection = AsyncMock(return_value=True)
        
        # Mock RAG service
        mock_rag.search = AsyncMock(return_value=[
            ("test_doc.txt", "Test chunk content", 0.9)
        ])
        mock_rag.get_context = AsyncMock(return_value="[Источник: test_doc.txt]\nTest chunk content")
        mock_rag.add_document = AsyncMock(return_value=3)
        mock_rag.check_connection = AsyncMock(return_value=True)
        mock_rag.get_collection_stats = AsyncMock(return_value={"document_count": 10})
        
        # Mock LLM service
        mock_llm.generate_answer = AsyncMock(return_value=(
            "SmartTask - это облачная платформа для управления проектами.",
            150,
            500
        ))
        
        # Mock DB check
        mock_db_check.return_value = True
        
        yield {
            "cache": mock_cache,
            "rag": mock_rag,
            "llm": mock_llm,
            "db_check": mock_db_check,
        }


@pytest.fixture
def mock_db_session():
    """Mock database session."""
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def test_client(mock_services, mock_db_session):
    """Create a test client with mocked dependencies."""
    # Patch the database session
    with patch('app.api.routes.get_db') as mock_get_db:
        async def get_mock_db():
            yield mock_db_session
        
        mock_get_db.return_value = get_mock_db()
        
        # Import app after patching
        from app.main import app
        
        with TestClient(app) as client:
            yield client


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "filename": "overview.txt",
            "content": """SmartTask Overview
            
            SmartTask is a cloud-based project management platform.
            It helps teams plan, track, and complete projects faster.
            
            Key features include:
            - Task boards (Kanban, Gantt)
            - Time management and deadlines
            - Collaboration and comments
            - Integrations (Slack, Google Drive, GitHub)
            - Progress and performance reports
            """
        },
        {
            "filename": "pricing.txt",
            "content": """SmartTask Pricing
            
            1. Free - up to 5 users, basic features
            2. Pro - $9/user/month, advanced reports and integrations
            3. Enterprise - custom pricing, SLA support and SSO
            """
        }
    ]


@pytest.fixture
def sample_questions():
    """Sample questions for testing."""
    return [
        {
            "question": "Что такое SmartTask?",
            "expected_keywords": ["платформа", "проект", "задач"]
        },
        {
            "question": "Сколько стоит Pro план?",
            "expected_keywords": ["$9", "месяц", "пользователь"]
        },
        {
            "question": "Какие интеграции поддерживаются?",
            "expected_keywords": ["Slack", "Google Drive", "GitHub"]
        }
    ]


# Configure pytest-asyncio
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
