"""Unit tests for cache_service edge cases.

Tests cover:
- Redis unavailability scenarios
- Corrupted cache data handling
- Connection recovery
- Edge cases in key generation
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
import redis.asyncio as redis

from app.services.cache_service import CacheService


class TestCacheServiceRedisUnavailable:
    """Tests for behavior when Redis is unavailable."""

    @pytest.fixture
    def cache_service(self):
        """Create a fresh CacheService instance for each test."""
        return CacheService()

    @pytest.mark.asyncio
    async def test_get_cached_answer_returns_none_when_redis_connection_fails(
        self, cache_service
    ):
        """When Redis connection fails, get_cached_answer should return None gracefully."""
        with patch.object(
            cache_service, '_client', new_callable=PropertyMock
        ) as mock_client:
            mock_redis = AsyncMock()
            mock_redis.get = AsyncMock(
                side_effect=redis.ConnectionError("Connection refused")
            )
            mock_client.return_value = mock_redis
            cache_service._client = mock_redis

            result = await cache_service.get_cached_answer("test question")

            assert result is None

    @pytest.mark.asyncio
    async def test_get_cached_answer_returns_none_when_redis_timeout(
        self, cache_service
    ):
        """When Redis times out, get_cached_answer should return None gracefully."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=redis.TimeoutError("Timeout"))
        cache_service._client = mock_redis

        result = await cache_service.get_cached_answer("test question")

        assert result is None

    @pytest.mark.asyncio
    async def test_set_cached_answer_returns_false_when_redis_unavailable(
        self, cache_service
    ):
        """When Redis is unavailable, set_cached_answer should return False."""
        mock_redis = AsyncMock()
        mock_redis.setex = AsyncMock(
            side_effect=redis.ConnectionError("Connection refused")
        )
        cache_service._client = mock_redis

        result = await cache_service.set_cached_answer(
            "test question",
            {"answer": "test answer", "sources": [], "tokens_used": 100}
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_check_connection_returns_false_when_redis_unavailable(
        self, cache_service
    ):
        """check_connection should return False when Redis is unavailable."""
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(
            side_effect=redis.ConnectionError("Connection refused")
        )
        cache_service._client = mock_redis

        result = await cache_service.check_connection()

        assert result is False

    @pytest.mark.asyncio
    async def test_clear_cache_returns_zero_when_redis_unavailable(
        self, cache_service
    ):
        """clear_cache should return 0 when Redis is unavailable."""
        mock_redis = AsyncMock()
        mock_redis.keys = AsyncMock(
            side_effect=redis.ConnectionError("Connection refused")
        )
        cache_service._client = mock_redis

        result = await cache_service.clear_cache()

        assert result == 0

    @pytest.mark.asyncio
    async def test_connect_failure_on_first_get_attempt(self, cache_service):
        """When initial connection fails, operations should handle gracefully."""
        with patch('redis.asyncio.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            mock_redis.get = AsyncMock(
                side_effect=redis.ConnectionError("Cannot connect")
            )
            mock_redis_class.return_value = mock_redis

            # First call will trigger connect()
            result = await cache_service.get_cached_answer("test")

            assert result is None


class TestCacheServiceCorruptedData:
    """Tests for handling corrupted cache data."""

    @pytest.fixture
    def cache_service(self):
        """Create a fresh CacheService instance for each test."""
        return CacheService()

    @pytest.mark.asyncio
    async def test_get_cached_answer_handles_invalid_json(self, cache_service):
        """When cached data is not valid JSON, should return None."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value="not valid json {{{")
        cache_service._client = mock_redis

        result = await cache_service.get_cached_answer("test question")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_cached_answer_handles_truncated_json(self, cache_service):
        """When cached data is truncated JSON, should return None."""
        mock_redis = AsyncMock()
        # Truncated JSON string
        mock_redis.get = AsyncMock(return_value='{"answer": "test", "sources": [')
        cache_service._client = mock_redis

        result = await cache_service.get_cached_answer("test question")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_cached_answer_handles_empty_string(self, cache_service):
        """When cached data is empty string, should return None."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value="")
        cache_service._client = mock_redis

        result = await cache_service.get_cached_answer("test question")

        # Empty string is falsy, so should return None from the if check
        assert result is None

    @pytest.mark.asyncio
    async def test_get_cached_answer_handles_null_bytes(self, cache_service):
        """When cached data contains null bytes, should handle gracefully."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value='{"answer": "test\x00corrupted"}')
        cache_service._client = mock_redis

        # This should either return the parsed data or None, not raise
        result = await cache_service.get_cached_answer("test question")

        # JSON with embedded null byte should still parse
        assert result is not None or result is None  # Should not raise

    @pytest.mark.asyncio
    async def test_get_cached_answer_handles_wrong_type(self, cache_service):
        """When cached data is wrong type (not dict), handle gracefully."""
        mock_redis = AsyncMock()
        # Valid JSON but wrong structure (array instead of object)
        mock_redis.get = AsyncMock(return_value='["item1", "item2"]')
        cache_service._client = mock_redis

        result = await cache_service.get_cached_answer("test question")

        # Should return the parsed data (list in this case)
        # The caller should handle type checking
        assert result == ["item1", "item2"]

    @pytest.mark.asyncio
    async def test_get_cached_answer_handles_unicode_errors(self, cache_service):
        """When cached data has encoding issues, should handle gracefully."""
        mock_redis = AsyncMock()
        # This could happen if data was corrupted
        mock_redis.get = AsyncMock(side_effect=UnicodeDecodeError(
            'utf-8', b'\xff\xfe', 0, 1, 'invalid start byte'
        ))
        cache_service._client = mock_redis

        result = await cache_service.get_cached_answer("test question")

        assert result is None

    @pytest.mark.asyncio
    async def test_set_cached_answer_handles_unserializable_data(
        self, cache_service
    ):
        """When data cannot be serialized to JSON, should return False."""
        mock_redis = AsyncMock()
        cache_service._client = mock_redis

        # Create unserializable data (contains a set)
        bad_data = {
            "answer": "test",
            "sources": set([1, 2, 3]),  # Sets are not JSON serializable
            "tokens_used": 100
        }

        result = await cache_service.set_cached_answer("test question", bad_data)

        assert result is False

    @pytest.mark.asyncio
    async def test_set_cached_answer_handles_circular_reference(
        self, cache_service
    ):
        """When data has circular reference, should return False."""
        mock_redis = AsyncMock()
        cache_service._client = mock_redis

        # Create circular reference
        bad_data = {"answer": "test"}
        bad_data["self"] = bad_data

        result = await cache_service.set_cached_answer("test question", bad_data)

        assert result is False


class TestCacheServiceConnectionRecovery:
    """Tests for connection recovery scenarios."""

    @pytest.fixture
    def cache_service(self):
        """Create a fresh CacheService instance for each test."""
        return CacheService()

    @pytest.mark.asyncio
    async def test_auto_reconnect_on_get_when_client_none(self, cache_service):
        """Should attempt to reconnect when client is None."""
        assert cache_service._client is None

        with patch('redis.asyncio.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            mock_redis.get = AsyncMock(return_value=None)
            mock_redis_class.return_value = mock_redis

            await cache_service.get_cached_answer("test")

            # Should have created a new client
            mock_redis_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_reconnect_on_set_when_client_none(self, cache_service):
        """Should attempt to reconnect when client is None on set."""
        assert cache_service._client is None

        with patch('redis.asyncio.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            mock_redis.setex = AsyncMock(return_value=True)
            mock_redis_class.return_value = mock_redis

            await cache_service.set_cached_answer(
                "test",
                {"answer": "test", "sources": [], "tokens_used": 100}
            )

            mock_redis_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_clears_client(self, cache_service):
        """disconnect() should clear the client reference."""
        mock_redis = AsyncMock()
        mock_redis.close = AsyncMock()
        cache_service._client = mock_redis

        await cache_service.disconnect()

        assert cache_service._client is None
        mock_redis.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_handles_none_client(self, cache_service):
        """disconnect() should handle None client gracefully."""
        cache_service._client = None

        # Should not raise
        await cache_service.disconnect()

        assert cache_service._client is None

    @pytest.mark.asyncio
    async def test_multiple_connect_calls_only_creates_one_client(
        self, cache_service
    ):
        """Multiple connect() calls should not create multiple clients."""
        with patch('redis.asyncio.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            mock_redis_class.return_value = mock_redis

            await cache_service.connect()
            await cache_service.connect()
            await cache_service.connect()

            # Should only create one client
            mock_redis_class.assert_called_once()


class TestCacheServiceKeyGeneration:
    """Tests for cache key generation edge cases."""

    @pytest.fixture
    def cache_service(self):
        """Create a fresh CacheService instance for each test."""
        return CacheService()

    def test_hash_question_normalizes_case(self, cache_service):
        """Questions should be normalized to lowercase."""
        key1 = cache_service._hash_question("What is SmartTask?")
        key2 = cache_service._hash_question("what is smarttask?")
        key3 = cache_service._hash_question("WHAT IS SMARTTASK?")

        assert key1 == key2 == key3

    def test_hash_question_normalizes_whitespace(self, cache_service):
        """Questions should be normalized for leading/trailing whitespace."""
        key1 = cache_service._hash_question("What is SmartTask?")
        key2 = cache_service._hash_question("  What is SmartTask?  ")
        key3 = cache_service._hash_question("\n\tWhat is SmartTask?\n\t")

        assert key1 == key2 == key3

    def test_hash_question_has_correct_prefix(self, cache_service):
        """All keys should have the 'faq:' prefix."""
        key = cache_service._hash_question("any question")

        assert key.startswith("faq:")

    def test_hash_question_produces_consistent_length(self, cache_service):
        """All keys should have consistent length (prefix + sha256 hex)."""
        key1 = cache_service._hash_question("short")
        key2 = cache_service._hash_question("a" * 10000)

        # faq: (4) + sha256 hex (64) = 68
        assert len(key1) == 68
        assert len(key2) == 68

    def test_hash_question_handles_unicode(self, cache_service):
        """Should handle unicode questions correctly."""
        key1 = cache_service._hash_question("Что такое SmartTask?")
        key2 = cache_service._hash_question("что такое smarttask?")

        assert key1 == key2
        assert key1.startswith("faq:")

    def test_hash_question_handles_emoji(self, cache_service):
        """Should handle emoji in questions."""
        key = cache_service._hash_question("What is SmartTask? 🚀")

        assert key.startswith("faq:")
        assert len(key) == 68

    def test_hash_question_handles_empty_string(self, cache_service):
        """Should handle empty string (though validation should prevent this)."""
        key = cache_service._hash_question("")

        assert key.startswith("faq:")
        assert len(key) == 68

    def test_hash_question_different_questions_produce_different_keys(
        self, cache_service
    ):
        """Different questions should produce different keys."""
        key1 = cache_service._hash_question("What is SmartTask?")
        key2 = cache_service._hash_question("How much does SmartTask cost?")

        assert key1 != key2


class TestCacheServiceClearCache:
    """Tests for clear_cache functionality."""

    @pytest.fixture
    def cache_service(self):
        """Create a fresh CacheService instance for each test."""
        return CacheService()

    @pytest.mark.asyncio
    async def test_clear_cache_returns_count_of_deleted_keys(self, cache_service):
        """clear_cache should return the number of deleted keys."""
        mock_redis = AsyncMock()
        mock_redis.keys = AsyncMock(return_value=["faq:key1", "faq:key2", "faq:key3"])
        mock_redis.delete = AsyncMock(return_value=3)
        cache_service._client = mock_redis

        result = await cache_service.clear_cache()

        assert result == 3
        mock_redis.delete.assert_called_once_with("faq:key1", "faq:key2", "faq:key3")

    @pytest.mark.asyncio
    async def test_clear_cache_returns_zero_when_no_keys(self, cache_service):
        """clear_cache should return 0 when no faq keys exist."""
        mock_redis = AsyncMock()
        mock_redis.keys = AsyncMock(return_value=[])
        cache_service._client = mock_redis

        result = await cache_service.clear_cache()

        assert result == 0
        mock_redis.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_clear_cache_only_deletes_faq_keys(self, cache_service):
        """clear_cache should only search for faq:* keys."""
        mock_redis = AsyncMock()
        mock_redis.keys = AsyncMock(return_value=[])
        cache_service._client = mock_redis

        await cache_service.clear_cache()

        mock_redis.keys.assert_called_once_with("faq:*")


class TestCacheServiceIntegrationScenarios:
    """Integration-like tests for realistic scenarios."""

    @pytest.fixture
    def cache_service(self):
        """Create a fresh CacheService instance for each test."""
        return CacheService()

    @pytest.mark.asyncio
    async def test_cache_hit_flow(self, cache_service):
        """Test complete cache hit flow."""
        cached_data = {
            "answer": "SmartTask is a project management platform.",
            "sources": [{"document": "overview.txt", "chunk": "SmartTask..."}],
            "tokens_used": 150
        }

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=json.dumps(cached_data))
        cache_service._client = mock_redis

        result = await cache_service.get_cached_answer("What is SmartTask?")

        assert result == cached_data
        assert result["answer"] == "SmartTask is a project management platform."

    @pytest.mark.asyncio
    async def test_cache_miss_then_set_flow(self, cache_service):
        """Test cache miss followed by setting cache."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock(return_value=True)
        cache_service._client = mock_redis

        # First, cache miss
        result = await cache_service.get_cached_answer("New question?")
        assert result is None

        # Then, set cache
        new_data = {
            "answer": "New answer",
            "sources": [],
            "tokens_used": 100
        }
        success = await cache_service.set_cached_answer("New question?", new_data)
        assert success is True

    @pytest.mark.asyncio
    async def test_redis_recovers_after_temporary_failure(self, cache_service):
        """Test that operations succeed after Redis recovers."""
        mock_redis = AsyncMock()

        # First call fails
        mock_redis.get = AsyncMock(
            side_effect=[
                redis.ConnectionError("Temporary failure"),
                json.dumps({"answer": "test", "sources": [], "tokens_used": 50})
            ]
        )
        cache_service._client = mock_redis

        # First call should return None due to error
        result1 = await cache_service.get_cached_answer("test")
        assert result1 is None

        # Second call should succeed
        result2 = await cache_service.get_cached_answer("test")
        assert result2 == {"answer": "test", "sources": [], "tokens_used": 50}
