"""Redis cache service for caching FAQ responses."""

import json
import hashlib
from typing import Optional
import redis.asyncio as redis
from app.config import get_settings
from app.utils import get_logger

logger = get_logger(__name__)
settings = get_settings()


class CacheService:
    """Service for caching FAQ responses in Redis."""
    
    def __init__(self):
        self._client: Optional[redis.Redis] = None
    
    async def connect(self) -> None:
        """Connect to Redis."""
        if self._client is None:
            self._client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                decode_responses=True,
            )
            logger.info("Connected to Redis", host=settings.redis_host, port=settings.redis_port)
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("Disconnected from Redis")
    
    @staticmethod
    def _hash_question(question: str) -> str:
        """Generate a hash key for a question."""
        normalized = question.lower().strip()
        return f"faq:{hashlib.sha256(normalized.encode()).hexdigest()}"
    
    async def get_cached_answer(self, question: str) -> Optional[dict]:
        """
        Get cached answer for a question.
        
        Args:
            question: The user's question
            
        Returns:
            Cached response dict or None if not found
        """
        if not self._client:
            await self.connect()
        
        key = self._hash_question(question)
        try:
            cached = await self._client.get(key)
            if cached:
                logger.info("Cache hit", question_hash=key[:20])
                return json.loads(cached)
            logger.debug("Cache miss", question_hash=key[:20])
            return None
        except Exception as e:
            logger.error("Error getting from cache", error=str(e))
            return None
    
    async def set_cached_answer(
        self, 
        question: str, 
        answer: dict,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache an answer for a question.
        
        Args:
            question: The user's question
            answer: The response dict to cache
            ttl: Time to live in seconds (default from settings)
            
        Returns:
            True if cached successfully
        """
        if not self._client:
            await self.connect()
        
        key = self._hash_question(question)
        ttl = ttl or settings.redis_cache_ttl
        
        try:
            await self._client.setex(key, ttl, json.dumps(answer))
            logger.info("Cached answer", question_hash=key[:20], ttl=ttl)
            return True
        except Exception as e:
            logger.error("Error setting cache", error=str(e))
            return False
    
    async def check_connection(self) -> bool:
        """Check if Redis connection is working."""
        try:
            if not self._client:
                await self.connect()
            await self._client.ping()
            return True
        except Exception as e:
            logger.error("Redis connection check failed", error=str(e))
            return False
    
    async def clear_cache(self) -> int:
        """Clear all FAQ cache entries. Returns number of keys deleted."""
        if not self._client:
            await self.connect()
        
        try:
            keys = await self._client.keys("faq:*")
            if keys:
                deleted = await self._client.delete(*keys)
                logger.info("Cleared cache", keys_deleted=deleted)
                return deleted
            return 0
        except Exception as e:
            logger.error("Error clearing cache", error=str(e))
            return 0


# Global cache service instance
cache_service = CacheService()
