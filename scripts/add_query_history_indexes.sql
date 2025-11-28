-- Migration: Add optimized indexes to query_history table
-- Created: 2024
-- Description: Adds indexes for common query patterns in the SmartTask FAQ service
--
-- Run this migration on existing databases to add the new indexes.
-- For new deployments, indexes are created automatically via SQLAlchemy.

-- Index for history listing: ORDER BY created_at DESC
-- This is the most common query pattern used by /api/history endpoint
CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_query_history_created_at_desc
ON query_history (created_at DESC);

-- Composite index for analytics queries
-- Useful for queries filtering by date range and aggregating token usage
CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_query_history_date_tokens
ON query_history (created_at, tokens_used);

-- Index for performance monitoring queries
-- Useful for finding slow queries or analyzing response time distribution
CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_query_history_response_time
ON query_history (response_time_ms);

-- Drop old indexes if they exist (from previous schema)
-- The simple index on question is not useful for Text columns without full-text search
DROP INDEX IF EXISTS ix_query_history_question;

-- Note: The old created_at index may still exist, we keep it as the new DESC index
-- is more specific for the common query pattern
-- DROP INDEX IF EXISTS ix_query_history_created_at;

-- Analyze the table to update statistics after index changes
ANALYZE query_history;
