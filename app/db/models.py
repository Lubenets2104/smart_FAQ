"""SQLAlchemy models for database tables."""

import uuid
from datetime import datetime
from sqlalchemy import Column, Integer, DateTime, Text, JSON, Index
from sqlalchemy.dialects.postgresql import UUID
from app.db.database import Base


class QueryHistory(Base):
    """
    Model for storing query history.

    Indexes:
    - Primary key on id (automatic)
    - ix_query_history_created_at_desc: For ORDER BY created_at DESC queries (history listing)
    - ix_query_history_tokens_created: Composite for analytics queries filtering by tokens
    - ix_query_history_response_time: For performance monitoring queries
    """

    __tablename__ = "query_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    tokens_used = Column(Integer, nullable=False, default=0)
    response_time_ms = Column(Integer, nullable=False, default=0)
    sources = Column(JSON, nullable=False, default=list)
    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    __table_args__ = (
        # Primary index for history listing: ORDER BY created_at DESC
        # Using postgresql_ops for descending order optimization
        Index(
            'ix_query_history_created_at_desc',
            created_at.desc(),
            postgresql_using='btree',
        ),
        # Composite index for analytics: filter by date range and aggregate tokens
        Index(
            'ix_query_history_date_tokens',
            created_at,
            tokens_used,
            postgresql_using='btree',
        ),
        # Index for performance monitoring queries
        Index(
            'ix_query_history_response_time',
            response_time_ms,
            postgresql_using='btree',
        ),
    )

    def __repr__(self) -> str:
        return f"<QueryHistory(id={self.id}, question={self.question[:50]}...)>"
