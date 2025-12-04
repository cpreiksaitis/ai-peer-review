"""Database models and operations for persisting reviews."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import Column, Integer, String, Text, Float, DateTime, create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


class ReviewRecord(Base):
    """Database model for a review session."""

    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Manuscript info
    manuscript_filename = Column(String(255))
    manuscript_title = Column(String(500))
    manuscript_text = Column(Text)  # Stored for reference

    # Review status
    status = Column(String(50), default="pending")  # pending, processing, completed, failed

    # Literature context
    literature_context = Column(Text)

    # Reviews (stored as JSON)
    initial_reviews = Column(Text)  # JSON dict
    debate_rounds = Column(Text)  # JSON list
    final_positions = Column(Text)  # JSON dict
    final_review = Column(Text)

    # Cost tracking
    total_cost_usd = Column(Float, default=0.0)
    total_tokens = Column(Integer, default=0)
    cost_breakdown = Column(Text)  # JSON dict

    # Recommendation
    recommendation = Column(String(50))  # Accept, Minor Revision, Major Revision, Reject

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "manuscript_filename": self.manuscript_filename,
            "manuscript_title": self.manuscript_title,
            "status": self.status,
            "final_review": self.final_review,
            "literature_context": self.literature_context or "",
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "recommendation": self.recommendation,
            "initial_reviews": json.loads(self.initial_reviews) if self.initial_reviews else {},
            "debate_rounds": json.loads(self.debate_rounds) if self.debate_rounds else [],
            "final_positions": json.loads(self.final_positions) if self.final_positions else {},
            "cost_breakdown": json.loads(self.cost_breakdown) if self.cost_breakdown else {},
        }


# Database setup
DATABASE_PATH = Path("data/reviews.db")


def get_database_url(path: Path = DATABASE_PATH) -> str:
    """Get the database URL."""
    return f"sqlite:///{path}"


def get_async_database_url(path: Path = DATABASE_PATH) -> str:
    """Get the async database URL."""
    return f"sqlite+aiosqlite:///{path}"


def init_db(path: Path = DATABASE_PATH):
    """Initialize the database synchronously."""
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(get_database_url(path))
    Base.metadata.create_all(engine)
    return engine


async def init_async_db(path: Path = DATABASE_PATH):
    """Initialize the database asynchronously."""
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_async_engine(get_async_database_url(path))
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine


def get_session(engine):
    """Get a synchronous session."""
    Session = sessionmaker(bind=engine)
    return Session()


def get_async_session_maker(engine) -> async_sessionmaker[AsyncSession]:
    """Get an async session maker."""
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


# CRUD operations
async def create_review(
    session: AsyncSession,
    manuscript_filename: str,
    manuscript_title: str,
    manuscript_text: str,
) -> ReviewRecord:
    """Create a new review record."""
    review = ReviewRecord(
        manuscript_filename=manuscript_filename,
        manuscript_title=manuscript_title,
        manuscript_text=manuscript_text,
        status="pending",
    )
    session.add(review)
    await session.commit()
    await session.refresh(review)
    return review


async def update_review_status(
    session: AsyncSession,
    review_id: int,
    status: str,
) -> Optional[ReviewRecord]:
    """Update review status."""
    review = await session.get(ReviewRecord, review_id)
    if review:
        review.status = status
        await session.commit()
        await session.refresh(review)
    return review


async def update_review_results(
    session: AsyncSession,
    review_id: int,
    initial_reviews: dict,
    debate_rounds: list,
    final_positions: dict,
    final_review: str,
    literature_context: str,
    cost_summary: dict,
    recommendation: str = "",
) -> Optional[ReviewRecord]:
    """Update review with results."""
    review = await session.get(ReviewRecord, review_id)
    if review:
        review.status = "completed"
        review.initial_reviews = json.dumps(initial_reviews)
        review.debate_rounds = json.dumps(debate_rounds)
        review.final_positions = json.dumps(final_positions)
        review.final_review = final_review
        review.literature_context = literature_context
        review.total_cost_usd = cost_summary.get("total_cost_usd", 0)
        review.total_tokens = cost_summary.get("total_tokens", 0)
        review.cost_breakdown = json.dumps(cost_summary)
        review.recommendation = recommendation
        await session.commit()
        await session.refresh(review)
    return review


async def get_review(session: AsyncSession, review_id: int) -> Optional[ReviewRecord]:
    """Get a review by ID."""
    return await session.get(ReviewRecord, review_id)


async def get_all_reviews(session: AsyncSession) -> list[ReviewRecord]:
    """Get all reviews."""
    from sqlalchemy import select
    result = await session.execute(
        select(ReviewRecord).order_by(ReviewRecord.created_at.desc())
    )
    return list(result.scalars().all())


async def delete_review(session: AsyncSession, review_id: int) -> bool:
    """Delete a review."""
    review = await session.get(ReviewRecord, review_id)
    if review:
        await session.delete(review)
        await session.commit()
        return True
    return False

