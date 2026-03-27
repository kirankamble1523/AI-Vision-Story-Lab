"""
database.py
───────────
SQLite database to store all image analysis history locally.
Uses synchronous SQLAlchemy (simpler for Streamlit apps).

Table: analyses
  - id, filename, caption, description, mood, objects,
    story, image_prompt, generated_image_url, timestamp
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import DATABASE_URL, MAX_HISTORY

# ── Engine ────────────────────────────────────────────────────────
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


# ── Model ─────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


class Analysis(Base):
    """One row per image analysis session."""

    __tablename__ = "analyses"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    filename            = Column(String(255), nullable=False)
    caption             = Column(Text, nullable=True)
    description         = Column(Text, nullable=True)
    mood                = Column(String(100), nullable=True)
    objects_json        = Column(Text, nullable=True)   # JSON list of detected objects
    story               = Column(Text, nullable=True)
    image_prompt        = Column(Text, nullable=True)   # prompt sent to DALL-E
    generated_image_url = Column(Text, nullable=True)
    timestamp           = Column(DateTime, default=datetime.utcnow)

    # ── Helpers ───────────────────────────────────────────────────
    def get_objects(self) -> List[str]:
        if not self.objects_json:
            return []
        try:
            return json.loads(self.objects_json)
        except Exception:
            return []

    def set_objects(self, items: List[str]) -> None:
        self.objects_json = json.dumps(items)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "filename": self.filename,
            "caption": self.caption,
            "description": self.description,
            "mood": self.mood,
            "objects": self.get_objects(),
            "story": self.story,
            "image_prompt": self.image_prompt,
            "generated_image_url": self.generated_image_url,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M") if self.timestamp else "",
        }


# ── Auto-create tables ────────────────────────────────────────────
def init_db() -> None:
    Base.metadata.create_all(bind=engine)


# ── CRUD helpers ──────────────────────────────────────────────────
def save_analysis(data: Dict[str, Any]) -> Analysis:
    """Insert a new Analysis row. Returns the saved object."""
    with SessionLocal() as db:
        record = Analysis(
            filename=data.get("filename", "unknown"),
            caption=data.get("caption"),
            description=data.get("description"),
            mood=data.get("mood"),
            story=data.get("story"),
            image_prompt=data.get("image_prompt"),
            generated_image_url=data.get("generated_image_url"),
            timestamp=datetime.utcnow(),
        )
        record.set_objects(data.get("objects", []))
        db.add(record)
        db.commit()
        db.refresh(record)

        # Trim old records if over limit
        _trim_history(db)
        return record


def _trim_history(db: Session) -> None:
    """Keep only the most recent MAX_HISTORY rows."""
    total = db.query(Analysis).count()
    if total > MAX_HISTORY:
        oldest_ids = (
            db.query(Analysis.id)
            .order_by(Analysis.timestamp.asc())
            .limit(total - MAX_HISTORY)
            .all()
        )
        ids = [r[0] for r in oldest_ids]
        db.query(Analysis).filter(Analysis.id.in_(ids)).delete(synchronize_session=False)
        db.commit()


def get_all_analyses() -> List[Dict[str, Any]]:
    """Return all analyses newest-first."""
    with SessionLocal() as db:
        rows = db.query(Analysis).order_by(Analysis.timestamp.desc()).all()
        return [r.to_dict() for r in rows]


def get_analysis_by_id(analysis_id: int) -> Optional[Dict[str, Any]]:
    with SessionLocal() as db:
        row = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        return row.to_dict() if row else None


def delete_all_analyses() -> None:
    with SessionLocal() as db:
        db.query(Analysis).delete()
        db.commit()