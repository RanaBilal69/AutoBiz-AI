"""
database.py — Lazy engine initialization.

FIXES:
  1. Engine is NEVER created at import time — no startup crash if DB is unreachable
  2. Supabase port fix: auto-corrects port 6543 → 5432 (pooler vs direct connection)
  3. Pool settings tuned for Supabase free tier

Correct Supabase URL format (use in HuggingFace Secrets as DATABASE_URL):
  postgresql://postgres.[PROJECT_REF]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:5432/postgres

Wrong format (causes ENOTFOUND error):
  postgresql://postgres.[PROJECT_REF]:[PASSWORD]@aws-1-[REGION].pooler.supabase.com:6543/postgres
"""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

load_dotenv()

_engine = None
_SessionLocal = None


def _build_engine():
    """Lazily build the SQLAlchemy engine on first use."""
    from app.core.db_config import get_current_db_url
    url = get_current_db_url()
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. "
            "Add it to .env (local) or HuggingFace Secrets (production)."
        )

    connect_args = {}

    # ── Supabase-specific fixes ───────────────────────────────
    if "supabase" in url.lower():
        # Force port 5432 (direct connection)
        # Port 6543 is the transaction pooler — it breaks SQLAlchemy prepared statements
        if ":6543/" in url:
            url = url.replace(":6543/", ":5432/")

        # Supabase requires SSL
        connect_args = {"sslmode": "require"}

    engine = create_engine(
        url,
        pool_pre_ping=True,     # ping before use — catches stale connections
        pool_size=3,            # small pool for Supabase free tier (max 60 connections)
        max_overflow=2,
        pool_recycle=280,       # recycle before Supabase's 300s idle timeout
        pool_timeout=30,
        connect_args=connect_args,
    )
    return engine


def get_engine():
    global _engine
    if _engine is None:
        _engine = _build_engine()
    return _engine


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a DB session."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=get_engine()
        )
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_connection() -> dict:
    """Lightweight health check — used by /health endpoint."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"connected": True, "error": None}
    except Exception as e:
        return {"connected": False, "error": str(e)[:300]}


def reset_engine():
    """Force engine rebuild — call after DB config changes."""
    global _engine, _SessionLocal
    if _engine:
        try:
            _engine.dispose()
        except Exception:
            pass
    _engine = None
    _SessionLocal = None


# ── Backward-compat shim ──────────────────────────────────────
# Some files do `from app.database import engine` — this lazy proxy
# satisfies that import without building the engine at import time.
class _LazyEngine:
    """Proxy that forwards all attribute access to the real engine."""
    def __getattr__(self, name):
        return getattr(get_engine(), name)

    def __repr__(self):
        return f"<LazyEngine wrapping {get_engine()!r}>"


engine = _LazyEngine()
# Do NOT define SessionLocal at module level — use get_db() dependency instead.