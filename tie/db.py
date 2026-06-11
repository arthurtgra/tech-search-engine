"""Engine/sessão SQLite com WAL (suporta collectors concorrentes)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from tie.config import DB_PATH
from tie.models import Base

_engine: Engine | None = None
_Session: sessionmaker[Session] | None = None


def _init() -> None:
    global _engine, _Session
    if _engine is not None:
        return
    _engine = create_engine(f"sqlite:///{DB_PATH}", future=True)

    @event.listens_for(_engine, "connect")
    def _set_pragma(dbapi_conn, _):  # noqa: ANN001
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()

    Base.metadata.create_all(_engine)
    _Session = sessionmaker(_engine, expire_on_commit=False, future=True)


@contextmanager
def get_session() -> Iterator[Session]:
    _init()
    assert _Session is not None
    session = _Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
