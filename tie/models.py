"""Modelo de dados unificado.

Camada Pydantic (RawDocument) = contrato entre collectors e normalização.
Camada SQLAlchemy = persistência do corpus normalizado, entidades, menções,
relações e séries temporais (TrendSnapshot) para cálculo de momentum.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field
from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# --------------------------------------------------------------------------- #
# Vocabulário controlado
# --------------------------------------------------------------------------- #
class EntityType(str, Enum):
    MODEL = "model"
    DATASET = "dataset"
    COMPANY = "company"
    INSTITUTION = "institution"
    FRAMEWORK = "framework"
    TOOL = "tool"
    TECHNIQUE = "technique"
    METHOD = "method"
    METRIC = "metric"
    BENCHMARK = "benchmark"


class RelationType(str, Enum):
    CO_OCCURS_WITH = "co_occurs_with"
    USED_BY = "used_by"
    CREATED_BY = "created_by"
    AFFILIATED_WITH = "affiliated_with"
    BENCHMARKED_ON = "benchmarked_on"


class Source(str, Enum):
    ARXIV = "arxiv"
    GITHUB = "github"
    HUGGINGFACE = "huggingface"
    OPENALEX = "openalex"


# --------------------------------------------------------------------------- #
# Contrato de coleta (Pydantic)
# --------------------------------------------------------------------------- #
class RawDocument(BaseModel):
    """Saída normalizada de qualquer collector."""

    source: Source
    source_id: str
    title: str
    text: str = ""  # abstract / readme / model card
    url: str = ""
    published_at: datetime | None = None
    # Métrica de popularidade específica da fonte (stars, downloads, citações).
    metric: float = 0.0
    metric_name: str = ""
    extra: dict = Field(default_factory=dict)

    def content_hash(self) -> str:
        raw = f"{self.source}|{self.source_id}|{self.title}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Persistência (SQLAlchemy)
# --------------------------------------------------------------------------- #
class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(32), index=True)
    source_id: Mapped[str] = mapped_column(String(255), index=True)
    title: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text, default="")
    url: Mapped[str] = mapped_column(Text, default="")
    published_at: Mapped[datetime | None] = mapped_column(DateTime, index=True)
    metric: Mapped[float] = mapped_column(Float, default=0.0)
    metric_name: Mapped[str] = mapped_column(String(32), default="")
    content_hash: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    extracted: Mapped[int] = mapped_column(Integer, default=0)  # flag de extração

    mentions: Mapped[list["Mention"]] = relationship(back_populates="document")


class Entity(Base):
    __tablename__ = "entities"
    __table_args__ = (UniqueConstraint("canonical_name", "type", name="uq_entity"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    canonical_name: Mapped[str] = mapped_column(String(255), index=True)
    type: Mapped[str] = mapped_column(String(32), index=True)
    aliases: Mapped[str] = mapped_column(Text, default="")  # JSON list
    first_seen: Mapped[datetime | None] = mapped_column(DateTime)
    last_seen: Mapped[datetime | None] = mapped_column(DateTime)
    mention_count: Mapped[int] = mapped_column(Integer, default=0)

    mentions: Mapped[list["Mention"]] = relationship(back_populates="entity")


class Mention(Base):
    __tablename__ = "mentions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    entity_id: Mapped[int] = mapped_column(ForeignKey("entities.id"), index=True)
    raw_text: Mapped[str] = mapped_column(String(255))
    confidence: Mapped[float] = mapped_column(Float, default=1.0)

    document: Mapped[Document] = relationship(back_populates="mentions")
    entity: Mapped[Entity] = relationship(back_populates="mentions")


class Relation(Base):
    __tablename__ = "relations"
    __table_args__ = (
        UniqueConstraint("src_entity_id", "dst_entity_id", "type", name="uq_relation"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    src_entity_id: Mapped[int] = mapped_column(ForeignKey("entities.id"), index=True)
    dst_entity_id: Mapped[int] = mapped_column(ForeignKey("entities.id"), index=True)
    type: Mapped[str] = mapped_column(String(32), index=True)
    weight: Mapped[float] = mapped_column(Float, default=0.0)
    evidence_doc_ids: Mapped[str] = mapped_column(Text, default="")  # JSON list


class TrendSnapshot(Base):
    """Série temporal por entidade: base auditável do cálculo de momentum."""

    __tablename__ = "trend_snapshots"
    __table_args__ = (
        UniqueConstraint("entity_id", "period_start", name="uq_snapshot"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    entity_id: Mapped[int] = mapped_column(ForeignKey("entities.id"), index=True)
    period_start: Mapped[datetime] = mapped_column(DateTime, index=True)
    doc_count: Mapped[int] = mapped_column(Integer, default=0)
    metric_sum: Mapped[float] = mapped_column(Float, default=0.0)
