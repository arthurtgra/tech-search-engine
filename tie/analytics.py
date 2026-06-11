"""Analítica: co-ocorrência, momentum e rankings — escopados a um subconjunto
de documentos (os relevantes à pergunta, vindos da busca semântica).

Todo número aqui é calculado sobre o corpus, com evidências rastreáveis.
Nenhum "score de confiança" inventado: usamos contagem de evidências e
momentum medido sobre janelas temporais reais.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import combinations

from sqlalchemy import func

from tie.config import Settings
from tie.db import get_session
from tie.models import Document, Entity, Mention, Relation


@dataclass
class RankedEntity:
    entity_id: int
    name: str
    type: str
    doc_count: int
    metric_sum: float
    momentum: float  # razão janela recente / anterior (>1 = crescendo)


@dataclass
class CoOccurrence:
    a: str
    b: str
    weight: int
    type_a: str
    type_b: str


def _doc_entity_map(session, doc_ids: list[int]):  # noqa: ANN001
    """{doc_id: set(entity_id)} e dados auxiliares de entidade/documento."""
    rows = (
        session.query(Mention.document_id, Mention.entity_id)
        .filter(Mention.document_id.in_(doc_ids))
        .all()
    )
    doc_ents: dict[int, set[int]] = defaultdict(set)
    for did, eid in rows:
        doc_ents[did].add(eid)
    return doc_ents


def build_relations(doc_ids: list[int]) -> int:
    """Persiste co-ocorrência ponderada entre entidades no subconjunto."""
    with get_session() as session:
        doc_ents = _doc_entity_map(session, doc_ids)
        weights: dict[tuple[int, int], list[int]] = defaultdict(list)
        for did, ents in doc_ents.items():
            for a, b in combinations(sorted(ents), 2):
                weights[(a, b)].append(did)

        # Recria relações de co-ocorrência para esse escopo.
        for (a, b), evidence in weights.items():
            rel = (
                session.query(Relation)
                .filter(
                    Relation.src_entity_id == a,
                    Relation.dst_entity_id == b,
                    Relation.type == "co_occurs_with",
                )
                .one_or_none()
            )
            if rel is None:
                rel = Relation(
                    src_entity_id=a, dst_entity_id=b, type="co_occurs_with",
                    weight=0.0, evidence_doc_ids="[]",
                )
                session.add(rel)
            rel.weight = float(len(evidence))
            rel.evidence_doc_ids = json.dumps(evidence[:20])
        return len(weights)


def rank_entities(
    settings: Settings, doc_ids: list[int], *, etype: str | None = None, top_n: int = 10
) -> list[RankedEntity]:
    half = settings.default_window_days // 2
    cutoff = datetime.utcnow() - timedelta(days=half)
    with get_session() as session:
        q = (
            session.query(
                Entity.id, Entity.canonical_name, Entity.type,
                func.count(Mention.id).label("dc"),
                func.coalesce(func.sum(Document.metric), 0.0).label("ms"),
            )
            .join(Mention, Mention.entity_id == Entity.id)
            .join(Document, Document.id == Mention.document_id)
            .filter(Mention.document_id.in_(doc_ids))
        )
        if etype:
            q = q.filter(Entity.type == etype)
        q = q.group_by(Entity.id).order_by(func.count(Mention.id).desc()).limit(top_n)
        ranked: list[RankedEntity] = []
        for eid, name, etype_, dc, ms in q.all():
            recent = (
                session.query(func.count(Mention.id))
                .join(Document, Document.id == Mention.document_id)
                .filter(
                    Mention.entity_id == eid,
                    Mention.document_id.in_(doc_ids),
                    Document.published_at >= cutoff,
                )
                .scalar()
            ) or 0
            older = dc - recent
            momentum = (recent / older) if older > 0 else (float(recent) if recent else 0.0)
            ranked.append(
                RankedEntity(eid, name, etype_, dc, float(ms), round(momentum, 2))
            )
        return ranked


def top_cooccurrences(doc_ids: list[int], *, top_n: int = 12) -> list[CoOccurrence]:
    with get_session() as session:
        doc_ents = _doc_entity_map(session, doc_ids)
        weights: dict[tuple[int, int], int] = defaultdict(int)
        for ents in doc_ents.values():
            for a, b in combinations(sorted(ents), 2):
                weights[(a, b)] += 1
        top = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        if not top:
            return []
        ids = {e for pair, _ in top for e in pair}
        meta = {
            e.id: (e.canonical_name, e.type)
            for e in session.query(Entity).filter(Entity.id.in_(ids)).all()
        }
        out: list[CoOccurrence] = []
        for (a, b), w in top:
            na, ta = meta.get(a, ("?", "?"))
            nb, tb = meta.get(b, ("?", "?"))
            out.append(CoOccurrence(na, nb, w, ta, tb))
        return out
