"""Busca semântica: embeddings locais (BGE) + LanceDB.

Escolha do modelo: BAAI/bge-small-en-v1.5 (384d). Melhor custo/qualidade entre
os locais para texto técnico curto; índice barato; suporta prefixo de query.
E5 é equivalente; Nomic só compensa com contexto longo (não é o gargalo aqui).
A interface fica isolada para troca trivial do modelo.
"""

from __future__ import annotations

from functools import lru_cache

import lancedb
import pyarrow as pa
from sentence_transformers import SentenceTransformer

from tie.config import LANCEDB_DIR, Settings
from tie.db import get_session
from tie.models import Document

_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_TABLE = "documents"


@lru_cache(maxsize=1)
def _model(name: str) -> SentenceTransformer:
    return SentenceTransformer(name)


def _schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("doc_id", pa.int64()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("title", pa.string()),
        ]
    )


def build_index(settings: Settings) -> int:
    """Embeda documentos ainda não indexados. Retorna nº embedados."""
    model = _model(settings.embed_model)
    db = lancedb.connect(str(LANCEDB_DIR))

    if _TABLE in db.table_names():
        tbl = db.open_table(_TABLE)
        indexed = {r["doc_id"] for r in tbl.search().select(["doc_id"]).limit(10**9).to_list()}
    else:
        tbl = db.create_table(_TABLE, schema=_schema(settings.embed_dim))
        indexed = set()

    with get_session() as session:
        docs = session.query(Document).all()
        pending = [d for d in docs if d.id not in indexed]
        if not pending:
            return 0
        texts = [f"{d.title}\n{d.text}"[:2000] for d in pending]
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        rows = [
            {"doc_id": d.id, "vector": v.tolist(), "title": d.title[:200]}
            for d, v in zip(pending, vectors)
        ]
    tbl.add(rows)
    return len(rows)


def semantic_search(
    settings: Settings, query: str, *, top_k: int = 200, min_similarity: float | None = None
) -> list[int]:
    """Retorna doc_ids relevantes à pergunta, cortando por similaridade de cosseno.

    Com métrica cosseno, `_distance = 1 - cos`; mantemos só docs com cos >= limiar.
    Isso filtra o que o filtro de categoria do collector não pegou.
    """
    threshold = settings.min_similarity if min_similarity is None else min_similarity
    db = lancedb.connect(str(LANCEDB_DIR))
    if _TABLE not in db.table_names():
        return []
    tbl = db.open_table(_TABLE)
    model = _model(settings.embed_model)
    qvec = model.encode(_QUERY_PREFIX + query, normalize_embeddings=True)
    hits = (
        tbl.search(qvec).metric("cosine").select(["doc_id"]).limit(top_k).to_list()
    )
    return [h["doc_id"] for h in hits if (1.0 - h["_distance"]) >= threshold]
