"""Extração de entidades.

Estratégia: LLM local (Ollama, JSON estruturado) como extrator principal, com
um dicionário-semente do domínio para ancorar tipos e garantir recall em nomes
canônicos conhecidos. Sem rede para Ollama -> cai para dicionário apenas.
"""

from __future__ import annotations

import json
import re

import ollama
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_fixed

from tie.config import Settings
from tie.db import get_session
from tie.models import Document, Entity, EntityType, Mention

# Dicionário-semente: nomes canônicos conhecidos do domínio de agentes.
# Serve de âncora de tipo e de recall mínimo se o LLM falhar.
SEED_ENTITIES: dict[str, EntityType] = {
    "langchain": EntityType.FRAMEWORK,
    "langgraph": EntityType.FRAMEWORK,
    "llamaindex": EntityType.FRAMEWORK,
    "autogen": EntityType.FRAMEWORK,
    "crewai": EntityType.FRAMEWORK,
    "openai gym": EntityType.FRAMEWORK,
    "react": EntityType.TECHNIQUE,
    "chain-of-thought": EntityType.TECHNIQUE,
    "tree-of-thoughts": EntityType.TECHNIQUE,
    "tool calling": EntityType.TECHNIQUE,
    "function calling": EntityType.TECHNIQUE,
    "rag": EntityType.TECHNIQUE,
    "reflexion": EntityType.TECHNIQUE,
    "gpt-4": EntityType.MODEL,
    "gpt-4o": EntityType.MODEL,
    "claude": EntityType.MODEL,
    "llama": EntityType.MODEL,
    "qwen": EntityType.MODEL,
    "mistral": EntityType.MODEL,
    "gemini": EntityType.MODEL,
    "openai": EntityType.COMPANY,
    "anthropic": EntityType.COMPANY,
    "google": EntityType.COMPANY,
    "deepmind": EntityType.COMPANY,
    "meta": EntityType.COMPANY,
    "microsoft": EntityType.COMPANY,
    "gaia": EntityType.BENCHMARK,
    "webarena": EntityType.BENCHMARK,
    "swe-bench": EntityType.BENCHMARK,
    "agentbench": EntityType.BENCHMARK,
}

_VALID_TYPES = {t.value for t in EntityType}

_SYSTEM = """You extract technology-intelligence entities from research/code text.
Return STRICT JSON: {"entities": [{"name": str, "type": str}]}.
Allowed types: model, dataset, company, institution, framework, tool, technique,
method, metric, benchmark.
Rules: only concrete named entities (not generic words like "model" or "system").
Normalize names to their common form. Max 15 entities. No commentary."""


class ExtractedEntity(BaseModel):
    name: str
    type: str


def _seed_match(text: str) -> list[ExtractedEntity]:
    low = text.lower()
    found: list[ExtractedEntity] = []
    for name, etype in SEED_ENTITIES.items():
        if re.search(r"\b" + re.escape(name) + r"\b", low):
            found.append(ExtractedEntity(name=name, type=etype.value))
    return found


@retry(stop=stop_after_attempt(2), wait=wait_fixed(1), reraise=True)
def _llm_extract(settings: Settings, text: str) -> list[ExtractedEntity]:
    client = ollama.Client(host=settings.ollama_host)
    resp = client.chat(
        model=settings.ollama_model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": text[:4000]},
        ],
        format="json",
        options={"temperature": 0.0},
    )
    payload = json.loads(resp["message"]["content"])
    out: list[ExtractedEntity] = []
    for item in payload.get("entities", []):
        try:
            ent = ExtractedEntity(**item)
        except (ValidationError, TypeError):
            continue
        if ent.type in _VALID_TYPES and len(ent.name) > 1:
            out.append(ent)
    return out


def extract_for_document(settings: Settings, title: str, text: str) -> list[ExtractedEntity]:
    combined = f"{title}\n{text}"
    entities = _seed_match(combined)
    if settings.use_llm_extraction:
        try:
            entities += _llm_extract(settings, combined)
        except Exception as exc:  # Ollama indisponível -> segue com seed
            print(f"[warn] LLM extraction falhou (usando só dicionário): {exc}")
    # Dedup por (nome lower, tipo)
    seen: set[tuple[str, str]] = set()
    unique: list[ExtractedEntity] = []
    for e in entities:
        key = (e.name.strip().lower(), e.type)
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique


def _upsert_entity(session, name: str, etype: str, doc_date) -> Entity:  # noqa: ANN001
    canonical = name.strip().lower()
    ent = (
        session.query(Entity)
        .filter(Entity.canonical_name == canonical, Entity.type == etype)
        .one_or_none()
    )
    if ent is None:
        ent = Entity(
            canonical_name=canonical, type=etype, aliases=json.dumps([name]),
            first_seen=doc_date, last_seen=doc_date, mention_count=0,
        )
        session.add(ent)
        session.flush()
    else:
        if doc_date:
            if ent.first_seen is None or doc_date < ent.first_seen:
                ent.first_seen = doc_date
            if ent.last_seen is None or doc_date > ent.last_seen:
                ent.last_seen = doc_date
    return ent


def extract_corpus(settings: Settings, limit: int | None = None) -> int:
    """Extrai entidades dos documentos ainda não processados. Retorna nº de docs.

    Uma transação curta POR documento: a chamada lenta ao LLM acontece FORA de
    qualquer transação de escrita, e o lock de escrita do SQLite é segurado só
    durante o commit de cada doc. Isso evita prender o banco por minutos (e o
    consequente 'database is locked' quando UI e CLI rodam juntos).
    """
    with get_session() as session:
        q = session.query(Document.id).filter(Document.extracted == 0)
        if limit:
            q = q.limit(limit)
        doc_ids = [row[0] for row in q.all()]

    processed = 0
    for doc_id in doc_ids:
        with get_session() as session:  # leitura curta
            doc = session.get(Document, doc_id)
            title, text, pub = doc.title, doc.text, doc.published_at

        extracted = extract_for_document(settings, title, text)  # LLM, sem lock

        with get_session() as session:  # escrita curta (1 commit por doc)
            for e in extracted:
                ent = _upsert_entity(session, e.name, e.type, pub)
                session.add(
                    Mention(
                        document_id=doc_id, entity_id=ent.id,
                        raw_text=e.name[:255], confidence=1.0,
                    )
                )
                ent.mention_count += 1
            doc = session.get(Document, doc_id)
            doc.extracted = 1
        processed += 1
    return processed
