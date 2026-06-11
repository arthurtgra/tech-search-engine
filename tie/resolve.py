"""Entity resolution determinística e auditável.

Mescla entidades do mesmo tipo cuja chave normalizada coincide
(ex.: "GPT-4", "gpt 4", "gpt-4" -> mesma entidade). Repointa menções,
acumula aliases e recalcula contagens. Sem merge silencioso por embedding
no v0 — a precisão da estatística depende disso ser explicável.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

from tie.db import get_session
from tie.models import Entity, Mention

_VERSION_SUFFIX = re.compile(r"[-_ ]?v?\d+(\.\d+)*$")


def _norm_key(name: str) -> str:
    s = name.lower().strip()
    s = s.replace("_", "-").replace(" ", "-")
    s = re.sub(r"-{2,}", "-", s)
    return s


def resolve_entities() -> dict[str, int]:
    """Mescla duplicatas. Retorna {'merged': n, 'remaining': m}."""
    merged = 0
    with get_session() as session:
        entities = session.query(Entity).all()
        groups: dict[tuple[str, str], list[Entity]] = defaultdict(list)
        for ent in entities:
            groups[(ent.type, _norm_key(ent.canonical_name))].append(ent)

        for (_type, _key), members in groups.items():
            if len(members) < 2:
                continue
            primary = max(members, key=lambda e: e.mention_count)
            aliases: set[str] = set(json.loads(primary.aliases or "[]"))
            for dup in members:
                if dup.id == primary.id:
                    continue
                aliases.update(json.loads(dup.aliases or "[]"))
                aliases.add(dup.canonical_name)
                session.query(Mention).filter(Mention.entity_id == dup.id).update(
                    {Mention.entity_id: primary.id}
                )
                primary.mention_count += dup.mention_count
                if dup.first_seen and (not primary.first_seen or dup.first_seen < primary.first_seen):
                    primary.first_seen = dup.first_seen
                if dup.last_seen and (not primary.last_seen or dup.last_seen > primary.last_seen):
                    primary.last_seen = dup.last_seen
                session.delete(dup)
                merged += 1
            primary.aliases = json.dumps(sorted(aliases))

        remaining = session.query(Entity).count()
    return {"merged": merged, "remaining": remaining}
