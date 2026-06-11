"""Configuração central. Tudo local, sem segredos obrigatórios.

GITHUB_TOKEN é opcional: sem ele a API do GitHub limita a 60 req/h.
Defina via variável de ambiente para subir para 5000 req/h.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
LANCEDB_DIR = DATA_DIR / "lancedb"
DB_PATH = DATA_DIR / "tie.db"


@dataclass(frozen=True)
class Settings:
    # Embeddings
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384

    # LLM local (Ollama)
    ollama_model: str = "qwen2.5:7b"
    ollama_host: str = "http://localhost:11434"
    use_llm_extraction: bool = True

    # Coleta
    github_token: str | None = field(default=None)
    arxiv_min_interval_s: float = 3.0  # arXiv pede ~3s entre requests
    http_timeout_s: float = 30.0
    user_agent: str = "tech-intel-engine/0.1 (research; +local)"

    # Janelas de análise
    default_window_days: int = 365
    trend_bucket_days: int = 30

    def __post_init__(self) -> None:
        for d in (DATA_DIR, CACHE_DIR, LANCEDB_DIR):
            d.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    return Settings(
        github_token=os.environ.get("GITHUB_TOKEN"),
        use_llm_extraction=os.environ.get("TIE_LLM", "1") != "0",
    )
