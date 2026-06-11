"""Registro de collectors disponíveis."""

from __future__ import annotations

from tie.collectors.arxiv import ArxivCollector
from tie.collectors.base import Collector
from tie.collectors.github import GithubCollector
from tie.collectors.huggingface import HuggingFaceCollector
from tie.config import Settings

ALL_COLLECTORS: list[type[Collector]] = [
    ArxivCollector,
    GithubCollector,
    HuggingFaceCollector,
]


def build_collectors(settings: Settings) -> list[Collector]:
    return [cls(settings) for cls in ALL_COLLECTORS]
