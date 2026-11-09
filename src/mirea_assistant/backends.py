"""Где работает модель: на этом компьютере (Ollama) или через интернет (API). Выбор — в настройках."""

from __future__ import annotations

from .api_llm import ApiModel
from .config import Config
from .llm import Ollama


def make_llm(cfg: Config) -> Ollama | ApiModel:
    if cfg.provider == "api" and cfg.api_key:
        return ApiModel(cfg.api_model, cfg.api_key, cfg.api_base)
    return Ollama(cfg.model)
