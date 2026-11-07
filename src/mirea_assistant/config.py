"""Настройки помощника: модель и порт. Файл пишет установщик (`mirea-assistant configure`)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_MODEL = "qwen3:4b"
DEFAULT_PORT = 8765


def home() -> Path:
    return Path(os.environ.get("MIREA_ASSISTANT_HOME") or Path.home() / ".mirea-assistant")


@dataclass
class Config:
    model: str = DEFAULT_MODEL
    port: int = DEFAULT_PORT

    @classmethod
    def load(cls) -> "Config":
        path = home() / "config.json"
        data = {}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
            except ValueError:
                data = {}
        cfg = cls(model=str(data.get("model") or DEFAULT_MODEL), port=int(data.get("port") or DEFAULT_PORT))
        if os.environ.get("MIREA_ASSISTANT_MODEL"):
            cfg.model = os.environ["MIREA_ASSISTANT_MODEL"]
        return cfg

    def save(self) -> Path:
        path = home() / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path


def model_for_ram(total_gb: float) -> str:
    """Самая крупная модель, которая поместится вместе с браузером и офисом."""
    if total_gb >= 15:
        return "qwen3:8b"
    if total_gb >= 7:
        return "qwen3:4b"
    return "qwen3:1.7b"
