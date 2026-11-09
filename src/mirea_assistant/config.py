"""Настройки помощника: модель и порт. Файл пишет установщик (`mirea-assistant configure`)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from .api_llm import DEFAULT_API_MODEL, OPENROUTER

# qwen3:4b в Ollama — это версия 2507 «с рассуждениями», которые нельзя выключить: на экзамене
# она не отвечала и за 5 минут. Инструктивная 4b — 25 из 28 за ~17 с, qwen3:8b — 28 из 28 за ~25 с
# (evals/cases.json, процессор GitHub Actions на 4 ядра).
DEFAULT_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
DEFAULT_PORT = 8765


def home() -> Path:
    return Path(os.environ.get("MIREA_ASSISTANT_HOME") or Path.home() / ".mirea-assistant")


@dataclass
class Config:
    model: str = DEFAULT_MODEL
    port: int = DEFAULT_PORT
    # Модель выбрана по памяти: при обновлении её можно сменить на лучшую. Выбранную вручную
    # (`configure --model`) обновление не трогает — раньше каждый повторный запуск установщика её затирал.
    auto_model: bool = True
    # Где работает модель: "ollama" — на этом компьютере, "api" — через интернет (OpenAI-совместимый API).
    provider: str = "ollama"
    api_base: str = OPENROUTER
    api_key: str = ""
    api_model: str = DEFAULT_API_MODEL

    @classmethod
    def load(cls) -> "Config":
        path = home() / "config.json"
        data = {}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
            except ValueError:
                data = {}
        cfg = cls(model=str(data.get("model") or DEFAULT_MODEL), port=int(data.get("port") or DEFAULT_PORT),
                  auto_model=data.get("auto_model") is not False,
                  provider="api" if data.get("provider") == "api" else "ollama",
                  api_base=str(data.get("api_base") or OPENROUTER), api_key=str(data.get("api_key") or ""),
                  api_model=str(data.get("api_model") or DEFAULT_API_MODEL))
        if os.environ.get("MIREA_ASSISTANT_API_KEY"):
            cfg.api_key = os.environ["MIREA_ASSISTANT_API_KEY"]
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
        return DEFAULT_MODEL
    return "qwen3:1.7b"
