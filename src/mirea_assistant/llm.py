"""Клиент локальной модели через Ollama (http://127.0.0.1:11434)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import httpx

DEFAULT_HOST = "http://127.0.0.1:11434"


def _error(r: httpx.Response) -> str:
    try:
        return str(r.json().get("error", ""))
    except ValueError:
        return ""


class LLMError(RuntimeError):
    """Локальная модель недоступна или ответила ошибкой — текст для пользователя."""


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class Reply:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict = field(default_factory=dict)


def ollama_host() -> str:
    host = os.environ.get("OLLAMA_HOST") or DEFAULT_HOST
    if "://" not in host:
        host = "http://" + host
    return host.rstrip("/").replace("0.0.0.0", "127.0.0.1")


class Ollama:
    def __init__(self, model: str, host: str | None = None, timeout: float = 300.0) -> None:
        self.model = model
        self.host = host or ollama_host()
        # Первый ответ на слабом ПК (загрузка модели в память) может занять минуту-две.
        # trust_env=False: httpx берёт системный прокси Windows из реестра и не учитывает исключение <local>,
        # и запросы к 127.0.0.1 уходили бы на прокси кафедры («ИИ не запущен»).
        self._client = httpx.AsyncClient(base_url=self.host, timeout=httpx.Timeout(timeout, connect=5.0),
                                         trust_env=False)

    async def close(self) -> None:
        await self._client.aclose()

    async def status(self) -> dict:
        """Работает ли Ollama и скачана ли модель."""
        try:
            r = await self._client.get("/api/tags", timeout=5.0)
            r.raise_for_status()
        except httpx.HTTPError:
            return {"provider": "ollama", "ollama": False, "model": self.model, "model_ready": False}
        names = {m.get("name") for m in r.json().get("models", [])} | {m.get("model") for m in r.json().get("models", [])}
        wanted = {self.model, self.model if ":" in self.model else self.model + ":latest"}
        return {"provider": "ollama", "ollama": True, "model": self.model, "model_ready": bool(names & wanted)}

    async def _post(self, body: dict) -> httpx.Response:
        try:
            return await self._client.post("/api/chat", json=body)
        except httpx.TimeoutException as e:
            raise LLMError("Локальная модель отвечает слишком долго. Повторите вопрос или закройте тяжёлые программы.") from e
        except httpx.HTTPError as e:
            raise LLMError("Не удаётся связаться с Ollama. Запустите приложение Ollama и повторите.") from e

    async def chat(self, messages: list[dict], tools: list[dict] | None = None, think: bool = False) -> Reply:
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            # Рассуждения qwen3 точнее на сложных вопросах, но на процессоре удлиняют ответ в разы:
            # включаются галочкой «Думать тщательнее».
            "think": think,
            # num_predict: предел длины ответа. Без него модель может «заговориться» и генерировать
            # текст бесконечно — на экзамене qwen3:4b не ответила за 40 минут.
            "options": {"temperature": 0.2, "num_ctx": 8192, "num_predict": 4096 if think else 1024},
            "keep_alive": "30m",
        }
        if tools:
            body["tools"] = tools
        r = await self._post(body)
        if r.status_code == 404:
            raise LLMError(f"Модель {self.model} не скачана. Выполните в PowerShell: ollama pull {self.model}")
        if r.status_code >= 400 and "think" in _error(r).lower():  # модель без режима рассуждений
            body.pop("think")
            r = await self._post(body)
        if r.status_code >= 400:
            raise LLMError(f"Ollama вернула ошибку: {_error(r) or r.status_code}")
        try:
            msg = r.json().get("message") or {}
        except ValueError as e:  # не JSON (например, страница прокси): понятная ошибка вместо текста исключения
            raise LLMError("Ollama вернула непонятный ответ. Перезапустите Ollama и повторите.") from e
        calls = []
        for c in msg.get("tool_calls") or []:
            fn = c.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):  # некоторые модели возвращают JSON строкой
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            if fn.get("name"):
                calls.append(ToolCall(fn["name"], args if isinstance(args, dict) else {}))
        return Reply(content=(msg.get("content") or "").strip(), tool_calls=calls, raw=msg)
