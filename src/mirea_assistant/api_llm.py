"""Модель через интернет: OpenAI-совместимый API (по умолчанию OpenRouter).

Тот же интерфейс, что у Ollama (chat/status/close), поэтому агент не знает, где работает модель.
Сообщения агента хранятся в формате Ollama — здесь они переводятся в формат OpenAI
(идентификаторы вызовов, аргументы строкой JSON) и обратно.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from . import __version__
from .llm import LLMError, Reply, ToolCall

OPENROUTER = "https://openrouter.ai/api/v1"
STATUS_TTL = 300.0  # страница спрашивает статус каждые 15 с — не ходим в платный API так часто

# Недорогие модели с вызовом инструментов и хорошим русским. Доступность проверяется по живому списку
# провайдера (models()): отсутствующие в нём не предлагаются.
RECOMMENDED = [
    "google/gemini-2.5-flash",
    "openai/gpt-4.1-mini",
    "deepseek/deepseek-chat-v3-0324",
    "qwen/qwen3-235b-a22b-2507",
]
DEFAULT_API_MODEL = RECOMMENDED[0]


def _error_text(r: httpx.Response) -> str:
    try:
        data = r.json()
    except ValueError:
        return ""
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict):
        return str(err.get("message") or "")
    return str(err or "")


def _friendly(r: httpx.Response, model: str) -> LLMError:
    detail = _error_text(r)
    code = r.status_code
    if code == 401:
        return LLMError("Ключ API не подошёл. Проверьте его в ⚙ Настройках.")
    if code == 402:
        return LLMError("На счёте API закончились средства. Пополните баланс у провайдера или переключитесь на локальную модель.")
    if code == 403:
        return LLMError("Провайдер API отказал в доступе. Возможно, сервис недоступен из вашей сети — "
                        "нужен VPN/прокси, или переключитесь на локальную модель.")
    if code == 404 or (code == 400 and "model" in detail.lower()):
        return LLMError(f"Модель {model} недоступна у провайдера API. Выберите другую в ⚙ Настройках.")
    if code == 429:
        return LLMError("Провайдер API просит подождать (слишком много запросов). Повторите через минуту.")
    return LLMError(f"Провайдер API вернул ошибку: {detail or code}")


def to_openai(messages: list[dict]) -> list[dict]:
    """Сообщения агента (формат Ollama) → формат OpenAI. Ответы инструментов идут по порядку вызовов."""
    out: list[dict] = []
    pending: list[str] = []
    n = 0
    for m in messages:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = c.get("function") or {}
                n += 1
                call_id = f"call_{n}"
                pending.append(call_id)
                args = fn.get("arguments")
                calls.append({"id": call_id, "type": "function",
                              "function": {"name": fn.get("name", ""),
                                           "arguments": args if isinstance(args, str) else json.dumps(args or {}, ensure_ascii=False)}})
            out.append({"role": "assistant", "content": m.get("content") or None, "tool_calls": calls})
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": pending.pop(0) if pending else f"call_{n}",
                        "content": m.get("content", "")})
        else:
            out.append({"role": role, "content": m.get("content", "")})
    return out


class ApiModel:
    """Модель по OpenAI-совместимому API. Системный прокси/VPN используется (в отличие от локальной Ollama)."""

    def __init__(self, model: str, api_key: str, base_url: str = OPENROUTER, timeout: float = 120.0) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.openrouter = "openrouter.ai" in self.base_url
        headers = {"Authorization": f"Bearer {api_key}", "X-Title": "Mirea Assistant", "User-Agent": f"mirea-assistant/{__version__}"}
        self._client = httpx.AsyncClient(base_url=self.base_url, headers=headers,
                                         timeout=httpx.Timeout(timeout, connect=10.0))
        self._status: tuple[float, dict] | None = None

    async def close(self) -> None:
        await self._client.aclose()

    async def _get_models(self) -> list[dict]:
        r = await self._client.get("/models", timeout=15.0)
        if r.status_code >= 400:
            raise _friendly(r, self.model)
        data = r.json().get("data") or []
        return [m for m in data if isinstance(m, dict) and m.get("id")]

    async def models(self) -> list[dict]:
        """Модели с вызовом инструментов (без него помощник работать не может), рекомендованные — первыми."""
        try:
            items = await self._get_models()
        except httpx.HTTPError as e:
            raise LLMError("Не удаётся связаться с провайдером API. Проверьте интернет (и VPN, если он нужен).") from e
        out = []
        for m in items:
            params = m.get("supported_parameters")
            if params is not None and "tools" not in params:
                continue
            price = m.get("pricing") or {}
            try:
                per_m = (float(price.get("prompt", 0)) * 1e6, float(price.get("completion", 0)) * 1e6)
            except (TypeError, ValueError):
                per_m = None
            out.append({"id": m["id"], "name": m.get("name") or m["id"], "recommended": m["id"] in RECOMMENDED,
                        "price_per_million": per_m})
        rank = {mid: i for i, mid in enumerate(RECOMMENDED)}
        out.sort(key=lambda x: (rank.get(x["id"], len(rank)), x["id"]))
        return out

    async def status(self) -> dict:
        """Ключ подходит и модель есть у провайдера. Кэшируется на STATUS_TTL секунд."""
        if self._status and time.monotonic() - self._status[0] < STATUS_TTL:
            return self._status[1]
        st: dict[str, Any] = {"provider": "api", "model": self.model, "ollama": True}
        try:
            ids = {m["id"] for m in await self._get_models()}
            st.update(model_ready=self.model in ids or not ids, error=None if self.model in ids or not ids
                      else f"Модели {self.model} нет у провайдера API — выберите другую в ⚙ Настройках.")
        except LLMError as e:
            st.update(model_ready=False, error=str(e))
        except (httpx.HTTPError, ValueError):
            st.update(model_ready=False, error="Не удаётся связаться с провайдером API. Проверьте интернет (и VPN, если он нужен).")
        self._status = (time.monotonic(), st)
        return st

    async def chat(self, messages: list[dict], tools: list[dict] | None = None, think: bool = False) -> Reply:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai(messages),
            "temperature": 0.2,
            "max_tokens": 4096 if think else 1500,
        }
        if tools:
            body["tools"] = tools  # формат TOOLS агента уже совпадает с OpenAI
        if self.openrouter:
            # только провайдеры, которые не хранят и не обучаются на запросах (в них ФИО и расписание)
            body["provider"] = {"data_collection": "deny"}
            if think:
                body["reasoning"] = {"effort": "medium"}
        try:
            r = await self._client.post("/chat/completions", json=body)
        except httpx.TimeoutException as e:
            raise LLMError("Модель API отвечает слишком долго. Повторите вопрос.") from e
        except httpx.HTTPError as e:
            raise LLMError("Не удаётся связаться с провайдером API. Проверьте интернет (и VPN, если он нужен).") from e
        if r.status_code >= 400:
            self._status = None
            raise _friendly(r, self.model)
        try:
            data = r.json()
            msg = (data.get("choices") or [{}])[0].get("message") or {}
        except (ValueError, AttributeError, IndexError) as e:
            raise LLMError("Провайдер API вернул непонятный ответ. Повторите вопрос.") from e
        if data.get("error"):
            raise LLMError(f"Провайдер API вернул ошибку: {_error_text(r) or data['error']}")
        calls = []
        for c in msg.get("tool_calls") or []:
            fn = c.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except ValueError:
                    args = {}
            if fn.get("name"):
                calls.append(ToolCall(fn["name"], args if isinstance(args, dict) else {}))
        return Reply(content=(msg.get("content") or "").strip(), tool_calls=calls, raw=msg)
