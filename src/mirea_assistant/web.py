"""Локальный веб-сервер: страница-чат и API на 127.0.0.1."""

from __future__ import annotations

import asyncio
import logging
import re
from importlib import resources
from typing import Any, Callable

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from . import __version__, agent, backends
from .config import Config
from .llm import LLMError, Ollama

logger = logging.getLogger(__name__)

# Адрес API: только https (ключ уходит в заголовке) или локальный OpenAI-совместимый сервер.
_API_BASE_RE = re.compile(r"^(https://[^\s/]+|http://(127\.0\.0\.1|localhost)(:\d+)?)(/[^\s]*)?$")
_API_KEY_RE = re.compile(r"^[\x21-\x7e]{8,500}$")
_MODEL_RE = re.compile(r"^[\w.:/@+-]{1,200}$")


def settings_view(cfg: Config) -> dict:
    """Настройки для страницы. Сам ключ на страницу не отдаётся — только признак и последние 4 символа."""
    key = cfg.api_key
    return {"provider": cfg.provider, "local_model": cfg.model, "api_base": cfg.api_base, "api_model": cfg.api_model,
            "api_key_set": bool(key), "api_key_hint": f"…{key[-4:]}" if len(key) >= 8 else ""}

APP_ID = "mirea-assistant"
MAX_BODY = 1_000_000  # история обрезается до 12 сообщений по 4000 символов; больше не нужно
_HEADERS = {
    "Cache-Control": "no-store",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; frame-ancestors 'none'",
}


def create_app(llm: Ollama, port: int, on_shutdown: Callable[[], None] | None = None,
               on_local: Callable[[], None] | None = None, make_llm: Callable[[Config], Any] = backends.make_llm) -> Starlette:
    allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
    origins = {f"http://{h}" for h in allowed}
    busy = asyncio.Lock()  # локальная модель обрабатывает один вопрос за раз
    state = {"llm": llm}  # модель меняется из настроек страницы без перезапуска
    page = resources.files("mirea_assistant").joinpath("chat.html").read_text(encoding="utf-8")

    def forbidden(request: Request) -> Response | None:
        # Защита от чужих сайтов: DNS rebinding (Host) и запросы с других страниц (Origin).
        if request.headers.get("host") not in allowed:
            return JSONResponse({"error": "forbidden host"}, status_code=403)
        if request.method != "POST":
            return None
        origin = request.headers.get("origin")
        if origin and origin not in origins:
            return JSONResponse({"error": "forbidden origin"}, status_code=403)
        # Запасная защита для браузеров/расширений, которые не шлют Origin: Sec-Fetch-Site ставит сам браузер.
        # Установщик (Invoke-RestMethod) этот заголовок не шлёт и проходит.
        if request.headers.get("sec-fetch-site", "same-origin") not in ("same-origin", "none"):
            return JSONResponse({"error": "forbidden origin"}, status_code=403)
        # Строго по типу: «text/plain; x=application/json» — «простой» запрос без preflight, раньше проходил.
        if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
            return JSONResponse({"error": "expected JSON"}, status_code=415)
        try:
            too_big = int(request.headers.get("content-length") or 0) > MAX_BODY
        except ValueError:
            too_big = True
        if too_big:
            return JSONResponse({"error": "Слишком длинный запрос."}, status_code=413)
        return None

    async def index(request: Request) -> Response:
        return forbidden(request) or HTMLResponse(page, headers=_HEADERS)

    async def status(request: Request) -> Response:
        if bad := forbidden(request):
            return bad
        return JSONResponse({"app": APP_ID, "version": __version__, "busy": busy.locked(), **await state["llm"].status()})

    async def get_settings(request: Request) -> Response:
        if bad := forbidden(request):
            return bad
        return JSONResponse(settings_view(Config.load()))

    async def save_settings(request: Request) -> Response:
        if bad := forbidden(request):
            return bad
        try:
            body = await request.json()
        except ValueError:
            body = None
        if not isinstance(body, dict) or body.get("provider") not in ("ollama", "api"):
            return JSONResponse({"error": "Некорректные настройки."}, status_code=400)
        cfg = Config.load()
        key = str(body.get("api_key") or "").strip()
        base = str(body.get("api_base") or "").strip().rstrip("/")
        model = str(body.get("api_model") or "").strip()
        if key and not _API_KEY_RE.match(key):
            return JSONResponse({"error": "Ключ API выглядит неправильно: скопируйте его целиком, без пробелов."},
                                status_code=400)
        if base and not _API_BASE_RE.match(base):
            return JSONResponse({"error": "Адрес API должен начинаться с https://"}, status_code=400)
        if model and not _MODEL_RE.match(model):
            return JSONResponse({"error": "Некорректное имя модели."}, status_code=400)
        if body.get("clear_key") is True:
            cfg.api_key = ""
        cfg.provider = body["provider"]
        cfg.api_key = key or cfg.api_key
        cfg.api_base = base or cfg.api_base
        cfg.api_model = model or cfg.api_model
        if cfg.provider == "api" and not cfg.api_key:
            return JSONResponse({"error": "Для работы через API нужен ключ."}, status_code=400)
        cfg.save()
        async with busy:  # не менять модель посреди ответа
            old, state["llm"] = state["llm"], make_llm(cfg)
            if hasattr(old, "close"):
                try:
                    await old.close()
                except Exception:  # noqa: BLE001 — закрытие старого клиента не должно мешать
                    logger.warning("closing previous model client failed", exc_info=True)
        if cfg.provider == "ollama" and on_local:
            on_local()
        logger.info("model backend: %s", cfg.provider)
        return JSONResponse(settings_view(cfg))

    async def models(request: Request) -> Response:
        if bad := forbidden(request):
            return bad
        current = state["llm"]
        if not hasattr(current, "models"):
            return JSONResponse({"error": "Список моделей доступен для работы через API."}, status_code=400)
        try:
            return JSONResponse({"models": await current.models()})
        except LLMError as e:
            return JSONResponse({"error": str(e)}, status_code=503)

    async def chat(request: Request) -> Response:
        if bad := forbidden(request):
            return bad
        try:
            body = await request.json()
        except ValueError:
            body = None
        messages = body.get("messages") if isinstance(body, dict) else None
        if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) for m in messages):
            return JSONResponse({"error": "Некорректный запрос."}, status_code=400)
        me = str(body.get("me") or "").strip()[:100] or None
        think = body.get("think") is True
        async with busy:
            try:
                result = await agent.answer(state["llm"], messages, me, think=think)
            except LLMError as e:
                return JSONResponse({"error": str(e)}, status_code=503)
            except ValueError as e:
                return JSONResponse({"error": str(e)}, status_code=400)
            except Exception:
                logger.exception("chat failed")
                return JSONResponse({"error": "Внутренняя ошибка помощника. Повторите вопрос."}, status_code=500)
        return JSONResponse({"answer": result.text, "steps": [s.public() for s in result.steps], "context": result.context})

    async def shutdown(request: Request) -> Response:
        if bad := forbidden(request):
            return bad
        if on_shutdown:
            asyncio.get_running_loop().call_later(0.3, on_shutdown)
        return JSONResponse({"ok": True})

    return Starlette(
        routes=[
            Route("/", index),
            Route("/api/status", status),
            Route("/api/chat", chat, methods=["POST"]),
            Route("/api/shutdown", shutdown, methods=["POST"]),
            Route("/api/settings", get_settings, methods=["GET"]),
            Route("/api/settings", save_settings, methods=["POST"]),
            Route("/api/models", models, methods=["GET"]),
        ]
    )
