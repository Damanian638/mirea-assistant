"""Локальный веб-сервер: страница-чат и API на 127.0.0.1."""

from __future__ import annotations

import asyncio
import logging
from importlib import resources
from typing import Callable

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from . import __version__, agent
from .llm import LLMError, Ollama

logger = logging.getLogger(__name__)

APP_ID = "mirea-assistant"
MAX_BODY = 1_000_000  # история обрезается до 12 сообщений по 4000 символов; больше не нужно
_HEADERS = {
    "Cache-Control": "no-store",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; frame-ancestors 'none'",
}


def create_app(llm: Ollama, port: int, on_shutdown: Callable[[], None] | None = None) -> Starlette:
    allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
    origins = {f"http://{h}" for h in allowed}
    busy = asyncio.Lock()  # локальная модель обрабатывает один вопрос за раз
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
        return JSONResponse({"app": APP_ID, "version": __version__, "busy": busy.locked(), **await llm.status()})

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
                result = await agent.answer(llm, messages, me, think=think)
            except LLMError as e:
                return JSONResponse({"error": str(e)}, status_code=503)
            except ValueError as e:
                return JSONResponse({"error": str(e)}, status_code=400)
            except Exception:
                logger.exception("chat failed")
                return JSONResponse({"error": "Внутренняя ошибка помощника. Повторите вопрос."}, status_code=500)
        return JSONResponse({"answer": result.text, "steps": [s.public() for s in result.steps]})

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
        ]
    )
