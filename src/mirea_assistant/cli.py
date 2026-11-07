"""Запуск помощника: `mirea-assistant` (с консолью) и `mirea-assistant-app` (ярлык, без консоли)."""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser

import httpx

from . import __version__
from .config import Config, home, model_for_ram
from .llm import Ollama, ollama_host
from .web import APP_ID, create_app

logger = logging.getLogger("mirea_assistant")


def _setup_logging(to_file: bool) -> None:
    if to_file:  # у оконного запуска нет консоли: sys.stdout/sys.stderr равны None
        path = home() / "assistant.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.FileHandler(path, encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stderr)
    from mirea_connect.logs import RedactingFormatter

    handler.setFormatter(RedactingFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def say(message: str, *, error: bool = False) -> None:
    """print, безопасный для оконного запуска, где stdout/stderr отсутствуют."""
    stream = sys.stderr if error else sys.stdout
    if stream is not None:
        print(message, file=stream)
    else:  # оконный запуск: сообщение остаётся в assistant.log
        (logger.error if error else logger.info)(message)


def running_instance(port: int) -> bool:
    """На порту уже работает наш помощник (повторный запуск ярлыка)."""
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/api/status", headers={"Host": f"127.0.0.1:{port}"}, timeout=2)
        return r.status_code == 200 and r.json().get("app") == APP_ID
    except (httpx.HTTPError, ValueError):
        return False


def port_free(port: int) -> bool:
    import socket

    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def ensure_ollama(wait_s: float = 20) -> bool:
    """Запустить Ollama в фоне, если она установлена, но не работает."""
    url = ollama_host() + "/api/tags"

    def up() -> bool:
        try:
            return httpx.get(url, timeout=2).status_code == 200
        except httpx.HTTPError:
            return False

    if up():
        return True
    exe = shutil.which("ollama")
    if not exe and sys.platform == "win32":
        candidate = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe")
        exe = candidate if os.path.exists(candidate) else None
    if not exe:
        logger.warning("ollama not found")
        return False
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen([exe, "serve"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=flags, start_new_session=sys.platform != "win32")
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if up():
            return True
        time.sleep(0.5)
    return False


def serve(port: int, open_browser: bool) -> int:
    import uvicorn

    cfg = Config.load()
    url = f"http://127.0.0.1:{port}/"
    if running_instance(port):
        if open_browser:
            webbrowser.open(url)
        say(f"Помощник уже работает: {url}")
        return 0
    if not port_free(port):
        say(f"Порт {port} занят другой программой. Запустите с другим портом: mirea-assistant --port {port + 1}",
            error=True)
        return 1

    threading.Thread(target=ensure_ollama, daemon=True).start()
    llm = Ollama(cfg.model)
    server: uvicorn.Server | None = None

    def stop() -> None:
        if server is not None:
            server.should_exit = True

    app = create_app(llm, port, on_shutdown=stop)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_config=None, access_log=False))

    if open_browser:
        def opener() -> None:
            for _ in range(100):
                if server.started:
                    webbrowser.open(url)
                    return
                time.sleep(0.1)

        threading.Thread(target=opener, daemon=True).start()
    say(f"Помощник МИРЭА: {url} (модель {cfg.model}). Остановить — Ctrl+C или кнопка «Выключить» на странице.")
    with contextlib.suppress(KeyboardInterrupt):
        server.run()
    return 0


def main(argv: list[str] | None = None, *, gui: bool = False) -> int:
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    p = argparse.ArgumentParser(prog="mirea-assistant", description="Помощник преподавателя МИРЭА с локальным ИИ")
    p.add_argument("--version", action="version", version=f"mirea-assistant {__version__}")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--no-browser", action="store_true", help="не открывать браузер")
    sub = p.add_subparsers(dest="cmd")
    c = sub.add_parser("configure", help="сохранить настройки (используется установщиком)")
    c.add_argument("--model")
    c.add_argument("--ram-gb", type=float, help="выбрать модель по объёму памяти")
    c.add_argument("--port", type=int, dest="cfg_port")
    sub.add_parser("status", help="проверить Ollama и модель")
    sub.add_parser("model", help="вывести имя выбранной модели (для установщика)")
    args = p.parse_args(argv)

    if args.cmd == "model":
        print(Config.load().model)
        return 0

    if args.cmd == "configure":
        cfg = Config.load()
        if args.ram_gb is not None:
            cfg.model = model_for_ram(args.ram_gb)
        if args.model:
            cfg.model = args.model
        if args.cfg_port:
            cfg.port = args.cfg_port
        path = cfg.save()
        print(f"Модель: {cfg.model}, порт: {cfg.port}. Настройки: {path}")
        return 0
    if args.cmd == "status":
        import asyncio

        cfg = Config.load()
        llm = Ollama(cfg.model)
        st = asyncio.run(llm.status())
        print(f"Ollama: {'работает' if st['ollama'] else 'не запущена'}; модель {cfg.model}: "
              f"{'скачана' if st['model_ready'] else 'не скачана'}")
        return 0 if st["model_ready"] else 1

    _setup_logging(to_file=gui)
    port = args.port or Config.load().port
    return serve(port, open_browser=not args.no_browser)


def main_gui() -> int:
    try:
        return main(gui=True)
    except Exception:
        logger.exception("assistant crashed")
        return 1
