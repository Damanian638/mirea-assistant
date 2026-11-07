"""Локальный сервер, запуск и настройки."""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from mirea_assistant import agent, cli, config, web
from mirea_assistant.llm import LLMError

PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"


class StubLLM:
    async def status(self):
        return {"ollama": True, "model": "qwen3:4b", "model_ready": True}


@pytest.fixture
def app(monkeypatch):
    stopped = []
    application = web.create_app(StubLLM(), PORT, on_shutdown=lambda: stopped.append(True))
    with TestClient(application, base_url=BASE) as c:
        c.stopped = stopped
        yield c


def test_page_and_status(app):
    r = app.get("/")
    assert r.status_code == 200 and "Помощник МИРЭА" in r.text and "frame-ancestors 'none'" in r.headers["content-security-policy"]
    s = app.get("/api/status").json()
    assert s["app"] == web.APP_ID and s["model_ready"] is True and s["busy"] is False


def test_chat_passes_history_and_me(app, monkeypatch):
    seen = {}

    async def fake_answer(llm, messages, me=None, think=False):
        seen.update(messages=messages, me=me, think=think)
        return agent.Answer("Ответ", [agent.Step("search", {"query": "ЭФБО"}, True, "Найдено: ЭФБО-01-24")])

    monkeypatch.setattr(agent, "answer", fake_answer)
    r = app.post("/api/chat", json={"messages": [{"role": "user", "content": "вопрос"}], "me": " Иванов И. И. ", "think": True})
    assert r.status_code == 200
    assert r.json() == {"answer": "Ответ", "steps": [{"tool": "search", "args": {"query": "ЭФБО"}, "ok": True,
                                                      "result": "Найдено: ЭФБО-01-24"}]}
    assert seen["me"] == "Иванов И. И." and seen["think"] is True


@pytest.mark.parametrize(
    "exc, code, fragment",
    [(LLMError("Запустите приложение Ollama"), 503, "Ollama"), (RuntimeError("boom"), 500, "Внутренняя ошибка")],
)
def test_chat_errors(app, monkeypatch, exc, code, fragment):
    async def fail(*a, **k):
        raise exc

    monkeypatch.setattr(agent, "answer", fail)
    r = app.post("/api/chat", json={"messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == code and fragment in r.json()["error"] and "boom" not in r.text


@pytest.mark.parametrize("body", ["not json", "[]", '{"messages": []}', '{"messages": ["x"]}', '{"messages": "x"}'])
def test_chat_rejects_bad_body(app, body):
    r = app.post("/api/chat", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 400


def test_foreign_sites_cannot_use_the_api(app):
    # чужой сайт в браузере: другой Origin, либо простой form POST без JSON, либо DNS rebinding
    msg = {"messages": [{"role": "user", "content": "x"}]}
    assert app.post("/api/chat", json=msg, headers={"Origin": "https://evil.example"}).status_code == 403
    assert app.post("/api/chat", content=json.dumps(msg), headers={"Content-Type": "text/plain"}).status_code == 415
    assert app.get("/api/status", headers={"Host": "evil.example"}).status_code == 403
    assert app.post("/api/shutdown", json={}, headers={"Origin": "https://evil.example"}).status_code == 403
    assert app.stopped == []


def test_shutdown_from_own_page(app):
    import time

    assert app.post("/api/shutdown", json={}, headers={"Origin": BASE}).status_code == 200
    time.sleep(0.5)
    assert app.stopped == [True]


# --- настройки и запуск ---


@pytest.mark.parametrize("ram, model", [(32, "qwen3:8b"), (15.8, "qwen3:8b"), (8, "qwen3:4b-instruct-2507-q4_K_M"),
                                       (7.9, "qwen3:4b-instruct-2507-q4_K_M"), (4, "qwen3:1.7b")])
def test_model_for_ram(ram, model):
    assert config.model_for_ram(ram) == model


def test_configure_writes_and_load_reads(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MIREA_ASSISTANT_HOME", str(tmp_path / "Иван Петров" / ".mirea-assistant"))
    monkeypatch.delenv("MIREA_ASSISTANT_MODEL", raising=False)
    assert config.Config.load().model == config.DEFAULT_MODEL
    assert cli.main(["configure", "--ram-gb", "16"]) == 0
    assert config.Config.load().model == "qwen3:8b"
    assert cli.main(["configure", "--model", "qwen3:1.7b", "--port", "9000"]) == 0
    cfg = config.Config.load()
    assert (cfg.model, cfg.port) == ("qwen3:1.7b", 9000)
    monkeypatch.setenv("MIREA_ASSISTANT_MODEL", "llama3.2")
    assert config.Config.load().model == "llama3.2"


def test_config_with_bom_or_garbage(tmp_path, monkeypatch):
    monkeypatch.setenv("MIREA_ASSISTANT_HOME", str(tmp_path))
    monkeypatch.delenv("MIREA_ASSISTANT_MODEL", raising=False)
    (tmp_path / "config.json").write_bytes(b'\xef\xbb\xbf{"model": "qwen3:8b"}')
    assert config.Config.load().model == "qwen3:8b"
    (tmp_path / "config.json").write_text("{", encoding="utf-8")
    assert config.Config.load().model == config.DEFAULT_MODEL


def test_second_launch_opens_existing_instance(monkeypatch):
    opened = []
    monkeypatch.setattr(cli, "running_instance", lambda port: True)
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: opened.append(url))
    assert cli.serve(PORT, open_browser=True) == 0
    assert opened == [f"http://127.0.0.1:{PORT}/"]


def test_busy_port_by_other_program(monkeypatch, capsys):
    monkeypatch.setattr(cli, "running_instance", lambda port: False)
    monkeypatch.setattr(cli, "port_free", lambda port: False)
    assert cli.serve(PORT, open_browser=False) == 1
    assert "--port 8766" in capsys.readouterr().err


def test_say_without_console(monkeypatch):
    monkeypatch.setattr("sys.stdout", None)
    monkeypatch.setattr("sys.stderr", None)
    cli.say("работает без консоли")
    cli.say("и ошибки тоже", error=True)
