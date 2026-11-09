"""Модель через интернет (OpenAI-совместимый API, по умолчанию OpenRouter) и её выбор в настройках."""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from starlette.testclient import TestClient

from mirea_assistant import agent, backends, cli, config, web
from mirea_assistant.api_llm import OPENROUTER, ApiModel, to_openai
from mirea_assistant.llm import LLMError, Ollama

PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"
KEY = "sk-or-v1-0123456789abcdef"


def test_messages_converted_to_openai_with_call_ids():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "timetable", "arguments": {"kind": "teacher"}}},
            {"function": {"name": "search", "arguments": {"query": "Ив"}}}]},
        {"role": "tool", "tool_name": "timetable", "content": "r1"},
        {"role": "tool", "tool_name": "search", "content": "r2"},
    ]
    out = to_openai(msgs)
    calls = out[2]["tool_calls"]
    assert [c["id"] for c in calls] == ["call_1", "call_2"] and calls[0]["type"] == "function"
    assert json.loads(calls[1]["function"]["arguments"]) == {"query": "Ив"}
    assert out[3] == {"role": "tool", "tool_call_id": "call_1", "content": "r1"}
    assert out[4]["tool_call_id"] == "call_2" and out[2]["content"] is None


@respx.mock
async def test_chat_request_and_tool_call_parsing():
    route = respx.post(f"{OPENROUTER}/chat/completions").mock(return_value=httpx.Response(200, json={
        "choices": [{"message": {"content": None, "tool_calls": [
            {"id": "x", "type": "function", "function": {"name": "free_rooms", "arguments": '{"date": "завтра", "pair": 3}'}}]}}]}))
    llm = ApiModel("google/gemini-2.5-flash", KEY)
    reply = await llm.chat([{"role": "user", "content": "q"}], agent.TOOLS)
    assert reply.tool_calls[0].name == "free_rooms" and reply.tool_calls[0].arguments == {"date": "завтра", "pair": 3}
    req = route.calls[0].request
    body = json.loads(req.content)
    assert req.headers["authorization"] == f"Bearer {KEY}"
    assert body["model"] == "google/gemini-2.5-flash" and body["tools"] == agent.TOOLS
    # в запросах ФИО и расписание: только провайдеры, которые не хранят данные
    assert body["provider"] == {"data_collection": "deny"} and "reasoning" not in body
    await llm.close()


@respx.mock
async def test_other_provider_gets_no_openrouter_fields():
    route = respx.post("https://api.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "Ответ"}}]}))
    llm = ApiModel("m", KEY, "https://api.example.com/v1/")
    assert (await llm.chat([{"role": "user", "content": "q"}], think=True)).content == "Ответ"
    body = json.loads(route.calls[0].request.content)
    assert "provider" not in body and "reasoning" not in body


@pytest.mark.parametrize(
    "response, fragment",
    [
        (httpx.Response(401, json={"error": {"message": "No auth"}}), "Ключ API не подошёл"),
        (httpx.Response(402, json={"error": {"message": "Insufficient credits"}}), "закончились средства"),
        (httpx.Response(403, json={"error": "Access denied by security policy."}), "VPN"),
        (httpx.Response(404, json={"error": {"message": "No endpoints found"}}), "недоступна у провайдера"),
        (httpx.Response(429, json={}), "подождать"),
        (httpx.Response(200, text="<html>"), "непонятный ответ"),
        (httpx.ConnectError("x"), "Не удаётся связаться"),
    ],
)
@respx.mock
async def test_api_errors_are_readable(response, fragment):
    route = respx.post(f"{OPENROUTER}/chat/completions")
    if isinstance(response, Exception):
        route.mock(side_effect=response)
    else:
        route.mock(return_value=response)
    with pytest.raises(LLMError, match=fragment):
        await ApiModel("m", KEY).chat([{"role": "user", "content": "q"}])


@respx.mock
async def test_status_cached_and_models_listed():
    route = respx.get(f"{OPENROUTER}/models").mock(return_value=httpx.Response(200, json={"data": [
        {"id": "google/gemini-2.5-flash", "supported_parameters": ["tools"], "pricing": {"prompt": "0.0000003", "completion": "0.0000025"}},
        {"id": "a/no-tools", "supported_parameters": ["temperature"]},
        {"id": "z/other", "supported_parameters": ["tools"]},
    ]}))
    llm = ApiModel("google/gemini-2.5-flash", KEY)
    st = await llm.status()
    assert st["provider"] == "api" and st["model_ready"] is True and st["error"] is None
    await llm.status()
    assert route.call_count == 1  # страница спрашивает статус каждые 15 с — в API ходим не чаще раза в 5 минут
    models = await llm.models()
    assert [m["id"] for m in models] == ["google/gemini-2.5-flash", "z/other"]  # без вызова инструментов — нельзя
    assert models[0]["recommended"] and models[0]["price_per_million"] == pytest.approx((0.3, 2.5))
    missing = ApiModel("x/gone", KEY)
    assert "нет у провайдера" in (await missing.status())["error"]


def test_config_keeps_api_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("MIREA_ASSISTANT_HOME", str(tmp_path))
    monkeypatch.delenv("MIREA_ASSISTANT_API_KEY", raising=False)
    monkeypatch.delenv("MIREA_ASSISTANT_MODEL", raising=False)
    assert cli.main(["configure", "--provider", "api", "--api-key", KEY, "--api-model", "openai/gpt-4.1-mini"]) == 0
    cfg = config.Config.load()
    assert (cfg.provider, cfg.api_key, cfg.api_model, cfg.api_base) == ("api", KEY, "openai/gpt-4.1-mini", OPENROUTER)
    assert isinstance(backends.make_llm(cfg), ApiModel)
    # обновление установщиком (configure --ram-gb) настройки API не теряет
    assert cli.main(["configure", "--ram-gb", "16"]) == 0
    assert config.Config.load().provider == "api"
    cfg.api_key = ""
    assert isinstance(backends.make_llm(cfg), Ollama)  # без ключа — локальная модель


class Stub:
    def __init__(self, name):
        self.name, self.closed = name, False

    async def status(self):
        return {"provider": self.name, "model": self.name, "model_ready": True, "ollama": True}

    async def close(self):
        self.closed = True

    async def models(self):
        return [{"id": "google/gemini-2.5-flash", "recommended": True, "price_per_million": None}]


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("MIREA_ASSISTANT_HOME", str(tmp_path))
    monkeypatch.delenv("MIREA_ASSISTANT_API_KEY", raising=False)
    made, local = [], []

    def make(cfg):
        made.append(cfg)
        return Stub(cfg.provider)

    first = Stub("ollama")
    application = web.create_app(first, PORT, on_local=lambda: local.append(True), make_llm=make)
    with TestClient(application, base_url=BASE) as c:
        c.made, c.local, c.first = made, local, first
        yield c


def test_switch_to_api_and_back_without_restart(app):
    assert app.get("/api/settings").json()["provider"] == "ollama"
    r = app.post("/api/settings", json={"provider": "api", "api_key": KEY, "api_model": "openai/gpt-4.1-mini"})
    assert r.status_code == 200
    view = r.json()
    assert view["api_key_set"] is True and view["api_key_hint"] == "…cdef" and KEY not in json.dumps(view)
    assert app.first.closed and app.get("/api/status").json()["provider"] == "api"
    assert app.get("/api/models").json()["models"][0]["id"] == "google/gemini-2.5-flash"
    # ключ не обязательно вводить заново
    assert app.post("/api/settings", json={"provider": "api", "api_model": "google/gemini-2.5-flash"}).status_code == 200
    assert config.Config.load().api_key == KEY
    assert app.post("/api/settings", json={"provider": "ollama"}).json()["provider"] == "ollama"
    assert app.local == [True]  # Ollama запускается, если она не работает


@pytest.mark.parametrize(
    "body, fragment",
    [
        ({"provider": "api"}, "нужен ключ"),
        ({"provider": "api", "api_key": "sk or with spaces"}, "Ключ API выглядит неправильно"),
        ({"provider": "api", "api_key": KEY, "api_base": "http://evil.example.com/v1"}, "https://"),
        ({"provider": "api", "api_key": KEY, "api_model": "m; rm -rf"}, "имя модели"),
        ({"provider": "cloud"}, "Некорректные"),
    ],
)
def test_settings_validated(app, body, fragment):
    r = app.post("/api/settings", json=body)
    assert r.status_code == 400 and fragment in r.json()["error"]


def test_settings_protected_from_other_sites(app):
    r = app.post("/api/settings", json={"provider": "api", "api_key": KEY}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    r = app.post("/api/settings", content=json.dumps({"provider": "api", "api_key": KEY}),
                 headers={"Content-Type": "text/plain"})
    assert r.status_code == 415


def test_page_has_provider_choice():
    page = (web.resources.files("mirea_assistant") / "chat.html").read_text(encoding="utf-8")
    assert 'name="provider" value="api"' in page and 'id="api-key" type="password"' in page
    assert "/api/settings" in page and "/api/models" in page
