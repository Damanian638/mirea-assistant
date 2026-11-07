"""Агент: локальная модель выбирает инструмент, программа считает, модель пересказывает.

Маленькие модели плохо считают даты и путаются в длинном JSON, поэтому:
календарь на три недели готов в системном промпте, аргументы нормализуются,
результаты инструментов передаются коротким текстом, а не сырым JSON.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable

from mirea_connect import services as S

from . import dates, slots
from .llm import LLMError, Ollama

logger = logging.getLogger(__name__)

MAX_ROUNDS = 5
WEEKDAY_NAMES = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
_KINDS = {
    "group": "group", "группа": "group", "groups": "group",
    "teacher": "teacher", "преподаватель": "teacher", "препод": "teacher", "teachers": "teacher",
    "classroom": "classroom", "аудитория": "classroom", "room": "classroom", "rooms": "classroom",
}
_WEEKDAYS = {"пн": 1, "вт": 2, "ср": 3, "чт": 4, "пт": 5, "сб": 6, "вс": 7}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "timetable",
            "description": "Расписание преподавателя, группы или аудитории на даты. Используй для вопросов «какие пары у…», «когда занята аудитория…».",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["teacher", "group", "classroom"]},
                    "name": {"type": "string", "description": "ФИО или «Фамилия И. О.», группа «ЭФБО-01-24», аудитория «А-9 (В-78)»"},
                    "date_from": {"type": "string", "description": "Первая дата: ГГГГ-ММ-ДД или слово, как сказал пользователь («завтра», «понедельник», «9 ноября»)"},
                    "days": {"type": "integer", "description": "Сколько дней показать, 1–14"},
                },
                "required": ["kind", "name", "date_from"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "free_rooms",
            "description": "Свободные аудитории в заданный день и время, можно в одном корпусе или проверить конкретные аудитории.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "ГГГГ-ММ-ДД или слово, как сказал пользователь («завтра», «понедельник»)"},
                    "start_time": {"type": "string", "description": "ЧЧ:ММ, начало пары по звонкам"},
                    "end_time": {"type": "string", "description": "ЧЧ:ММ, конец пары"},
                    "campus": {"type": "string", "description": "Корпус: В-78, В-86, С-20, МП-1, СГ-22"},
                    "rooms": {"type": "array", "items": {"type": "string"}, "description": "Конкретные аудитории для проверки"},
                },
                "required": ["date", "start_time", "end_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_reschedule_slots",
            "description": (
                "Найти, куда перенести пару: пары, в которые свободны преподаватель и все группы, и свободные аудитории. "
                "Используй для вопросов «куда перенести», «когда можно поставить отработку/консультацию»."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "teacher": {"type": "string", "description": "Преподаватель"},
                    "groups": {"type": "array", "items": {"type": "string"}, "description": "Группы, у которых пара"},
                    "date_from": {"type": "string", "description": "ГГГГ-ММ-ДД или слово («следующий понедельник»)"},
                    "date_to": {"type": "string", "description": "ГГГГ-ММ-ДД или слово («следующая суббота»), не дальше 21 дня от date_from"},
                    "campus": {"type": "string", "description": "Корпус, где нужна аудитория"},
                    "pairs": {"type": "array", "items": {"type": "integer"}, "description": "Подходящие номера пар 1–7"},
                    "weekdays": {"type": "array", "items": {"type": "integer"}, "description": "Дни недели 1=пн … 6=сб"},
                },
                "required": ["teacher", "date_from"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Найти точное название группы, ФИО преподавателя или аудитории по части названия.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["teacher", "group", "classroom"]},
                    "query": {"type": "string"},
                },
                "required": ["kind", "query"],
            },
        },
    },
]


# --- нормализация аргументов от маленькой модели ---


def _str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _list(v: Any) -> list[str]:
    if v is None or v == "":
        return []
    if isinstance(v, str):
        return [x.strip() for x in v.replace(";", ",").split(",") if x.strip()]
    if isinstance(v, (list, tuple)):
        return [str(x).strip() for x in v if str(x).strip()]
    return [str(v)]


def _ints(v: Any, names: dict[str, int] | None = None) -> list[int] | None:
    out = []
    for x in _list(v):
        key = x.casefold()[:2]
        if names and key in names:
            out.append(names[key])
        else:
            try:
                out.append(int(float(x)))
            except ValueError:
                raise S.ServiceError(f"Ожидалось число, получено «{x}».") from None
    return out or None


def _kind(v: Any) -> str:
    k = _KINDS.get((_str(v) or "").casefold())
    if not k:
        raise S.ServiceError("kind должен быть teacher, group или classroom.")
    return k


# --- компактный текст результатов для модели ---


def _fmt_day(iso: str) -> str:
    dt = datetime.fromisoformat(iso)
    return f"{dt.strftime('%d.%m')} {slots.WEEKDAYS[dt.weekday()]}"


def format_timetable(data: dict) -> str:
    events = data["events"]
    start = f"{data['from']} ({_fmt_day(data['from'])})" if data.get("from") else "всё доступное"
    head = f"Расписание: {data['name']} ({data['kind']}), с {start} на {data.get('days')} дн."
    if not events:
        return head + "\nЗанятий нет."
    lines = [head]
    for e in events[:60]:
        loc = f" — {e['location']}" if e.get("location") else ""
        extra = f" [{e['description'][:80]}]" if e.get("description") else ""
        end = e["end"][11:16] if e.get("end") else "?"
        lines.append(f"{_fmt_day(e['start'])} {e['start'][11:16]}–{end} {e['summary']}{loc}{extra}")
    if len(events) > 60:
        lines.append(f"… и ещё {len(events) - 60} занятий (сузьте диапазон дат)")
    return "\n".join(lines)


def format_free_rooms(data: dict) -> str:
    head = f"{data['date']} ({_fmt_day(data['date'])}) {data['from']}–{data['to']}"
    if "rooms" in data:
        return head + "\n" + "\n".join(f"{r['room']}: {'свободна' if r['free'] else 'занята'}" for r in data["rooms"])
    rooms = [f"{r['room']}" + (f" ({r['campus']})" if r.get("campus") and not data.get("campus") else "") for r in data["free_rooms"]]
    text = f"{head}, корпус {data.get('campus') or 'любой'}: свободно {data['total_free']}\n" + ", ".join(rooms[:40])
    if data["total_free"] > 40:
        text += f" … и ещё {data['total_free'] - 40}"
    if data.get("available_campuses") is not None:
        text += f"\nВ этом корпусе ничего не найдено. Корпуса со свободными аудиториями: {', '.join(data['available_campuses'])}"
    return text


def format_slots(data: dict) -> str:
    head = (f"Перенос: преподаватель {data['teacher']}, группы: {', '.join(data['groups']) or 'не указаны'}, "
            f"период {data['period']}, корпус {data.get('campus') or 'любой'}. Найдено вариантов: {data['slots_found']}.")
    if not data["slots"]:
        return head + "\nСвободных для всех пар в этот период нет — предложите расширить период или убрать ограничения."
    lines = [head]
    for s in data["slots"][:15]:
        rooms = ", ".join(s["free_rooms"]) if s["free_rooms"] else "аудитории не проверялись"
        total = f" (всего свободно {s['free_rooms_total']})" if s.get("free_rooms_total") else ""
        notes = f" ⚠ {'; '.join(s['notes'])}" if s["notes"] else ""
        lines.append(f"{_fmt_day(s['date'])} ({s['date']}), {s['pair']}-я пара {s['time']}: {rooms}{total}{notes}")
    if len(data["slots"]) > 15:
        lines.append(f"… и ещё {len(data['slots']) - 15} вариантов")
    lines.append(data["note"])
    return "\n".join(lines)


def format_search(items: list[dict]) -> str:
    if not items:
        return "Ничего не найдено."
    return "Найдено: " + "; ".join(i["name"] for i in items[:20])


# --- исполнение инструментов ---


def _date(v: Any) -> str | None:
    return dates.resolve(v, datetime.now(slots.MSK))


async def run_tool(name: str, args: dict) -> str:
    if name == "timetable":
        days = int(args.get("days") or 7)
        data = await S.timetable(_kind(args.get("kind")), _str(args.get("name")) or "", max(1, min(days, 14)),
                                 _date(args.get("date_from")))
        return format_timetable(data)
    if name == "free_rooms":
        data = await S.free_rooms(_list(args.get("rooms")) or None, _date(args.get("date")),
                                  _str(args.get("start_time")) or "", _str(args.get("end_time")) or "",
                                  _str(args.get("campus")))
        return format_free_rooms(data)
    if name == "find_reschedule_slots":
        data = await slots.find_slots(
            _str(args.get("teacher")) or "",
            _list(args.get("groups")),
            _date(args.get("date_from")),
            _date(args.get("date_to")),
            _str(args.get("campus")),
            _ints(args.get("pairs")),
            _ints(args.get("weekdays"), _WEEKDAYS),
        )
        return format_slots(data)
    if name == "search":
        return format_search(await S.search(_kind(args.get("kind")), _str(args.get("query")) or ""))
    raise S.ServiceError(f"Неизвестный инструмент {name}.")


# --- системный промпт ---


def calendar_lines(today: datetime, days: int = 21) -> list[str]:
    out = []
    for i in range(-1, days):
        d = (today + timedelta(days=i)).date()
        mark = {-1: " (вчера)", 0: " (сегодня)", 1: " (завтра)"}.get(i, "")
        week = " — следующая неделя" if d.weekday() == 0 and i > 0 else ""
        out.append(f"{WEEKDAY_NAMES[d.weekday()]} {d.strftime('%d.%m.%Y')} = {d.isoformat()}{mark}{week}")
    return out


def system_prompt(me: str | None = None, now: datetime | None = None) -> str:
    now = now or datetime.now(slots.MSK)
    bells = "; ".join(f"{n}-я {a}–{b}" for n, (a, b) in slots.BELLS.items())
    who = (f"Пользователь — преподаватель {me}. «Мои пары», «у меня» — это расписание преподавателя {me}."
           if me else "Если пользователь спрашивает про «мои пары», а ФИО неизвестно — спроси ФИО.")
    return f"""Ты — помощник преподавателя РТУ МИРЭА. Отвечай по-русски, коротко и по делу, списком, если вариантов несколько.

Правила:
- Расписание, аудитории и даты бери ТОЛЬКО из инструментов. Ничего не придумывай.
- Даты передавай как сказал пользователь («завтра», «понедельник», «9 ноября») или ГГГГ-ММ-ДД из календаря ниже. В ответе называй дату и день недели из результата инструмента.
- Если инструмент вернул ОШИБКУ со списком вариантов — покажи варианты и спроси, кого/что имели в виду.
- Если не хватает данных (кто, какая группа, когда) — задай один уточняющий вопрос.
- Для переноса пары и поиска времени для отработки используй find_reschedule_slots. Напомни, что перенос согласуется с учебным отделом.
- Свободная по расписанию аудитория может быть физически закрыта — упомяни это, если предлагаешь аудиторию.
- {who}

Время московское. Звонки: {bells}.
Корпуса: «Вернадского 78» = В-78, «Вернадского 86» = В-86, «Стромынка» = С-20, «Малая Пироговская» = МП-1, «Соколиная гора» = СГ-22.

Календарь:
""" + "\n".join(calendar_lines(now))


# --- цикл ---


@dataclass
class Step:
    tool: str
    args: dict
    ok: bool
    result: str

    def public(self) -> dict:
        return {"tool": self.tool, "args": self.args, "ok": self.ok, "result": self.result[:2000]}


@dataclass
class Answer:
    text: str
    steps: list[Step] = field(default_factory=list)


ToolRunner = Callable[[str, dict], Awaitable[str]]


async def answer(llm: Ollama, history: list[dict], me: str | None = None, tool_runner: ToolRunner = run_tool) -> Answer:
    """history — сообщения чата [{role: user|assistant, content}], последнее — вопрос пользователя."""
    clean = [{"role": m["role"], "content": str(m.get("content", ""))[:4000]}
             for m in history[-12:] if m.get("role") in ("user", "assistant")]
    if not clean or clean[-1]["role"] != "user":
        raise ValueError("последним должно быть сообщение пользователя")
    messages = [{"role": "system", "content": system_prompt(me)}, *clean]
    steps: list[Step] = []
    seen: set[str] = set()

    for _ in range(MAX_ROUNDS):
        reply = await llm.chat(messages, TOOLS)
        if not reply.tool_calls:
            return Answer(reply.content or "Не получилось сформулировать ответ. Переформулируйте вопрос.", steps)
        messages.append({"role": "assistant", "content": reply.content,
                         "tool_calls": [{"function": {"name": c.name, "arguments": c.arguments}} for c in reply.tool_calls]})
        for call in reply.tool_calls:
            key = call.name + json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)
            if key in seen:  # маленькие модели иногда зацикливаются на одном вызове
                result, ok = "Этот запрос уже выполнен, результат выше. Ответь пользователю.", True
            else:
                seen.add(key)
                try:
                    result, ok = await tool_runner(call.name, call.arguments), True
                except LLMError:
                    raise
                except Exception as e:  # ошибка — модели, чтобы она объяснила или переспросила
                    result, ok = f"ОШИБКА: {S.friendly_error(e)}", False
            steps.append(Step(call.name, call.arguments, ok, result))
            messages.append({"role": "tool", "tool_name": call.name, "content": result})

    # лимит раундов: просим ответить по уже полученным данным
    messages.append({"role": "user", "content": "Ответь на исходный вопрос по уже полученным данным, без новых запросов."})
    final = await llm.chat(messages)
    return Answer(final.content or "Не получилось получить ответ. Уточните вопрос.", steps)
