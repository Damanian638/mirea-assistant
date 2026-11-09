"""Агент: локальная модель выбирает инструмент, программа считает, модель пересказывает.

Маленькие модели плохо считают даты и путаются в длинном JSON, поэтому:
календарь на три недели готов в системном промпте, аргументы нормализуются,
результаты инструментов передаются коротким текстом, а не сырым JSON.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable

from mirea_connect import services as S

from . import dates, lessons, slots
from .llm import LLMError, Ollama

logger = logging.getLogger(__name__)

MAX_ROUNDS = 5
WEEKDAY_NAMES = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
_KINDS = {
    "group": "group", "группа": "group", "groups": "group",
    "teacher": "teacher", "преподаватель": "teacher", "препод": "teacher", "teachers": "teacher",
    "classroom": "classroom", "аудитория": "classroom", "room": "classroom", "rooms": "classroom",
}
# Первые две буквы сокращения или полного названия: «четверг»[:2] == «че», а не «чт».
_WEEKDAYS = {"пн": 1, "по": 1, "вт": 2, "ср": 3, "чт": 4, "че": 4, "пт": 5, "пя": 5, "сб": 6, "су": 6, "вс": 7, "во": 7}

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
                    "pair": {"type": "integer", "description": "Номер пары 1–7, если пользователь назвал пару — тогда время не нужно"},
                    "start_time": {"type": "string", "description": "ЧЧ:ММ, если пользователь назвал время, а не пару"},
                    "end_time": {"type": "string", "description": "ЧЧ:ММ"},
                    "campus": {"type": "string", "description": "Корпус: В-78, В-86, С-20, МП-1, СГ-22"},
                    "rooms": {"type": "array", "items": {"type": "string"}, "description": "Конкретные аудитории для проверки"},
                    "building": {"type": "string", "description": "Буква корпуса внутри кампуса, если пользователь её назвал: «корпус Б» → «Б»"},
                    "floor": {"type": "integer", "description": "Этаж, если пользователь его назвал"},
                },
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_reschedule_slots",
            "description": (
                "Куда перенести пару или когда поставить отработку. Если пользователь говорит, КАКУЮ пару переносит "
                "(дата и номер пары), передай lesson_date и lesson_pairs — программа сама найдёт группы, корпус, "
                "подберёт столько пар подряд и отсортирует варианты по удобству. Группы и корпус в этом случае не передавай."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "teacher": {"type": "string", "description": "Преподаватель; для «у меня» — ФИО пользователя"},
                    "lesson_date": {"type": "string", "description": "Дата пары, которую переносим («9 ноября», «понедельник», ГГГГ-ММ-ДД)"},
                    "lesson_pairs": {"type": "array", "items": {"type": "integer"}, "description": "Номера переносимых пар в тот день, например [1, 2]"},
                    "date_from": {"type": "string", "description": "С какой даты искать новое время (если пользователь назвал период)"},
                    "date_to": {"type": "string", "description": "По какую дату искать (не дальше 21 дня от date_from)"},
                    "allowed_pairs": {"type": "array", "items": {"type": "integer"}, "description": "Только если пользователь сам ограничил НОВОЕ время номерами пар («можно только на 3–5 паре»)"},
                    "weekdays": {"type": "array", "items": {"type": "integer"}, "description": "Только если пользователь ограничил дни недели: 1=пн … 6=сб"},
                    "groups": {"type": "array", "items": {"type": "string"}, "description": "Группы — только если переносимая пара не указана (отработка для конкретных групп)"},
                    "campus": {"type": "string", "description": "Кампус (В-78, В-86, С-20, МП-1, СГ-22) — только если пользователь его назвал"},
                    "building": {"type": "string", "description": "Буква корпуса внутри кампуса: «корпус Б» → «Б»"},
                    "floor": {"type": "integer", "description": "Этаж аудитории, если пользователь его назвал"},
                },
                "required": ["teacher"],
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


def format_lessons(tt: lessons.Timetable, start: datetime, days: int) -> str:
    end = start + timedelta(days=days)
    items = [x for x in tt.lessons if start <= x.start < end]
    head = f"Расписание: {tt.name} ({tt.kind}), с {_fmt_day(start.date().isoformat())} {start.date().isoformat()} на {days} дн."
    if not items:
        return head + "\nЗанятий нет."
    lines = [head]
    for x in items[:60]:
        pair = slots.pair_of(x)
        parts = [f"{_fmt_day(x.day.isoformat())} {pair}-я пара {x.start.strftime('%H:%M')}–{x.end.strftime('%H:%M')}" if pair
                 else f"{_fmt_day(x.day.isoformat())} {x.start.strftime('%H:%M')}–{x.end.strftime('%H:%M')}",
                 x.subject + (f" ({x.kind_ru})" if x.kind_ru else "")]
        if x.rooms:
            parts.append(x.location())
        if x.groups and tt.kind != "group":
            parts.append("группы: " + ", ".join(x.groups))
        if x.teachers and tt.kind != "teacher":
            parts.append("преподаватель: " + ", ".join(x.teachers))
        lines.append(" — ".join(parts))
    if len(items) > 60:
        lines.append(f"… и ещё {len(items) - 60} занятий (сузьте диапазон дат)")
    return "\n".join(lines)


def format_free_rooms(data: dict) -> str:
    head = f"{data['date']} ({_fmt_day(data['date'])}) {data['from']}–{data['to']}"
    if "rooms" in data:
        return head + "\n" + "\n".join(f"{r['room']}: {'свободна' if r['free'] else 'занята'}" for r in data["rooms"])
    rooms = [f"{r['room']}" + (f" ({r['campus']})" if r.get("campus") and not data.get("campus") else "") for r in data["free_rooms"]]
    where = data.get("place") or f"корпус {data.get('campus') or 'любой'}"
    text = f"{head}, {where}: свободных аудиторий {data['total_free']}\n" + ", ".join(rooms[:40])
    if data["total_free"] > 40:
        text += f" … и ещё {data['total_free'] - 40}"
    if data.get("available_campuses") is not None:
        text += f"\nВ этом корпусе ничего не найдено. Корпуса со свободными аудиториями: {', '.join(data['available_campuses'])}"
    return text


def format_slots(data: dict) -> str:
    lines = []
    if data.get("moving"):
        lines.append(f"Переносим: {data['moving_date']}: {data['moving']}.")
    lines.append(f"Преподаватель {data['teacher']}; группы: {', '.join(data['groups']) or 'не указаны'}; "
                 f"аудитории: {slots.place_text(data.get('campus'), data.get('building'), data.get('floor')) or 'любой корпус'}; "
                 f"пар подряд: {data['pairs_in_a_row']}; "
                 f"период поиска {data['period']}. Найдено вариантов: {data['slots_found']}.")
    if not data["slots"]:
        lines.append("Подходящего времени, когда свободны все, нет — предложите расширить период поиска.")
        return "\n".join(lines)
    lines.append("Варианты, от самых удобных:")
    for i, sl in enumerate(data["slots"][:10], 1):
        pairs = "–".join(map(str, (sl["pairs"][0], sl["pairs"][-1]))) if len(sl["pairs"]) > 1 else str(sl["pairs"][0])
        rooms = ", ".join(sl["free_rooms"]) if sl["free_rooms"] else "аудитории не проверялись"
        same = " (та же аудитория свободна)" if sl.get("same_room") else ""
        total = f", всего свободных аудиторий {sl['free_rooms_total']}" if sl.get("free_rooms_total") else ""
        notes = f" — {'; '.join(sl['notes'])}" if sl["notes"] else ""
        lines.append(f"{i}. {_fmt_day(sl['date'])} ({sl['date']}), {pairs} пара {sl['time']}: {rooms}{same}{total}{notes}")
    if len(data["slots"]) > 10:
        lines.append(f"… и ещё {len(data['slots']) - 10} вариантов")
    lines.append(data["note"])
    return "\n".join(lines)


def format_search(items: list[dict]) -> str:
    if not items:
        return "Ничего не найдено."
    return "Найдено: " + "; ".join(i["name"] for i in items[:20])


# --- исполнение инструментов ---


def clock() -> datetime:
    """Текущее время по Москве; экзамен модели подменяет его, чтобы «понедельник» был всегда одним днём."""
    return datetime.now(slots.MSK)


def _date(v: Any) -> str | None:
    return dates.resolve(v, clock())


# Корпус словами → код из расписания. Поиск аудиторий понимает только коды: «Вернадского 78» дал бы 0 свободных.
_CAMPUSES = [
    (("вернадского 78", "вернадского, 78", "вернадского, д.78", "в78", "в 78"), "В-78"),
    (("вернадского 86", "вернадского, 86", "вернадского, д.86", "в86", "в 86"), "В-86"),
    (("стромынк", "с20", "с 20"), "С-20"),
    (("пироговск", "мп1", "мп 1"), "МП-1"),
    (("соколин", "сг22", "сг 22"), "СГ-22"),
]


def _campus(v: Any) -> str | None:
    raw = _str(v)
    if not raw:
        return None
    low = S._norm_letters(raw).replace("-", "").replace("пр.", "").replace("проспект", "").strip()
    compact = low.replace(" ", "")
    for keys, code in _CAMPUSES:
        if any(k.replace(" ", "") in compact for k in keys):
            return code
    if low in ("любой", "все", "any", "всё"):
        return None
    return raw


def _building(v: Any) -> str | None:
    """«Б», «корпус Б», «б» → «Б». Коды кампусов («В-78») сюда не относятся."""
    raw = (_str(v) or "").casefold().replace("корпус", "").strip(" «»\"'.")
    return raw.upper() if len(raw) == 1 and raw.isalpha() else None


def _floor(v: Any) -> int | None:
    try:
        n = int(float(str(v).strip())) if _str(v) else None
    except ValueError:
        return None
    return n if n and 1 <= n <= 30 else None


def _filter_rooms(data: dict, building: str | None, floor: int | None) -> dict:
    if not (building or floor) or "free_rooms" not in data:
        return data
    fitting = [r for r in data["free_rooms"] if slots.room_fits(r["room"], building, floor)]
    return {**data, "free_rooms": fitting, "total_free": len(fitting),
            "place": slots.place_text(data.get("campus"), building, floor)}


async def run_tool(name: str, args: dict) -> str:
    if name == "timetable":
        days = int(args.get("days") or 7)
        if days < 1:
            raise S.ServiceError("days должно быть от 1 до 14.")
        start = S._start_of(_date(args.get("date_from")))
        tt = await lessons.fetch(_kind(args.get("kind")), _str(args.get("name")) or "")
        return format_lessons(tt, start, min(days, 14))
    if name == "free_rooms":
        data = await S.free_rooms(_list(args.get("rooms")) or None, _date(args.get("date")),
                                  _str(args.get("start_time")) or "", _str(args.get("end_time")) or "",
                                  _campus(args.get("campus")))
        return format_free_rooms(_filter_rooms(data, _building(args.get("building")), _floor(args.get("floor"))))
    if name == "find_reschedule_slots":
        data = await slots.find_slots(
            _str(args.get("teacher")) or "",
            lesson_date=_date(args.get("lesson_date")),
            lesson_pairs=_ints(args.get("lesson_pairs")),
            groups=_list(args.get("groups")),
            date_from=_date(args.get("date_from")),
            date_to=_date(args.get("date_to")),
            campus=_campus(args.get("campus")),
            allowed_pairs=_ints(args.get("allowed_pairs") or args.get("pairs")),
            weekdays=_ints(args.get("weekdays"), _WEEKDAYS),
            building=_building(args.get("building")),
            floor=_floor(args.get("floor")),
        )
        return format_slots(data)
    if name == "search":
        return format_search(await S.search(_kind(args.get("kind")), _str(args.get("query")) or ""))
    raise S.ServiceError(f"Неизвестный инструмент {name}.")


# Имя-пример в промпте, когда ФИО пользователя неизвестно. Экзамен: qwen3:8b подставила пример
# («Иванов Иван Иванович») в ответ на «какие у меня пары» — такой вызов программа не выполняет.
EXAMPLE_TEACHER = "Петров Пётр Петрович"
_MY_WORDS = re.compile(r"\b(у меня|мне|меня|я|мо(й|я|ё|е|и|ю|ей|его|ему|ем|им|их|ими))\b", re.I)

_ME_WORDS = {"я", "мне", "меня", "мой", "мои", "моё", "мое", "моих", "me", "user", "пользователь"}


def _means_me(value: Any) -> bool:
    """Пусто, «я/мои» или заготовка из примера («<ФИО пользователя>») — значит, речь о пользователе.

    Прогон на Windows: модель qwen3:0.6b передала teacher="<ФИО пользователя>" дословно из примера."""
    v = (_str(value) or "").casefold()
    return not v or v in _ME_WORDS or "<" in v or "фио" in v or "пользовател" in v


def needs_my_name(name: str, args: dict, me: str | None, user_text: str) -> bool:
    """Вопрос про «мои пары», а ФИО пользователя неизвестно — сначала спросить его."""
    if me or name not in ("timetable", "find_reschedule_slots"):
        return False
    if name == "timetable" and _KINDS.get((_str(args.get("kind")) or "teacher").casefold()) != "teacher":
        return False
    who = args.get("teacher" if name == "find_reschedule_slots" else "name")
    example = S._norm(EXAMPLE_TEACHER)
    v = (_str(who) or "").casefold()
    if v and (S._norm(v).startswith(example[:6]) or "<" in v or "фио" in v):
        return True  # имя из примера или заготовка — настоящим ФИО быть не может
    return _means_me(who) and bool(_MY_WORDS.search(user_text or ""))


def with_me(name: str, args: dict, me: str | None) -> dict:
    """«У меня» без ФИО: подставить преподавателя из настроек."""
    if not me:
        return args
    if name == "find_reschedule_slots" and _means_me(args.get("teacher")):
        return {**args, "teacher": me}
    if name == "timetable" and _means_me(args.get("name")) and _KINDS.get((_str(args.get("kind")) or "teacher").casefold()) == "teacher":
        return {**args, "kind": "teacher", "name": me}
    return args


def fix_dates(name: str, args: dict, user_dates: list[str]) -> dict:
    """Если пользователь назвал ровно один день, а модель передала другой — верна дата пользователя."""
    if len(user_dates) != 1:
        return args
    wanted = user_dates[0]
    key = {"timetable": "date_from", "free_rooms": "date"}.get(name)
    if name == "find_reschedule_slots" and _str(args.get("lesson_date")):
        # Дата переносимой пары: ошибка модели здесь молча переносит другое занятие с другой группой.
        # Если названный день — это новое время (date_from/date_to), дату пары не трогаем.
        if wanted not in (_date(args.get("date_from")), _date(args.get("date_to"))):
            key = "lesson_date"
    if not key:
        return args
    given = _date(args.get(key))
    if given == wanted:
        return args
    fixed = dict(args)
    fixed[key] = wanted
    logger.info("date corrected for %s: %s -> %s", name, args.get(key), wanted)
    return fixed


def fix_pair_time(name: str, args: dict, user_pairs: list[int], user_text: str) -> dict:
    """free_rooms: номер пары → время по звонкам. Если пользователь назвал одну пару, верна она.

    Экзамен qwen3:8b: «на 7 паре» → модель передала время 5-й пары."""
    if name != "free_rooms":
        return args
    out = dict(args)
    pair = out.pop("pair", None)
    try:
        pair = int(pair) if pair not in (None, "") else None
    except (TypeError, ValueError):
        pair = None
    explicit_time = re.search(r"\b\d{1,2}:\d{2}\b", user_text or "")
    if len(user_pairs) == 1 and not explicit_time:
        pair = user_pairs[0]
    if pair in slots.BELLS:
        start, end = slots.BELLS[pair]
        if (out.get("start_time"), out.get("end_time")) != (start, end):
            logger.info("pair time for free_rooms: %s-%s -> pair %s", out.get("start_time"), out.get("end_time"), pair)
        out["start_time"], out["end_time"] = start, end
    return out


def effective_args(name: str, args: dict, me: str | None, user_dates: list[str],
                   user_pairs: list[int] | None = None, user_text: str = "") -> dict:
    """Аргументы, с которыми программа реально вызовет инструмент (после подстановки ФИО и исправления даты)."""
    out = fix_dates(name, with_me(name, args, me), user_dates)
    out = fix_pair_time(name, out, user_pairs or [], user_text)
    out = fix_place(name, out, user_text)
    out = drop_invented_period(name, out, user_dates, user_text)
    if _str(out.get("campus")):
        out = {**out, "campus": _campus(out["campus"])}
    return out


def fix_place(name: str, args: dict, user_text: str) -> dict:
    """Корпус-буква и этаж считает программа. Экзамен (скриншот): «в корпусе Б на втором этаже» —
    модель передала campus="Б", а этаж потеряла; предложены аудитории А-128-а, А-16…"""
    if name not in ("free_rooms", "find_reschedule_slots") or _list(args.get("rooms")):
        return args
    out = dict(args)
    letter = _building(out.get("campus"))
    if letter:  # «Б» — корпус внутри кампуса, а не кампус
        out.pop("campus")
        out.setdefault("building", letter)
    building, floor = dates.mentioned_place(user_text)
    if building:
        out["building"] = building
    if floor:
        out["floor"] = floor
    if out.get("building") and not _str(out.get("campus")) and "вернадск" in (user_text or "").casefold():
        out["campus"] = "В-78" if "86" not in user_text else "В-86"
    return out


# \bнедел: не «понедельник»
_PERIOD_WORDS = re.compile(r"\bнедел|месяц|период|\bдо\b|\bпо\b|\bс\s+\d|ближайш|числ", re.I)


def drop_invented_period(name: str, args: dict, user_dates: list[str], user_text: str) -> dict:
    """Срок поиска, который пользователь не называл, не ограничивает перенос.
    Скриншот: «перенести 1 и 2 пары в понедельник» — модель сама поставила date_from=date_to=10.11,
    и вариантов осталось два, оба вечером."""
    if name != "find_reschedule_slots" or not (_str(args.get("date_from")) or _str(args.get("date_to"))):
        return args
    given = {_date(args.get("date_from")), _date(args.get("date_to"))} - {None}
    if given & set(user_dates) or _PERIOD_WORDS.search(user_text or ""):
        return args
    logger.info("invented search period dropped: %s - %s", args.get("date_from"), args.get("date_to"))
    return {k: v for k, v in args.items() if k not in ("date_from", "date_to")}


# --- системный промпт ---


def calendar_lines(today: datetime, days: int = 21) -> list[str]:
    out = []
    mondays = 0
    for i in range(-1, days):
        d = (today + timedelta(days=i)).date()
        mark = {-1: " (вчера)", 0: " (сегодня)", 1: " (завтра)"}.get(i, "")
        week = ""
        if d.weekday() == 0 and i > 0:
            # Пометка только у ближайшего понедельника: раньше «следующая неделя» стояла у каждого,
            # и модель считала «следующую неделю» с 16 ноября вместо 9-го.
            mondays += 1
            week = {1: " — начало следующей недели", 2: " — начало недели через одну"}.get(mondays, "")
        out.append(f"{WEEKDAY_NAMES[d.weekday()]} {d.strftime('%d.%m.%Y')} = {d.isoformat()}{mark}{week}")
    return out


def system_prompt(me: str | None = None, now: datetime | None = None) -> str:
    now = now or clock()
    bells = "; ".join(f"{n}-я {a}–{b}" for n, (a, b) in slots.BELLS.items())
    who = (f"Пользователь — преподаватель {me}. «Мои пары», «у меня» — это расписание преподавателя {me}."
           if me else "ФИО пользователя неизвестно. Если он спрашивает про «мои пары» или «у меня» — сначала спроси ФИО, "
           f"не подставляй имя из примеров ниже ({EXAMPLE_TEACHER} — только пример).")
    t = me or EXAMPLE_TEACHER
    return f"""Ты — помощник преподавателя РТУ МИРЭА. Отвечай по-русски, коротко и по делу, списком, если вариантов несколько.

Правила:
- Расписание, аудитории и даты бери ТОЛЬКО из инструментов. Ничего не придумывай.
- Даты передавай как сказал пользователь («завтра», «понедельник», «9 ноября») или ГГГГ-ММ-ДД из календаря ниже. В ответе называй дату и день недели из результата инструмента.
- Если инструмент вернул ОШИБКУ со списком вариантов — покажи варианты и спроси, кого/что имели в виду.
- Если не хватает данных (кто, какая группа, когда) — задай один уточняющий вопрос.
- Для переноса пары и поиска времени для отработки используй find_reschedule_slots. Напомни, что перенос согласуется с учебным отделом.
- Корпус передавай, только если пользователь его назвал. Кампус — В-78 и т. п.; буква корпуса («корпус Б») — building, этаж — floor.
- Период поиска (date_from/date_to) передавай, только если пользователь сам назвал срок.
- Числа в результатах — количество аудиторий, а не мест.
- Если результат говорит, что условие пользователя (корпус, этаж, день) не выполнено, скажи об этом прямо и предложи, как ослабить условие.
- Строка «[Параметры поиска: …]» в твоих прошлых ответах — с чем ты искал. На уточнение («а в среду?», «а в корпусе А?») повтори тот же поиск, изменив только названное. Сам эту строку не пиши.
- Группы, предметы и аудитории называй только из результатов инструментов, полностью («РИБО-02-23»).
- Свободная по расписанию аудитория может быть физически закрыта — упомяни это, если предлагаешь аудиторию.
- {who}

Время московское. Звонки: {bells}.
Корпуса: «Вернадского 78» = В-78, «Вернадского 86» = В-86, «Стромынка» = С-20, «Малая Пироговская» = МП-1, «Соколиная гора» = СГ-22.

Примеры (как вызывать инструменты):
- «Куда перенести 1 и 2 пару у меня 9 ноября?» → find_reschedule_slots(teacher="{t}", lesson_date="9 ноября", lesson_pairs=[1, 2])
- «…проверь с 16 по 22 ноября» → то же + date_from="16 ноября", date_to="22 ноября"
- «Когда поставить отработку для ЭФБО-01-24 на этой неделе?» → find_reschedule_slots(teacher="{t}", groups=["ЭФБО-01-24"], date_from="сегодня")
- «У какой группы у меня пары в понедельник?» → timetable(kind="teacher", name="{t}", date_from="понедельник", days=1)
- «Перенеси мою 3 пару в пятницу, можно только на понедельник или вторник» → find_reschedule_slots(teacher="{t}", lesson_date="пятница", lesson_pairs=[3], weekdays=[1, 2])
- «Свободные аудитории завтра на 3 паре» → free_rooms(date="завтра", pair=3)
- «Свободна ли А-214 в среду с 10:40 до 12:10?» → free_rooms(date="среда", start_time="10:40", end_time="12:10", rooms=["А-214"])

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

    @property
    def context(self) -> str:
        """Параметры последних успешных поисков — страница добавит их к этому ответу в истории,
        чтобы уточняющий вопрос («а в среду?») продолжил тот же поиск."""
        done = [s for s in self.steps if s.ok and not s.result.startswith("Этот запрос уже выполнен")][-2:]
        if not done:
            return ""
        calls = "; ".join(f"{s.tool} {json.dumps(s.args, ensure_ascii=False, sort_keys=True)}" for s in done)
        return f"[Параметры поиска: {calls}]"[:800]


_CONTEXT_LINE = re.compile(r"^\s*\[Параметры поиска:.*$", re.M)


def _clean(text: str) -> str:
    return _CONTEXT_LINE.sub("", text or "").strip()


ToolRunner = Callable[[str, dict], Awaitable[str]]


async def answer(llm: Ollama, history: list[dict], me: str | None = None, tool_runner: ToolRunner = run_tool,
                 think: bool = False) -> Answer:
    """history — сообщения чата [{role: user|assistant, content}], последнее — вопрос пользователя."""
    clean = [{"role": m["role"], "content": str(m.get("content", ""))[:4000]}
             for m in history[-12:] if m.get("role") in ("user", "assistant")]
    if not clean or clean[-1]["role"] != "user":
        raise ValueError("последним должно быть сообщение пользователя")
    messages = [{"role": "system", "content": system_prompt(me)}, *clean]
    steps: list[Step] = []
    seen: set[str] = set()
    user_dates = dates.mentioned(clean[-1]["content"], clock())
    user_pairs = dates.mentioned_pairs(clean[-1]["content"])

    for _ in range(MAX_ROUNDS):
        reply = await llm.chat(messages, TOOLS, think=think)
        if not reply.tool_calls:
            return Answer(_clean(reply.content) or "Не получилось сформулировать ответ. Переформулируйте вопрос.", steps)
        messages.append({"role": "assistant", "content": reply.content,
                         "tool_calls": [{"function": {"name": c.name, "arguments": c.arguments}} for c in reply.tool_calls]})
        for call in reply.tool_calls:
            call.arguments = effective_args(call.name, call.arguments, me, user_dates, user_pairs, clean[-1]["content"])
            key = call.name + json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)
            if needs_my_name(call.name, call.arguments, me, clean[-1]["content"]):
                result = ("ОШИБКА: ФИО пользователя неизвестно. Не вызывай инструмент: спроси, как зовут пользователя "
                          "(или предложи вписать ФИО в ⚙ Настройки).")
                steps.append(Step(call.name, call.arguments, False, result))
                messages.append({"role": "tool", "tool_name": call.name, "content": result})
                continue
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
    final = await llm.chat(messages, think=think)
    return Answer(_clean(final.content) or "Не получилось получить ответ. Уточните вопрос.", steps)
