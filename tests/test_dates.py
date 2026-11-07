"""Даты словами. «Сегодня» — суббота 07.11.2026."""

from __future__ import annotations

from datetime import date

import pytest

from mirea_assistant import agent, dates

TODAY = date(2026, 11, 7)  # суббота


@pytest.mark.parametrize(
    "text, expected",
    [
        ("понедельник", "2026-11-09"),  # прогон на Windows: модель 0.6b передала тут среду 11.11
        ("в понедельник", "2026-11-09"),
        ("Пн", "2026-11-09"),
        ("среду", "2026-11-11"),
        ("в пятницу", "2026-11-13"),
        ("суббота", "2026-11-07"),  # сегодня суббота — сегодняшний день
        ("воскресенье", "2026-11-08"),
        ("следующий понедельник", "2026-11-09"),
        ("в среду на следующей неделе", "2026-11-11"),
        ("следующая суббота", "2026-11-14"),
        ("сегодня", "2026-11-07"),
        ("завтра", "2026-11-08"),
        ("послезавтра", "2026-11-09"),
        ("9 ноября", "2026-11-09"),
        ("9 ноября 2026", "2026-11-09"),
        ("15 января", "2027-01-15"),  # январь, названный в ноябре, — следующего года
        ("1 октября", "2026-10-01"),  # недавнее прошлое остаётся этим годом
        ("09.11", "2026-11-09"),
        ("09.11.2026", "2026-11-09"),
        ("9.11.26", "2026-11-09"),
        ("2026-11-09", "2026-11-09"),
        ("", None),
        (None, None),
        ("когда-нибудь", "когда-нибудь"),  # не распознано — пусть отклонит проверка формата
        ("31.02", "31.02"),
    ],
)
def test_resolve(text, expected):
    assert dates.resolve(text, TODAY) == expected


def test_monday_during_the_week_is_this_week():
    wednesday = date(2026, 11, 11)
    assert dates.resolve("пятница", wednesday) == "2026-11-13"
    assert dates.resolve("понедельник", wednesday) == "2026-11-16"
    assert dates.resolve("следующая пятница", wednesday) == "2026-11-20"


async def test_tool_receives_resolved_date(monkeypatch):
    seen = {}

    async def free_rooms(rooms, day, start, end, campus=None):
        seen["date"] = day
        return {"date": day, "from": start, "to": end, "campus": campus, "total_free": 0, "free_rooms": []}

    monkeypatch.setattr(agent.S, "free_rooms", free_rooms)
    monkeypatch.setattr(agent, "_date", lambda v: dates.resolve(v, TODAY))
    text = await agent.run_tool("free_rooms", {"date": "в понедельник", "start_time": "10:40", "end_time": "12:10", "campus": "В-78"})
    assert seen["date"] == "2026-11-09"
    assert text.startswith("2026-11-09 (09.11 Пн) 10:40–12:10")  # модель видит день недели в результате


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Какие аудитории свободны в понедельник на 2-й паре в В-78?", ["2026-11-09"]),
        ("пары группы на завтра", ["2026-11-08"]),
        ("что 9 ноября у ЭФБО-01-24", ["2026-11-09"]),
        ("расписание на 09.11", ["2026-11-09"]),
        ("в среду на следующей неделе", ["2026-11-11"]),
        ("с понедельника по среду", ["2026-11-09", "2026-11-11"]),  # два дня — диапазон, дату модели не правим
        ("найди среди свободных аудиторий", []),  # «среди» — не среда
        ("куда перенести пару на следующей неделе", []),
    ],
)
def test_mentioned(text, expected):
    assert dates.mentioned(text, TODAY) == expected


def test_model_date_corrected_when_user_named_one_day(monkeypatch):
    monkeypatch.setattr(agent, "_date", lambda v: dates.resolve(v, TODAY))
    # прогон на Windows: «в понедельник», а модель передала сегодняшнюю субботу
    fixed = agent.fix_dates("free_rooms", {"date": "2026-11-07", "start_time": "10:40"}, ["2026-11-09"])
    assert fixed == {"date": "2026-11-09", "start_time": "10:40"}
    assert agent.fix_dates("free_rooms", {"date": "понедельник"}, ["2026-11-09"]) == {"date": "понедельник"}
    assert agent.fix_dates("free_rooms", {"date": "2026-11-07"}, ["2026-11-09", "2026-11-11"]) == {"date": "2026-11-07"}
    assert agent.fix_dates("find_reschedule_slots", {"date_from": "2026-11-07"}, ["2026-11-09"]) == {"date_from": "2026-11-07"}
