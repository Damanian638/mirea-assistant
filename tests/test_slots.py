"""Поиск времени для переноса пары — на подменённом расписании, без сети.

2030-01-07 — понедельник. «Сейчас» — воскресенье 2030-01-06 12:00.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from mirea_connect import services as S

from mirea_assistant import lessons as L
from mirea_assistant import slots

MSK = slots.MSK
TEACHER = "Коняшкин Георгий Викторович"


def les(day: str, pair: int, subject="ПКР", groups=("РИБО-02-23",), room="Б-202", campus="В-78", kind="laboratoryWork"):
    a, b = slots.BELLS[pair]
    return L.Lesson(start=datetime.fromisoformat(f"{day}T{a}").replace(tzinfo=MSK),
                    end=datetime.fromisoformat(f"{day}T{b}").replace(tzinfo=MSK),
                    subject=subject, kind=kind, rooms=[room], campus=campus, groups=list(groups), pair=pair)


@pytest.fixture
def world(monkeypatch):
    state = {
        "teacher": [les("2030-01-07", 1), les("2030-01-07", 2), les("2030-01-07", 5, "Введение", ("РРБО-01-26",), "А-7"),
                    les("2030-01-08", 3, "ПЭПиС", ("ЭФБО-08-24",), "В-317")],
        "РИБО-02-23": [les("2030-01-07", 1), les("2030-01-07", 2), les("2030-01-08", 1, "Матан", ())],
        "ЭФБО-01-24": [],
        "РРБО-01-26": [les("2030-01-07", 5, "Введение", ("РРБО-01-26",), "А-7")],
        "rooms": lambda day, start, end: [{"room": "Б-202", "campus": "В-78"}, {"room": "А-9", "campus": "В-78"},
                                          {"room": "10", "campus": "С-20"}],
        "room_calls": [],
    }

    async def fetch(kind, query):
        if kind == "teacher":
            return L.Timetable("teacher", TEACHER, state["teacher"])
        if query not in state:
            raise S.ServiceError(f"По запросу «{query}» ничего не найдено")
        return L.Timetable("group", query, state[query])

    async def free_rooms(rooms, date, start, end, campus=None):
        state["room_calls"].append((date, start, end, campus))
        free = state["rooms"](date, start, end)
        if campus:
            free = [f for f in free if S._norm(f["campus"]) == S._norm(campus)]
        return {"total_free": len(free), "free_rooms": free}

    monkeypatch.setattr(L, "fetch", fetch)
    monkeypatch.setattr(S, "free_rooms", free_rooms)

    class FixedDT(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2030, 1, 6, 12, 0, tzinfo=MSK).astimezone(tz)

    monkeypatch.setattr(slots, "datetime", FixedDT)
    return state


def by_day(res, day):
    return [s["pairs"] for s in res["slots"] if s["date"] == day]


async def test_moving_lessons_found_with_groups_campus_and_block(world):
    """Реальный случай: «перенести 1 и 2 пару 9 ноября» — модель не знала групп и отвечала «нет слотов»."""
    res = await slots.find_slots("Коняшкин", lesson_date="2030-01-07", lesson_pairs=[1, 2], need_room=False)
    assert res["groups"] == ["РИБО-02-23"] and res["campus"] == "В-78" and res["pairs_in_a_row"] == 2
    assert "1-я пара ПКР (лабораторная), группы РИБО-02-23, Б-202 (В-78)" in res["moving"]
    assert res["moving_date"] == "7 января (Пн)"
    assert all(len(s["pairs"]) == 2 for s in res["slots"])  # две пары подряд
    # 8 января: у группы 1-я пара, у преподавателя 3-я → 1–2, 2–3, 3–4 не подходят
    assert [1, 2] not in by_day(res, "2030-01-08") and [2, 3] not in by_day(res, "2030-01-08")
    assert [4, 5] in by_day(res, "2030-01-08")
    # 7 января: 1–2 — сама пара, 5 — лекция → 3–4 подходит, 4–5 и 5–6 нет
    assert by_day(res, "2030-01-07") == [[3, 4], [6, 7]]


async def test_period_defaults_to_two_weeks_from_lesson(world):
    res = await slots.find_slots("Коняшкин", lesson_date="2030-01-07", lesson_pairs=[1, 2], need_room=False)
    assert res["period"] == "2030-01-07 — 2030-01-20"
    res = await slots.find_slots("Коняшкин", lesson_date="2030-01-07", lesson_pairs=[1, 2],
                                 date_from="2030-01-14", date_to="2030-01-19", need_room=False)
    assert {s["date"] for s in res["slots"]} <= {f"2030-01-{d}" for d in range(14, 20)}


async def test_convenient_days_first(world):
    res = await slots.find_slots("Коняшкин", lesson_date="2030-01-07", lesson_pairs=[1, 2], need_room=False)
    first = res["slots"][0]
    assert first["date"] in ("2030-01-07", "2030-01-08")  # дни, когда преподаватель и так в университете
    assert "в этот день у преподавателя уже есть пары" in first["notes"]
    assert any("без «окна»" in n for n in first["notes"])
    later = [s for s in res["slots"] if s["date"] == "2030-01-09"]
    assert later and "отдельный приезд преподавателя в университет" in later[0]["notes"]


async def test_same_room_preferred_and_checked_for_whole_block(world):
    res = await slots.find_slots("Коняшкин", lesson_date="2030-01-07", lesson_pairs=[1, 2])
    s0 = res["slots"][0]
    assert s0["free_rooms"][0] == "Б-202" and s0["same_room"] is True
    assert all(c == "В-78" for *_, c in world["room_calls"])
    # окно аудиторий — на обе пары сразу (от начала первой до конца второй)
    assert any(start == "12:40" and end == "15:50" for _, start, end, _ in world["room_calls"])


async def test_blocks_without_rooms_dropped(world):
    world["rooms"] = lambda day, start, end: [] if start == "12:40" else [{"room": "А-9", "campus": "В-78"}]
    res = await slots.find_slots("Коняшкин", lesson_date="2030-01-07", lesson_pairs=[1, 2])
    # проверенные блоки с 12:40 (аудиторий нет) выброшены; непроверенные (за пределом 30 проверок) остаются без аудиторий
    assert not any(s["time"].startswith("12:40") and s["free_rooms_total"] is not None for s in res["slots"])
    assert any(s["time"].startswith("14:20") and s["free_rooms"] == ["А-9"] for s in res["slots"])


async def test_unknown_lesson_lists_that_days_pairs(world):
    with pytest.raises(S.ServiceError, match="нет 3-й пары.*1-я пара ПКР.*РИБО-02-23"):
        await slots.find_slots("Коняшкин", lesson_date="2030-01-07", lesson_pairs=[3])
    with pytest.raises(S.ServiceError, match="нет занятий"):
        await slots.find_slots("Коняшкин", lesson_date="2030-01-09", lesson_pairs=[1])


async def test_single_lesson_and_non_adjacent_pairs_move_one_by_one(world):
    res = await slots.find_slots("Коняшкин", lesson_date="2030-01-07", lesson_pairs=[1, 5], need_room=False)
    assert res["pairs_in_a_row"] == 1 and set(res["groups"]) == {"РИБО-02-23", "РРБО-01-26"}


async def test_allowed_pairs_and_weekdays_restrict_new_time(world):
    res = await slots.find_slots("Коняшкин", lesson_date="2030-01-07", lesson_pairs=[1, 2], allowed_pairs=[3, 4, 5],
                                 weekdays=[2], need_room=False)
    assert {s["weekday"] for s in res["slots"]} == {"Вт"}
    assert all(set(s["pairs"]) <= {3, 4, 5} for s in res["slots"])


async def test_explicit_groups_without_lesson(world):
    res = await slots.find_slots("Коняшкин", groups=["ЭФБО-01-24"], date_from="2030-01-08", date_to="2030-01-08",
                                 need_room=False)
    assert res["moving"] is None and res["pairs_in_a_row"] == 1
    assert sorted(by_day(res, "2030-01-08")) == [[1], [2], [4], [5], [6], [7]]
    assert by_day(res, "2030-01-08")[:2] == [[2], [4]]  # вплотную к 3-й паре преподавателя — выше


async def test_touching_boundary_is_not_a_conflict(world):
    world["teacher"] = [L.Lesson(start=datetime(2030, 1, 8, 10, 30, tzinfo=MSK), end=datetime(2030, 1, 8, 10, 40, tzinfo=MSK),
                                 subject="x")]
    res = await slots.find_slots("Коняшкин", date_from="2030-01-08", date_to="2030-01-08", allowed_pairs=[1, 2], need_room=False)
    assert by_day(res, "2030-01-08") == [[1], [2]]


async def test_sunday_and_past_skipped(world):
    res = await slots.find_slots("Коняшкин", date_from="2030-01-05", date_to="2030-01-06", need_room=False)
    assert res["slots"] == []  # суббота 5-го уже прошла, 6-е — воскресенье


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"date_from": "2030-01-07", "date_to": "2030-03-01"}, "не больше"),
        ({"date_from": "2030-01-07", "date_to": "2030-01-01"}, "не раньше"),
        ({"date_from": "07.01.2030"}, "ГГГГ-ММ-ДД"),
        ({"allowed_pairs": [0, 8]}, "1–7"),
        ({"lesson_date": "2030-01-07", "lesson_pairs": [9]}, "1–7"),
        ({"weekdays": [9]}, "1–7"),
    ],
)
async def test_validation(world, kwargs, match):
    with pytest.raises(S.ServiceError, match=match):
        await slots.find_slots("Коняшкин", **kwargs)


def test_parse_raw_mirea_payload_keeps_groups():
    """Формат ответа get_schedule_for_entity (снят с реального ответа МИРЭА)."""
    payload = [{"type": "__lesson_schedule__", "subject": "Программно-конфигурируемые радиотехнологии",
                "lesson_type": "laboratoryWork", "lesson_bells": {"number": 1, "end_time": "10:30", "start_time": "09:00"},
                "groups": ["РИБО-02-23"], "dates": ["14-09-2026", "09-11-2026"],
                "teachers": [{"name": TEACHER}],
                "classrooms": [{"uid": "83", "name": "Б-202", "campus": {"name": "Проспект Вернадского, д.78", "short_name": "В-78"}}]},
               {"type": "other"}]
    out = L.parse(payload)
    assert len(out) == 2
    x = out[1]
    assert x.start.isoformat() == "2026-11-09T09:00:00+03:00" and x.pair == 1
    assert x.groups == ["РИБО-02-23"] and x.campus == "В-78" and x.location() == "Б-202 (В-78)"
    assert x.teachers == [TEACHER] and x.kind_ru == "лабораторная"
    assert L.parse({"data": payload})[0].groups == ["РИБО-02-23"]
