"""Поиск окон для переноса пары — на подменённом расписании, без сети."""

from __future__ import annotations

from datetime import datetime

import pytest
from mirea_connect import services as S
from pymirea.schedule import Schedule, ScheduleEvent

from mirea_assistant import slots

MSK = slots.MSK


def ev(day: str, start: str, end: str | None, what="Физика", where="А-9 (В-78)") -> ScheduleEvent:
    s = datetime.fromisoformat(f"{day}T{start}").replace(tzinfo=MSK)
    e = datetime.fromisoformat(f"{day}T{end}").replace(tzinfo=MSK) if end else None
    return ScheduleEvent(start=s, end=e, summary=what, location=where)


@pytest.fixture
def world(monkeypatch):
    """2030-01-07 — понедельник. Расписания и свободные аудитории задаются тестом."""
    state = {
        "teacher": [ev("2030-01-07", "09:00", "10:30"), ev("2030-01-07", "12:40", "14:10", where="330 (С-20)")],
        "ЭФБО-01-24": [ev("2030-01-07", "10:40", "12:10")],
        "ЭФБО-02-24": [ev("2030-01-08", "09:00", None)],  # без конца — считаем 90 минут
        "rooms": lambda day, start: [{"room": "А-9", "campus": "В-78"}, {"room": "10", "campus": "С-20"}],
        "room_calls": [],
    }

    async def fetch_entity(kind, query):
        key = "teacher" if kind == "teacher" else query
        if key not in state:
            raise S.ServiceError(f"По запросу «{query}» ничего не найдено")
        name = "Иванов Илья Иванович" if kind == "teacher" else query
        return Schedule(events=state[key], entity_type=kind, entity_name=name)

    async def free_rooms(rooms, date, start, end, campus=None):
        state["room_calls"].append((date, start, campus))
        free = state["rooms"](date, start)
        if campus:
            free = [f for f in free if S._norm(f["campus"]) == S._norm(campus)]
        return {"total_free": len(free), "free_rooms": free}

    monkeypatch.setattr(S, "fetch_entity", fetch_entity)
    monkeypatch.setattr(S, "free_rooms", free_rooms)

    class FixedDT(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2030, 1, 6, 12, 0, tzinfo=MSK).astimezone(tz)

    monkeypatch.setattr(slots, "datetime", FixedDT)
    return state


def _pairs(res, day):
    return [s["pair"] for s in res["slots"] if s["date"] == day]


async def test_teacher_and_group_busy_pairs_are_excluded(world):
    res = await slots.find_slots("Иванов", ["ЭФБО-01-24"], "2030-01-07", "2030-01-07")
    assert _pairs(res, "2030-01-07") == [4, 5, 6, 7]  # 1 и 3 — преподаватель, 2 — группа
    assert res["teacher"] == "Иванов Илья Иванович" and res["groups"] == ["ЭФБО-01-24"]


async def test_several_groups_and_event_without_end(world):
    res = await slots.find_slots("Иванов", ["ЭФБО-01-24", "ЭФБО-02-24"], "2030-01-08", "2030-01-08")
    assert _pairs(res, "2030-01-08") == [2, 3, 4, 5, 6, 7]


async def test_touching_boundary_is_not_a_conflict(world):
    world["teacher"] = [ev("2030-01-07", "10:30", "10:40")]  # ровно между 1-й и 2-й парой
    res = await slots.find_slots("Иванов", None, "2030-01-07", "2030-01-07", pairs=[1, 2])
    assert _pairs(res, "2030-01-07") == [1, 2]
    world["teacher"] = [ev("2030-01-07", "10:29", "10:41")]
    res = await slots.find_slots("Иванов", None, "2030-01-07", "2030-01-07", pairs=[1, 2])
    assert _pairs(res, "2030-01-07") == []


async def test_sunday_skipped_and_weekday_filter(world):
    res = await slots.find_slots("Иванов", None, "2030-01-12", "2030-01-14", pairs=[4], need_room=False)
    assert [s["date"] for s in res["slots"]] == ["2030-01-12", "2030-01-14"]  # сб и пн, без вс
    res = await slots.find_slots("Иванов", None, "2030-01-07", "2030-01-13", pairs=[4], weekdays=[3], need_room=False)
    assert [s["weekday"] for s in res["slots"]] == ["Ср"]


async def test_past_pairs_are_not_offered(world):
    res = await slots.find_slots("Иванов", None, "2030-01-06", "2030-01-06", need_room=False)
    assert res["slots"] == []  # воскресенье
    world_now = datetime(2030, 1, 7, 13, 0, tzinfo=MSK)
    slots.datetime.now = classmethod(lambda cls, tz=None: world_now)  # 13:00 понедельника
    res = await slots.find_slots("Иванов", None, "2030-01-07", "2030-01-07", need_room=False)
    assert min(_pairs(res, "2030-01-07")) == 4


async def test_rooms_filtered_by_campus_and_slots_without_rooms_dropped(world):
    world["rooms"] = lambda day, start: [] if start == "14:20" else [{"room": "А-9", "campus": "В-78"}]
    res = await slots.find_slots("Иванов", None, "2030-01-07", "2030-01-07", campus="B-78")  # латиница
    by_pair = {s["pair"]: s for s in res["slots"]}
    assert 4 not in by_pair  # нет аудиторий — перенос невозможен
    assert by_pair[5]["free_rooms"] == ["А-9"]
    assert all(c == "B-78" for _, _, c in world["room_calls"])


async def test_rooms_without_campus_show_campus_label(world):
    res = await slots.find_slots("Иванов", None, "2030-01-07", "2030-01-07", pairs=[5])
    assert res["slots"][0]["free_rooms"] == ["А-9 (В-78)", "10 (С-20)"]


async def test_neighbouring_pair_in_other_campus_is_flagged(world):
    res = await slots.find_slots("Иванов", None, "2030-01-07", "2030-01-07", campus="В-78", pairs=[2, 4, 6])
    notes = {s["pair"]: s["notes"] for s in res["slots"]}
    assert any("3-я пара" in n and "С-20" in n for n in notes[4])  # 3-я пара в С-20, а перенос на В-78
    assert any("3-я пара" in n for n in notes[2])
    assert notes[6] == []  # соседних пар нет
    # 1-я пара в В-78 — тот же корпус, не предупреждаем
    assert not any("1-я пара" in n for n in notes[2])


async def test_room_service_failure_keeps_slot_with_note(world):
    async def broken(*a, **k):
        raise S.ServiceError("Сервис свободных аудиторий недоступен")

    S.free_rooms = broken
    res = await slots.find_slots("Иванов", None, "2030-01-07", "2030-01-07", pairs=[5])
    assert res["slots"][0]["notes"] == ["свободные аудитории не удалось проверить"]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"date_from": "2030-01-07", "date_to": "2030-03-01"}, "не больше"),
        ({"date_from": "2030-01-07", "date_to": "2030-01-01"}, "не раньше"),
        ({"date_from": "07.01.2030"}, "ГГГГ-ММ-ДД"),
        ({"pairs": [0, 8]}, "от 1 до 7"),
        ({"weekdays": [9]}, "от 1"),
    ],
)
async def test_validation(world, kwargs, match):
    with pytest.raises(S.ServiceError, match=match):
        await slots.find_slots("Иванов", **kwargs)


async def test_ambiguous_teacher_error_propagates(world, monkeypatch):
    async def ambiguous(kind, query):
        raise S.ServiceError("По запросу «Иванов» найдено несколько преподавателей: А; Б.")

    monkeypatch.setattr(S, "fetch_entity", ambiguous)
    with pytest.raises(S.ServiceError, match="несколько"):
        await slots.find_slots("Иванов")
