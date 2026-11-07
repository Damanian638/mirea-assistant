"""Поиск времени для переноса пары: считает программа, а не ИИ.

Берём расписание преподавателя и групп целиком, отмечаем занятые пары по звонкам
МИРЭА и для каждой пары, где все свободны, спрашиваем свободные аудитории.
Ответ детерминирован: ИИ только передаёт параметры и пересказывает результат.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from mirea_connect import services as S

MSK = S.MSK
WEEKDAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]

# Звонки РТУ МИРЭА: номер пары → (начало, конец).
BELLS: dict[int, tuple[str, str]] = {
    1: ("09:00", "10:30"),
    2: ("10:40", "12:10"),
    3: ("12:40", "14:10"),
    4: ("14:20", "15:50"),
    5: ("16:20", "17:50"),
    6: ("18:00", "19:30"),
    7: ("19:40", "21:10"),
}

MAX_DAYS = 21
MAX_ROOM_CHECKS = 40  # запросов свободных аудиторий за один поиск
ROOMS_SHOWN = 8
_CAMPUS_RE = re.compile(r"\(([^()]+)\)\s*$")


@dataclass
class Busy:
    start: datetime
    end: datetime
    who: str
    what: str
    location: str | None


@dataclass
class Slot:
    day: date
    pair: int
    free_rooms: list[str] = field(default_factory=list)
    rooms_total: int | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def start(self) -> str:
        return BELLS[self.pair][0]

    @property
    def end(self) -> str:
        return BELLS[self.pair][1]

    def as_dict(self) -> dict:
        return {
            "date": self.day.isoformat(),
            "weekday": WEEKDAYS[self.day.weekday()],
            "pair": self.pair,
            "time": f"{self.start}–{self.end}",
            "free_rooms": self.free_rooms,
            "free_rooms_total": self.rooms_total,
            "notes": self.notes,
        }


def _at(day: date, hhmm: str) -> datetime:
    h, m = (int(x) for x in hhmm.split(":"))
    return datetime(day.year, day.month, day.day, h, m, tzinfo=MSK)


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=MSK)


def _campus(location: str | None) -> str | None:
    m = _CAMPUS_RE.search(location or "")
    return m.group(1).strip() if m else None


def busy_from_schedule(schedule, who: str) -> list[Busy]:
    out = []
    for e in schedule.events:
        start = _aware(e.start)
        end = _aware(e.end) if e.end else start + timedelta(minutes=90)
        out.append(Busy(start, end, who, e.summary or "", e.location))
    return out


def pair_conflicts(busy: list[Busy], day: date, pair: int) -> list[Busy]:
    a, b = _at(day, BELLS[pair][0]), _at(day, BELLS[pair][1])
    return [x for x in busy if x.start < b and x.end > a]  # граница касанием — не пересечение


def _parse_pairs(pairs: list[int] | None) -> list[int]:
    if not pairs:
        return list(BELLS)
    bad = [p for p in pairs if p not in BELLS]
    if bad:
        raise S.ServiceError(f"Номер пары должен быть от 1 до 7 (получено: {bad}).")
    return sorted(set(pairs))


def _days(date_from: str | None, date_to: str | None, weekdays: list[int] | None) -> list[date]:
    start = S._start_of(date_from, "date_from").date()
    end = S._start_of(date_to, "date_to").date() if date_to else start + timedelta(days=6)
    if end < start:
        raise S.ServiceError("date_to должна быть не раньше date_from.")
    if (end - start).days + 1 > MAX_DAYS:
        raise S.ServiceError(f"Диапазон слишком большой: не больше {MAX_DAYS} дней за один поиск.")
    if weekdays and any(w not in range(1, 8) for w in weekdays):
        raise S.ServiceError("weekdays — номера дней недели от 1 (пн) до 7 (вс).")
    out = []
    d = start
    while d <= end:
        if d.weekday() != 6 and (not weekdays or d.weekday() + 1 in weekdays):  # воскресенье пропускаем
            out.append(d)
        d += timedelta(days=1)
    return out


def _neighbour_notes(slot_day: date, pair: int, teacher_busy: list[Busy], campus: str | None) -> list[str]:
    """Предупредить, если соседняя пара преподавателя в другом корпусе — не успеть переехать."""
    notes = []
    for other in (pair - 1, pair + 1):
        if other not in BELLS:
            continue
        for b in pair_conflicts(teacher_busy, slot_day, other):
            c = _campus(b.location)
            if campus and c and S._norm(c) != S._norm(campus):
                notes.append(f"соседняя {other}-я пара преподавателя в корпусе {c}")
    return notes


async def _free_rooms(day: date, pair: int, campus: str | None) -> tuple[list[str], int] | None:
    try:
        data = await S.free_rooms(None, day.isoformat(), BELLS[pair][0], BELLS[pair][1], campus)
    except S.ServiceError:
        return None
    rooms = [f"{r['room']} ({r['campus']})" if r.get("campus") and not campus else r["room"] for r in data["free_rooms"]]
    return rooms, data["total_free"]


async def find_slots(
    teacher: str,
    groups: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    campus: str | None = None,
    pairs: list[int] | None = None,
    weekdays: list[int] | None = None,
    need_room: bool = True,
) -> dict:
    """Пары, в которые свободны преподаватель и все группы, с подходящими аудиториями."""
    pair_list = _parse_pairs(pairs)
    days = _days(date_from, date_to, weekdays)
    groups = [g for g in (groups or []) if g and g.strip()]

    teacher_schedule = await S.fetch_entity("teacher", teacher)
    group_schedules = [await S.fetch_entity("group", g) for g in groups]
    teacher_busy = busy_from_schedule(teacher_schedule, teacher_schedule.entity_name or teacher)
    busy = list(teacher_busy)
    for g, sched in zip(groups, group_schedules):
        busy += busy_from_schedule(sched, sched.entity_name or g)

    now = datetime.now(MSK)
    candidates = [
        Slot(d, p)
        for d in days
        for p in pair_list
        if _at(d, BELLS[p][0]) > now and not pair_conflicts(busy, d, p)
    ]
    for s in candidates:
        s.notes = _neighbour_notes(s.day, s.pair, teacher_busy, campus)

    checked = 0
    if need_room and candidates:
        sem = asyncio.Semaphore(4)

        async def fill(s: Slot) -> None:
            async with sem:
                res = await _free_rooms(s.day, s.pair, campus)
            if res is None:
                s.notes.append("свободные аудитории не удалось проверить")
            else:
                s.free_rooms, s.rooms_total = res[0][:ROOMS_SHOWN], res[1]

        to_check = candidates[:MAX_ROOM_CHECKS]
        checked = len(to_check)
        await asyncio.gather(*(fill(s) for s in to_check))
        # без аудитории перенос невозможен; непроверенные оставляем с пометкой
        candidates = [s for s in candidates[:checked] if s.rooms_total != 0] + candidates[checked:]

    return {
        "teacher": teacher_schedule.entity_name,
        "groups": [s.entity_name for s in group_schedules],
        "campus": campus,
        "period": f"{days[0].isoformat()} — {days[-1].isoformat()}" if days else None,
        "slots_found": len(candidates),
        "slots": [s.as_dict() for s in candidates],
        "rooms_checked_for": checked if need_room else 0,
        "note": "Свободно по расписанию МИРЭА; перенос нужно согласовать с учебным отделом.",
    }
