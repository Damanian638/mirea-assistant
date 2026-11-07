"""Поиск времени для переноса пары: считает программа, а не ИИ.

Достаточно сказать, какую пару переносим («1 и 2 пара 9 ноября»): программа сама
находит эти занятия, их группы и корпус, подбирает столько пар подряд, сколько
переносится, и ставит выше удобные варианты — дни, когда преподаватель и группы
и так в университете, без «окон». ИИ только передаёт параметры и пересказывает.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from mirea_connect import services as S

from . import lessons as L

MSK = S.MSK
WEEKDAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
MONTHS_GEN = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября",
              "ноября", "декабря"]

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
MAX_ROOM_CHECKS = 30  # запросов свободных аудиторий за один поиск
ROOMS_SHOWN = 8


def human_date(d: date) -> str:
    return f"{d.day} {MONTHS_GEN[d.month - 1]} ({WEEKDAYS[d.weekday()]})"


@dataclass
class Busy:
    start: datetime
    end: datetime
    who: str


@dataclass
class Slot:
    day: date
    pairs: list[int]
    free_rooms: list[str] = field(default_factory=list)
    rooms_total: int | None = None
    same_room: bool = False
    notes: list[str] = field(default_factory=list)
    score: int = 0

    @property
    def start(self) -> str:
        return BELLS[self.pairs[0]][0]

    @property
    def end(self) -> str:
        return BELLS[self.pairs[-1]][1]

    def as_dict(self) -> dict:
        return {
            "date": self.day.isoformat(),
            "weekday": WEEKDAYS[self.day.weekday()],
            "pairs": self.pairs,
            "time": f"{self.start}–{self.end}",
            "free_rooms": self.free_rooms,
            "free_rooms_total": self.rooms_total,
            "same_room": self.same_room,
            "notes": self.notes,
        }


def _at(day: date, hhmm: str) -> datetime:
    h, m = (int(x) for x in hhmm.split(":"))
    return datetime(day.year, day.month, day.day, h, m, tzinfo=MSK)


def pair_of(lesson: L.Lesson) -> int | None:
    if lesson.pair in BELLS:
        return lesson.pair
    hhmm = lesson.start.astimezone(MSK).strftime("%H:%M")
    return next((n for n, (a, _) in BELLS.items() if a == hhmm), None)


def busy_of(lessons: list[L.Lesson], who: str) -> list[Busy]:
    return [Busy(x.start, x.end, who) for x in lessons]


def conflicts(busy: list[Busy], day: date, pairs: list[int]) -> list[Busy]:
    a, b = _at(day, BELLS[pairs[0]][0]), _at(day, BELLS[pairs[-1]][1])
    return [x for x in busy if x.start < b and x.end > a]  # касание границы — не пересечение


def _ints(values: list[int] | None, allowed: range, what: str) -> list[int] | None:
    if not values:
        return None
    bad = [v for v in values if v not in allowed]
    if bad:
        raise S.ServiceError(f"{what}: допустимо {allowed.start}–{allowed.stop - 1} (получено: {bad}).")
    return sorted(set(values))


def _describe(lessons: list[L.Lesson]) -> str:
    return "; ".join(
        f"{pair_of(x)}-я пара {x.subject}" + (f" ({x.kind_ru})" if x.kind_ru else "")
        + (f", группы {', '.join(x.groups)}" if x.groups else "") + (f", {x.location()}" if x.rooms else "")
        for x in lessons
    )


def _pick_lessons(tt: L.Timetable, day: date, pairs: list[int] | None) -> list[L.Lesson]:
    that_day = [x for x in tt.lessons if x.day == day]
    if not that_day:
        raise S.ServiceError(f"{human_date(day)} у преподавателя {tt.name} нет занятий — проверьте дату.")
    chosen = [x for x in that_day if not pairs or pair_of(x) in pairs]
    missing = sorted(set(pairs or []) - {pair_of(x) for x in chosen})
    if missing or not chosen:
        raise S.ServiceError(
            f"{human_date(day)} у преподавателя {tt.name} нет {', '.join(map(str, missing))}-й пары. "
            f"В этот день: {_describe(that_day)}."
        )
    return chosen


async def find_slots(
    teacher: str,
    lesson_date: str | None = None,
    lesson_pairs: list[int] | None = None,
    groups: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    campus: str | None = None,
    allowed_pairs: list[int] | None = None,
    weekdays: list[int] | None = None,
    need_room: bool = True,
) -> dict:
    """Куда перенести пару: время, когда свободны преподаватель и все группы, и аудитории."""
    lesson_pairs = _ints(lesson_pairs, range(1, 8), "Номер переносимой пары")
    allowed = _ints(allowed_pairs, range(1, 8), "Номер пары") or list(BELLS)
    weekdays = _ints(weekdays, range(1, 8), "День недели")
    lesson_day = S._start_of(lesson_date, "lesson_date").date() if lesson_date else None
    start = S._start_of(date_from, "date_from").date() if date_from else (lesson_day or datetime.now(MSK).date())
    end = S._start_of(date_to, "date_to").date() if date_to else start + timedelta(days=13 if lesson_day else 6)
    if end < start:
        raise S.ServiceError("date_to должна быть не раньше date_from.")
    if (end - start).days + 1 > MAX_DAYS:
        raise S.ServiceError(f"Диапазон слишком большой: не больше {MAX_DAYS} дней за один поиск.")

    tt = await L.fetch("teacher", teacher)
    moving = _pick_lessons(tt, lesson_day, lesson_pairs) if lesson_day else []
    group_names = list(dict.fromkeys([g for x in moving for g in x.groups] + [g.strip() for g in groups or [] if g.strip()]))
    if campus is None and moving:
        campus = Counter(x.campus for x in moving if x.campus).most_common(1)[0][0] if any(x.campus for x in moving) else None
    moved_pairs = sorted({pair_of(x) for x in moving if pair_of(x)})
    together = len(moved_pairs) > 1 and moved_pairs == list(range(moved_pairs[0], moved_pairs[-1] + 1))
    block = len(moved_pairs) if together else 1
    original_rooms = {r for x in moving for r in x.rooms}

    group_tts = [await L.fetch("group", g) for g in group_names]
    teacher_busy = busy_of(tt.lessons, tt.name)
    teacher_other = busy_of([x for x in tt.lessons if x not in moving], tt.name)  # без переносимых пар
    busy = teacher_busy + [b for g in group_tts for b in busy_of(g.lessons, g.name)]
    now = datetime.now(MSK)
    teacher_days = Counter(x.day for x in tt.lessons if x not in moving)
    group_days = {x.day for g in group_tts for x in g.lessons}

    candidates: list[Slot] = []
    d = start
    while d <= end:
        if d.weekday() != 6 and (not weekdays or d.weekday() + 1 in weekdays):
            for p in allowed:
                ps = list(range(p, p + block))
                if ps[-1] > 7 or any(x not in allowed for x in ps):
                    continue
                if _at(d, BELLS[p][0]) <= now or conflicts(busy, d, ps):
                    continue
                s = Slot(d, ps)
                if teacher_days[d]:
                    s.score += 3
                    s.notes.append("в этот день у преподавателя уже есть пары")
                    before = p > 1 and conflicts(teacher_other, d, [p - 1])
                    after = ps[-1] < 7 and conflicts(teacher_other, d, [ps[-1] + 1])
                    if before or after:
                        s.score += 2
                        s.notes.append("вплотную к другой паре преподавателя — без «окна»")
                else:
                    s.notes.append("отдельный приезд преподавателя в университет")
                if group_names:
                    if d in group_days:
                        s.score += 1
                    else:
                        s.notes.append("у группы в этот день нет других занятий")
                if ps[-1] >= 6:
                    s.score -= 1
                    s.notes.append("вечернее время")
                candidates.append(s)
        d += timedelta(days=1)
    candidates.sort(key=lambda s: (-s.score, s.day, s.pairs[0]))

    checked = 0
    if need_room and candidates:
        sem = asyncio.Semaphore(4)

        async def fill(s: Slot) -> None:
            async with sem:
                try:
                    # одним окном на весь блок: аудитория свободна на все пары подряд
                    data = await S.free_rooms(None, s.day.isoformat(), s.start, s.end, campus)
                except S.ServiceError:
                    s.notes.append("свободные аудитории не удалось проверить")
                    return
            rooms = [f"{r['room']} ({r['campus']})" if r.get("campus") and not campus else r["room"]
                     for r in data["free_rooms"]]
            same = [r for r in rooms if r.split(" (")[0] in original_rooms]
            s.same_room = bool(same)
            if same:
                s.score += 2
            s.free_rooms = (same + [r for r in rooms if r not in same])[:ROOMS_SHOWN]
            s.rooms_total = data["total_free"]

        to_check = candidates[:MAX_ROOM_CHECKS]
        checked = len(to_check)
        await asyncio.gather(*(fill(s) for s in to_check))
        candidates = [s for s in to_check if s.rooms_total != 0] + candidates[checked:]
        candidates.sort(key=lambda s: (-s.score, s.day, s.pairs[0]))

    return {
        "teacher": tt.name,
        "moving": _describe(moving) if moving else None,
        "moving_date": human_date(lesson_day) if lesson_day else None,
        "groups": group_names,
        "campus": campus,
        "pairs_in_a_row": block,
        "period": f"{start.isoformat()} — {end.isoformat()}",
        "slots_found": len(candidates),
        "slots": [s.as_dict() for s in candidates],
        "rooms_checked_for": checked if need_room else 0,
        "note": "Свободно по расписанию МИРЭА; перенос нужно согласовать с учебным отделом.",
    }
