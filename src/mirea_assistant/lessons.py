"""Занятия с группами.

pymirea при разборе расписания отбрасывает список групп, а без него нельзя ответить
«у какой группы пара» и подобрать перенос: модель начинала угадывать («РИБО, РРБО»).
Здесь тот же запрос get_schedule_for_entity разбирается целиком.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from mirea_connect import services as S
from pymirea import schedule as pm
from pymirea.schedule import MireaSchedule

logger = logging.getLogger(__name__)

LESSON_TYPES = {
    "lecture": "лекция", "practice": "практика", "laboratoryWork": "лабораторная",
    "laboratory": "лабораторная", "exam": "экзамен", "consultation": "консультация", "credit": "зачёт",
}


@dataclass
class Lesson:
    start: datetime
    end: datetime
    subject: str
    kind: str | None = None
    rooms: list[str] = field(default_factory=list)
    campus: str | None = None
    groups: list[str] = field(default_factory=list)
    teachers: list[str] = field(default_factory=list)
    pair: int | None = None

    @property
    def day(self) -> date:
        return self.start.date()

    @property
    def kind_ru(self) -> str:
        return LESSON_TYPES.get(self.kind or "", self.kind or "")

    def location(self) -> str:
        return ", ".join(f"{r} ({self.campus})" if self.campus else r for r in self.rooms)


@dataclass
class Timetable:
    kind: str
    name: str
    lessons: list[Lesson]


def _msk(dt: datetime) -> datetime:
    """iCal pymirea может отдать время без часового пояса: сравнение с московским тогда падает с TypeError."""
    return dt.replace(tzinfo=S.MSK) if dt.tzinfo is None else dt


def _items(payload) -> list[dict]:
    data = payload.get("data") if isinstance(payload, dict) else payload
    return [i for i in data or [] if isinstance(i, dict) and i.get("type") == "__lesson_schedule__"]


def parse(payload) -> list[Lesson]:
    out: list[Lesson] = []
    for it in _items(payload):
        bells = it.get("lesson_bells") or {}
        rooms, campus = [], None
        for r in it.get("classrooms") or []:
            if isinstance(r, dict) and r.get("name"):
                rooms.append(r["name"])
                c = r.get("campus") or {}
                campus = campus or (c.get("short_name") or c.get("name") if isinstance(c, dict) else None)
            elif isinstance(r, str) and r:
                rooms.append(r)
        groups = [g if isinstance(g, str) else (g.get("name") or "") for g in it.get("groups") or []]
        teachers = [t if isinstance(t, str) else (t.get("name") or "") for t in it.get("teachers") or []]
        for d in it.get("dates") or []:
            start = pm._combine_date_time(d, bells.get("start_time"))
            if not start:
                continue
            end = pm._combine_date_time(d, bells.get("end_time")) or start + timedelta(minutes=90)
            out.append(Lesson(start=start, end=end, subject=it.get("subject") or "Занятие", kind=it.get("lesson_type"),
                              rooms=rooms, campus=campus, groups=[g for g in groups if g],
                              teachers=[t for t in teachers if t], pair=bells.get("number")))
    out.sort(key=lambda x: x.start)
    return out


async def fetch(kind: str, query: str) -> Timetable:
    """Найти сущность (с выбором при неоднозначности) и загрузить занятия с группами."""
    query = S._query(kind, query)
    sched = MireaSchedule()
    try:
        found = [f for f in await S._search(sched, kind, query) if f.name]
        if not found:
            raise S.ServiceError(
                f"По запросу «{query}» ничего не найдено ({kind}). Проверьте написание; "
                "если оно верное, сервис расписания мог быть временно недоступен — повторите позже."
            )
        pick = S._pick(found, kind, query)
        payload, err = await sched._rpc("get_schedule_for_entity", {
            "p_entity_type": kind, "p_entity": pick.uid or pick.name, "p_organization_id": pm.SCHEDULE_ORGANIZATION_ID,
        })
    finally:
        await sched.close()
    if payload is None:
        # pymirea не бросает исключение при таймауте, 5xx или открытом circuit breaker, а возвращает None.
        # Пустое расписание здесь значило бы «группа свободна всю неделю» — и перенос на занятое время.
        raise S.ServiceError(f"Не удалось загрузить расписание: {pick.name} ({err or 'сервис расписания недоступен'}). "
                             "Повторите через минуту.")
    lessons = parse(payload)
    if not lessons:
        # Запасной путь: формат ответа изменился — берём события pymirea (без групп), чтобы не сломаться.
        logger.warning("raw schedule without lessons for %s, falling back", kind)
        sched_obj = await S.fetch_entity(kind, query)
        lessons = [Lesson(start=_msk(e.start), end=_msk(e.end) if e.end else _msk(e.start) + timedelta(minutes=90),
                          subject=e.summary or "", rooms=[e.location] if e.location else []) for e in sched_obj.events]
        lessons.sort(key=lambda x: x.start)
    return Timetable(kind=kind, name=pick.name, lessons=lessons)
