"""Даты словами → ГГГГ-ММ-ДД. Считает программа: маленькие модели ошибаются в днях недели.

Прогон на Windows показал: на вопрос «в понедельник» модель qwen3:0.6b передала дату среды.
Теперь модель может передать слово как есть («понедельник», «завтра», «9 ноября»).
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

# Полные формы слов: «среди» или «субботник» не должны считаться днём недели.
_WEEKDAY_RE = [
    re.compile(r"(пн|понедельник[аеу]?)"), re.compile(r"(вт|вторник[аеу]?)"), re.compile(r"(ср|сред[аеуы])"),
    re.compile(r"(чт|четверг[аеу]?)"), re.compile(r"(пт|пятниц[аеуы])"), re.compile(r"(сб|суббот[аеуы])"),
    re.compile(r"(вс|воскресень[еяю])"),
]
_MONTHS = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "мая": 5, "май": 5, "июн": 6, "июл": 7,
    "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}
_RELATIVE = {"сегодня": 0, "завтра": 1, "послезавтра": 2, "вчера": -1}


def _weekday(word: str) -> int | None:
    for n, rx in enumerate(_WEEKDAY_RE):
        if rx.fullmatch(word):
            return n
    return None


def _month(word: str) -> int | None:
    for key, n in _MONTHS.items():
        if word.startswith(key):
            return n
    return None


def _with_year(today: date, month: int, day: int) -> date:
    d = date(today.year, month, day)
    # «9 января», сказанное в ноябре, — это следующий год
    return date(today.year + 1, month, day) if (today - d).days > 60 else d


def resolve(value: object, today: date | datetime) -> str | None:
    """Вернуть ГГГГ-ММ-ДД; нераспознанный текст — без изменений (его отклонит проверка формата)."""
    if isinstance(today, datetime):
        today = today.date()
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return raw
    text = raw.casefold().replace("ё", "е")
    words = re.findall(r"[а-яa-z]+|\d+", text)

    m = re.fullmatch(r"(\d{1,2})\.(\d{1,2})(?:\.(\d{2,4}))?", raw)
    if m:
        day, month = int(m.group(1)), int(m.group(2))
        try:
            if m.group(3):
                year = int(m.group(3))
                return date(year + 2000 if year < 100 else year, month, day).isoformat()
            return _with_year(today, month, day).isoformat()
        except ValueError:
            return raw

    for w in words:
        if w in _RELATIVE:
            return (today + timedelta(days=_RELATIVE[w])).isoformat()

    nums = [w for w in words if w.isdigit()]
    months = [_month(w) for w in words if not w.isdigit()]
    month = next((x for x in months if x), None)
    if nums and month:
        try:
            return _with_year(today, month, int(nums[0])).isoformat()
        except ValueError:
            return raw

    wd = next((x for x in (_weekday(w) for w in words) if x is not None), None)
    if wd is not None:
        if any(w.startswith("следующ") for w in words):
            # день на следующей неделе (неделя — с понедельника)
            monday_next = today + timedelta(days=7 - today.weekday())
            return (monday_next + timedelta(days=wd)).isoformat()
        return (today + timedelta(days=(wd - today.weekday()) % 7)).isoformat()  # ближайший, включая сегодня
    return raw


def mentioned(text: str, today: date | datetime) -> list[str]:
    """Даты, названные в сообщении пользователя («в понедельник», «завтра», «9 ноября», «09.11»)."""
    if isinstance(today, datetime):
        today = today.date()
    low = (text or "").casefold().replace("ё", "е")
    words = re.findall(r"[а-яa-z]+|\d+", low)
    nxt = any(w.startswith("следующ") for w in words)
    found: list[str] = []
    for m in re.finditer(r"\b\d{1,2}\.\d{1,2}(?:\.\d{2,4})?\b|\b\d{4}-\d{2}-\d{2}\b", low):
        found.append(resolve(m.group(0), today) or "")
    for i, w in enumerate(words):
        if w in _RELATIVE:
            found.append((today + timedelta(days=_RELATIVE[w])).isoformat())
        elif _weekday(w) is not None:
            found.append(resolve(("следующий " if nxt else "") + w, today) or "")
        elif w.isdigit() and i + 1 < len(words) and _month(words[i + 1]):
            found.append(resolve(f"{w} {words[i + 1]}", today) or "")
    return [d for d in dict.fromkeys(found) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d)]
