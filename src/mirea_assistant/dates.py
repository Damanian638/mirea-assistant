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
    # «9 января», сказанное в ноябре, — это следующий год; «28 декабря», сказанное 10 января, — прошлый
    if (today - d).days > 60:
        return date(today.year + 1, month, day)
    if (d - today).days > 300:
        return date(today.year - 1, month, day)
    return d


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

    iso = re.search(r"\b\d{4}-\d{2}-\d{2}\b", raw)
    if iso:  # «понедельник 2026-11-16» — точная дата важнее слова
        return iso.group(0)
    m = re.search(r"\b(\d{1,2})\.(\d{1,2})(?:\.(\d{2,4}))?\b", raw)
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
        if any(w.startswith("прошл") for w in words):
            # «в прошлую пятницу» — день прошлой недели (раньше получалась будущая пятница)
            monday_prev = today - timedelta(days=today.weekday() + 7)
            return (monday_prev + timedelta(days=wd)).isoformat()
        return (today + timedelta(days=(wd - today.weekday()) % 7)).isoformat()  # ближайший, включая сегодня
    return raw


def mentioned(text: str, today: date | datetime) -> list[str]:
    """Даты, названные в сообщении пользователя («в понедельник», «завтра», «9 ноября», «09.11»)."""
    if isinstance(today, datetime):
        today = today.date()
    low = (text or "").casefold().replace("ё", "е")
    words = re.findall(r"[а-яa-z]+|\d+", low)
    nxt = any(w.startswith("следующ") for w in words)
    prev = any(w.startswith("прошл") for w in words)
    found: list[str] = []
    for m in re.finditer(r"\b\d{1,2}\.\d{1,2}(?:\.\d{2,4})?\b|\b\d{4}-\d{2}-\d{2}\b", low):
        found.append(resolve(m.group(0), today) or "")
    for i, w in enumerate(words):
        if w in _RELATIVE:
            found.append((today + timedelta(days=_RELATIVE[w])).isoformat())
        elif _weekday(w) is not None:
            found.append(resolve(("следующий " if nxt else "прошлый " if prev else "") + w, today) or "")
        elif w.isdigit() and i + 1 < len(words) and _month(words[i + 1]):
            found.append(resolve(f"{w} {words[i + 1]}", today) or "")
    found = [d for d in dict.fromkeys(found) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d)]
    if not found:
        # «на следующей неделе» → её понедельник; «на этой неделе» → сегодня.
        # Экзамен qwen3:8b: «на следующую неделю» в субботу 7.11 → модель брала 16.11 вместо 9.11.
        if re.search(r"следующ\w*\s+недел", low):
            found.append((today + timedelta(days=7 - today.weekday())).isoformat())
        elif re.search(r"(эт\w+|текущ\w+)\s+недел", low):
            found.append(today.isoformat())
    return found


_PAIR_WORDS = {"перв": 1, "втор": 2, "трет": 3, "четверт": 4, "пят": 5, "шест": 6, "седьм": 7}

# Номер пары: цифра 1–7, не часть даты или времени («25.11», «10:40»), с окончанием «-я», «-й»…
_PAIR_NUM = r"(?<![\d.:])[1-7](?![\d.:])(?:-?(?:я|й|ю|ой|ей))?"
# Без вложенных неоднозначных повторов: старое выражение на «1 , 1 , … 1 x» работало экспоненциально долго
# и останавливало сервер (ReDoS).
_PAIR_LIST_RE = re.compile(rf"({_PAIR_NUM}(?:\s*(?:,|и|или)\s*{_PAIR_NUM})*)\s*пар")
# Порядковое слово целиком: «вторник пару» и «пятницу пару» — не 2-я и не 5-я пара.
_PAIR_WORD_RE = re.compile(
    r"\b(перв|втор|трет|четверт|пят|шест|седьм)(?:ая|ой|ую|ое|ые|ых|ей|ья|ьей|ью|ье|ий|ым|ом)?\s+пар"
)


def mentioned_pairs(text: str) -> list[int]:
    """Номера пар в вопросе: «на 7 паре», «3-я пара», «первой паре». «1 и 2 пары» — обе."""
    low = (text or "").casefold().replace("ё", "е")[:4000]
    found: list[int] = []
    for m in _PAIR_LIST_RE.finditer(low):
        found += [int(x) for x in re.findall(r"\d", m.group(1))]
    for m in _PAIR_WORD_RE.finditer(low):
        found.append(_PAIR_WORDS[m.group(1)])
    return [n for n in dict.fromkeys(found) if 1 <= n <= 7]
