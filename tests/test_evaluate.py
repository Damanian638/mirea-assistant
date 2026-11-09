"""Экзамен модели: подсчёт и формат вопросов (без настоящей модели)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from mirea_assistant import agent, evaluate
from mirea_assistant.llm import Reply, ToolCall

SPEC = json.loads((Path(__file__).resolve().parents[1] / "evals" / "cases.json").read_text(encoding="utf-8"))
TODAY = datetime.fromisoformat(SPEC["today"])
TOOLS = {t["function"]["name"]: set(t["function"]["parameters"]["properties"]) for t in agent.TOOLS}


def test_cases_are_well_formed():
    ids = [c["id"] for c in SPEC["cases"]]
    assert len(ids) == len(set(ids)) and len(ids) >= 25
    for c in SPEC["cases"]:
        want = c["expect"]["tool"]
        for t in want if isinstance(want, list) else [want] if want else []:
            assert t in TOOLS, c["id"]
        for key in c["expect"].get("checks", {}):
            op, _, fld = key.partition(":")
            assert op in ("date", "set", "eq", "has", "absent") or key == "time", (c["id"], key)
            if fld and fld != "*":
                assert any(fld in TOOLS[t] for t in (want if isinstance(want, list) else [want])), (c["id"], fld)


def test_today_is_saturday():
    assert TODAY.weekday() == 5 and TODAY.date().isoformat() == "2026-11-07"


@pytest.mark.parametrize(
    "tool, args, ok",
    [
        ("find_reschedule_slots", {"teacher": "Коняшкин Георгий Викторович", "lesson_date": "9 ноября", "lesson_pairs": [1, 2]}, True),
        ("find_reschedule_slots", {"teacher": "Коняшкин Г.В.", "lesson_date": "2026-11-09", "lesson_pairs": ["2", "1"]}, True),
        # реальная ошибка qwen3:8b: номера пар как ограничение нового времени
        ("find_reschedule_slots", {"teacher": "Коняшкин", "date_from": "2026-11-09", "pairs": [1, 2]}, False),
        ("find_reschedule_slots", {"teacher": "Коняшкин", "lesson_date": "2026-11-11", "lesson_pairs": [1, 2]}, False),
        ("find_reschedule_slots", {"teacher": "Коняшкин", "lesson_date": "9 ноября", "lesson_pairs": [1, 2], "campus": "В-78"}, False),
        ("timetable", {"kind": "teacher", "name": "Коняшкин", "date_from": "2026-11-09"}, False),
    ],
)
def test_check_move_case(tool, args, ok):
    case = next(c for c in SPEC["cases"] if c["id"] == "move-1")
    assert (evaluate.check(case["expect"], tool, args, TODAY) == []) is ok


def test_check_free_rooms_time_and_campus():
    case = next(c for c in SPEC["cases"] if c["id"] == "room-1")
    good = {"date": "понедельник", "start_time": "9:00", "end_time": "10:30"}
    assert evaluate.check(case["expect"], "free_rooms", good, TODAY) == []
    # реальная ошибка qwen3:8b: подставила корпус, которого пользователь не называл
    bad = evaluate.check(case["expect"], "free_rooms", {**good, "campus": "В-78"}, TODAY)
    assert bad and "campus" in bad[0]


def test_check_clarification_cases():
    case = next(c for c in SPEC["cases"] if c["id"] == "ask-1")
    assert evaluate.check(case["expect"], None, {}, TODAY) == []
    assert evaluate.check(case["expect"], "timetable", {"kind": "teacher"}, TODAY)


class ScriptedLLM:
    def __init__(self, reply):
        self.reply = reply

    async def chat(self, messages, tools=None, think=False):
        return self.reply


async def test_run_case_captures_first_call_after_program_fixes(monkeypatch):
    monkeypatch.setattr(agent, "clock", lambda: TODAY)
    case = next(c for c in SPEC["cases"] if c["id"] == "room-2")
    # модель ошиблась с датой (сегодняшняя суббота) и написала корпус словами — программа исправит
    llm = ScriptedLLM(Reply("", [ToolCall("free_rooms", {"date": "2026-11-07", "start_time": "12:40", "end_time": "14:10",
                                                         "campus": "Вернадского 78"})]))
    r = await evaluate.run_case(llm, case, SPEC["me"], TODAY)
    assert r.tool == "free_rooms" and r.args["date"] == "2026-11-09" and r.raw_args["date"] == "2026-11-07"
    assert r.ok, r.problems  # campus «Вернадского 78» засчитывается: программа поймёт как В-78


async def test_run_case_placeholder_teacher_fixed(monkeypatch):
    monkeypatch.setattr(agent, "clock", lambda: TODAY)
    case = next(c for c in SPEC["cases"] if c["id"] == "move-3")
    llm = ScriptedLLM(Reply("", [ToolCall("find_reschedule_slots", {"teacher": "<ФИО пользователя>", "lesson_date": "понедельник",
                                                                    "lesson_pairs": [5]})]))
    r = await evaluate.run_case(llm, case, SPEC["me"], TODAY)
    assert r.ok, r.problems and r.args["teacher"] == SPEC["me"]


async def test_text_answer_counts_as_clarification(monkeypatch):
    monkeypatch.setattr(agent, "clock", lambda: TODAY)
    case = next(c for c in SPEC["cases"] if c["id"] == "ask-2")
    r = await evaluate.run_case(ScriptedLLM(Reply("Какую пару и когда нужно перенести?")), case, SPEC["me"], TODAY)
    assert r.ok and r.tool is None and "Какую пару" in r.answer


def test_report_lists_errors_and_groups():
    good = evaluate.Result(SPEC["cases"][0], True, "find_reschedule_slots", {"a": 1}, {}, "", 3.0)
    bad = evaluate.Result(SPEC["cases"][1], False, "timetable", {"b": 2}, {"b": 3}, "", 5.0, ["вызван timetable"])
    text = evaluate.report([good, bad], "qwen3:4b", False)
    assert "# Экзамен модели qwen3:4b: 1 из 2" in text and "| Перенос пары | 1 из 2 |" in text
    assert "**move-2**" in text and "модель передала" in text and "- вызван timetable" in text


def test_live_suspicions_flag_invented_rooms_groups_and_dates():
    from mirea_assistant import evaluate

    results = "2026-11-10 (10.11 Вт) 3-я пара: Б-202, А-9\nгруппы: РИБО-02-23"
    # прогон 09.11: модель выдала расписание дня из текста ошибки за варианты переноса на 16–17 ноября
    answer = "1. 16 ноября — Б-202, В-317, группа ЭФБО-06-24. 2. 10.11 — А-9, РИБО-02-23"
    got = evaluate.suspicions(answer, results, "куда перенести пару")
    assert got == ["аудитория В-317 не из результатов", "группа ЭФБО-06-24 не из результатов",
                   "дата 16.11 не из результатов"]
    assert evaluate.suspicions("В-78, Б-202, 10 ноября", results, "") == []


def test_history_is_sent_before_question():
    from mirea_assistant import evaluate

    case = {"q": "а в корпусе Б?", "history": [{"role": "user", "content": "аудитории в В-78"},
                                               {"role": "assistant", "content": "…"}]}
    assert [m["content"] for m in evaluate._messages(case)] == ["аудитории в В-78", "…", "а в корпусе Б?"]
