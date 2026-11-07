"""Экзамен для модели: понимает ли она вопросы преподавателей.

Проверяется не текст ответа, а первый вызов инструмента: какой инструмент и с какими
параметрами (после тех же исправлений, что делает программа: ФИО из настроек, дата
из слов пользователя). Сеть МИРЭА не используется — вызов перехватывается.

    python -m mirea_assistant.evaluate evals/cases.json --model qwen3:4b [--think] [--report report.md]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from mirea_connect import services as S

from . import agent, dates
from .llm import LLMError, Ollama


class _Captured(BaseException):
    """Остановить агента после первого вызова инструмента (BaseException — агент его не перехватывает)."""

    def __init__(self, name: str, args: dict) -> None:
        self.name, self.call_args = name, args


@dataclass
class Result:
    case: dict
    ok: bool
    tool: str | None
    args: dict
    raw_args: dict
    answer: str
    seconds: float
    problems: list[str] = field(default_factory=list)


def _norm(v: Any) -> str:
    return S._norm(str(v)) if v is not None else ""


def _values(v: Any) -> list[str]:
    if v is None or v == "":
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return [str(v)]


def check(expect: dict, tool: str | None, args: dict, today: datetime) -> list[str]:
    """Список расхождений; пустой — ответ верный."""
    want = expect.get("tool")
    if want is None:
        return [] if tool is None else [f"нужно было уточнить у пользователя, а модель вызвала {tool}"]
    allowed = want if isinstance(want, list) else [want]
    if tool is None:
        return [f"модель не вызвала инструмент (ожидался {' или '.join(allowed)})"]
    if tool not in allowed:
        return [f"вызван {tool}, ожидался {' или '.join(allowed)}"]
    problems = []
    for key, value in (expect.get("checks") or {}).items():
        op, _, fld = key.partition(":")
        got = args.get(fld) if fld != "*" else None
        if op == "date":
            d = dates.resolve(got, today) if got not in (None, "") else None
            if d != value:
                problems.append(f"{fld}: ожидалось {value}, получено {got!r}" + (f" (= {d})" if d and d != got else ""))
        elif op == "set":
            gv = sorted(_norm(x) for x in _values(got))
            if gv != sorted(_norm(x) for x in value):
                problems.append(f"{fld}: ожидалось {value}, получено {got!r}")
        elif op == "eq":
            if _norm(got) != _norm(value):
                problems.append(f"{fld}: ожидалось {value!r}, получено {got!r}")
        elif op == "has":
            hay = " ".join(_norm(x) for vals in (args.values() if fld == "*" else [got]) for x in _values(vals))
            for needle in value if isinstance(value, list) else [value]:
                if _norm(needle) not in hay:
                    problems.append(f"{fld}: нет «{needle}» (получено {got if fld != '*' else args!r})")
        elif op == "absent":
            if _values(got):
                problems.append(f"{fld} не должно передаваться, получено {got!r}")
        elif key == "time":
            start, end = value
            gs, ge = str(args.get("start_time") or ""), str(args.get("end_time") or "")
            if gs.zfill(5) != start or ge.zfill(5) != end:
                problems.append(f"время: ожидалось {start}–{end}, получено {gs}–{ge}")
        else:
            problems.append(f"неизвестная проверка {key}")
    return problems


async def run_case(llm, case: dict, me: str | None, today: datetime) -> Result:
    async def runner(name: str, args: dict) -> str:
        raise _Captured(name, args)

    raw: dict = {}

    class Spy:
        """Запоминает аргументы модели до исправлений программы."""

        async def chat(self, messages, tools=None, think=False):
            reply = await llm.chat(messages, tools, think=think)
            if reply.tool_calls:
                raw.update(reply.tool_calls[0].arguments)
            return reply

    started = time.monotonic()
    tool, args, answer = None, {}, ""
    try:
        result = await agent.answer(Spy(), [{"role": "user", "content": case["q"]}], me=me, tool_runner=runner,
                                    think=case.get("think", False) or getattr(llm, "think_default", False))
        answer = result.text
    except _Captured as c:
        tool, args = c.name, c.call_args
    seconds = time.monotonic() - started
    problems = check(case["expect"], tool, args, today)
    return Result(case, not problems, tool, args, dict(raw), answer, seconds, problems)


def report(results: list[Result], model: str, think: bool) -> str:
    ok = sum(r.ok for r in results)
    total_s = sum(r.seconds for r in results)
    lines = [f"# Экзамен модели {model}{' (думать тщательнее)' if think else ''}: {ok} из {len(results)}", ""]
    groups: dict[str, list[Result]] = {}
    for r in results:
        groups.setdefault(r.case.get("group", "Прочее"), []).append(r)
    lines.append("| Тема | Верно |\n|---|---|")
    for g, rs in groups.items():
        lines.append(f"| {g} | {sum(r.ok for r in rs)} из {len(rs)} |")
    lines.append(f"\nВремя: всего {total_s:.0f} с, в среднем {total_s / max(len(results), 1):.1f} с на вопрос.\n")
    bad = [r for r in results if not r.ok]
    if bad:
        lines.append("## Ошибки\n")
        for r in bad:
            lines.append(f"**{r.case['id']}** «{r.case['q']}»  ")
            lines.append(f"вызов: `{r.tool}` {json.dumps(r.args, ensure_ascii=False)}  ")
            if r.raw_args and r.raw_args != r.args:
                lines.append(f"модель передала: {json.dumps(r.raw_args, ensure_ascii=False)}  ")
            if r.tool is None and r.answer:
                lines.append(f"ответ: {r.answer[:200]!r}  ")
            for p in r.problems:
                lines.append(f"- {p}")
            lines.append("")
    lines.append("## Все вопросы\n\n| | Вопрос | Вызов | с |\n|---|---|---|---|")
    for r in results:
        call = f"`{r.tool}` {json.dumps(r.args, ensure_ascii=False)}" if r.tool else "уточнение"
        lines.append(f"| {'✅' if r.ok else '❌'} | {r.case['q']} | {call.replace('|', '/')} | {r.seconds:.0f} |")
    return "\n".join(lines) + "\n"


async def main_async(args) -> int:
    spec = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    today = datetime.fromisoformat(spec["today"])
    agent.clock = lambda: today  # «понедельник» на экзамене всегда 9 ноября
    cases = [c for c in spec["cases"] if not args.only or c["id"] in args.only.split(",")]
    llm = Ollama(args.model)
    llm.think_default = args.think  # type: ignore[attr-defined]
    st = await llm.status()
    if not st["model_ready"]:
        print(f"Модель {args.model} недоступна: {st}", file=sys.stderr)
        return 2
    results = []
    for i, case in enumerate(cases, 1):
        me = case.get("me", spec.get("me")) if "me" in case else spec.get("me")
        try:
            r = await run_case(llm, case, me, today)
        except LLMError as e:
            r = Result(case, False, None, {}, {}, "", 0.0, [f"ошибка модели: {e}"])
        results.append(r)
        mark = "OK  " if r.ok else "FAIL"
        print(f"[{i}/{len(cases)}] {mark} {r.seconds:5.1f}s {case['id']}: {r.tool} {json.dumps(r.args, ensure_ascii=False)}"
              + ("" if r.ok else f"  <- {'; '.join(r.problems)}"), flush=True)
    await llm.close()
    text = report(results, args.model, args.think)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(text)
    ok = sum(r.ok for r in results)
    print(f"\nИтог: {ok} из {len(results)}")
    return 0 if ok / max(len(results), 1) >= args.min_score else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m mirea_assistant.evaluate")
    p.add_argument("cases", nargs="?", default="evals/cases.json")
    p.add_argument("--model", default=os.environ.get("MIREA_ASSISTANT_MODEL", "qwen3:4b"))
    p.add_argument("--think", action="store_true", help="режим «Думать тщательнее»")
    p.add_argument("--only", help="id вопросов через запятую")
    p.add_argument("--report", help="записать отчёт в markdown")
    p.add_argument("--min-score", type=float, default=0.0, help="минимальная доля верных для кода возврата 0")
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    return asyncio.run(main_async(p.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
