"""Общие подмены для тестов."""

from __future__ import annotations

import pytest

from mirea_assistant import cli


@pytest.fixture(autouse=True)
def no_message_box(monkeypatch):
    """Настоящее окно MessageBox остановило бы прогон тестов до нажатия «ОК»."""
    shown: list[str] = []
    monkeypatch.setattr(cli, "message_box", shown.append, raising=False)
    return shown
