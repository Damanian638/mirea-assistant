"""install.ps1: Windows здесь не запустить, поэтому фиксируем обязательные защиты статически."""

from __future__ import annotations

import re
from pathlib import Path

PS = (Path(__file__).resolve().parents[1] / "install.ps1").read_text(encoding="utf-8")
LINES = [ln for ln in PS.splitlines() if ln.strip() and not ln.strip().startswith("#")]


def test_every_native_step_checks_exit_code():
    call = re.compile(r'^\s*(\$\w+\s*=\s*)?(uv\s|powershell\s|winget\s|&\s*"\$(Bin|Ollama)")')
    native = [i for i, ln in enumerate(LINES) if call.search(ln)]
    assert len(native) >= 5  # uv, powershell, winget, ollama pull и т. д.
    for i in native:
        tail = "\n".join(LINES[i : i + 3])
        assert "$LASTEXITCODE" in tail or "$Installed" in tail, f"нет проверки после: {LINES[i]}"


def test_success_only_at_the_end_and_failures_throw():
    done = [i for i, ln in enumerate(LINES) if "Готово" in ln]
    assert len(done) == 1 and done[0] == len(LINES) - 1
    assert "throw" in PS and not re.search(r"^\s*exit\b", PS, re.M)


def test_windows_specifics():
    assert "[Console]::OutputEncoding" in PS and "UTF8" in PS  # кириллица в пути профиля
    assert "--reinstall" in PS
    assert 'GetFolderPath("Desktop")' in PS  # рабочий стол в OneDrive / с кириллицей
    assert "mirea-assistant-app.exe" in PS  # ярлык без окна консоли
    assert "Win32_ComputerSystem" in PS and "configure --ram-gb" in PS
    assert "Get-PSDrive" in PS  # проверка места до скачивания модели
    assert "/VERYSILENT" in PS and "winget install --id Ollama.Ollama" in PS
