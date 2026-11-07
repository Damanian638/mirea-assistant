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


def test_update_stops_running_assistant_first():
    """Найдено прогоном на Windows: повторная установка падала с Access is denied, пока помощник запущен."""
    stop = PS.index("Stop-Assistant\n\n")
    install = PS.index("uv tool install --quiet")
    assert stop < install
    assert "/api/shutdown" in PS and "Stop-Process" in PS


def test_shortcut_name_survives_non_russian_windows():
    """Найдено прогоном на Windows с английской локалью: WScript.Shell сохранял «???????? ?????.lnk»."""
    assert "mirea-assistant-tmp.lnk" in PS and "Move-Item -LiteralPath $Tmp" in PS
    assert 'CreateShortcut((Join-Path $Dir "Помощник' not in PS


def test_stop_assistant_kills_only_the_assistant():
    """Ревью: фильтр «*python*» по всему ~\\.local\bin убивал python пользователя (uv python install --default)."""
    stop = PS[PS.index("function Stop-Assistant"): PS.index("Stop-Assistant\n\n")]
    assert '-like "*python*"' not in stop
    assert '"mirea-assistant*.exe"' in stop and '"$AppDir\\"' in stop


def test_uv_queries_do_not_abort_on_stderr_in_ps51():
    """Ревью: в PS 5.1 `uv tool dir 2>$null` при ErrorActionPreference=Stop падает от любого предупреждения uv."""
    stop = PS[PS.index("function Stop-Assistant"): PS.index("Stop-Assistant\n\n")]
    cont = stop.index('$ErrorActionPreference = "Continue"')
    assert cont < stop.index("uv tool dir 2>$null") < stop.index('$ErrorActionPreference = "Stop"')


def test_disk_space_checked_only_when_model_must_be_downloaded():
    """Ревью: при обновлении с уже скачанной qwen3:8b и 5 ГБ свободных установка падала «Мало места»."""
    assert "if (-not $HaveModel)" in PS and PS.index("if (-not $HaveModel)") < PS.index("Get-PSDrive")
