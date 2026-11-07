# Помощник МИРЭА — установка и обновление одной командой (Windows, PowerShell):
#   powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/Damanian638/mirea-assistant/main/install.ps1 | iex"
# Ставит: uv (Python), программу-помощника, Ollama (локальный ИИ), модель по объёму памяти, ярлык на рабочий стол.
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # без этого Invoke-WebRequest в Windows PowerShell качает в разы медленнее
$Src = if ($env:MIREA_ASSISTANT_SRC) { $env:MIREA_ASSISTANT_SRC } else { "https://github.com/Damanian638/mirea-assistant/archive/refs/heads/main.zip" }
$Issues = "https://github.com/Damanian638/mirea-assistant/issues"

# uv и Ollama печатают пути в UTF-8; без этого путь C:\Users\Иван читается кракозябрами.
$PrevOutputEncoding = [Console]::OutputEncoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Fail([string]$Message) {
    [Console]::OutputEncoding = $PrevOutputEncoding
    Write-Host ""
    Write-Host "Ошибка: $Message" -ForegroundColor Red
    Write-Host "Установка не завершена. Если ошибка повторяется, сообщите о ней: $Issues"
    # throw, а не exit: при `irm | iex` в открытом окне exit закрыл бы PowerShell пользователя.
    throw "Помощник МИРЭА: $Message"
}

function Step([string]$Text) { Write-Host ""; Write-Host "-> $Text" -ForegroundColor Cyan }

# --- 1. uv (менеджер Python) ---
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Step "Ставлю uv (менеджер Python от Astral)..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    if ($LASTEXITCODE -ne 0) { Fail "Не удалось установить uv. Проверьте подключение к интернету." }
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Fail "uv не найден после установки. Откройте новое окно PowerShell и повторите." }
}

# --- 2. Сама программа ---
Step "Ставлю программу «Помощник МИРЭА»..."
uv tool install --force --reinstall --python 3.12 $Src
if ($LASTEXITCODE -ne 0) {
    Fail ("Не удалось установить программу (см. сообщения выше). Если там «Отказано в доступе» или " +
          "«used by another process» — помощник сейчас запущен: Закройте его кнопкой «Выключить помощника» " +
          "на его странице и повторите команду.")
}
$BinDir = uv tool dir --bin
if ($LASTEXITCODE -ne 0 -or -not $BinDir) { Fail "Не удалось определить каталог программ uv." }
$BinDir = ($BinDir | Select-Object -First 1).Trim()
$Bin = Join-Path $BinDir "mirea-assistant.exe"
$App = Join-Path $BinDir "mirea-assistant-app.exe"  # запуск без чёрного окна
if (-not (Test-Path -LiteralPath $Bin)) { Fail "Не найдена установленная программа: $Bin" }
if (-not (Test-Path -LiteralPath $App)) { $App = $Bin }

# --- 3. Ollama (в ней работает локальный ИИ) ---
$OllamaDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
$Ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
if (-not $Ollama -and (Test-Path -LiteralPath (Join-Path $OllamaDir "ollama.exe"))) { $Ollama = Join-Path $OllamaDir "ollama.exe" }
if (-not $Ollama) {
    Step "Ставлю Ollama — программу, в которой работает локальный ИИ (около 1 ГБ)..."
    $Installed = $false
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install --id Ollama.Ollama -e --silent --accept-source-agreements --accept-package-agreements
        $Installed = ($LASTEXITCODE -eq 0)
    }
    if (-not $Installed) {
        $Setup = Join-Path $env:TEMP "OllamaSetup.exe"
        try { Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $Setup -UseBasicParsing }
        catch { Fail "Не удалось скачать Ollama. Проверьте интернет или установите вручную с https://ollama.com/download и повторите." }
        $p = Start-Process -FilePath $Setup -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART" -Wait -PassThru
        if ($p.ExitCode -ne 0) { Fail "Установщик Ollama завершился с ошибкой (код $($p.ExitCode))." }
    }
    $env:Path = "$OllamaDir;$env:Path"
    $Ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
    if (-not $Ollama) { Fail "Ollama не найдена после установки. Перезагрузите компьютер и повторите команду." }
}

# Ollama должна работать в фоне; обычно она запускается сама, иначе запускаем.
function Test-Ollama { try { Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 3 | Out-Null; $true } catch { $false } }
if (-not (Test-Ollama)) {
    Start-Process -FilePath $Ollama -ArgumentList "serve" -WindowStyle Hidden
    for ($i = 0; $i -lt 30 -and -not (Test-Ollama); $i++) { Start-Sleep -Seconds 1 }
    if (-not (Test-Ollama)) { Fail "Ollama не запустилась. Откройте Ollama из меню «Пуск», подождите минуту и повторите команду." }
}

# --- 4. Модель по объёму памяти ---
$RamGb = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
if ($env:MIREA_ASSISTANT_MODEL) { & "$Bin" configure --model $env:MIREA_ASSISTANT_MODEL } else { & "$Bin" configure --ram-gb $RamGb }
if ($LASTEXITCODE -ne 0) { Fail "Не удалось сохранить настройки помощника." }
$Model = (& "$Bin" model | Select-Object -Last 1).Trim()
if ($LASTEXITCODE -ne 0 -or -not $Model) { Fail "Не удалось определить модель." }

$Drive = Get-PSDrive -Name ($env:USERPROFILE.Substring(0, 1))
$FreeGb = [math]::Round($Drive.Free / 1GB, 1)
$NeedGb = if ($Model -like "*8b*") { 7 } elseif ($Model -like "*1.7b*") { 2 } else { 4 }
if ($FreeGb -lt $NeedGb) { Fail "Мало места на диске: свободно $FreeGb ГБ, для модели $Model нужно около $NeedGb ГБ." }

Step "Скачиваю модель ИИ $Model (оперативной памяти: $RamGb ГБ). Это несколько гигабайт, может занять 5–30 минут..."
& "$Ollama" pull $Model
if ($LASTEXITCODE -ne 0) { Fail "Не удалось скачать модель $Model. Проверьте интернет и повторите команду: скачивание продолжится." }

# --- 5. Ярлыки ---
Step "Создаю ярлык «Помощник МИРЭА» на рабочем столе..."
try {
    $Shell = New-Object -ComObject WScript.Shell
    foreach ($Dir in @([Environment]::GetFolderPath("Desktop"), [Environment]::GetFolderPath("Programs"))) {
        $Lnk = $Shell.CreateShortcut((Join-Path $Dir "Помощник МИРЭА.lnk"))
        $Lnk.TargetPath = $App
        $Lnk.WorkingDirectory = $env:USERPROFILE
        $Lnk.Description = "Помощник преподавателя РТУ МИРЭА"
        $Lnk.Save()
    }
} catch {
    Write-Host "Ярлык создать не удалось. Запускайте помощника командой: $Bin" -ForegroundColor Yellow
}

# --- 6. Запуск ---
[Console]::OutputEncoding = $PrevOutputEncoding
Start-Process -FilePath $App
Write-Host ""
Write-Host "Готово. Помощник открывается в браузере; дальше запускайте его ярлыком «Помощник МИРЭА» на рабочем столе."
