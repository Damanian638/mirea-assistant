# Помощник МИРЭА — установка и обновление одной командой (Windows, PowerShell):
#   irm https://raw.githubusercontent.com/Damanian638/mirea-assistant/main/install.ps1 | iex
# Ставит: uv (Python), программу, Ollama (локальный ИИ), модель по объёму памяти, ярлык на рабочем столе.
# Повторный запуск обновляет программу; уже скачанное не скачивается заново.
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # без этого Invoke-WebRequest в Windows PowerShell качает в разы медленнее
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
$Src = if ($env:MIREA_ASSISTANT_SRC) { $env:MIREA_ASSISTANT_SRC } else { "https://github.com/Damanian638/mirea-assistant/archive/refs/heads/main.zip" }
$Issues = "https://github.com/Damanian638/mirea-assistant/issues"

# uv и Ollama печатают пути в UTF-8; без этого путь C:\Users\Иван читается кракозябрами.
$PrevOutputEncoding = [Console]::OutputEncoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Fail([string]$Message) {
    [Console]::OutputEncoding = $PrevOutputEncoding
    Write-Host ""
    Write-Host "Ошибка: $Message" -ForegroundColor Red
    Write-Host "Установка не завершена. Повторите команду; если ошибка не исчезает, сообщите о ней: $Issues"
    # throw, а не exit: при `irm | iex` в открытом окне exit закрыл бы PowerShell пользователя.
    throw "Помощник МИРЭА: $Message"
}

$script:StepNo = 0
function Step([string]$Text) {
    $script:StepNo++
    Write-Host ""
    Write-Host "[$($script:StepNo)/5] $Text" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Установка «Помощника МИРЭА». Займёт 5–30 минут, окно не закрывайте." -ForegroundColor Green

# --- 1. Python (через uv) и сама программа ---
Step "Программа «Помощник МИРЭА»"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "    ставлю uv — установщик Python..."
    powershell -NoProfile -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" | Out-Null
    if ($LASTEXITCODE -ne 0) { Fail "Не удалось установить uv. Проверьте подключение к интернету." }
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Fail "uv не найден после установки. Откройте новое окно PowerShell и повторите." }
}
# Обновление: работающий помощник держит свои файлы, и uv не может их заменить («Access is denied»).
function Stop-Assistant {
    $port = 8765
    $cfg = Join-Path $env:USERPROFILE ".mirea-assistant\config.json"
    if (Test-Path -LiteralPath $cfg) {
        try { $p = (Get-Content -Raw -Encoding UTF8 -LiteralPath $cfg | ConvertFrom-Json).port; if ($p) { $port = [int]$p } } catch {}
    }
    try {
        Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$port/api/shutdown" -ContentType "application/json" -Body "{}" -TimeoutSec 3 | Out-Null
        Write-Host "    выключаю запущенного помощника для обновления..."
        Start-Sleep -Seconds 2
    } catch {}
    $dirs = @()
    $tools = uv tool dir 2>$null
    if ($LASTEXITCODE -eq 0 -and $tools) { $dirs += Join-Path ($tools | Select-Object -First 1).Trim() "mirea-assistant" }
    $bins = uv tool dir --bin 2>$null
    if ($LASTEXITCODE -eq 0 -and $bins) { $dirs += ($bins | Select-Object -First 1).Trim() }
    Get-Process -ErrorAction SilentlyContinue | Where-Object {
        $path = $_.Path
        $path -and ($dirs | Where-Object { $path.StartsWith($_, [StringComparison]::OrdinalIgnoreCase) }) -and
            ($path -like "*mirea-assistant*" -or $path -like "*python*")
    } | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
}
Stop-Assistant

Write-Host "    скачиваю и устанавливаю..."
uv tool install --quiet --force --reinstall --python 3.12 $Src
if ($LASTEXITCODE -ne 0) {
    Fail ("Не удалось установить программу (см. сообщения выше). Если там «Отказано в доступе» или " +
          "«used by another process» — помощник сейчас запущен: Закройте его ссылкой «Выключить помощника» " +
          "на его странице и повторите команду.")
}
$BinDir = uv tool dir --bin
if ($LASTEXITCODE -ne 0 -or -not $BinDir) { Fail "Не удалось определить каталог программ uv." }
$BinDir = ($BinDir | Select-Object -First 1).Trim()
$Bin = Join-Path $BinDir "mirea-assistant.exe"
$App = Join-Path $BinDir "mirea-assistant-app.exe"  # запуск без чёрного окна
if (-not (Test-Path -LiteralPath $Bin)) { Fail "Не найдена установленная программа: $Bin" }
if (-not (Test-Path -LiteralPath $App)) { $App = $Bin }
Write-Host "    готово"

# --- 2. Ollama (в ней работает локальный ИИ) ---
Step "Ollama — программа, в которой работает ИИ"
$OllamaDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
$Ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
if (-not $Ollama -and (Test-Path -LiteralPath (Join-Path $OllamaDir "ollama.exe"))) { $Ollama = Join-Path $OllamaDir "ollama.exe" }
if ($Ollama) {
    Write-Host "    уже установлена"
} else {
    Write-Host "    скачиваю и устанавливаю (около 1 ГБ)..."
    $Installed = $false
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install --id Ollama.Ollama -e --silent --accept-source-agreements --accept-package-agreements | Out-Null
        $Installed = ($LASTEXITCODE -eq 0)
    }
    if (-not $Installed) {
        $Setup = Join-Path $env:TEMP "OllamaSetup.exe"
        try { Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $Setup -UseBasicParsing }
        catch { Fail "Не удалось скачать Ollama. Проверьте интернет или установите её вручную с https://ollama.com/download и повторите." }
        $p = Start-Process -FilePath $Setup -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART" -Wait -PassThru
        if ($p.ExitCode -ne 0) { Fail "Установщик Ollama завершился с ошибкой (код $($p.ExitCode))." }
    }
    $env:Path = "$OllamaDir;$env:Path"
    $Ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
    if (-not $Ollama) { Fail "Ollama не найдена после установки. Перезагрузите компьютер и повторите команду." }
    Write-Host "    готово"
}

# Ollama должна работать в фоне; обычно она запускается сама, иначе запускаем.
function Test-Ollama { try { Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 3 | Out-Null; $true } catch { $false } }
if (-not (Test-Ollama)) {
    Start-Process -FilePath $Ollama -ArgumentList "serve" -WindowStyle Hidden
    for ($i = 0; $i -lt 30 -and -not (Test-Ollama); $i++) { Start-Sleep -Seconds 1 }
    if (-not (Test-Ollama)) { Fail "Ollama не запустилась. Откройте Ollama из меню «Пуск», подождите минуту и повторите команду." }
}

# --- 3. Выбор модели по объёму памяти ---
Step "Подбираю модель ИИ под этот компьютер"
$RamGb = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
if ($env:MIREA_ASSISTANT_MODEL) { & "$Bin" configure --model $env:MIREA_ASSISTANT_MODEL | Out-Null } else { & "$Bin" configure --ram-gb $RamGb | Out-Null }
if ($LASTEXITCODE -ne 0) { Fail "Не удалось сохранить настройки помощника." }
$Model = (& "$Bin" model | Select-Object -Last 1).Trim()
if ($LASTEXITCODE -ne 0 -or -not $Model) { Fail "Не удалось определить модель." }
$Drive = Get-PSDrive -Name ($env:USERPROFILE.Substring(0, 1))
$FreeGb = [math]::Round($Drive.Free / 1GB, 1)
$NeedGb = if ($Model -like "*8b*") { 7 } elseif ($Model -like "*1.7b*") { 2 } else { 4 }
if ($FreeGb -lt $NeedGb) { Fail "Мало места на диске: свободно $FreeGb ГБ, для модели нужно около $NeedGb ГБ. Освободите место и повторите." }
Write-Host "    памяти $RamGb ГБ — модель $Model"

# --- 4. Скачивание модели ---
Step "Скачиваю модель $Model (несколько гигабайт; самая долгая часть)"
& "$Ollama" pull $Model
if ($LASTEXITCODE -ne 0) { Fail "Не удалось скачать модель. Проверьте интернет и повторите команду: скачивание продолжится с места остановки." }

# --- 5. Ярлык и запуск ---
Step "Ярлык на рабочем столе и запуск"
$Made = 0
$Shell = New-Object -ComObject WScript.Shell
foreach ($Dir in @([Environment]::GetFolderPath("Desktop"), [Environment]::GetFolderPath("Programs"))) {
    if (-not $Dir) { continue }
    try {
        if (-not (Test-Path -LiteralPath $Dir)) { New-Item -ItemType Directory -Path $Dir -Force | Out-Null }
        $Lnk = $Shell.CreateShortcut((Join-Path $Dir "Помощник МИРЭА.lnk"))
        $Lnk.TargetPath = $App
        $Lnk.WorkingDirectory = $env:USERPROFILE
        $Lnk.Description = "Помощник преподавателя РТУ МИРЭА"
        $Lnk.Save()
        $Made++
    } catch {
        Write-Host "    ярлык в «$Dir» не создан: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}
if ($Made -eq 0) { Write-Host "    запускайте помощника командой: $App" -ForegroundColor Yellow } else { Write-Host "    готово" }
[Console]::OutputEncoding = $PrevOutputEncoding
Start-Process -FilePath $App
Write-Host ""
Write-Host "Готово! Помощник открывается в браузере. Дальше запускайте его ярлыком «Помощник МИРЭА» на рабочем столе." -ForegroundColor Green
