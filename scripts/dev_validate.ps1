# 项目离线校验一键入口（PowerShell 7）。
#   用法: pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\dev_validate.ps1
# 只跑不需要游戏的两条门：pytest tests/ 与 scripts\m5_offline_validate.py。
# 为什么要有它：AGENTS.md 的"常用验证"要求这两条每次改动都跑，手敲容易漏一项
# 或漏掉 `-o addopts=`（本项目 pytest.ini 带 addopts，漏了会跑成另一套配置）。
[CmdletBinding()]
param(
    [switch]$Fast,          # 只跑失败即停的快速模式（-x）
    [string]$Repo = (Split-Path -Parent $PSScriptRoot)
)
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$ErrorActionPreference = 'Continue'

$py = Join-Path $Repo '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) {
    Write-Error "python venv not found: $py"
    exit 2
}
Push-Location $Repo
try {
    Write-Host "=== pytest tests/ ===" -ForegroundColor Cyan
    $t0 = [Diagnostics.Stopwatch]::StartNew()
    $pytestArgs = @('-m', 'pytest', 'tests/', '-o', 'addopts=', '-q')
    if ($Fast) { $pytestArgs += '-x' }
    & $py @pytestArgs 2>&1 | Select-Object -Last 4
    $pytestRc = $LASTEXITCODE
    Write-Host ("pytest rc={0} wall={1}s" -f $pytestRc,
                [math]::Round($t0.Elapsed.TotalSeconds, 1))

    Write-Host "=== scripts/m5_offline_validate.py ===" -ForegroundColor Cyan
    & $py 'scripts\m5_offline_validate.py' 2>&1 | Select-Object -Last 12
    $offRc = $LASTEXITCODE

    Write-Host ("RESULT: pytest={0} offline_validate={1}" -f `
        $(if ($pytestRc -eq 0) { 'PASS' } else { 'FAIL' }),
        $(if ($offRc -eq 0) { 'PASS' } else { 'FAIL' })) -ForegroundColor `
        $(if (($pytestRc -eq 0) -and ($offRc -eq 0)) { 'Green' } else { 'Red' })
    exit ([int]($pytestRc -ne 0) -bor ([int]($offRc -ne 0)))
}
finally {
    Pop-Location
}
