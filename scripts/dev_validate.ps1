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
# Console 编码不会改变 Python 的 locale；父进程与子进程必须一起用 UTF-8。
# 否则 subprocess(text=True) 的 GBK 读取线程会失败，stdout 甚至变成 None。
$savedPythonUtf8 = $env:PYTHONUTF8
$savedPythonIoEncoding = $env:PYTHONIOENCODING
Push-Location $Repo
try {
    $env:PYTHONUTF8 = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    Write-Host "=== pytest tests/ ===" -ForegroundColor Cyan
    $t0 = [Diagnostics.Stopwatch]::StartNew()
    # 全量输出落盘：原来只 Select-Object -Last 4，失败用例名被截掉——实测踩到
    # 一次 1 failed 的门禁，事后无法定位是哪个用例（只知道总数），只能重跑碰运气。
    $logDir = Join-Path $Repo 'logs\experiments'
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $gateLog = Join-Path $logDir ("gate_" + $stamp + ".log")
    $pytestArgs = @('-m', 'pytest', 'tests/', '-o', 'addopts=', '-q')
    if ($Fast) { $pytestArgs += '-x' }
    & $py @pytestArgs 2>&1 | Tee-Object -FilePath $gateLog | Select-Object -Last 4
    $pytestRc = $LASTEXITCODE
    if ($pytestRc -ne 0) {
        Write-Host "--- 失败用例（完整日志见 $gateLog）---" -ForegroundColor Red
        Select-String -Path $gateLog -Pattern '^(FAILED|ERROR)' |
            ForEach-Object { Write-Host $_.Line -ForegroundColor Red }
    }
    Write-Host ("pytest 日志: {0}" -f $gateLog)
    Write-Host ("pytest rc={0} wall={1}s" -f $pytestRc,
                [math]::Round($t0.Elapsed.TotalSeconds, 1))

    Write-Host "=== scripts/m5_offline_validate.py ===" -ForegroundColor Cyan
    & $py 'scripts\m5_offline_validate.py' 2>&1 | Select-Object -Last 12
    $offRc = $LASTEXITCODE

    # 技术债棘轮：只许减少、不许增加（静默 except / print / god-function 行数）。
    # 它不是"风格检查"：静默吞异常会把"传感器没数据"变成一条正常的空结果。
    Write-Host "=== 技术债棘轮（scripts/m5_debt_meter.py --check）===" -ForegroundColor Cyan
    & $py 'scripts\m5_debt_meter.py' '--check' 2>&1 | Select-Object -Last 8
    $debtRc = $LASTEXITCODE

    # 文档索引新鲜度：新增/改名文档后忘记重生成，索引就会指向不存在的入口。
    # 2026-11-06 实测：索引停在 224 篇，实际已有 306 篇（82 篇无入口）。
    Write-Host "=== 文档索引（scripts/m5_docs_index.py --check）===" -ForegroundColor Cyan
    & $py 'scripts\m5_docs_index.py' '--check' 2>&1 | Select-Object -Last 4
    $idxRc = $LASTEXITCODE

    Write-Host ("RESULT: pytest={0} offline_validate={1} debt={2} docs_index={3}" -f `
        $(if ($pytestRc -eq 0) { 'PASS' } else { 'FAIL' }),
        $(if ($offRc -eq 0) { 'PASS' } else { 'FAIL' }),
        $(if ($debtRc -eq 0) { 'PASS' } else { 'FAIL' }),
        $(if ($idxRc -eq 0) { 'PASS' } else { 'FAIL' })) -ForegroundColor `
        $(if (($pytestRc -eq 0) -and ($offRc -eq 0) -and ($debtRc -eq 0) -and ($idxRc -eq 0)) { 'Green' } else { 'Red' })
    exit ([int]($pytestRc -ne 0) -bor ([int]($offRc -ne 0)) -bor ([int]($debtRc -ne 0)) -bor ([int]($idxRc -ne 0))
}
finally {
    $env:PYTHONUTF8 = $savedPythonUtf8
    $env:PYTHONIOENCODING = $savedPythonIoEncoding
    Pop-Location
}
