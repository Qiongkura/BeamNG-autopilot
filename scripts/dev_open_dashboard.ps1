# 打开"实验看板"（静态 HTML，10 个分区）——不是 8760 那个实时监控页。
#
#   用法: pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\dev_open_dashboard.ps1
#         pwsh ... -File scripts\dev_open_dashboard.ps1 -RunId t14_plateau_arm5 -Open
#
# 它做三件事：挑一个运行目录（默认取最新有 checkpoint 的）→ 渲染看板 HTML 到
# logs\dashboards\ → 打印文件路径与 file:// URL（-Open 则顺手打开）。
# 只读：不训练、不接触游戏、不改运行产物（只往 logs\dashboards\ 写渲染结果）。
[CmdletBinding()]
param(
    [string]$RunId = "",
    [string]$ExperimentsRoot = "",
    [string]$Champion = "baseline",
    [switch]$Open
)
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$ErrorActionPreference = 'Continue'

$repo = Split-Path -Parent $PSScriptRoot
if (-not $ExperimentsRoot) { $ExperimentsRoot = Join-Path $repo 'logs\experiments' }
$py = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { Write-Error "python venv not found: $py"; exit 2 }

if (-not $RunId) {
    # 最新一个"有 checkpoint 或判定"的运行目录（排除纯日志目录）
    $cand = Get-ChildItem -Directory $ExperimentsRoot |
        Where-Object {
            (Test-Path (Join-Path $_.FullName 'champion.json')) -or
            (Test-Path (Join-Path $_.FullName 'events.jsonl')) -or
            (Get-ChildItem -Path $_.FullName -Filter 'checkpoint_last.pt' -Recurse -ErrorAction SilentlyContinue |
                Select-Object -First 1)
        } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $cand) { Write-Error "no run directory under $ExperimentsRoot"; exit 3 }
    $RunId = $cand.Name
}
$runDir = Join-Path $ExperimentsRoot $RunId
if (-not (Test-Path $runDir)) { Write-Error "run not found: $runDir"; exit 3 }

$outDir = Join-Path $repo 'logs\dashboards'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$out = Join-Path $outDir "$RunId.html"
$manifest = Join-Path $runDir 'rounds_dataset.json'
$argsList = @((Join-Path $repo 'scripts\m5_seg_dashboard.py'), 'render',
              '--run-dir', $runDir, '--out', $out)
if (Test-Path $manifest) { $argsList += @('--manifest', $manifest) }
if ($Champion) { $argsList += @('--champion', $Champion) }

& $py @argsList
if ($LASTEXITCODE -ne 0) { Write-Error "render failed (rc=$LASTEXITCODE)"; exit $LASTEXITCODE }

# 根 URL 直接就是看板本身（不是运行清单）：把这一份同时写成 index.html
Copy-Item -Force $out (Join-Path $outDir 'index.html')

Write-Host ""
Write-Host "看板已生成: $out" -ForegroundColor Green
Write-Host "根地址即本页: http://127.0.0.1:8761/（每次渲染自动指向最新运行）"
$url = 'file:///' + ($out -replace '\\', '/')
Write-Host "浏览器可开: $url"
Write-Host "（这是静态页面，关掉不影响任何训练；8760 那个是另一个实时监控程序）"
if ($Open) { Start-Process $url }
