# 一键标注一个任务包（E1 负例包 / 正例包），逐视角打开标注器。
#
# 为什么要它：`m5_annotate_package.py` 生成的包是"**原地**标注"（输出写回同一个
# 视角目录），一个包有 4 个视角 = 4 条命令；漏标任何一个视角，`m5_annotation_
# readiness.py` 会判"未就绪"（缺 label），那一轮就白等了。这个脚本把 4 条命令
# 串起来：关掉一个窗口自动进下一个视角。
#
# 用法（PowerShell 7）:
#   # 先看会执行什么（不开窗口）
#   pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\annotate_pkg.ps1 -Package logs\experiments\annotate_pkg_e1_jv_20260927 -List
#   # 真开始（负例包：按"确认无线"核对）
#   pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\annotate_pkg.ps1 -Package logs\experiments\annotate_pkg_e1_jv_20260927 -Reviewer owner
#   # 正例包：画漆线
#   pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\annotate_pkg.ps1 -Package logs\experiments\annotate_pkg_e2_it3_20260927 -Reviewer owner
#   # 标注完做就绪核对（不通过就别往下跑训练）
#   .venv\Scripts\python.exe scripts\m5_annotation_readiness.py --dir <包>\front_main --dir <包>\pillar_left ...
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Package,
    [string]$Prefill = "logs\m5_seg\seg_model\best.pt",
    [switch]$NoPrefill,
    [string]$Reviewer = "",
    [string[]]$Views = @(),
    [switch]$List,
    [string]$Repo = (Split-Path -Parent $PSScriptRoot)
)
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$ErrorActionPreference = 'Continue'

$py = Join-Path $Repo '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { Write-Error "python venv not found: $py"; exit 2 }
# 子进程**不缓冲**输出：实测踩到——标注器被强关（点窗口的 X / 被杀）时，Python 的
# 缓冲 stdout 会连同"为什么崩"一起丢掉，日志里只剩一句退出码，无法定位。
$env:PYTHONUNBUFFERED = '1'
# 窗口被强关时 OpenCV 可能直接结束进程（不会走我们的收尾）——所以每跑完一个视角
# 都重新检查一次进程/窗口状态，并明确告诉用户"关窗口请用 q，不要点 X"。
$env:PYTHONIOENCODING = 'utf-8'
$pkgRoot = Join-Path $Repo $Package
if (-not (Test-Path $pkgRoot)) { Write-Error "package not found: $pkgRoot"; exit 3 }

# 视角 = 含 frame_*.npz 的直接子目录（按名字排序，确定性）
$found = @()
if ($Views.Count -gt 0) {
    $found = $Views
} else {
    foreach ($d in (Get-ChildItem -Path $pkgRoot -Directory | Sort-Object Name)) {
        if (Get-ChildItem -Path $d.FullName -Filter 'frame_*.npz' -ErrorAction SilentlyContinue) {
            $found += $d.Name
        }
    }
}
if ($found.Count -eq 0) {
    Write-Error "包里没有含 frame_*.npz 的视角目录：$pkgRoot（空包不许静默通过）"
    exit 4
}

$prefillArgs = @()
if (-not $NoPrefill) {
    $ck = Join-Path $Repo $Prefill
    if (Test-Path $ck) { $prefillArgs = @('--prefill-model', $Prefill) }
    else { Write-Host "[annotate] 预填权重不存在，改为从空白画：$ck" -ForegroundColor Yellow }
}
$reviewerArgs = @()
if ($Reviewer) { $reviewerArgs = @('--reviewer', $Reviewer) }

Write-Host ("[annotate] 包 {0}：{1} 个视角 -> {2}" -f $Package, $found.Count, ($found -join ', ')) -ForegroundColor Cyan
$i = 0
foreach ($v in $found) {
    $i++
    $dir = Join-Path $pkgRoot $v
    $argv = @('scripts\m5_annotate_manual.py', '--frames-dir', "$Package\$v",
              '--out', "$Package\$v") + $prefillArgs + $reviewerArgs
    if ($List) {
        Write-Host ("[annotate] {0}/{1} 将执行: {2} {3}" -f $i, $found.Count, $py, ($argv -join ' '))
        continue
    }
    Write-Host ("[annotate] {0}/{1} 打开 {2}（关掉窗口后自动进下一个）" -f $i, $found.Count, $v) -ForegroundColor Cyan
    Push-Location $Repo
    Write-Host ("[annotate] 提示：画完/退出请按 q（点窗口右上角的 X 可能直接杀掉进程、" -f $null) -ForegroundColor DarkGray
    Write-Host ("[annotate]       留下关不掉的空窗口）。画线用鼠标拖动；工具栏按钮也能点。" -f $null) -ForegroundColor DarkGray
    try { & $py -u @argv } finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) {
        Write-Host ("[annotate] {0} 退出码 {1}：停下来，别继续标下一个（先查这个视角）" -f $v, $LASTEXITCODE) -ForegroundColor Red
        exit 5
    }
}
if ($List) { Write-Host "[annotate] -List：未打开任何窗口" -ForegroundColor Cyan }
else {
    Write-Host "[annotate] 全部视角已跑完。下一步（就绪核对，不通过不要往下跑）：" -ForegroundColor Green
    Write-Host ("  .venv\Scripts\python.exe scripts\m5_annotation_readiness.py " + (($found | ForEach-Object { "--dir $Package\$_" }) -join ' '))
}
