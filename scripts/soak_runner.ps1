# W3 8 小时持续运行验收：监督器（PowerShell 7）。
#
# 方案 §8.2：「常驻入口由项目自有调度器或 Windows 任务计划程序唤起**同一入口**。
# 任务计划程序只负责启动，互斥和恢复由库负责。」所以验收形状是：监督器反复唤起
# `m5_seg_autoloop.py run`，每次是一段"连续运行"（受 max_wall_minutes 约束），
# 总窗口 8 小时；每轮的结果（rc/耗时）写进 soak_iterations.jsonl，供事后判定。
#
# rc 的含义（都记下来，不吞）：0 正常结束；1 训练/流程失败；4 互斥（锁/租约被占，
# 说明互斥在工作）；5 资源门未通过；6/7 采集前置或身份审计未过；8 停止条件满足。
#
# 用法:
#   pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\soak_runner.ps1 `
#       -RunId t14_soak_20260926 -Config logs\experiments\t14_soak_20260926\loop_config.json `
#       -Hours 8 -GapSeconds 300
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$RunId,
    [Parameter(Mandatory = $true)][string]$Config,
    [double]$Hours = 8,
    [int]$GapSeconds = 300,
    [int]$MaxIterations = 0,
    [string]$Repo = (Split-Path -Parent $PSScriptRoot)
)
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$ErrorActionPreference = 'Continue'

$py = Join-Path $Repo '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { Write-Error "python venv not found: $py"; exit 2 }
$entry = Join-Path $Repo 'scripts\m5_seg_autoloop.py'
$runDir = Join-Path $Repo ("logs\experiments\" + $RunId)
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$iterLog = Join-Path $runDir 'soak_iterations.jsonl'
$deadline = (Get-Date).AddHours($Hours)

Push-Location $Repo
try {
    Write-Host ("[soak] run={0} window={1}h gap={2}s -> {3}" -f `
        $RunId, $Hours, $GapSeconds, $runDir)
    $iter = 0
    while ((Get-Date) -lt $deadline) {
        if ($MaxIterations -gt 0 -and $iter -ge $MaxIterations) { break }
        $iter++
        $t0 = Get-Date
        $log = Join-Path $runDir ("soak_iter_{0:d3}.log" -f $iter)
        & $py $entry 'run' '--run-id' $RunId '--no-dry-run' '--config' $Config `
            *>> $log
        $rc = $LASTEXITCODE
        $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 2)
        $rec = [ordered]@{
            iter = $iter; rc = $rc; minutes = $mins
            at = (Get-Date).ToString('s'); log = (Split-Path -Leaf $log)
        }
        Add-Content -Path $iterLog -Value ($rec | ConvertTo-Json -Compress) `
            -Encoding UTF8
        Write-Host ("[soak] iter {0}: rc={1} {2} min" -f $iter, $rc, $mins)
        if ((Get-Date) -ge $deadline) { break }
        Start-Sleep -Seconds $GapSeconds
    }
    $done = [ordered]@{
        run_id = $RunId; iterations = $iter; hours = $Hours
        finished_at = (Get-Date).ToString('s')
        note = "窗口结束；逐轮 rc/耗时见 soak_iterations.jsonl"
    }
    Add-Content -Path $iterLog -Value ($done | ConvertTo-Json -Compress) `
        -Encoding UTF8
    Write-Host ("[soak] 完成 {0} 轮，窗口 {1}h" -f $iter, $Hours)
}
finally { Pop-Location }
