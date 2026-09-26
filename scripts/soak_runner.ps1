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
    # 给了前缀就"每轮一个新 run"（见文件头注释）；缺省沿用同一个 RunId
    [string]$RunIdPrefix = "",
    # 给了目录就"每轮一个不同的单因子提议"：目录里放 proposals_iNN.json，
    # 按轮次取（用完从头循环）。这样 4 小时里每轮都是新因子，不重复同一训练。
    [string]$ProposalDir = "",
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
    Write-Host ("[soak] run={0}{1} window={2}h gap={3}s -> {4}" -f `
        $RunId, $(if ($RunIdPrefix) { " (每轮新 run: ${RunIdPrefix}_iNN)" } else { "" }),
        $Hours, $GapSeconds, $runDir)
    $iter = 0
    while ((Get-Date) -lt $deadline) {
        if ($MaxIterations -gt 0 -and $iter -ge $MaxIterations) { break }
        $iter++
        $t0 = Get-Date
        $log = Join-Path $runDir ("soak_iter_{0:d3}.log" -f $iter)
        if ($RunIdPrefix) {
            $iterRun = "{0}_i{1:d2}" -f $RunIdPrefix, $iter
            $iterCfg = Join-Path $runDir ("loop_config_i{0:d2}.json" -f $iter)
            Copy-Item -LiteralPath $Config -Destination $iterCfg -Force
            if ($ProposalDir) {
                $cand = Get-ChildItem -Path $ProposalDir -Filter 'proposals_i*.json' |
                    Sort-Object Name
                if ($cand.Count -gt 0) {
                    $pick = $cand[($iter - 1) % $cand.Count].FullName
                    $blob = Get-Content -Raw -Encoding UTF8 $iterCfg | ConvertFrom-Json
                    $blob.proposals = $pick
                    ($blob | ConvertTo-Json -Depth 6) | Set-Content -Encoding UTF8 $iterCfg
                    Write-Host ("[soak] iter {0}: 因子 <- {1}" -f $iter,
                                (Split-Path -Leaf $pick))
                }
            }
        } else {
            $iterRun = $RunId
            $iterCfg = $Config
        }
        & $py $entry 'run' '--run-id' $iterRun '--no-dry-run' '--config' $iterCfg `
            *>> $log
        $rc = $LASTEXITCODE
        $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 2)
        $rec = [ordered]@{
            iter = $iter; rc = $rc; minutes = $mins; run_id = $iterRun
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
