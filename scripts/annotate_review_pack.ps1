# 打开 W1 §6.3 复核包（起步 14 帧，按采集分成 7 个包）。
#
# 为什么逐个包打开：标注包的身份是"一次采集一个 meta（map_name/source_id）"，
# 一次只能标注一个采集的帧；窗口关掉后自动进下一个包。
#
# 输出写到**新目录**（默认 <PackDir>\reviewed\<包名>\<视角>），源包保持原样——
# 方案 §6.3 要求"人工编辑输出到新数据版本目录，保留输入/输出哈希和审阅差异"，
# 所以源包不覆盖，事后我按两个目录算哈希与差异。
#
# 用法:
#   # 先看会执行什么（不开窗口）
#   pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\annotate_review_pack.ps1 -List
#   # 真开始（逐个包，关掉一个自动进下一个）
#   pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\annotate_review_pack.ps1
#   # 只做某几个包
#   pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\annotate_review_pack.ps1 -Only pkg_town,pkg_wide
[CmdletBinding()]
param(
    [string]$PackDir = "logs\experiments\review_pack_20260926",
    [string]$OutRoot = "",
    [string]$Prefill = "logs\experiments\t14_e0_20260925\seed43\checkpoint_last.pt",
    [switch]$NoPrefill,
    [string[]]$Only = @(),
    [string]$Reviewer = "",
    [switch]$Full,               # 标注**全量**包（packages_full -> reviewed_full）      # 复核人标识（写进 meta.json 的 annotation.reviewer）
    [switch]$List,
    [string]$View = "front_main",
    [string]$Repo = (Split-Path -Parent $PSScriptRoot)
)
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$ErrorActionPreference = 'Continue'

$py = Join-Path $Repo '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { Write-Error "python venv not found: $py"; exit 2 }
$packRoot = Join-Path $Repo $PackDir
$pkgSub = if ($Full) { 'packages_full' } else { 'packages' }
$pkgRoot = Join-Path $packRoot $pkgSub
if (-not (Test-Path $pkgRoot)) { Write-Error "no packages under $pkgRoot"; exit 3 }
if (-not $OutRoot) {
    $OutRoot = if ($Full) { Join-Path $packRoot 'reviewed_full' }
               else { Join-Path $packRoot 'reviewed' }
}

# 类别提示：按**完整路径**匹配（不同采集里同名帧很常见，只按文件名查会串味——
# 实测踩到：砾石土路的帧被显示成"清晰漆线"）。文件名只作兜底。
$byPath = @{}
$byName = @{}
foreach ($wsName in @('review_worksheet.json', 'review_worksheet_full.json')) {
    $wsPath = Join-Path $packRoot $wsName
    if (-not (Test-Path $wsPath)) { continue }
    $ws = Get-Content -Raw -Encoding UTF8 $wsPath | ConvertFrom-Json
    foreach ($row in $ws.rows) {
        if (-not $row.path) { continue }
        $byPath[$row.path] = $row.category
        $k = Split-Path -Leaf $row.path
        if (-not $byName.ContainsKey($k)) { $byName[$k] = $row.category }
    }
}

$prefillPath = Join-Path $Repo $Prefill
if (-not $NoPrefill -and -not (Test-Path $prefillPath)) {
    Write-Host "[annotate] 预填模型不存在，改为空白画：$prefillPath" -ForegroundColor Yellow
    $NoPrefill = $true
}

Push-Location $Repo
try {
    $all = Get-ChildItem -Directory $pkgRoot -Filter 'pkg_*' | Sort-Object Name
    $pkgs = $all
    if ($Only.Count -gt 0) {
        # `pwsh -File` 会把 `-Only a,b` 当**一个字符串**传进来（不做数组解析），
        # 所以这里自己按逗号拆——否则匹配为空（实测踩到："没有匹配的包"）。
        $pats = @()
        foreach ($o in $Only) {
            $pats += @(($o -split ',') | ForEach-Object { $_.Trim() } | Where-Object { $_ })
        }
        $pkgs = $all | Where-Object {
            $n = $_.Name
            @($pats | Where-Object { $n -like "*$_*" }).Count -gt 0
        }
    }
    if ($pkgs.Count -eq 0) {
        Write-Host "[annotate] -Only 没匹配到包。可用包名：" -ForegroundColor Yellow
        foreach ($d in $all) { Write-Host ("  " + $d.Name) }
        exit 4
    }

    Write-Host ("[annotate] 共 {0} 个包；输出 -> {1}" -f $pkgs.Count, $OutRoot)
    Write-Host "[annotate] 键位：1=line 2=road 3=背景/擦除 · b 画笔/填充 · u 撤销 · c 清空 · a 上一帧 · s 保存并下一帧 · q 退出"
    Write-Host "[annotate] 规则：只标你**实际看过并确认**的区域；没看过留未知，不要当'确定没有漆线'"
    $i = 0
    foreach ($pkg in $pkgs) {
        $i++
        $frames = Get-ChildItem (Join-Path $pkg.FullName $View) -Filter 'frame_*.npz'
        $cats = @{}
        foreach ($f in $frames) {
            $c = if ($byPath.ContainsKey($f.FullName)) { $byPath[$f.FullName] }
                 elseif ($byName.ContainsKey($f.Name)) { $byName[$f.Name] }
                 else { '未标注类别' }
            $cats[$c] = ($cats[$c] + 1)
        }
        $catTxt = ($cats.GetEnumerator() | Sort-Object Name | ForEach-Object { "$($_.Key)×$($_.Value)" }) -join '、'
        $outDir = Join-Path (Join-Path $OutRoot $pkg.Name) $View
        $argv = @('scripts\m5_annotate_manual.py', '--frames-dir',
                  (Join-Path $pkg.FullName $View), '--out', $outDir)
        if (-not $NoPrefill) { $argv += @('--prefill-model', $Prefill) }
        if ($Reviewer) { $argv += @('--reviewer', $Reviewer) }
        Write-Host ""
        Write-Host ("[{0}/{1}] {2}（{3} 帧：{4}）" -f $i, $pkgs.Count, $pkg.Name, $frames.Count, $catTxt) -ForegroundColor Cyan
        Write-Host ("  输出 -> {0}" -f $outDir)
        Write-Host ("  命令： & '{0}' {1}" -f $py, ($argv -join ' '))
        if ($List) { continue }
        & $py @argv
        Write-Host ("  [完成] {0}（rc={1}）" -f $pkg.Name, $LASTEXITCODE)
    }
    Write-Host ""
    Write-Host "[annotate] 全部结束。输出在 $OutRoot —— 告诉我一声，我做审计（身份/覆盖/四个计数）与输入输出差异"
}
finally { Pop-Location }
