# Rebuild manuscripts and export matching PDFs with isolated Edge profiles.
[CmdletBinding()]
param([switch]$IncludeSupplement)
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv\Scripts\python.exe'
$edgePaths = @(
    (Join-Path ${env:ProgramFiles(x86)} 'Microsoft\Edge\Application\msedge.exe'),
    (Join-Path $env:ProgramFiles 'Microsoft\Edge\Application\msedge.exe')
)
$edge = $edgePaths | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $edge) { throw 'Microsoft Edge executable was not found.' }
$savedUtf8 = $env:PYTHONUTF8
Push-Location $repo
try {
    $env:PYTHONUTF8 = '1'
    & $python 'docs/paper/build_paper.py' --figs-src logs/paper_composites logs/paper_figures --wide-all
    if ($LASTEXITCODE -ne 0) { throw 'English manuscript build failed.' }
    & $python 'docs/paper/build_paper.py' --md docs/paper/paper_zh.md --tex docs/paper/main_zh.tex --html logs/paper_build/paper_zh.html --lang zh --figs logs/paper_build/figures_zh --img-dir figures_zh --figs-src logs/paper_composites_zh logs/paper_figures_zh --wide-all
    if ($LASTEXITCODE -ne 0) { throw 'Chinese manuscript build failed.' }
    $documents = @('paper', 'paper_zh')
    if ($IncludeSupplement) {
        & $python 'docs/paper/build_paper.py' --md docs/paper/supplement.md --tex docs/paper/supplement.tex --html logs/paper_build/supplement.html --fig-prefix S --figs-src logs/paper_figures --wide-all
        if ($LASTEXITCODE -ne 0) { throw 'English supplement build failed.' }
        & $python 'docs/paper/build_paper.py' --md docs/paper/supplement_zh.md --tex docs/paper/supplement_zh.tex --html logs/paper_build/supplement_zh.html --lang zh --fig-prefix S --figs logs/paper_build/figures_zh --img-dir figures_zh --figs-src logs/paper_figures_zh --wide-all
        if ($LASTEXITCODE -ne 0) { throw 'Chinese supplement build failed.' }
        $documents += @('supplement', 'supplement_zh')
    }
    foreach ($document in $documents) {
        $html = Join-Path $repo ("logs\paper_build\" + $document + '.html')
        $pdf = Join-Path $repo ("docs\paper\T16_" + $document + '.pdf')
        $freshPdf = Join-Path $repo ('logs\paper_build\export_' + $document + '_' + [guid]::NewGuid().ToString('N') + '.pdf')
        $profile = Join-Path $repo ('logs\paper_build\edge_export_' + $document + '_' + [guid]::NewGuid().ToString('N'))
        $uri = ([uri]$html).AbsoluteUri
        $edgeArgs = @('--headless=new', '--disable-gpu', '--no-pdf-header-footer',
                      ('"--user-data-dir=' + $profile + '"'),
                      ('"--print-to-pdf=' + $freshPdf + '"'), ('"' + $uri + '"'))
        $exportProcess = Start-Process -FilePath $edge -ArgumentList $edgeArgs -WindowStyle Hidden -PassThru
        if (-not $exportProcess.WaitForExit(120000)) {
            & taskkill.exe /PID $exportProcess.Id /T /F | Out-Null
            throw ("PDF export timed out: " + $document)
        }
        if ($exportProcess.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $freshPdf)) {
            throw ("PDF export failed: " + $document)
        }
        if ((Get-Item -LiteralPath $freshPdf).Length -lt 1000) { throw 'PDF output is incomplete.' }
        Move-Item -LiteralPath $freshPdf -Destination $pdf -Force
        Write-Host ("Exported: " + $pdf)
    }
}
finally {
    $env:PYTHONUTF8 = $savedUtf8
    Pop-Location
}
