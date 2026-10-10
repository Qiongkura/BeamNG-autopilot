# PowerShell 7 entry; all arguments are forwarded to the audited library runner.
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv\Scripts\python.exe'
$savedUtf8 = $env:PYTHONUTF8
try {
    $env:PYTHONUTF8 = '1'
    Push-Location $repo
    try {
        & $python (Join-Path $PSScriptRoot 'm5_pose_sweep.py') @args
        $rc = $LASTEXITCODE
    }
    finally { Pop-Location }
}
finally { $env:PYTHONUTF8 = $savedUtf8 }
exit $rc
