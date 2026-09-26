' BeamNG Autopilot - training ledger (read-only), double-click to use.
' Scans every training run under logs\ and renders ONE self-contained HTML
' page with the whole history: runs, epochs, per-run dev metrics, decisions.
' It lists facts and does NOT rank runs across experiments (they use different
' data, splits and epoch counts), so read the banner on the page.
' All the logic lives in scripts\m5_training_view.py history.

Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

projectRoot = fso.GetParentFolderName(WScript.ScriptFullName)
pythonExe  = projectRoot & "\.venv\Scripts\python.exe"
viewPy     = projectRoot & "\scripts\m5_training_view.py"

If Not fso.FileExists(pythonExe) Then
    MsgBox "Python environment not found:" & vbCrLf & pythonExe, 48, "Training ledger"
    WScript.Quit 1
End If

If Not fso.FileExists(viewPy) Then
    MsgBox "Launcher script not found:" & vbCrLf & viewPy, 48, "Training ledger"
    WScript.Quit 1
End If

shell.CurrentDirectory = projectRoot

' 0 = hidden console (the browser is the window); wait for the scan to end.
' Scanning ~300 runs takes a few seconds.
rc = shell.Run("""" & pythonExe & """ """ & viewPy & """ history", 0, True)

Const RC_CTRL_C = -1073741510   ' 0xC000013A: console closed / Ctrl+C - normal stop

If rc <> 0 And rc <> RC_CTRL_C Then
    ' Normal stop (window closed / Ctrl+C) is not a failure: do not nag.
    ' The ledger renders an empty page when nothing was trained, so any
    ' non-zero code here means the render itself crashed.
    MsgBox "Ledger render failed (exit code " & rc & ")." & vbCrLf & vbCrLf _
         & "To see the reason, run this in a terminal:" & vbCrLf & _
           "  .venv\Scripts\python.exe scripts\m5_training_view.py history" _
         , 48, "Training ledger"
    WScript.Quit rc
End If
