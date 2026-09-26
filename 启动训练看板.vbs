' BeamNG Autopilot - training dashboard (read-only), double-click to use.
' Renders the newest experiment run to one self-contained HTML page and opens
' it in the default browser.  All the logic (which run, what to pass, how to
' report "no data") lives in scripts\m5_training_view.py - this file only
' locates the interpreter, runs it, and surfaces a failure that would
' otherwise happen in a hidden console.

Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

projectRoot = fso.GetParentFolderName(WScript.ScriptFullName)
pythonExe  = projectRoot & "\.venv\Scripts\python.exe"
viewPy     = projectRoot & "\scripts\m5_training_view.py"

If Not fso.FileExists(pythonExe) Then
    MsgBox "Python environment not found:" & vbCrLf & pythonExe, 48, "Training dashboard"
    WScript.Quit 1
End If

If Not fso.FileExists(viewPy) Then
    MsgBox "Launcher script not found:" & vbCrLf & viewPy, 48, "Training dashboard"
    WScript.Quit 1
End If

shell.CurrentDirectory = projectRoot

' 0 = hidden console (the browser is the window); wait for the render to end.
rc = shell.Run("""" & pythonExe & """ """ & viewPy & """ dashboard", 0, True)

Const RC_CTRL_C = -1073741510   ' 0xC000013A: 关闭控制台/按下 Ctrl+C —— 正常停止

If rc <> 0 And rc <> RC_CTRL_C Then
    ' 正常停止（关窗/Ctrl+C）不是失败：不要弹这个框。
    MsgBox "Dashboard render failed (exit code " & rc & ")." & vbCrLf & vbCrLf _
         & "To see the reason, run this in a terminal:" & vbCrLf & _
           "  .venv\Scripts\python.exe scripts\m5_training_view.py dashboard" _
         & vbCrLf & vbCrLf _
         & "exit 2 = no run with an event log yet (run the experiment loop" & vbCrLf _
         & "         first; it writes logs\experiments\<run_id>\events.jsonl)." _
         , 48, "Training dashboard"
    WScript.Quit rc
End If
