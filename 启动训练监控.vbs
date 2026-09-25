' BeamNG Autopilot - live training monitor (read-only), double-click to use.
' Serves the newest run that has per-step metrics and opens the page in the
' default browser.  All the logic lives in scripts\m5_training_view.py.
'
' 1 = visible console: this one blocks (it IS the server), so the console is
' where you read the URL and press Ctrl+C / close the window to stop it.
' Double-clicking it again while it runs just reopens the browser tab.

Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

projectRoot = fso.GetParentFolderName(WScript.ScriptFullName)
pythonExe  = projectRoot & "\.venv\Scripts\python.exe"
viewPy     = projectRoot & "\scripts\m5_training_view.py"

If Not fso.FileExists(pythonExe) Then
    MsgBox "Python environment not found:" & vbCrLf & pythonExe, 48, "Training monitor"
    WScript.Quit 1
End If

If Not fso.FileExists(viewPy) Then
    MsgBox "Launcher script not found:" & vbCrLf & viewPy, 48, "Training monitor"
    WScript.Quit 1
End If

shell.CurrentDirectory = projectRoot

rc = shell.Run("""" & pythonExe & """ """ & viewPy & """ monitor", 1, True)

Const RC_CTRL_C = -1073741510   ' 0xC000013A: 关闭控制台/按下 Ctrl+C —— 正常停止

If rc <> 0 And rc <> RC_CTRL_C Then
    ' 正常停止（关窗/Ctrl+C）不是失败：不要弹这个框。
    MsgBox "Monitor did not start (exit code " & rc & ")." & vbCrLf & vbCrLf _
         & "To see the reason, run this in a terminal:" & vbCrLf & _
           "  .venv\Scripts\python.exe scripts\m5_training_view.py monitor" _
         & vbCrLf & vbCrLf _
         & "exit 2 = no run has per-step metrics yet.  They are only written" & vbCrLf _
         & "         when training is started with --metrics-run <run_id>." & vbCrLf _
         & "exit 3 = no port could be opened at all (rare)." & vbCrLf & vbCrLf _
         & "If port 8760 is already taken, the script does NOT open that page" & vbCrLf _
         & "unless it is showing this same run - it picks a free port instead" & vbCrLf _
         & "and prints the URL it used." _
         , 48, "Training monitor"
    WScript.Quit rc
End If
