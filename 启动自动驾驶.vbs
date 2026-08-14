' BeamNG Autopilot - double-click to open the control console (no terminal).
' Uses the project venv python with a hidden console window.

Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

projectRoot = fso.GetParentFolderName(WScript.ScriptFullName)
pythonExe  = projectRoot & "\.venv\Scripts\python.exe"
launcherPy = projectRoot & "\scripts\m5_launcher.py"

If Not fso.FileExists(pythonExe) Then
    MsgBox "Python environment not found:" & vbCrLf & pythonExe, 48, "BeamNG Autopilot"
    WScript.Quit 1
End If

If Not fso.FileExists(launcherPy) Then
    MsgBox "Launcher script not found:" & vbCrLf & launcherPy, 48, "BeamNG Autopilot"
    WScript.Quit 1
End If

shell.CurrentDirectory = projectRoot
' 0 = hidden console window; the tkinter GUI window still shows normally.
shell.Run """" & pythonExe & """ """ & launcherPy & """", 0, False
