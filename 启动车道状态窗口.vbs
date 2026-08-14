' BeamNG Lane State Window - double-click to keep the live lane-state
' viewer open.  Hidden console; the OpenCV window still shows normally.
' If the viewer crashes it is restarted after a short delay.

Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

projectRoot = fso.GetParentFolderName(WScript.ScriptFullName)
pythonExe  = projectRoot & "\.venv\Scripts\python.exe"
viewerPy   = projectRoot & "\scripts\m5_lane_state_view.py"

If Not fso.FileExists(pythonExe) Then
    MsgBox "Python environment not found:" & vbCrLf & pythonExe, 48, "BeamNG Lane State"
    WScript.Quit 1
End If

If Not fso.FileExists(viewerPy) Then
    MsgBox "Viewer script not found:" & vbCrLf & viewerPy, 48, "BeamNG Lane State"
    WScript.Quit 1
End If

shell.CurrentDirectory = projectRoot
If shell.Environment("Process")("BEAMNG_PORT") = "" Then
    shell.Environment("Process")("BEAMNG_PORT") = "64257"
End If

' 0 = hidden console; the OpenCV window is a normal visible window.
' Exit code 0 means the user closed the window on purpose -> stop.
Do
    rc = shell.Run("""" & pythonExe & """ """ & viewerPy _
                   & """ --runtime tech", 0, True)
    If rc = 0 Then
        Exit Do
    End If
    WScript.Sleep 2000
Loop
