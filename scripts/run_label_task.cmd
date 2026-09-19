@echo off
setlocal
pushd "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo [label] Python environment not found: .venv\Scripts\python.exe
  echo Run this command from the project root or create the venv first.
  pause
  popd
  exit /b 2
)
".venv\Scripts\python.exe" "scripts\m5_run_label_task.py" --review-incomplete %*
set "rc=%ERRORLEVEL%"
echo.
echo [label] exit code: %rc%
pause
popd
exit /b %rc%
