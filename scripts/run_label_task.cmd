@echo off
REM 一条命令跑标注：导出待标注帧 -> 打开标注器 -> 打印训练命令。
REM 双击本文件即可（Windows）。规则见包内 RULES.txt。
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo [label] 找不到 .venv\Scripts\python.exe，请先创建虚拟环境并安装依赖
  pause
  exit /b 2
)
".venv\Scripts\python.exe" scripts\m5_run_label_task.py %*
echo.
pause
