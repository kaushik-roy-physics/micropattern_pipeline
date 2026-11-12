@echo off
rem ===========================================================================
rem Start the pipeline GUI (Windows). Double-click this file.
rem The first start builds a private Python environment in your user folder
rem (a few minutes, needs internet). Later starts take seconds. The GUI opens
rem in your web browser and is only reachable from this computer.
rem ===========================================================================
setlocal
set "HERE=%~dp0"
if defined GUI_VENV (set "VENV=%GUI_VENV%") else (set "VENV=%USERPROFILE%\.micropattern_pipeline_gui_venv")

set "PY="
where py >nul 2>nul && py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul && set "PY=py -3"
if not defined PY (
  where python >nul 2>nul && python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo Python 3.10 or newer is needed. Install it from https://www.python.org/downloads/
  echo ^(tick "Add python.exe to PATH" in the installer^) and try again.
  pause
  exit /b 1
)

if not exist "%VENV%\Scripts\python.exe" (
  echo First start: creating the GUI's Python environment in %VENV% ...
  %PY% -m venv "%VENV%" || (echo Could not create %VENV% & pause & exit /b 1)
)
fc /b "%HERE%requirements-gui.txt" "%VENV%\requirements-gui.txt" >nul 2>nul
if errorlevel 1 (
  echo Installing / updating packages ^(one-off, a few minutes^) ...
  "%VENV%\Scripts\python.exe" -m pip install --upgrade pip --quiet
  "%VENV%\Scripts\python.exe" -m pip install -r "%HERE%requirements-gui.txt"
  if errorlevel 1 (echo Package installation failed - check the internet connection. & pause & exit /b 1)
  copy /y "%HERE%requirements-gui.txt" "%VENV%\requirements-gui.txt" >nul
)

rem Streamlit asks for an e-mail address on its very first run and waits at that
rem prompt before serving anything; an existing credentials file skips it.
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
  if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
  (echo [general]& echo email = "")> "%USERPROFILE%\.streamlit\credentials.toml"
)

echo Starting the GUI - it opens in your browser. Close this window to stop it.
"%VENV%\Scripts\python.exe" -m streamlit run "%HERE%gui\app.py" --server.address localhost --browser.gatherUsageStats false
pause
