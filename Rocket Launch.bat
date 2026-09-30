@echo off
setlocal EnableDelayedExpansion
title Rocket Launch Control Agent
REM ---------------------------------------------------------------------------
REM  Rocket Launch Control Agent: sets up, starts the demo, opens the browser.
REM  A double-click works.
REM
REM  1. Python environment: .venv from requirements.txt, created on first start.
REM  2. The trained models: layer-4\model.pkl. The first start builds them with
REM     run_factory.py (about 10 minutes); later starts reuse them.
REM  3. The analyst's model on 127.0.0.1:8001. A server that already runs there
REM     is reused and never stopped. When nothing answers and LLAMA_SERVER and
REM     LLAMA_MODEL name a llama.cpp llama-server and a GGUF file, it is started
REM     under the model name in llm_config.toml. Without a model the demo still
REM     runs; only the analyst chat stays offline.
REM  4. The dashboard on http://localhost:8765, which opens the browser.
REM ---------------------------------------------------------------------------

pushd "%~dp0" || exit /b 1
set "WARN="
echo.
echo   Rocket Launch Control Agent
echo.

REM ------------------------------------------------------------ 1. Python
if exist ".venv\Scripts\python.exe" goto :have_venv
set "PYEXE="
where py >nul 2>nul && set "PYEXE=py -3"
if not defined PYEXE where python >nul 2>nul && set "PYEXE=python"
if not defined PYEXE goto :no_python
echo   [1/4] creating the Python environment in .venv (first start only)
!PYEXE! -m venv .venv || goto :no_python
".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt || goto :pip_failed
goto :models
:have_venv
echo   [1/4] Python environment: .venv

REM ------------------------------------------------------------ 2. the models
:models
if exist "layer-4\model.pkl" (
  echo   [2/4] trained models: layer-4\model.pkl
  goto :llm
)
echo   [2/4] building the models from 3,000 simulated launches (first start only,
echo         about 10 minutes)
".venv\Scripts\python.exe" run_factory.py || goto :factory_failed

REM ------------------------------------------------------------ 3. the analyst's model
:llm
set "WANT="
for /f "tokens=2 delims==" %%a in ('findstr /r /c:"^model *=" "llm_config.toml"') do set "WANT=%%a"
set "WANT=!WANT: =!"
set "WANT=!WANT:"=!"
if not defined WANT set "WANT=qwen3.8-27b"
call :model_state
if "!STATE!"=="ok" (
  echo   [3/4] !WANT! is serving on 8001.
  goto :dashboard
)
if "!STATE!"=="other" goto :model_other
if not defined LLAMA_SERVER goto :model_none
if not defined LLAMA_MODEL goto :model_none
if not exist "!LLAMA_SERVER!" goto :model_none
if not exist "!LLAMA_MODEL!" goto :model_none
echo   [3/4] starting !WANT! on 8001. Loading the model takes a minute or two.
start "!WANT! :8001" /min "!LLAMA_SERVER!" -m "!LLAMA_MODEL!" --alias !WANT! --host 127.0.0.1 --port 8001 -c 32768 -np 1 -ngl 99 --jinja
set /a "TRIES=100"
:model_wait
call :sleep 3
call :model_state
if "!STATE!"=="ok" (
  echo         ready.
  goto :dashboard
)
set /a "TRIES-=1"
if !TRIES! GTR 0 goto :model_wait
echo         The model did not come up within 5 minutes; see its window.
echo         The demo starts without the analyst chat.
set "WARN=yes"
goto :dashboard

:model_other
echo   [3/4] The model server on 8001 has !LOADED! loaded, not !WANT!.
echo         It keeps running: this launcher stops nothing. The analyst chat stays
echo         offline until !WANT! runs there, or until llm_config.toml names it.
set "WARN=yes"
goto :dashboard

:model_none
echo   [3/4] No model server answers on 8001, and none is set up to start here.
echo         The demo starts without the analyst chat. See "The analyst chat" in
echo         README.md to add it.
set "WARN=yes"

REM ------------------------------------------------------------ 4. the dashboard
:dashboard
call :answers 8765 /api/analyst/status
if "!UP!"=="yes" (
  echo   [4/4] The dashboard is already on 8765: opening it.
  start "" "http://localhost:8765/"
  goto :done
)
echo   [4/4] starting the dashboard on 8765. The browser opens by itself.
start "Rocket Launch Control Agent :8765" /min cmd /k .venv\Scripts\python.exe ControlRoom\serve.py

:done
echo.
echo   The demo is at http://localhost:8765
echo   Each server runs in its own window; close a window to stop that server.
popd
if defined WARN (
  echo.
  pause
) else (
  call :sleep 8
)
exit /b 0

:no_python
echo   Python 3.11 or newer is needed and was not found. Install it from
echo   https://www.python.org and run this again.
goto :fail
:pip_failed
echo   Installing the packages in requirements.txt failed; see the messages above.
goto :fail
:factory_failed
echo   Building the models failed; see the messages above.
goto :fail
:fail
echo.
popd
pause
exit /b 1

REM ------------------------------------------------------------------- helpers
:model_state
REM  STATE=ok when WANT is loaded on 8001; other when another model is (LOADED
REM  names it); none when nothing answers. llama.cpp answers ANY model name with
REM  whatever it has loaded, so /v1/models is the one place to ask.
set "STATE=none"
set "LOADED="
for /f "delims=" %%m in ('powershell -NoProfile -Command "try { (Invoke-RestMethod -TimeoutSec 4 -Uri http://127.0.0.1:8001/v1/models).data.id -join ' ' } catch { '' }" 2^>nul') do set "LOADED=%%m"
if not defined LOADED exit /b
set "STATE=other"
for %%m in (!LOADED!) do if /i "%%m"=="!WANT!" set "STATE=ok"
exit /b

:answers
REM  UP=yes when http://127.0.0.1:%1%2 answers with a success status.
set "UP=no"
curl -sf -m 3 -o nul "http://127.0.0.1:%~1%~2" 2>nul && set "UP=yes"
exit /b

:sleep
REM  ping instead of timeout: timeout fails when input is redirected.
set /a "PINGS=%~1+1"
ping -n !PINGS! 127.0.0.1 >nul 2>nul
exit /b
