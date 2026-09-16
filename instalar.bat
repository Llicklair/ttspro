@echo off
setlocal EnableExtensions
rem ttspro - one-shot install on Windows: tools, Python env, model weights, browser demo.
rem Safe to run again: every step skips what is already there.
rem   instalar.bat            install everything, then offer to start the demo
rem   instalar.bat --sin-demo install only
rem ASCII and CRLF on purpose: cmd re-reads this file by byte offset after every
rem call/goto, and multibyte characters or LF-only endings make it misparse lines.

cd /d "%~dp0"
echo.
echo == ttspro: install ==
echo.

rem ---------------------------------------------------------------- tools (winget)
where winget >nul 2>nul
if errorlevel 1 (
  echo winget is not available. Install "App Installer" from the Microsoft Store, or install
  echo uv, Node.js 22+ and eSpeak NG by hand, then run this script again.
  goto :fallo
)

call :necesita uv astral-sh.uv
call :necesita node OpenJS.NodeJS.LTS
call :necesita espeak-ng eSpeak-NG.eSpeak-NG

rem winget does not refresh PATH in the shell that called it: add the usual folders for this run.
set "PATH=%USERPROFILE%\.local\bin;%ProgramFiles%\nodejs;%ProgramFiles%\eSpeak NG;%PATH%"

where uv >nul 2>nul
if errorlevel 1 (
  echo uv still not on PATH: open a new terminal and run this script again.
  goto :fallo
)
where node >nul 2>nul
if errorlevel 1 (
  echo node still not on PATH: open a new terminal and run this script again.
  goto :fallo
)

rem ---------------------------------------------------------------- python side
echo.
echo == Python environment (uv sync) ==
call uv sync --extra dev --extra export --extra voz
if errorlevel 1 goto :fallo

echo.
echo == Model weights ==
echo    Nothing is required here any more (ADR 0012). The page loads its engine by itself:
echo    Pocket TTS (177 MB) comes from Hugging Face into the browser cache, and it is the one
echo    that can clone a voice from a recording.
echo.
echo    Supertonic is the other engine: reads a touch better, 44.1 kHz and 31 languages, but
echo    cannot clone. 398 MB, and only worth keeping locally if you are going to use it.
if exist "models\supertonic\onnx\vector_estimator.onnx" (
  echo    models\supertonic is already here, skipping.
  goto :web
)
choice /c YN /n /m "   Download Supertonic now (398 MB)? [Y/N] "
if errorlevel 2 goto :web
call uv run python -m ttspro.supertonic.descargar
if errorlevel 1 goto :fallo

:web
rem ---------------------------------------------------------------- browser side
echo.
echo == Browser runtime (npm) ==
pushd web
if exist "node_modules\" (
  rem npm ci wipes node_modules first, which fails while a dev server holds esbuild.exe
  call npm install --no-audit --no-fund
) else (
  call npm ci --no-audit --no-fund
)
if errorlevel 1 (
  popd
  goto :fallo
)
call npm run preparar
if errorlevel 1 (
  popd
  goto :fallo
)
popd

echo.
echo == Done ==
echo    demo:   cd web  then  npm run dev     and open http://localhost:5173
echo    tests:  uv run pytest tests -q
echo    measuring: add --extra eval (Whisper, jiwer); uv sync prunes it if you leave it out
echo.

if /i "%~1"=="--sin-demo" exit /b 0
choice /C SN /N /M "Start the demo now? [S/N] "
if errorlevel 2 exit /b 0
start "" http://localhost:5173
cd web
call npm run dev
exit /b 0

exit /b 0

rem ---------------------------------------------------------------- when something fails
rem Double-clicked, the window closes the instant the script exits and the reason
rem is never seen. %cmdcmdline% carries "/c" only in that case: pause there, and
rem not when run from a console that stays open anyway.
:fallo
echo.
echo Algo ha fallado: el motivo esta en las lineas de arriba.
echo %cmdcmdline% | find /i "/c" >nul && pause
exit /b 1

rem ---------------------------------------------------------------- helpers
:necesita
rem %1 = executable to look for, %2 = winget package id
where %1 >nul 2>nul
if not errorlevel 1 (
  echo    %1: found
  goto :eof
)
if /i "%1"=="espeak-ng" (
  if exist "%ProgramFiles%\eSpeak NG\espeak-ng.exe" (
    echo    %1: found
    goto :eof
  )
)
echo    %1: installing %2 with winget
winget install --id %2 -e --silent --accept-source-agreements --accept-package-agreements
if errorlevel 1 echo    %1: winget failed. If it is only espeak-ng, the demo still works; the Python tests will not.
goto :eof
