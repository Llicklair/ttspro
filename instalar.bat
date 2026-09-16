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
echo    Nothing is required here: with no local copy the page pulls the engine straight from
echo    Hugging Face into the browser cache, and the model that clones a voice from a recording
echo    always comes that way.
echo.
echo    A local copy (398 MB) only makes every later reload faster, and it is what the Python
echo    tests need: without it, 27 of them skip.
if exist "models\supertonic\onnx\vector_estimator.onnx" (
  echo    models\supertonic is already here, skipping.
  goto :web
)
rem Dos cosas que `choice` a secas hace mal aqui:
rem  - con stdin redirigido (instalacion automatizada) escribe en stderr un ERROR
rem    que no viene a cuento y devuelve 255; el `errorlevel 2` lo tomaba por un
rem    "No" de carambola, no por diseno. `2>nul` calla el ruido y el 255 sigue
rem    cayendo del lado seguro, que es no descargar.
rem  - sin /t se queda esperando para siempre a alguien que quiza no esta.
rem Veinte segundos y, si nadie contesta, N: la pagina funciona igual sin la copia.
choice /c YN /n /t 20 /d N /m "   Download it now (398 MB)? [Y/N, N en 20s] " 2>nul
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
