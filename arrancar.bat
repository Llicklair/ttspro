@echo off
setlocal EnableExtensions
rem ttspro - start the page on Windows: checks the models, prepares the static
rem files, launches the dev server and opens the browser. Run instalar.bat once first.
rem   arrancar.bat            page at http://localhost:5173
rem   arrancar.bat --lib      the library + example page instead (http://127.0.0.1:8080)
rem ASCII and CRLF on purpose: cmd re-reads this file by byte offset after every
rem call/goto, and multibyte characters or LF-only endings make it misparse lines.

cd /d "%~dp0"
set "PATH=%USERPROFILE%\.local\bin;%ProgramFiles%\nodejs;%PATH%"

where uv >nul 2>nul
if errorlevel 1 (
  echo uv is not on PATH: run instalar.bat first.
  goto :fallo
)
where node >nul 2>nul
if errorlevel 1 (
  echo node is not on PATH: run instalar.bat first.
  goto :fallo
)
if not exist ".venv\" (
  echo No .venv yet: run instalar.bat first.
  goto :fallo
)
if not exist "web\node_modules\" (
  echo No web\node_modules yet: run instalar.bat first.
  goto :fallo
)

rem The engine is four ONNX graphs, 398 MB (ADR 0012). They are NOT required to
rem start: the page pulls them straight from Hugging Face with its own button,
rem which is the no-install path. Keeping them locally only makes the page fast,
rem so a missing models\supertonic is a question, never a failure.
if exist "models\supertonic\onnx\vector_estimator.onnx" goto :listo
echo.
echo == The engine is not downloaded yet (398 MB) ==
echo    The page works without it: it pulls the graphs from Hugging Face by itself,
echo    and the button "usar Hugging Face" forces that on purpose. No install needed.
echo    Keeping a local copy is faster on every reload.
echo.
rem `/t 20 /d N` y `2>nul` por lo mismo que en instalar.bat: sin una consola
rem delante, `choice` a secas escribe un ERROR que no viene a cuento y se queda
rem esperando a alguien que no esta.
choice /c YN /n /t 20 /d N /m "Download it now? [Y/N, N en 20s] " 2>nul
if errorlevel 2 goto :listo
call uv run python -m ttspro.supertonic.descargar
if errorlevel 1 goto :fallo

:listo
if /i "%~1"=="--lib" goto :lib

echo == ttspro: page at http://localhost:5173  (Ctrl+C stops it) ==
start "" http://localhost:5173/
cd web
call npm run dev
if errorlevel 1 goto :fallo
exit /b 0

:lib
echo == ttspro: library + example at http://127.0.0.1:8080  (Ctrl+C stops it) ==
cd web
call npm run build:lib
if errorlevel 1 goto :fallo
start "" http://127.0.0.1:8080/
call npm run servir
if errorlevel 1 goto :fallo
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
