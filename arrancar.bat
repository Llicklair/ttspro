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
  exit /b 1
)
where node >nul 2>nul
if errorlevel 1 (
  echo node is not on PATH: run instalar.bat first.
  exit /b 1
)
if not exist ".venv\" (
  echo No .venv yet: run instalar.bat first.
  exit /b 1
)
if not exist "web\node_modules\" (
  echo No web\node_modules yet: run instalar.bat first.
  exit /b 1
)

if exist "models\tts.onnx" if exist "models\conversor.onnx" if exist "models\voz.onnx" goto :listo
echo == Models missing: building them from public weights (about a minute) ==
call uv run python -m ttspro.export.modelos --sin-encoder
if errorlevel 1 exit /b 1

:listo
if /i "%~1"=="--lib" goto :lib

echo == ttspro: page at http://localhost:5173  (Ctrl+C stops it) ==
start "" http://localhost:5173/
cd web
call npm run dev
exit /b %errorlevel%

:lib
echo == ttspro: library + example at http://127.0.0.1:8080  (Ctrl+C stops it) ==
cd web
call npm run build:lib
if errorlevel 1 exit /b 1
start "" http://127.0.0.1:8080/
call npm run servir
exit /b %errorlevel%
