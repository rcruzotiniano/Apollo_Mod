@echo off
cd /d %~dp0

:menu
cls
echo =========================
echo        APOLLO IA
echo =========================
echo.
echo 1 - Ejecutar proceso
echo 2 - Salir
echo.
set /p option=Selecciona una opcion: 

if "%option%"=="1" goto run
if "%option%"=="2" exit

goto menu

:run
cls
echo Ejecutando...
echo.

venv\Scripts\python.exe app.py

echo.
echo =========================
echo Proceso terminado
echo =========================
echo.
echo 1 - Volver a ejecutar
echo 2 - Salir
echo.
set /p option2=Selecciona una opcion: 

if "%option2%"=="1" goto menu
if "%option2%"=="2" exit

goto menu