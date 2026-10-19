@echo off
setlocal
cd /d "%~dp0"
set "PY="
for %%v in (3.12 3.13 3.14) do (
    if not defined PY (
        py -%%v -c "import sys" >nul 2>nul && set "PY=py -%%v"
    )
)
if not defined PY set "PY=python"
%PY% reproduce.py %*
exit /b %errorlevel%
