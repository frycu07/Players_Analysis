@echo off
echo Instalacja programu Analiza Statystyk Pilkarzy
echo.

python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt

echo.
echo Instalacja zakonczona!
echo Aby uruchomic program, uzyj pliku start.bat
pause 