#!/bin/bash
echo "Instalacja programu Analiza Statystyk Pilkarzy"
echo

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

echo
echo "Instalacja zakonczona!"
echo "Aby uruchomic program, uzyj pliku start.sh"
read -p "Nacisnij Enter, aby zakonczyc..." 