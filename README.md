# Analiza Statystyk Piłkarzy

## Wymagania
- Python 3.9 lub nowszy
- pip (menedżer pakietów Pythona)

## Instalacja

### Windows

1. Pobierz i zainstaluj Python 3.9 lub nowszy ze strony [python.org](https://www.python.org/downloads/)
2. Pobierz to repozytorium jako ZIP i rozpakuj
3. Kliknij dwukrotnie na `install.bat`
4. Po instalacji, kliknij dwukrotnie na `start.bat` aby uruchomić program

### macOS/Linux

1. Otwórz terminal
2. Upewnij się, że masz zainstalowanego Pythona 3.9 lub nowszy:
   ```bash
   python3 --version
   ```
3. Pobierz to repozytorium
4. Przejdź do katalogu z programem
5. Nadaj uprawnienia wykonywania skryptom:
   ```bash
   chmod +x install.sh
   chmod +x start.sh
   ```
6. Uruchom instalację:
   ```bash
   ./install.sh
   ```
7. Uruchom program:
   ```bash
   ./start.sh
   ```

## Użytkowanie

1. Po uruchomieniu programu, otworzy się on w domyślnej przeglądarce
2. Kliknij "Wybierz plik Excel z danymi" i wybierz swój plik
3. Używaj filtrów i wykresów do analizy danych

## Format pliku Excel

Plik Excel powinien zawierać następujące kolumny:
- Player (Imię i nazwisko zawodnika)
- Team within selected timeframe (Drużyna)
- Position (Pozycja)
- Minutes played (Rozegrane minuty)
- Inne statystyki (bramki, asysty, itp.)

## Wsparcie

W razie problemów, sprawdź czy:
1. Masz zainstalowanego Pythona w odpowiedniej wersji
2. Wszystkie pliki są w tym samym katalogu
3. Plik Excel ma odpowiedni format
