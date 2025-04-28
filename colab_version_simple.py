# Filtrowanie danych
def filter_data(df, team, position, minutes):
    if df.empty:
        return df
    
    filtered = df.copy()
    
    try:
        if team != 'Wszystkie':
            filtered = filtered[filtered['Team within selected timeframe'] == team]
        if position != 'Wszystkie':
            # Modyfikacja filtrowania po pozycji - sprawdzamy czy pozycja jest w liście pozycji zawodnika
            filtered = filtered[filtered['Position'].apply(lambda x: position in [pos.strip() for pos in str(x).split(',')])]
        filtered = filtered[filtered['Minutes played'] >= minutes]
        
        if filtered.empty:
            print("Filtry zwróciły pusty zbiór danych")
            return df  # Zwracamy oryginalne dane zamiast pustego zbioru
        
    except Exception as e:
        print(f"Błąd podczas filtrowania: {str(e)}")
        return df
    
    return filtered

# Wyświetlanie informacji o danych
teams = df['Team within selected timeframe'].dropna().unique()

# Modyfikacja wyświetlania pozycji - zbieramy wszystkie unikalne pozycje z komórek
all_positions = set()
for positions_str in df['Position'].dropna():
    # Dzielimy string pozycji na pojedyncze pozycje
    individual_positions = [pos.strip() for pos in str(positions_str).split(',')]
    # Dodajemy każdą pojedynczą pozycję do zbioru
    for pos in individual_positions:
        if pos:  # Sprawdzamy czy pozycja nie jest pustym stringiem
            all_positions.add(pos)
positions = sorted(list(all_positions))  # Sortujemy alfabetycznie

print(f"Liczba zawodników: {len(df)}")
print(f"Liczba drużyn: {len(teams)}")
print(f"Liczba pozycji: {len(positions)}")

# Wybór drużyny
print("\nWybierz drużynę:")
for i, team in enumerate(['Wszystkie'] + list(teams)):
    print(f"{i}: {team}")
team_idx = int(input("Podaj numer drużyny: "))
selected_team = 'Wszystkie' if team_idx == 0 else list(teams)[team_idx-1]

# Wybór pozycji
print("\nWybierz pozycję:")
for i, position in enumerate(['Wszystkie'] + positions):
    print(f"{i}: {position}")
position_idx = int(input("Podaj numer pozycji: "))
selected_position = 'Wszystkie' if position_idx == 0 else positions[position_idx-1] 