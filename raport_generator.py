import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from docx import Document
from docx.shared import Inches, Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_ORIENT
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as path_effects

def generate_report(df):
    st.title("Raport drużyny")
    
    # Predefiniowane zestawy statystyk dla różnych pozycji
    position_stat_sets = {
        "Obrońcy - środkowi": [
            "Successful defensive actions per 90",
            "Defensive duels per 90",
            "Defensive duels won, %",
            "Aerial duels per 90",
            "Aerial duels won, %",
            "Fouls per 90",
            "Passes per 90",
            "Forward passes per 90",
            "Short / medium passes per 90",
            "Long passes per 90",
            "Passes to final third per 90"
        ],
        "Boczni obrońcy - DEF": [
            "Successful defensive actions per 90",
            "Defensive duels per 90",
            "Defensive duels won, %",
            "Aerial duels per 90",
            "Aerial duels won, %",
            "Fouls per 90"
        ],
        "Boczni obrońcy - OFF": [
            "Successful attacking actions per 90",
            "Shots per 90",
            "Shots on target, %",
            "Crosses per 90",
            "Crosses to goalie box per 90",
            "Dribbles per 90",
            "Offensive duels per 90",
            "Offensive duels won, %",
            "Progressive runs per 90",
            "Forward passes per 90",
            "Back passes per 90",
            "Accelerations per 90",
            "xA per 90",
            "Passes to final third per 90",
            "Passes to penalty area per 90",
            "Deep completions per 90",
            "Deep completed crosses per 90"
        ],
        "Środkowi Pomocnicy - DEF": [
            "Duels per 90",
            "Duels won, %",
            "Successful defensive actions per 90",
            "Defensive duels per 90",
            "Defensive duels won, %",
            "Aerial duels per 90",
            "Aerial duels won, %",
            "Sliding tackles per 90",
            "Interceptions per 90",
            "Fouls per 90"
        ],
        "Środkowi Pomocnicy - OFF": [
            "Assists",
            "xA",
            "Successful attacking actions per 90",
            "xG per 90",
            "Shots per 90",
            "Shots on target, %",
            "Dribbles per 90",
            "Offensive duels per 90",
            "Offensive duels won, %",
            "Touches in box per 90",
            "Progressive runs per 90",
            "Received passes per 90",
            "Passes per 90",
            "Forward passes per 90",
            "Short / medium passes per 90",
            "Long passes per 90",
            "Second assists per 90",
            "Key passes per 90",
            "Passes to final third per 90",
            "Passes to penalty area per 90",
            "Deep completions per 90"
        ],
        "Skrzydłowi / 10": [
            "xA",
            "Successful attacking actions per 90",
            "xG per 90",
            "Shots per 90",
            "Crosses per 90",
            "Crosses to goalie box per 90",
            "Dribbles per 90",
            "Offensive duels per 90",
            "Offensive duels won, %",
            "Touches in box per 90",
            "Progressive runs per 90",
            "Received passes per 90",
            "Passes per 90",
            "Forward passes per 90",
            "Received long passes per 90",
            "Second assists per 90",
            "Key passes per 90",
            "Passes to final third per 90",
            "Passes to penalty area per 90",
            "Deep completions per 90"
        ],
        "Napastnicy": [
            "Goals per 90",
            "xG per 90",
            "Aerial duels per 90",
            "Shots per 90",
            "Shots on target, %",
            "Dribbles per 90",
            "Offensive duels per 90",
            "Touches in box per 90",
            "Progressive runs per 90",
            "Accelerations per 90",
            "Received passes per 90",
            "Received long passes per 90",
            "xA per 90",
            "Key passes per 90",
            "Passes to final third per 90",
            "Passes to penalty area per 90",
            "Deep completions per 90",
        ]
    }
    
    # Wybór drużyny
    teams = df['Team within selected timeframe'].unique()
    selected_team = st.selectbox('Wybierz drużynę:', teams)
    
    # Filtrowanie danych dla wybranej drużyny
    team_df = df[df['Team within selected timeframe'] == selected_team]
    
    # Wybór zawodników
    players = team_df['Player'].unique()
    selected_players = st.multiselect('Wybierz zawodników:', players, default=players[:min(4, len(players))])
    
    if not selected_players:
        st.warning("Wybierz przynajmniej jednego zawodnika")
        return
        
    # Filtrowanie danych dla wybranych zawodników
    players_df = team_df[team_df['Player'].isin(selected_players)]
    
    # Wybór zestawów statystyk dla każdego zawodnika
    player_stats = {}
    for player in selected_players:
        st.subheader(f"Statystyki dla {player}")
        selected_position_sets = st.multiselect(
            f'Wybierz zestawy statystyk dla {player}:',
            list(position_stat_sets.keys()),
            key=f"stats_{player}"
        )
        if selected_position_sets:
            player_stats[player] = selected_position_sets
    
    # Pole do wprowadzenia uwag
    st.subheader("Uwagi do raportu")
    notes = st.text_area("Wpisz uwagi (np. wnioski, zalecenia):", 
                         "1. Do głowy najczęściej skacze Paluszek jeśli gra.\n2. W rozegraniu największy udział ma Szota\n3. Matsenko jest najlepszy jeśli chodzi o ilość i skutecznosc poijedynkow 1 na 1, ale tym samym najwięcej fauluje.")
    
    # Przycisk generowania raportu
    if st.button("Generuj raport"):
        # Tworzenie dokumentu Word
        doc = Document()
        
        # Ustawienie orientacji strony na poziomą
        section = doc.sections[0]
        section.orientation = WD_ORIENT.LANDSCAPE
        section.page_width = Inches(11.69)  # A4 szerokość
        section.page_height = Inches(8.27)  # A4 wysokość
        
        # Dla każdego zestawu statystyk
        for position_set, stats in position_stat_sets.items():
            # Znajdź zawodników, którzy mają ten zestaw statystyk
            players_with_set = [p for p, sets in player_stats.items() if position_set in sets]
            
            if players_with_set:
                # Normalizacja danych dla wykresu
                numeric_data = df[stats].copy()
                numeric_norm = pd.DataFrame()
                
                for col in stats:
                    non_zero_mask = numeric_data[col] > 0
                    non_zero_count = non_zero_mask.sum()
                    
                    if non_zero_count > 0:
                        numeric_data.loc[non_zero_mask, f'{col}_rank'] = numeric_data[col][non_zero_mask].rank(ascending=False, method='min')
                        numeric_norm[f'{col}_percentile'] = numeric_data[f'{col}_rank'].apply(
                            lambda x: ((non_zero_count - x + 1) / non_zero_count * 100) if x > 0 else 0
                        )
                        numeric_data.loc[~non_zero_mask, f'{col}_rank'] = 0
                        numeric_norm.loc[~non_zero_mask, f'{col}_percentile'] = 0
                    else:
                        numeric_data[f'{col}_rank'] = 0
                        numeric_norm[f'{col}_percentile'] = 0
                
                # Stwórz duży obraz dla całej strony
                fig = plt.figure(figsize=(11.69, 8.27), dpi=300)
                
                # Dodaj czerwony gradient tła
                plt_background = fig.add_axes([0, 0, 1, 1])
                background_gradient = np.ones((100, 100, 3))
                for i in range(100):
                    background_gradient[:, i, 0] = 1.0  # R
                    background_gradient[:, i, 1] = 0.3 - i * 0.003  # G (od 0.3 do 0)
                    background_gradient[:, i, 2] = 0.3 - i * 0.003  # B (od 0.3 do 0)
                plt_background.imshow(background_gradient, aspect='auto')
                plt_background.axis('off')
                
                # Dodaj tytuł strony
                title_ax = fig.add_axes([0.1, 0.85, 0.8, 0.1])
                title_ax.text(0.5, 0.5, f"{position_set}", color='white', fontsize=36, ha='center', va='center', 
                              fontweight='bold')
                title_ax.axis('off')
                
                # Dodaj uwagi
                notes_ax = fig.add_axes([0.1, 0.75, 0.8, 0.1])
                notes_lines = notes.split('\n')
                y_pos = 0.9
                for line in notes_lines:
                    notes_ax.text(0.0, y_pos, line, color='white', fontsize=14, ha='left', va='center')
                    y_pos -= 0.3
                notes_ax.axis('off')
                
                # Dodaj grid dla statystyk zawodników
                n_players = len(players_with_set)
                if n_players <= 2:
                    cols, rows = n_players, 1
                else:
                    cols, rows = 2, (n_players + 1) // 2
                
                # Układ dla statystyk zawodników
                player_areas = []
                for i in range(n_players):
                    row = i // cols
                    col = i % cols
                    
                    # Oblicz położenie i wymiary obszaru dla zawodnika
                    x = 0.05 + col * (0.9 / cols)
                    y = 0.05 + (rows - row - 1) * (0.65 / rows)
                    w = 0.9 / cols - 0.02
                    h = 0.65 / rows - 0.02
                    
                    # Dodaj nazwę zawodnika
                    name_ax = fig.add_axes([x, y + h - 0.05, w, 0.05])
                    name_ax.text(0.5, 0.5, players_with_set[i], color='white', fontsize=18, ha='center', va='center', 
                                 fontweight='bold')
                    name_ax.axis('off')
                    
                    # Dodaj obszar na tabelę statystyk
                    table_ax = fig.add_axes([x, y, w, h - 0.05])
                    table_ax.axis('off')
                    
                    # Przygotuj dane zawodnika
                    player = players_with_set[i]
                    player_data = players_df[players_df['Player'] == player]
                    
                    # Utwórz tabelę statystyk
                    table_data = []
                    for stat in stats:
                        try:
                            player_value = player_data[stat].iloc[0] if not player_data.empty else 0
                            player_percentile = numeric_norm.loc[player_data.index[0], f'{stat}_percentile'] if not player_data.empty else 0
                            rank_value = numeric_data.loc[player_data.index[0], f'{stat}_rank'] if not player_data.empty else len(df)
                        except Exception as e:
                            player_value = 0
                            player_percentile = 0
                            rank_value = len(df)
                        
                        rank = int(rank_value) if not pd.isna(rank_value) else len(df)
                        total_players = len(df)
                        
                        # Dodaj wiersz do tabeli
                        table_data.append([
                            stat, 
                            player_percentile, 
                            f"{player_value:.1f} ({rank}/{total_players}, {player_percentile:.1f}th percentile)"
                        ])
                    
                    # Rysuj tabelę
                    rows_in_table = len(table_data)
                    cell_height = 1.0 / (rows_in_table + 1)  # +1 dla nagłówka
                    
                    # Rysuj nagłówki
                    table_ax.text(0.0, 1.0 - 0.5 * cell_height, "Statystyka", fontsize=10, ha='left', va='center', fontweight='bold')
                    table_ax.text(0.5, 1.0 - 0.5 * cell_height, "Wykres", fontsize=10, ha='center', va='center', fontweight='bold')
                    table_ax.text(0.85, 1.0 - 0.5 * cell_height, "Wartość", fontsize=10, ha='right', va='center', fontweight='bold')
                    
                    # Rysuj linie poziome nagłówka
                    table_ax.axhline(y=1.0, xmin=0, xmax=1, color='black', linewidth=1)
                    table_ax.axhline(y=1.0 - cell_height, xmin=0, xmax=1, color='black', linewidth=1)
                    
                    # Rysuj linie pionowe
                    table_ax.axvline(x=0, ymin=0, ymax=1, color='black', linewidth=1)
                    table_ax.axvline(x=0.4, ymin=0, ymax=1, color='black', linewidth=1)
                    table_ax.axvline(x=0.7, ymin=0, ymax=1, color='black', linewidth=1)
                    table_ax.axvline(x=1.0, ymin=0, ymax=1, color='black', linewidth=1)
                    
                    # Rysuj wiersze tabeli
                    for j, (stat, percentile, value_text) in enumerate(table_data):
                        y_pos = 1.0 - (j + 1.5) * cell_height
                        
                        # Nazwa statystyki
                        table_ax.text(0.02, y_pos, stat, fontsize=8, ha='left', va='center')
                        
                        # Pasek percentyla
                        bar_color = sns.color_palette('RdYlGn', as_cmap=True)(percentile/100)
                        rect = patches.Rectangle((0.42, y_pos - 0.3*cell_height), 0.26 * percentile/100, 0.6*cell_height,
                                                linewidth=1, edgecolor='none', facecolor=bar_color)
                        table_ax.add_patch(rect)
                        
                        # Wartość
                        table_ax.text(0.98, y_pos, value_text, fontsize=8, ha='right', va='center')
                        
                        # Linia pozioma pod wierszem
                        table_ax.axhline(y=1.0 - (j + 2) * cell_height, xmin=0, xmax=1, color='black', linewidth=0.5)
                
                # Zapisz obraz
                img_stream = io.BytesIO()
                plt.savefig(img_stream, format='png', bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor())
                plt.close()
                img_stream.seek(0)
                
                # Dodaj obraz do dokumentu na pełną stronę
                doc.add_picture(img_stream, width=Inches(11.5))
                
                # Dodaj podział strony
                doc.add_page_break()
        
        # Zapisz dokument
        doc_stream = io.BytesIO()
        doc.save(doc_stream)
        doc_stream.seek(0)
        
        # Przygotuj link do pobrania
        b64 = base64.b64encode(doc_stream.getvalue()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{b64}" download="raport_{selected_team}.docx">Pobierz raport</a>'
        st.markdown(href, unsafe_allow_html=True) 