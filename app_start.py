import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from rankings import show_rankings  # Dodajemy import modułu rankings
from raport_generator import generate_report
from team_stats import analyze_team_stats  # Dodajemy import modułu team_stats
from Player_form import show_player_form  # Dodajemy import modułu Player_form


# Wczytanie danych
@st.cache_data
def load_data(file):
    try:
        # Wczytaj dane z pliku
        df = pd.read_excel(file).fillna(0)

        # Wczytaj nazwy kolumn z pliku Kolumny.xlsx
        try:
            kolumny_df = pd.read_excel("Kolumny.xlsx")

            # Jeśli plik Kolumny.xlsx ma kolumny, użyj ich do nadpisania nazw kolumn w df
            if not kolumny_df.empty and len(kolumny_df.columns) > 0:
                # Sprawdź czy liczba kolumn się zgadza
                if len(df.columns) == len(kolumny_df.columns):
                    # Get the first row of kolumny_df if it contains the actual column names
                    if kolumny_df.shape[0] > 0:
                        # Check if the first row contains the actual column names
                        first_row = kolumny_df.iloc[0].tolist()

                        # If the first row contains strings and not all are NaN, use it as column names
                        if all(isinstance(x, str) for x in first_row if pd.notna(x)) and any(pd.notna(x) for x in first_row):
                            df.columns = first_row
                        else:
                            # Use the column names from kolumny_df
                            df.columns = kolumny_df.columns
                    else:
                        # Use the column names from kolumny_df
                        df.columns = kolumny_df.columns
                else:
                    st.warning(f"Liczba kolumn w pliku ({len(df.columns)}) nie zgadza się z liczbą kolumn w pliku Kolumny.xlsx ({len(kolumny_df.columns)}). Nazwy kolumn nie zostały nadpisane.")
        except Exception as e:
            st.warning(f"Nie udało się wczytać pliku Kolumny.xlsx: {str(e)}. Nazwy kolumn nie zostały nadpisane.")

        return df
    except Exception as e:
        st.error(f"Błąd podczas wczytywania pliku: {str(e)}")
        return pd.DataFrame()


# Główna funkcja aplikacji
def main():
    st.title('Analiza Statystyk Piłkarzy')

    # Wybór trybu aplikacji
    app_mode = st.sidebar.radio('Wybierz tryb:', ['Analiza Zawodników', 'Analiza Drużyny', 'Porównanie Drużyn', 'Rankingi', 'Raport', 'Formularze Zawodników'])

    # Obsługa trybu Formularze Zawodników osobno
    if app_mode == 'Formularze Zawodników':
        show_player_form()
        return

    # Dodanie uploadera plików dla pozostałych trybów
    uploaded_file = st.file_uploader("Wybierz plik Excel z danymi", type=['xlsx', 'xls'])

    if uploaded_file is None:
        st.warning("Proszę wybrać plik z danymi.")
        return

    df = load_data(uploaded_file)

    if df.empty:
        st.warning("Nie udało się wczytać danych z pliku.")
        return

    # Inicjalizacja zmiennych filtrów z wartościami domyślnymi
    selected_teams = 'Wszystkie'
    selected_positions = []
    selected_minutes = int(df['Minutes played'].min())
    filter_mode = 'Tylko dla filtrowanych zawodników'

    # Pokazuj filtry tylko dla trybów Analiza Zawodników i Rankingi
    if app_mode == 'Analiza Zawodników' or app_mode == 'Rankingi':
        # Filtr drużyn (Team within selected timeframe) - menu rozwijane
        teams = df['Team within selected timeframe'].dropna().unique()
        selected_teams = st.selectbox('Wybierz drużynę:', options=['Wszystkie'] + list(teams))

        # Filtr pozycji - menu rozwijane z możliwością wyboru wielu pozycji
        all_positions = set()
        for positions_str in df['Position'].dropna():
            # Dzielimy string pozycji na pojedyncze pozycje
            individual_positions = [pos.strip() for pos in str(positions_str).split(',')]
            # Dodajemy każdą pojedynczą pozycję do zbioru
            for pos in individual_positions:
                if pos:  # Sprawdzamy czy pozycja nie jest pustym stringiem
                    all_positions.add(pos)
        positions = sorted(list(all_positions))  # Sortujemy alfabetycznie
        selected_positions = st.multiselect('Wybierz pozycje:', options=positions, default=[])

        # Filtr liczby rozegranych minut - suwak
        min_minutes = int(df['Minutes played'].min())
        max_minutes = int(df['Minutes played'].max())
        selected_minutes = st.slider('Minimalna liczba minut:', min_value=min_minutes, max_value=max_minutes,
                                    value=min_minutes)

        # Przełącznik dla trybu filtrowania
        filter_mode = st.radio('Wybierz tryb:', ['Tylko dla filtrowanych zawodników', 'Dla wszystkich zawodników w lidze'])

    if app_mode == 'Analiza Drużyny':
        analyze_team_stats(df)
        return
    elif app_mode == 'Porównanie Drużyn':
        from team_stats import compare_teams
        compare_teams(df)
        return
    elif app_mode == 'Rankingi':
        show_rankings(df, selected_teams, selected_positions, selected_minutes, filter_mode)
        return
    elif app_mode == 'Raport':
        generate_report(df)
        return

    # Filtr danych według wybranych kryteriów
    def filter_data(df, teams, positions, minutes, filter_mode):
        if df.empty:
            return df

        filtered = df.copy()

        try:
            if filter_mode == 'Tylko dla filtrowanych zawodników':
                if teams != 'Wszystkie':
                    filtered = filtered[filtered['Team within selected timeframe'] == teams]
                if positions:  # Jeśli wybrano jakieś pozycje
                    # Sprawdzamy czy którakolwiek z wybranych pozycji znajduje się w liście pozycji zawodnika
                    filtered = filtered[filtered['Position'].apply(
                        lambda x: any(pos in [p.strip() for p in str(x).split(',')] for pos in positions)
                    )]
                filtered = filtered[filtered['Minutes played'] >= minutes]
            else:
                # Dla trybu "Dla wszystkich zawodników w lidze" filtrujemy tylko po minutach
                filtered = filtered[filtered['Minutes played'] >= minutes]

            if filtered.empty:
                st.warning("Filtry zwróciły pusty zbiór danych")
                return df  # Zwracamy oryginalne dane zamiast pustego zbioru

        except Exception as e:
            st.error(f"Błąd podczas filtrowania: {str(e)}")
            return df

        return filtered

    filtered_df = filter_data(df, selected_teams, selected_positions, selected_minutes, filter_mode)

    if filtered_df.empty:
        st.warning("Brak zawodników spełniających kryteria! Zmień filtry.")
        return

    # Lista zawodników do wyboru - filtrowana według wybranej drużyny i pozycji
    players_for_selection = filtered_df['Player'].unique()

    if len(players_for_selection) == 0:
        st.warning("Brak zawodników spełniających wybrane kryteria drużyny i pozycji.")
        return

    # Wybór zawodnika
    player = st.selectbox('Wybierz zawodnika:', players_for_selection)
    player_data = filtered_df[filtered_df['Player'] == player].copy()

    # Dodajemy wybór do 4 zawodników do porównania dla wykresu radarowego (łącznie 5 zawodników)
    show_comparison = st.checkbox('Porównaj z innymi zawodnikami')
    comparison_players = []
    comparison_players_data = {}
    if show_comparison:
        comparison_players = st.multiselect('Wybierz zawodników do porównania (max 4):', 
                                       [p for p in players_for_selection if p != player],
                                       max_selections=4)

    if player_data.empty:
        st.warning("Brak danych dla wybranego zawodnika! Spróbuj zmienić filtry.")
        return

    # Statystyki do tabeli (nie na wykresy)
    table_stats = ['Weight', 'Age', 'Height', 'Market value']

    # Predefiniowane zestawy statystyk dla różnych pozycji
    position_stat_sets = {
        "Środkowi Obrońcy": [
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

    # Wybór predefiniowanego zestawu statystyk
    selected_position_set = st.selectbox(
        'Wybierz zestaw statystyk dla pozycji:',
        ['Wszystkie statystyki'] + list(position_stat_sets.keys())
    )

    # Wybór statystyk do wykresu (checkboxy)
    all_stats = [col for col in filtered_df.select_dtypes(include=['number']).columns if col not in table_stats]
    st.sidebar.header('Wybierz statystyki')

    # Aktualizacja wybranych statystyk na podstawie wybranego zestawu
    if selected_position_set == 'Wszystkie statystyki':
        select_all = st.sidebar.checkbox('Wybierz wszystkie', value=True)
        if select_all:
            selected_stats = all_stats
        else:
            selected_stats = [stat for stat in all_stats if st.sidebar.checkbox(stat)]
    else:
        selected_stats = position_stat_sets[selected_position_set]
        st.sidebar.write("Statystyki zostały automatycznie wybrane dla wybranej pozycji")

    # Wybór rodzaju wykresu
    chart_type = st.radio('Wybierz typ wykresu:', ['Wykres słupkowy', 'Radar Chart'])

    # Wyświetlanie danych ogólnych w tabeli
    st.subheader('Dane ogólne')
    non_numeric_data = player_data.select_dtypes(exclude=['number']).drop(
        columns=['Player', 'Position', 'Team within selected timeframe'], errors='ignore')
    numeric_table_data = player_data[table_stats].T.astype(str)
    all_table_data = pd.concat([non_numeric_data.T, numeric_table_data], axis=0)
    all_table_data.columns = ['Wartość']
    st.write(all_table_data)

    # Normalizacja danych dla wykresów
    numeric_data = filtered_df[selected_stats].copy()

    # Przygotowanie słowników do przechowywania danych przed utworzeniem DataFrame
    percentile_data = {}
    rank_data = {}

    for col in numeric_data.columns:
        # Filtrujemy zawodników z wartością > 0
        non_zero_mask = numeric_data[col] > 0
        non_zero_count = non_zero_mask.sum()

        if non_zero_count > 0:
            # Obliczamy ranking tylko dla zawodników z wartością > 0
            rank_series = numeric_data[col][non_zero_mask].rank(ascending=False, method='min')
            numeric_data.loc[non_zero_mask, f'{col}_rank'] = rank_series

            # Obliczamy percentyl jako (pozycja / liczba zawodników z wartością > 0) * 100
            percentile_series = numeric_data[f'{col}_rank'].apply(
                lambda x: ((non_zero_count - x + 1) / non_zero_count * 100) if x > 0 else 0
            )
            percentile_data[f'{col}_percentile'] = percentile_series

            # Dla zawodników z wartością 0 ustawiamy rank i percentyl na 0
            numeric_data.loc[~non_zero_mask, f'{col}_rank'] = 0
        else:
            # Jeśli wszyscy mają 0, ustawiamy ranking i percentyl na 0
            numeric_data[f'{col}_rank'] = 0
            percentile_data[f'{col}_percentile'] = pd.Series(0, index=numeric_data.index)

    # Tworzenie DataFrame z wszystkimi kolumnami percentile na raz
    numeric_norm = pd.DataFrame(percentile_data)

    # Łączenie wszystkich danych w jeden DataFrame
    rank_columns = [col + '_rank' for col in selected_stats]
    filtered_df = pd.concat([filtered_df, numeric_norm, numeric_data[rank_columns]], axis=1).copy()
    player_data = filtered_df[filtered_df['Player'] == player].copy()

    # Aktualizacja danych dla porównywanych zawodników jeśli są wybrani
    if show_comparison and comparison_players:
        for comp_player in comparison_players:
            comparison_players_data[comp_player] = filtered_df[filtered_df['Player'] == comp_player].copy()

    # Wyświetlanie wykresów
    st.subheader('Wizualizacja Statystyk')

    if chart_type == 'Wykres słupkowy':
        # Definicje szerokości kolumn i inicjalizacja tabeli HTML
        col_widths = {'stat': '200px', 'bar': '300px', 'value': '250px'}
        table_html = '<table style="width:100%; border-collapse: collapse;">'
        table_html += '<tr><th style="width:{}; text-align: left; border: 1px solid black;">Statystyka</th>'.format(
            col_widths['stat'])
        table_html += '<th style="width:{}; text-align: left; border: 1px solid black;">Wykres</th>'.format(
            col_widths['bar'])
        table_html += '<th style="width:{}; text-align: left; border: 1px solid black;">Wartość</th></tr>'.format(
            col_widths['value'])

        for stat in selected_stats:
            percentile_col = stat + '_percentile'
            rank_col = stat + '_rank'

            try:
                player_value = player_data[stat].iloc[0] if not player_data.empty else 0
                player_percentile = player_data[percentile_col].iloc[0] if not player_data.empty else 0
                rank_value = player_data[rank_col].iloc[0] if not player_data.empty else len(filtered_df)
            except Exception as e:
                player_value = 0
                player_percentile = 0
                rank_value = len(filtered_df)

            rank = int(rank_value) if not pd.isna(rank_value) else len(filtered_df)
            total_players = len(filtered_df)

            color = sns.color_palette('RdYlGn', as_cmap=True)(player_percentile / 100)
            rgb_color = f'rgb({int(color[0] * 255)}, {int(color[1] * 255)}, {int(color[2] * 255)})'

            bar_html = f'<div style="width:{col_widths["bar"]}; background:lightgrey; border:1px solid black; height:20px; position:relative;">'
            bar_html += f'<div style="width:{player_percentile}%; background:{rgb_color}; height:100%;"></div></div>'

            value_display = f'{player_value} ({rank}/{total_players}, {player_percentile:.1f}th percentile)'

            table_html += f'<tr><td style="width:{col_widths["stat"]}; border: 1px solid black;">{stat}</td>'
            table_html += f'<td style="width:{col_widths["bar"]}; border: 1px solid black;">{bar_html}</td>'
            table_html += f'<td style="width:{col_widths["value"]}; border: 1px solid black;">{value_display}</td></tr>'

        table_html += '</table>'
        st.markdown(table_html, unsafe_allow_html=True)

    elif chart_type == 'Radar Chart':
        # Wybieramy maksymalnie 25 statystyk dla czytelności wykresu
        if len(selected_stats) > 25:
            st.warning('Wybrano zbyt wiele statystyk. Dla wykresu radarowego pokazanych zostanie pierwsze 25.')
            radar_stats = selected_stats[:25]
        else:
            radar_stats = selected_stats

        # Usuwamy duplikaty ze statystyk radarowych
        radar_stats = list(dict.fromkeys(radar_stats))

        # Definiujemy tytuł wykresu
        chart_title = f"Statystyki zawodnika: {player}"
        if show_comparison and comparison_players:
            comparison_names = ", ".join(comparison_players)
            chart_title += f" vs {comparison_names}"

        fig = go.Figure()

        # Dodaj głównego zawodnika
        radar_values = []
        hover_texts = []
        for stat in radar_stats:
            percentile_col = stat + '_percentile'
            percentile = player_data[percentile_col].iloc[0] if not player_data.empty else 0
            actual_value = player_data[stat].iloc[0] if not player_data.empty else "-"
            radar_values.append(percentile)
            hover_texts.append(f"Wartość: {actual_value}<br>Percentyl: {percentile:.1f}%")

        fig.add_trace(go.Scatterpolar(
            r=radar_values,
            theta=radar_stats,
            fill='toself',
            name=player,
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
            fillcolor='rgba(99, 110, 250, 0.2)',
            line=dict(color='rgb(99, 110, 250)')
        ))

        # Dodaj porównywanych zawodników
        if show_comparison and comparison_players:
            # Kolory dla porównywanych zawodników
            colors = [
                ('rgba(239, 85, 59, 0.2)', 'rgb(239, 85, 59)'),
                ('rgba(0, 204, 150, 0.2)', 'rgb(0, 204, 150)'),
                ('rgba(171, 99, 250, 0.2)', 'rgb(171, 99, 250)'),
                ('rgba(255, 161, 90, 0.2)', 'rgb(255, 161, 90)')
            ]

            for i, comp_player in enumerate(comparison_players):
                if comp_player in comparison_players_data:
                    comp_data = comparison_players_data[comp_player]

                    # Używamy różnych kolorów dla każdego porównywanego zawodnika
                    fillcolor, linecolor = colors[i % len(colors)]

                    radar_values_comp = []
                    hover_texts_comp = []
                    for stat in radar_stats:
                        percentile_col = stat + '_percentile'
                        percentile = comp_data[percentile_col].iloc[0] if not comp_data.empty else 0
                        actual_value = comp_data[stat].iloc[0] if not comp_data.empty else "-"
                        radar_values_comp.append(percentile)
                        hover_texts_comp.append(f"Wartość: {actual_value}<br>Percentyl: {percentile:.1f}%")

                    fig.add_trace(go.Scatterpolar(
                        r=radar_values_comp,
                        theta=radar_stats,
                        fill='toself',
                        name=comp_player,
                        text=hover_texts_comp,
                        hovertemplate="%{text}<extra></extra>",
                        fillcolor=fillcolor,
                        line=dict(color=linecolor)
                    ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title=chart_title
        )

        st.plotly_chart(fig)

        # Tabela z wartościami
        st.write("Dokładne wartości:")

        # Modyfikacja tabeli HTML aby pokazać wybranych zawodników
        col_widths = {'stat': '200px', 'bar1': '200px', 'value1': '80px'}

        # Dodajemy kolumny dla porównywanych zawodników
        if show_comparison and comparison_players:
            for i, comp_player in enumerate(comparison_players):
                col_widths[f'bar{i+2}'] = '200px'
                col_widths[f'value{i+2}'] = '80px'

        # Tworzymy nagłówek tabeli
        table_html = '<table style="width:100%; border-collapse: collapse;">'
        table_html += '<tr><th style="width:{}; text-align: left; border: 1px solid black;">Statystyka</th>'.format(
            col_widths['stat'])
        table_html += '<th style="width:{}; text-align: left; border: 1px solid black;">{}</th>'.format(
            col_widths['bar1'], player)
        table_html += '<th style="width:{}; text-align: left; border: 1px solid black;">Wartość</th>'.format(
            col_widths['value1'])

        # Dodajemy nagłówki dla porównywanych zawodników
        if show_comparison and comparison_players:
            for i, comp_player in enumerate(comparison_players):
                table_html += '<th style="width:{}; text-align: left; border: 1px solid black;">{}</th>'.format(
                    col_widths[f'bar{i+2}'], comp_player)
                table_html += '<th style="width:{}; text-align: left; border: 1px solid black;">Wartość</th>'.format(
                    col_widths[f'value{i+2}'])
        table_html += '</tr>'

        # Wypełniamy tabelę danymi
        for stat in radar_stats:
            percentile_col = stat + '_percentile'
            try:
                # Dane głównego zawodnika
                value1 = player_data[stat].iloc[0] if not player_data.empty else "-"
                percentile1 = player_data[percentile_col].iloc[0] if not player_data.empty else 0
                color1 = sns.color_palette('RdYlGn', as_cmap=True)(percentile1 / 100)
                rgb_color1 = f'rgb({int(color1[0] * 255)}, {int(color1[1] * 255)}, {int(color1[2] * 255)})'

                bar_html1 = f'<div style="width:{col_widths["bar1"]}; background:lightgrey; border:1px solid black; height:20px; position:relative;">'
                bar_html1 += f'<div style="width:{percentile1}%; background:{rgb_color1}; height:100%;"></div></div>'

                # Rozpocznij wiersz tabeli
                table_html += f'<tr><td style="width:{col_widths["stat"]}; border: 1px solid black;">{stat}</td>'
                table_html += f'<td style="width:{col_widths["bar1"]}; border: 1px solid black;">{bar_html1}</td>'
                table_html += f'<td style="width:{col_widths["value1"]}; border: 1px solid black;">{value1}</td>'

                # Dodaj dane dla porównywanych zawodników
                if show_comparison and comparison_players:
                    for i, comp_player in enumerate(comparison_players):
                        if comp_player in comparison_players_data:
                            comp_data = comparison_players_data[comp_player]
                            value_comp = comp_data[stat].iloc[0] if not comp_data.empty else "-"
                            percentile_comp = comp_data[percentile_col].iloc[0] if not comp_data.empty else 0
                            color_comp = sns.color_palette('RdYlGn', as_cmap=True)(percentile_comp / 100)
                            rgb_color_comp = f'rgb({int(color_comp[0] * 255)}, {int(color_comp[1] * 255)}, {int(color_comp[2] * 255)})'

                            bar_html_comp = f'<div style="width:{col_widths[f"bar{i+2}"]}; background:lightgrey; border:1px solid black; height:20px; position:relative;">'
                            bar_html_comp += f'<div style="width:{percentile_comp}%; background:{rgb_color_comp}; height:100%;"></div></div>'

                            table_html += f'<td style="width:{col_widths[f"bar{i+2}"]}; border: 1px solid black;">{bar_html_comp}</td>'
                            table_html += f'<td style="width:{col_widths[f"value{i+2}"]}; border: 1px solid black;">{value_comp}</td>'
                        else:
                            # Jeśli brak danych dla tego zawodnika
                            table_html += f'<td style="width:{col_widths[f"bar{i+2}"]}; border: 1px solid black;">-</td>'
                            table_html += f'<td style="width:{col_widths[f"value{i+2}"]}; border: 1px solid black;">-</td>'

                table_html += '</tr>'

            except Exception:
                table_html += f'<tr><td style="border: 1px solid black;">{stat}</td>'
                table_html += '<td style="border: 1px solid black;">-</td><td style="border: 1px solid black;">-</td>'

                # Dodaj puste komórki dla porównywanych zawodników w przypadku błędu
                if show_comparison and comparison_players:
                    for i in range(len(comparison_players)):
                        table_html += '<td style="border: 1px solid black;">-</td><td style="border: 1px solid black;">-</td>'

                table_html += '</tr>'

        table_html += '</table>'
        st.markdown(table_html, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
