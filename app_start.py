import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from rankings import show_rankings  # Dodajemy import modułu rankings
from raport_generator import generate_report


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
    app_mode = st.sidebar.radio('Wybierz tryb:', ['Analiza Zawodników', 'Rankingi', 'Raport'])

    # Dodanie uploadera plików
    uploaded_file = st.file_uploader("Wybierz plik Excel z danymi", type=['xlsx', 'xls'])

    if uploaded_file is None:
        st.warning("Proszę wybrać plik z danymi.")
        return

    df = load_data(uploaded_file)

    if df.empty:
        st.warning("Nie udało się wczytać danych z pliku.")
        return

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

    if app_mode == 'Rankingi':
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
    players_for_selection = df.copy()
    if selected_teams != 'Wszystkie':
        players_for_selection = players_for_selection[players_for_selection['Team within selected timeframe'] == selected_teams]
    if selected_positions:
        players_for_selection = players_for_selection[players_for_selection['Position'].apply(
            lambda x: any(pos in [p.strip() for p in str(x).split(',')] for pos in selected_positions)
        )]
    players_for_selection = players_for_selection['Player'].unique()

    if len(players_for_selection) == 0:
        st.warning("Brak zawodników spełniających wybrane kryteria drużyny i pozycji.")
        return

    # Wybór zawodnika
    player = st.selectbox('Wybierz zawodnika:', players_for_selection)
    player_data = filtered_df[filtered_df['Player'] == player].copy()

    # Dodajemy wybór drugiego zawodnika dla wykresu radarowego
    show_comparison = st.checkbox('Porównaj z innym zawodnikiem')
    if show_comparison:
        player2 = st.selectbox('Wybierz drugiego zawodnika:', 
                             [p for p in players_for_selection if p != player])
        player2_data = filtered_df[filtered_df['Player'] == player2].copy()

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
    numeric_norm = pd.DataFrame()

    for col in numeric_data.columns:
        # Filtrujemy zawodników z wartością > 0
        non_zero_mask = numeric_data[col] > 0
        non_zero_count = non_zero_mask.sum()

        if non_zero_count > 0:
            # Obliczamy ranking tylko dla zawodników z wartością > 0
            numeric_data.loc[non_zero_mask, f'{col}_rank'] = numeric_data[col][non_zero_mask].rank(ascending=False, method='min')

            # Obliczamy percentyl jako (pozycja / liczba zawodników z wartością > 0) * 100
            numeric_norm[f'{col}_percentile'] = numeric_data[f'{col}_rank'].apply(
                lambda x: ((non_zero_count - x + 1) / non_zero_count * 100) if x > 0 else 0
            )

            # Dla zawodników z wartością 0 ustawiamy rank i percentyl na 0
            numeric_data.loc[~non_zero_mask, f'{col}_rank'] = 0
            numeric_norm.loc[~non_zero_mask, f'{col}_percentile'] = 0
        else:
            # Jeśli wszyscy mają 0, ustawiamy ranking i percentyl na 0
            numeric_data[f'{col}_rank'] = 0
            numeric_norm[f'{col}_percentile'] = 0

    filtered_df = pd.concat([filtered_df, numeric_norm], axis=1)
    filtered_df = pd.concat([filtered_df, numeric_data[[col + '_rank' for col in selected_stats]]], axis=1)
    player_data = filtered_df[filtered_df['Player'] == player].copy()

    # Aktualizacja danych dla drugiego zawodnika jeśli jest wybrany
    if show_comparison:
        player2_data = filtered_df[filtered_df['Player'] == player2].copy()

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

        # Przygotowanie danych do wykresu radarowego dla pierwszego zawodnika
        radar_values = []
        hover_texts = []
        for stat in radar_stats:
            percentile_col = stat + '_percentile'
            try:
                percentile = player_data[percentile_col].iloc[0] if not player_data.empty else 0
                actual_value = player_data[stat].iloc[0] if not player_data.empty else 0
                radar_values.append(percentile)
                hover_texts.append(f"Wartość: {actual_value}<br>Percentyl: {percentile:.1f}%")
            except Exception:
                radar_values.append(0)
                hover_texts.append("Brak danych")

        # Tworzenie wykresu radarowego
        fig = go.Figure()

        # Dodanie pierwszego zawodnika
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

        # Dodanie drugiego zawodnika jeśli wybrano porównanie
        if show_comparison:
            radar_values2 = []
            hover_texts2 = []
            for stat in radar_stats:
                percentile_col = stat + '_percentile'
                try:
                    percentile = player2_data[percentile_col].iloc[0] if not player2_data.empty else 0
                    actual_value = player2_data[stat].iloc[0] if not player2_data.empty else 0
                    radar_values2.append(percentile)
                    hover_texts2.append(f"Wartość: {actual_value}<br>Percentyl: {percentile:.1f}%")
                except Exception:
                    radar_values2.append(0)
                    hover_texts2.append("Brak danych")

            fig.add_trace(go.Scatterpolar(
                r=radar_values2,
                theta=radar_stats,
                fill='toself',
                name=player2,
                text=hover_texts2,
                hovertemplate="%{text}<extra></extra>",
                fillcolor='rgba(239, 85, 59, 0.2)',
                line=dict(color='rgb(239, 85, 59)')
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title=f'Profil zawodnika: {player}' + (f' vs {player2}' if show_comparison else '')
        )

        st.plotly_chart(fig)

        # Tabela z wartościami
        st.write("Dokładne wartości:")

        # Modyfikacja tabeli HTML aby pokazać obu zawodników
        col_widths = {'stat': '200px', 'bar1': '200px', 'value1': '80px'}
        if show_comparison:
            col_widths['bar2'] = '200px'
            col_widths['value2'] = '80px'

        table_html = '<table style="width:100%; border-collapse: collapse;">'
        table_html += '<tr><th style="width:{}; text-align: left; border: 1px solid black;">Statystyka</th>'.format(
            col_widths['stat'])
        table_html += '<th style="width:{}; text-align: left; border: 1px solid black;">{}</th>'.format(
            col_widths['bar1'], player)
        table_html += '<th style="width:{}; text-align: left; border: 1px solid black;">Wartość</th>'.format(
            col_widths['value1'])
        if show_comparison:
            table_html += '<th style="width:{}; text-align: left; border: 1px solid black;">{}</th>'.format(
                col_widths['bar2'], player2)
            table_html += '<th style="width:{}; text-align: left; border: 1px solid black;">Wartość</th>'.format(
                col_widths['value2'])
        table_html += '</tr>'

        for stat in radar_stats:
            percentile_col = stat + '_percentile'
            try:
                # Dane pierwszego zawodnika
                value1 = player_data[stat].iloc[0] if not player_data.empty else 0
                percentile1 = player_data[percentile_col].iloc[0] if not player_data.empty else 0
                color1 = sns.color_palette('RdYlGn', as_cmap=True)(percentile1 / 100)
                rgb_color1 = f'rgb({int(color1[0] * 255)}, {int(color1[1] * 255)}, {int(color1[2] * 255)})'

                bar_html1 = f'<div style="width:{col_widths["bar1"]}; background:lightgrey; border:1px solid black; height:20px; position:relative;">'
                bar_html1 += f'<div style="width:{percentile1}%; background:{rgb_color1}; height:100%;"></div></div>'

                # Rozpocznij wiersz tabeli
                table_html += f'<tr><td style="width:{col_widths["stat"]}; border: 1px solid black;">{stat}</td>'
                table_html += f'<td style="width:{col_widths["bar1"]}; border: 1px solid black;">{bar_html1}</td>'
                table_html += f'<td style="width:{col_widths["value1"]}; border: 1px solid black;">{value1}</td>'

                # Dodaj dane drugiego zawodnika jeśli wybrano porównanie
                if show_comparison:
                    value2 = player2_data[stat].iloc[0] if not player2_data.empty else 0
                    percentile2 = player2_data[percentile_col].iloc[0] if not player2_data.empty else 0
                    color2 = sns.color_palette('RdYlGn', as_cmap=True)(percentile2 / 100)
                    rgb_color2 = f'rgb({int(color2[0] * 255)}, {int(color2[1] * 255)}, {int(color2[2] * 255)})'

                    bar_html2 = f'<div style="width:{col_widths["bar2"]}; background:lightgrey; border:1px solid black; height:20px; position:relative;">'
                    bar_html2 += f'<div style="width:{percentile2}%; background:{rgb_color2}; height:100%;"></div></div>'

                    table_html += f'<td style="width:{col_widths["bar2"]}; border: 1px solid black;">{bar_html2}</td>'
                    table_html += f'<td style="width:{col_widths["value2"]}; border: 1px solid black;">{value2}</td>'

                table_html += '</tr>'

            except Exception:
                table_html += f'<tr><td style="border: 1px solid black;">{stat}</td>'
                table_html += '<td style="border: 1px solid black;">-</td><td style="border: 1px solid black;">-</td>'
                if show_comparison:
                    table_html += '<td style="border: 1px solid black;">-</td><td style="border: 1px solid black;">-</td>'
                table_html += '</tr>'

        table_html += '</table>'
        st.markdown(table_html, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
